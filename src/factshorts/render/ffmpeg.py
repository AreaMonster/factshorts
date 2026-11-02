"""ffmpeg/ffprobe wrapper: frame pipe, encoder detection, concat, mux and probing."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

HW_ENCODERS = {"nvenc": "h264_nvenc", "amf": "h264_amf", "qsv": "h264_qsv"}


class FFmpegError(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegError("ffmpeg not found on PATH (Windows: winget install Gyan.FFmpeg)")
    return path


def ffprobe_bin() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise FFmpegError("ffprobe not found on PATH (it ships with ffmpeg)")
    return path


def run(args: list[str], what: str) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.run(args, capture_output=True, check=False)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-8:]
        raise FFmpegError(f"{what} failed: " + " | ".join(tail))
    return proc


@lru_cache(maxsize=8)
def encoder_works(codec: str) -> bool:
    """Listed encoders can still fail (no GPU/driver), so try a real 1-frame encode."""
    proc = subprocess.run(
        [
            ffmpeg_bin(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=256x256:r=30:d=0.1",
            "-frames:v",
            "1",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def pick_encoder(preference: str = "auto") -> str:
    if preference == "libx264":
        return "libx264"
    if preference != "auto":
        codec = HW_ENCODERS[preference]
        if encoder_works(codec):
            return codec
        log.warning("encoder %s not working, falling back to libx264", codec)
        return "libx264"
    for codec in HW_ENCODERS.values():
        if encoder_works(codec):
            return codec
    return "libx264"


def video_codec_args(codec: str, quality: int, fps: int) -> list[str]:
    common = [
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(fps * 2),
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
    ]
    if codec == "h264_nvenc":
        specific = [
            "-preset",
            "p5",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(quality),
            "-b:v",
            "0",
            "-profile:v",
            "high",
        ]
    elif codec == "h264_amf":
        specific = [
            "-quality",
            "quality",
            "-rc",
            "cqp",
            "-qp_i",
            str(quality),
            "-qp_p",
            str(quality + 2),
            "-profile:v",
            "high",
        ]
    elif codec == "h264_qsv":
        specific = ["-preset", "slower", "-global_quality", str(quality), "-profile:v", "high"]
    else:
        specific = ["-preset", "medium", "-crf", str(quality), "-profile:v", "high"]
    return ["-c:v", codec, *specific, *common]


class FrameWriter:
    """Pipe raw RGB frames into ffmpeg, producing a video-only segment."""

    def __init__(self, out: Path, width: int, height: int, fps: int, codec: str, quality: int):
        self.out = out
        args = [
            ffmpeg_bin(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv",
            *video_codec_args(codec, quality, fps),
            "-an",
            str(out),
        ]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0

    def write(self, rgb: bytes) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(rgb)
        except BrokenPipeError as exc:
            raise FFmpegError(self._error()) from exc
        self.frames += 1

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise FFmpegError(self._error())

    def _error(self) -> str:
        assert self.proc.stderr is not None
        err = self.proc.stderr.read().decode("utf-8", "replace").strip()
        return f"ffmpeg encode of {self.out.name} failed: {err[-600:]}"


class FrameReader:
    """Decode a video to raw RGB frames, one at a time (repeats the last frame at the end)."""

    def __init__(self, path: Path, width: int, height: int) -> None:
        self.size = width * height * 3
        self.proc = subprocess.Popen(
            [
                ffmpeg_bin(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vf",
                f"scale={width}:{height}:in_color_matrix=bt709:in_range=tv",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            stdout=subprocess.PIPE,
        )
        self.last: bytes | None = None

    def read(self) -> bytes:
        assert self.proc.stdout is not None
        data = self.proc.stdout.read(self.size)
        if len(data) == self.size:
            self.last = data
        if self.last is None:
            raise FFmpegError("background clip has no frames")
        return self.last

    def close(self) -> None:
        if self.proc.stdout:
            self.proc.stdout.close()
        self.proc.kill()
        self.proc.wait()


def concat(segments: list[Path], out: Path) -> None:
    listing = out.with_suffix(".txt")
    listing.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in segments), encoding="utf-8"
    )
    run(
        [
            ffmpeg_bin(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(out),
        ],
        "concat",
    )


def mux(video: Path, audio: Path, out: Path, duration: float, audio_bitrate: str) -> None:
    """Add the mastered audio (padded/trimmed to the video length) to the video."""
    run(
        [
            ffmpeg_bin(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-af",
            "apad",
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            f"{duration:.3f}",
            "-movflags",
            "+faststart",
            str(out),
        ],
        "mux",
    )


def probe(path: Path) -> dict[str, Any]:
    proc = run(
        [ffprobe_bin(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        "ffprobe",
    )
    return json.loads(proc.stdout)


def stream_durations(path: Path) -> dict[str, float]:
    """Duration per stream type ('video', 'audio') and of the container ('format')."""
    info = probe(path)
    out: dict[str, float] = {"format": float(info["format"].get("duration", 0))}
    for s in info["streams"]:
        if s["codec_type"] in ("video", "audio") and "duration" in s:
            out.setdefault(s["codec_type"], float(s["duration"]))
    return out
