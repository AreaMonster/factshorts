"""Final audio: narration + background music (ducked under the voice), two-pass loudness."""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from factshorts.config import ASSETS_DIR, AudioConfig
from factshorts.render.ffmpeg import ffmpeg_bin, run

log = logging.getLogger(__name__)

SR = 48000
MUSIC_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

# Chord progressions as semitone offsets from the key root (major triads, relative minors).
PROGRESSIONS = [
    [[0, 4, 7], [-3, 0, 4], [-7, -3, 0], [-5, -1, 2]],  # I vi IV V
    [[-3, 0, 4], [-7, -3, 0], [0, 4, 7], [-5, -1, 2]],  # vi IV I V
    [[0, 4, 7], [-5, -1, 2], [-3, 0, 4], [-7, -3, 0]],  # I V vi IV
]
ROOTS_HZ = [110.0, 123.47, 130.81, 146.83, 164.81]  # A2 B2 C3 D3 E3


def ambient_pad(duration: float, seed: int, sr: int = SR) -> np.ndarray:
    """A calm, original pad (stereo float32): slow chords of detuned soft tones plus air."""
    rng = np.random.default_rng(seed)
    root = ROOTS_HZ[seed % len(ROOTS_HZ)]
    progression = PROGRESSIONS[(seed // 7) % len(PROGRESSIONS)]
    chord_s, fade_s = 6.0, 2.0
    n = int(duration * sr) + 1
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros((n, 2), dtype=np.float32)
    n_chords = math.ceil(duration / chord_s) + 1
    for c in range(n_chords):
        start = c * chord_s - fade_s / 2
        env = np.clip((t - start) / fade_s, 0, 1) * np.clip(
            (start + chord_s + fade_s - t) / fade_s, 0, 1
        )
        if not env.any():
            continue
        env = (env**2).astype(np.float32)
        for k, semis in enumerate(progression[c % len(progression)]):
            f = root * 2 ** (semis / 12)
            for detune, pan in ((-0.004, 0.25), (0.004, 0.75)):
                phase = rng.uniform(0, 2 * np.pi)
                tone = np.sin(2 * np.pi * f * (1 + detune) * t + phase)
                tone += 0.25 * np.sin(2 * np.pi * 2 * f * (1 + detune) * t + phase)  # octave
                amp = 0.06 / (1 + 0.3 * k)
                out[:, 0] += amp * (1 - pan) * env * tone
                out[:, 1] += amp * pan * env * tone
    # Soft filtered noise for "air", and a slow swell so it never feels static.
    noise = rng.normal(0, 1, (n, 2)).astype(np.float32)
    kernel = np.ones(64, dtype=np.float32) / 64
    for ch in range(2):
        noise[:, ch] = np.convolve(noise[:, ch], kernel, mode="same")
    swell = 0.85 + 0.15 * np.sin(2 * np.pi * t / 23.0)
    out = (out + 0.015 * noise) * swell[:, None]
    fade = np.clip(np.minimum(t / 1.5, (duration - t) / 2.0), 0, 1)[:, None]
    out *= fade
    peak = float(np.abs(out).max()) or 1.0
    return (out / peak * 0.5).astype(np.float32)


def music_files() -> list[Path]:
    folder = ASSETS_DIR / "music"
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in MUSIC_EXTS)


def decode_looped(path: Path, duration: float) -> np.ndarray:
    """Decode a music file to stereo float32 at 48 kHz, looped to `duration`."""
    proc = subprocess.run(
        [
            ffmpeg_bin(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-stream_loop",
            "-1",
            "-i",
            str(path),
            "-t",
            f"{duration:.3f}",
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            str(SR),
            "-",
        ],
        capture_output=True,
        check=True,
    )
    data = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).copy()
    t = np.arange(len(data)) / SR
    data *= np.clip(np.minimum(t / 1.5, (duration - t) / 2.0), 0, 1)[:, None]
    return data


def write_wav(path: Path, data: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def make_music(out: Path, duration: float, seed: int) -> str:
    """Write the background music bed; returns where it came from (for the credits)."""
    files = music_files()
    if files:
        path = files[seed % len(files)]
        write_wav(out, decode_looped(path, duration))
        return f"music: {path.name} (assets/music)"
    write_wav(out, ambient_pad(duration, seed))
    return "music: generated ambient pad (original, no copyright)"


def _measure(args: list[str]) -> dict:
    proc = subprocess.run(args, capture_output=True, check=False)
    text = proc.stderr.decode("utf-8", "replace")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.S)
    if proc.returncode != 0 or not m:
        raise RuntimeError("loudness measurement failed: " + text[-400:])
    return json.loads(m.group(0))


def master(
    narration: Path,
    out: Path,
    duration: float,
    lufs: float,
    cfg: AudioConfig,
    music: Path | None = None,
) -> dict:
    """Narration (+ ducked music) → stereo 48 kHz WAV at `lufs` (two-pass loudnorm).

    Returns the measured loudness of the result."""
    inputs = ["-i", str(narration)]
    voice = (
        f"[0:a]aresample={SR},pan=stereo|c0=c0|c1=c0,apad,atrim=0:{duration:.3f},asplit=2[v1][v2]"
    )
    if music is not None:
        inputs += ["-i", str(music)]
        graph = (
            f"{voice};"
            f"[1:a]aresample={SR},volume={cfg.music_volume_db}dB,apad,atrim=0:{duration:.3f}[m];"
            f"[m][v2]sidechaincompress=threshold={cfg.duck_threshold}:ratio={cfg.duck_ratio}:"
            "attack=20:release=450[ducked];"
            "[v1][ducked]amix=inputs=2:normalize=0[mix]"
        )
    else:
        graph = f"{voice};[v2]anullsink;[v1]anull[mix]"
    base = [ffmpeg_bin(), "-hide_banner", "-nostats", *inputs]
    target = f"I={lufs}:TP=-1.5:LRA=11"
    m = _measure(
        [
            *base,
            "-filter_complex",
            f"{graph};[mix]loudnorm={target}:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    second = (
        f"{graph};[mix]loudnorm={target}:measured_I={m['input_i']}:"
        f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
        f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,"
        f"aresample={SR}[out]"
    )
    run(
        [
            *base[:2],
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            second,
            "-map",
            "[out]",
            "-ac",
            "2",
            "-ar",
            str(SR),
            "-c:a",
            "pcm_s16le",
            str(out),
        ],
        "audio mastering",
    )
    return m


def loudness(path: Path) -> float:
    """Integrated loudness (LUFS) of a file, via ffmpeg's EBU R128 meter."""
    proc = subprocess.run(
        [
            ffmpeg_bin(),
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", proc.stderr.decode("utf-8", "replace"))
    return float(found[-1]) if found else float("nan")
