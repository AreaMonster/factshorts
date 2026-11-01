"""Render scenes to frames in parallel worker processes, concat, then mux with narration."""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from factshorts.config import RenderConfig
from factshorts.core.captions import caption_chunks
from factshorts.models import AudioPlan, RenderResult, Script, WordTiming
from factshorts.render import ffmpeg
from factshorts.render.background import Background
from factshorts.render.easing import ease_out_back, ease_out_cubic, progress
from factshorts.render.layout import SafeZone, fit_text
from factshorts.render.themes import Theme, load_theme
from factshorts.render.timeline import segment_frame_ranges, total_frames

log = logging.getLogger(__name__)

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class SegmentSpec:
    segment_id: str
    role: str
    visual: str
    on_screen_text: str
    start_frame: int
    end_frame: int
    words: list[dict[str, float | str]] = field(default_factory=list)
    background: str | None = None  # AI background clip with exactly this segment's frames


@dataclass
class RenderSpec:
    width: int
    height: int
    fps: int
    theme: str
    codec: str
    quality: int
    total_frames: int
    title: str
    source_label: str
    segment: SegmentSpec
    out: str


# --- drawing helpers ---------------------------------------------------------------


def _text_layer(
    lines: list[str],
    font,
    line_h: int,
    fill: tuple[int, int, int],
    shadow: bool = True,
    stroke: int = 0,
    stroke_fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    widths = [font.getlength(line) for line in lines]
    pad = 30
    w = int(max(widths)) + pad * 2
    h = line_h * len(lines) + pad * 2
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for i, line in enumerate(lines):
        x = (w - widths[i]) / 2
        y = pad + i * line_h
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(*fill, 255),
            stroke_width=stroke,
            stroke_fill=(*stroke_fill, 255),
        )
    if not shadow:
        return layer
    shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    alpha = layer.getchannel("A").filter(ImageFilter.GaussianBlur(10))
    shadow_layer.putalpha(alpha.point(lambda a: int(a * 0.6)))
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.alpha_composite(shadow_layer, (0, 6))
    out.alpha_composite(layer)
    return out


def _paste(
    frame: Image.Image,
    layer: Image.Image,
    cx: float,
    cy: float,
    alpha: float = 1.0,
    scale: float = 1.0,
) -> None:
    if alpha <= 0.01:
        return
    if abs(scale - 1.0) > 0.005:
        w, h = max(1, int(layer.width * scale)), max(1, int(layer.height * scale))
        layer = layer.resize((w, h), Image.Resampling.BILINEAR)
    if alpha < 0.99:
        layer = layer.copy()
        layer.putalpha(layer.getchannel("A").point(lambda a: int(a * alpha)))
    frame.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)), layer)


def _scrim(width: int, height: int, s: float) -> np.ndarray:
    """Darkening mask so white text stays readable on bright illustrations:
    a fade at the top (title text) and a soft band behind the captions."""
    y = np.arange(height, dtype=np.float32)
    top = 0.50 * np.clip(1 - y / (760 * s), 0, 1) ** 1.4
    cap_center = height * 0.8 - 250 * s
    band = 0.38 * np.exp(-(((y - cap_center) / (210 * s)) ** 2))
    dark = np.clip(top + band, 0, 0.6)
    return (1 - dark)[:, None, None]


class SegmentPainter:
    def __init__(self, spec: RenderSpec) -> None:
        self.spec = spec
        self.seg = spec.segment
        self.theme: Theme = load_theme(spec.theme)
        self.zone = SafeZone.for_size(spec.width, spec.height)
        self.s = spec.width / 1080
        self.clip: ffmpeg.FrameReader | None = None
        if self.seg.background:
            self.clip = ffmpeg.FrameReader(Path(self.seg.background), spec.width, spec.height)
            self.scrim = _scrim(spec.width, spec.height, self.s)
        else:
            self.bg = Background(self.theme, spec.width, spec.height)
        self.speed = self.theme.animation.speed
        self.words = [WordTiming(**w) for w in self.seg.words]  # type: ignore[arg-type]
        self.chunks = caption_chunks(self.words)
        self._caption_cache: dict[tuple[int, int], Image.Image] = {}
        c = self.theme.colors
        self.text_rgb, self.accent_rgb = c.rgb("text"), c.rgb("accent")
        top = c.rgb("background_top")
        self.track_rgb = tuple(int(v * 0.7 + 255 * 0.3) for v in top)
        self.main_layer = self._main_layer()
        self.number_match = NUMBER_RE.search(self.seg.on_screen_text)
        self.extra = self._end_card() if self.seg.role == "outro" else None

    # main on-screen text
    def _main_layer(self, text: str | None = None) -> Image.Image | None:
        text = (text if text is not None else self.seg.on_screen_text).strip()
        if not text:
            return None
        big = self.seg.role in ("hook", "outro") or self.seg.visual == "big_number"
        # Over illustrations the text sits higher and smaller so the picture stays visible.
        on_image = self.clip is not None
        size = (150 if big else 120) if on_image else (190 if big else 150)
        font, lines, line_h = fit_text(
            text.upper(),
            self.theme.display_font,
            max_width=self.zone.inner_width,
            max_height=int((400 if on_image else 560) * self.s),
            max_size=int(size * self.s),
            min_size=int(60 * self.s),
            max_lines=4,
            line_spacing=0.95,
        )
        color = self.accent_rgb if self.seg.visual == "big_number" else self.text_rgb
        return _text_layer(lines, font, line_h, color)

    def _end_card(self) -> Image.Image:
        font = self.theme.body_font(int(34 * self.s), self.theme.fonts.small_weight)
        lines = [self.spec.source_label, "Text: CC BY-SA 4.0"]
        line_h = int(sum(font.getmetrics()) * 1.3)
        return _text_layer(lines, font, line_h, self.theme.colors.rgb("muted"), shadow=False)

    def _caption_layer(self, chunk_i: int, active: int) -> Image.Image:
        key = (chunk_i, active)
        if key in self._caption_cache:
            return self._caption_cache[key]
        words = [w.text for w in self.chunks[chunk_i]]
        size = int(66 * self.s)
        font = self.theme.body_font(size)
        while font.getlength(" ".join(words)) > self.zone.inner_width and size > 36:
            size -= 4
            font = self.theme.body_font(size)
        stroke = max(3, int(size * 0.11))
        space = font.getlength(" ")
        total = sum(font.getlength(w) for w in words) + space * (len(words) - 1)
        pad = 40
        h = int(sum(font.getmetrics())) + pad * 2
        layer = Image.new("RGBA", (int(total) + pad * 2, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        x = pad
        for i, w in enumerate(words):
            wl = font.getlength(w)
            if i == active:
                draw.rounded_rectangle(
                    (x - 12, pad - 6, x + wl + 12, h - pad + 10),
                    radius=18,
                    fill=(*self.accent_rgb, 235),
                )
                draw.text((x, pad), w, font=font, fill=(20, 20, 30, 255))
            else:
                draw.text(
                    (x, pad),
                    w,
                    font=font,
                    fill=(*self.theme.colors.rgb("caption"), 255),
                    stroke_width=stroke,
                    stroke_fill=(*self.theme.colors.rgb("caption_stroke"), 255),
                )
            x += wl + space
        self._caption_cache[key] = layer
        return layer

    def paint(self, frame_index: int) -> bytes:
        spec, seg = self.spec, self.seg
        t = frame_index / spec.fps
        local = (frame_index - seg.start_frame) / spec.fps
        seg_len = (seg.end_frame - seg.start_frame) / spec.fps
        if self.clip is not None:
            raw = np.frombuffer(self.clip.read(), dtype=np.uint8)
            raw = raw.reshape(spec.height, spec.width, 3)
            frame = Image.fromarray((raw * self.scrim).astype(np.uint8), "RGB")
        else:
            frame = Image.fromarray(self.bg.frame(t), "RGB")
        draw = ImageDraw.Draw(frame)

        # progress bar
        bar_y, bar_h = int(48 * self.s), int(10 * self.s)
        x0, x1 = self.zone.left, spec.width - self.zone.left
        draw.rounded_rectangle(
            (x0, bar_y, x1, bar_y + bar_h), radius=bar_h // 2, fill=self.track_rgb
        )
        fill_x = x0 + (x1 - x0) * min(1.0, (frame_index + 1) / spec.total_frames)
        draw.rounded_rectangle(
            (x0, bar_y, fill_x, bar_y + bar_h), radius=bar_h // 2, fill=self.accent_rgb
        )

        # main text: slide/scale/fade in, fade out at the end
        intro = 0.4 / self.speed
        p_in = ease_out_cubic(progress(local, 0.0, intro))
        p_out = 1 - progress(local, seg_len - 0.22, 0.22)
        layer = self.main_layer
        if seg.visual == "big_number" and self.number_match and local < 0.9:
            layer = self._count_up_layer(progress(local, 0.0, 0.9))
        if layer is not None:
            base_y = 330 if self.clip is not None else 560
            cy = self.zone.top + int(base_y * self.s) - (1 - p_in) * 70 * self.s
            scale = 0.9 + 0.1 * ease_out_back(progress(local, 0.0, intro))
            _paste(frame, layer, self.zone.center_x, cy, alpha=min(p_in, p_out), scale=scale)
        if self.extra is not None:
            _paste(
                frame,
                self.extra,
                self.zone.center_x,
                self.zone.top + int(900 * self.s),
                alpha=min(ease_out_cubic(progress(local, 0.3, 0.5)), p_out),
            )

        # captions (word-by-word highlight) in the lower safe area
        for ci, chunk in enumerate(self.chunks):
            if chunk[0].start <= t < self._chunk_end(ci):
                active = next((i for i, w in enumerate(chunk) if w.start <= t < w.end), -1)
                if active < 0:
                    active = len(chunk) - 1
                pop = 0.92 + 0.08 * ease_out_back(progress(t, chunk[0].start, 0.12))
                cap = self._caption_layer(ci, active)
                _paste(
                    frame, cap, self.zone.center_x, self.zone.bottom - int(250 * self.s), scale=pop
                )
                break
        return frame.tobytes()

    def _chunk_end(self, ci: int) -> float:
        end = self.chunks[ci][-1].end
        if ci + 1 < len(self.chunks):
            return min(self.chunks[ci + 1][0].start, end + 0.35)
        return end + 0.35

    def _count_up_layer(self, p: float) -> Image.Image | None:
        assert self.number_match is not None
        raw = self.number_match.group(0)
        target = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        value = target * ease_out_cubic(p)
        shown = f"{value:,.{decimals}f}" if "," in raw else f"{value:.{decimals}f}"
        text = self.seg.on_screen_text.replace(raw, shown, 1)
        return self._main_layer(text)


def render_segment(spec_dict: dict) -> str:
    seg = SegmentSpec(**spec_dict.pop("segment"))
    spec = RenderSpec(segment=seg, **spec_dict)
    painter = SegmentPainter(spec)
    writer = ffmpeg.FrameWriter(
        Path(spec.out), spec.width, spec.height, spec.fps, spec.codec, spec.quality
    )
    try:
        for i in range(seg.start_frame, seg.end_frame):
            writer.write(painter.paint(i))
    finally:
        writer.close()
        if painter.clip is not None:
            painter.clip.close()
    return spec.out


# --- orchestration ---------------------------------------------------------------


def build_specs(
    script: Script,
    audio: AudioPlan,
    cfg: RenderConfig,
    codec: str,
    out_dir: Path,
    source_label: str,
    theme: str | None = None,
    backgrounds: dict[str, str] | None = None,
) -> list[RenderSpec]:
    total = total_frames(audio, cfg.fps)
    ranges = segment_frame_ranges(audio, cfg.fps)
    by_id = {s.id: s for s in script.segments}
    specs: list[RenderSpec] = []
    for i, sa in enumerate(audio.segments):
        seg = by_id[sa.segment_id]
        start, end = ranges[sa.segment_id]
        words = [w.model_dump() for s in sa.sentences for w in s.words]
        specs.append(
            RenderSpec(
                width=cfg.width,
                height=cfg.height,
                fps=cfg.fps,
                theme=theme or cfg.theme,
                codec=codec,
                quality=cfg.video_quality,
                total_frames=total,
                title=script.title,
                source_label=source_label,
                segment=SegmentSpec(
                    segment_id=seg.id,
                    role=seg.role,
                    visual=seg.visual,
                    on_screen_text=seg.on_screen_text,
                    start_frame=start,
                    end_frame=end,
                    words=words,
                    background=(backgrounds or {}).get(seg.id),
                ),
                out=str(out_dir / f"{i:02d}_{seg.id}.mp4"),
            )
        )
    return specs


def worker_count(cfg: RenderConfig, codec: str, n_segments: int) -> int:
    n = cfg.workers or max(1, (os.cpu_count() or 2) // 2)
    if codec != "libx264":
        n = min(n, cfg.max_hw_sessions)
    return max(1, min(n, n_segments))


def render_video(
    script: Script,
    audio: AudioPlan,
    audio_dir: Path,
    render_dir: Path,
    cfg: RenderConfig,
    source_label: str,
    theme: str | None = None,
    backgrounds: dict[str, str] | None = None,
) -> RenderResult:
    render_dir.mkdir(parents=True, exist_ok=True)
    codec = ffmpeg.pick_encoder(cfg.encoder)
    specs = build_specs(script, audio, cfg, codec, render_dir, source_label, theme, backgrounds)
    workers = worker_count(cfg, codec, len(specs))
    log.info(
        "rendering %d segments, %d frames, encoder %s, %d workers",
        len(specs),
        specs[0].total_frames,
        codec,
        workers,
    )
    payloads = [asdict(s) for s in specs]
    if workers == 1:
        outs = [render_segment(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outs = list(pool.map(render_segment, payloads))
    video_only = render_dir / "video_only.mp4"
    ffmpeg.concat([Path(o) for o in outs], video_only)
    final = render_dir / "final.mp4"
    duration = specs[0].total_frames / cfg.fps
    ffmpeg.mux(
        video_only, audio_dir / audio.wav, final, duration, cfg.lufs_target, cfg.audio_bitrate
    )
    return RenderResult(
        video=str(final), duration=duration, frames=specs[0].total_frames, encoder=codec
    )
