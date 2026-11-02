"""Thumbnail: the opening illustration (or a video frame) with the title in big letters."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from factshorts.render.backgrounds import cover
from factshorts.render.ffmpeg import ffmpeg_bin
from factshorts.render.layout import SafeZone, fit_text
from factshorts.render.themes import Theme


def frame_at(video: Path, t: float) -> Image.Image:
    proc = subprocess.run(
        [
            ffmpeg_bin(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t:.2f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    from io import BytesIO

    return Image.open(BytesIO(proc.stdout)).convert("RGB")


def make_thumbnail(
    base: Image.Image, title: str, theme: Theme, out: Path, width: int = 1080, height: int = 1920
) -> Path:
    img = cover(base, width, height)
    # Darken the upper half so the title pops, keep the picture visible below.
    arr = np.asarray(img).astype(np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    arr *= 1 - 0.6 * np.clip(1 - y / 0.6, 0, 1) ** 1.2
    img = Image.fromarray(arr.astype(np.uint8))
    zone = SafeZone.for_size(width, height)
    s = width / 1080
    font, lines, line_h = fit_text(
        title.upper(),
        theme.display_font,
        max_width=zone.inner_width,
        max_height=int(700 * s),
        max_size=int(230 * s),
        min_size=int(90 * s),
        max_lines=4,
        line_spacing=0.95,
    )
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    y0 = zone.top + int(120 * s)
    accent = theme.colors.rgb("accent")
    for i, line in enumerate(lines):
        x = zone.center_x - font.getlength(line) / 2
        color = accent if i == len(lines) - 1 else theme.colors.rgb("text")
        draw.text(
            (x, y0 + i * line_h),
            line,
            font=font,
            fill=(*color, 255),
            stroke_width=max(4, int(8 * s)),
            stroke_fill=(0, 0, 0, 255),
        )
    shadow = layer.getchannel("A").filter(ImageFilter.GaussianBlur(14))
    shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shade.putalpha(shadow.point(lambda a: int(a * 0.55)))
    result = img.convert("RGBA")
    result.alpha_composite(shade, (0, int(8 * s)))
    result.alpha_composite(layer)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(out, "JPEG", quality=90)
    return out
