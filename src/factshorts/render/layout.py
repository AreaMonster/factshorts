"""Vertical-video safe zone and text fitting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PIL import ImageFont

FontFactory = Callable[[int], ImageFont.FreeTypeFont]


@dataclass(frozen=True)
class SafeZone:
    """Area not covered by TikTok/Shorts/Reels UI (bottom ~20 %, right-edge buttons)."""

    width: int = 1080
    height: int = 1920
    left: int = 70
    right_margin: int = 160
    top: int = 130
    bottom_fraction: float = 0.20

    @property
    def right(self) -> int:
        return self.width - self.right_margin

    @property
    def bottom(self) -> int:
        return int(self.height * (1 - self.bottom_fraction))

    @property
    def inner_width(self) -> int:
        return self.right - self.left

    @property
    def center_x(self) -> int:
        return (self.left + self.right) // 2

    def contains(self, box: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = box
        return x0 >= self.left and x1 <= self.right and y0 >= self.top and y1 <= self.bottom

    @classmethod
    def for_size(cls, width: int, height: int) -> SafeZone:
        s = width / 1080
        return cls(
            width=width,
            height=height,
            left=int(70 * s),
            right_margin=int(160 * s),
            top=int(130 * s),
        )


def text_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    return font.getlength(text)


def wrap(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and font.getlength(trial) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def fit_text(
    text: str,
    font_factory: FontFactory,
    max_width: float,
    max_height: float,
    max_size: int,
    min_size: int = 24,
    max_lines: int = 4,
    line_spacing: float = 1.05,
) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """Largest font size at which `text` wraps into the box. Returns (font, lines, line_height)."""
    size = max_size
    while True:
        font = font_factory(size)
        lines = wrap(text, font, max_width)
        ascent, descent = font.getmetrics()
        line_h = int((ascent + descent) * line_spacing)
        fits = (
            len(lines) <= max_lines
            and line_h * len(lines) <= max_height
            and all(font.getlength(line) <= max_width for line in lines)
        )
        if fits or size <= min_size:
            return font, lines, line_h
        size = max(min_size, int(size * 0.92))
