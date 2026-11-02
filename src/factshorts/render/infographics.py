"""Animated infographics per scene type, drawn as RGBA layers over the scene's picture.

Data comes from the scene's own (fact-checked) on-screen text and narration: numbers for
comparisons, years for timelines, comma-separated items for lists."""

from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image, ImageDraw

from factshorts.render.easing import ease_out_back, ease_out_cubic, progress
from factshorts.render.layout import wrap
from factshorts.render.themes import Theme
from factshorts.verify.normalize import Quantity, normalize_chars, parse_quantities


@dataclass
class Infographic:
    kind: str
    values: list[Quantity]
    items: list[str]
    caption: str = ""


def plan_infographic(visual: str, on_screen_text: str, narration: str) -> Infographic | None:
    """Decide what to draw for a scene; None when its data does not fit the scene type."""
    combined = f"{on_screen_text}. {narration}"
    numbers = [q for q in parse_quantities(combined) if not q.is_year]
    years = sorted(
        {int(q.value) for q in parse_quantities(f"{on_screen_text}. {narration}") if q.is_year}
    )
    if visual == "comparison":
        pair = _comparable_pair(numbers)
        if pair:
            labels = [_label_before(normalize_chars(combined), q) for q in pair]
            return Infographic("comparison", pair, labels)
    if visual == "timeline" and years:
        return Infographic("timeline", [], [str(y) for y in years[:4]])
    if visual == "icon_list":
        items = [
            i.strip(" .") for i in re.split(r",|;|\band\b|&|\+", on_screen_text) if i.strip(" .")
        ]
        if 2 <= len(items) <= 5:
            return Infographic("list", [], [i[0].upper() + i[1:] for i in items])
    if visual == "quote" and on_screen_text.strip():
        return Infographic("quote", [], [on_screen_text.strip()])
    if visual == "map_pin":
        return Infographic("pin", [], [])
    return None


def _label_before(text: str, q: Quantity) -> str:
    """A capitalized word right before the number names what it belongs to ('Venus 464 °C')."""
    words = re.findall(r"[\w\u00C0-\u024F'-]+", text[: q.start])
    if words and words[-1][:1].isupper() and words[-1].lower() not in {"vs", "and", "the"}:
        return f"{words[-1]} {q.raw}"
    return q.raw


def _comparable_pair(numbers: list[Quantity]) -> list[Quantity] | None:
    """Two numbers in the same unit (or both unitless), largest difference first."""
    best = None
    for i, a in enumerate(numbers):
        for b in numbers[i + 1 :]:
            if a.unit == b.unit and a.value > 0 and b.value > 0 and a.value != b.value:
                best = best or [a, b]
    return best


def _rounded(draw: ImageDraw.ImageDraw, box: tuple, r: int, fill: tuple) -> None:
    x0, _, x1, _ = box
    if x1 - x0 < 2 * r:
        r = max(1, int((x1 - x0) / 2))
    draw.rounded_rectangle(box, radius=r, fill=fill)


def panel(width: int, height: int, s: float) -> Image.Image:
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=int(36 * s), fill=(12, 14, 24, 165)
    )
    return layer


def draw_infographic(
    info: Infographic, theme: Theme, t: float, width: int, s: float, source_label: str = ""
) -> Image.Image:
    """Layer for local time `t` (seconds since the scene started)."""
    accent = (*theme.colors.rgb("accent"), 255)
    text = (*theme.colors.rgb("text"), 255)
    muted = (*theme.colors.rgb("muted"), 255)
    body = theme.body_font(int(46 * s))
    small = theme.body_font(int(34 * s), theme.fonts.small_weight)
    big = theme.display_font(int(96 * s))

    if info.kind == "comparison":
        a, b = info.values
        layer = panel(width, int(330 * s), s)
        d = ImageDraw.Draw(layer)
        top = max(a.value, b.value)
        for i, q in enumerate((a, b)):
            p = ease_out_cubic(progress(t, 0.35 + 0.25 * i, 0.9))
            y = int((60 + 130 * i) * s)
            full = width - int(80 * s)
            length = max(int(18 * s), int(full * (q.value / top) * p))
            _rounded(
                d,
                (int(40 * s), y + int(56 * s), int(40 * s) + length, y + int(104 * s)),
                int(24 * s),
                accent if i == 0 else text,
            )
            label = info.items[i] if info.items else q.raw
            d.text((int(40 * s), y), label, font=body, fill=text)
        return layer

    if info.kind == "timeline":
        layer = panel(width, int(260 * s), s)
        d = ImageDraw.Draw(layer)
        half = max(big.getlength(year) for year in info.items) / 2
        x0, x1, y = int(40 * s + half), int(width - 40 * s - half), int(150 * s)
        p = ease_out_cubic(progress(t, 0.3, 1.0))
        d.line((x0, y, x0 + (x1 - x0) * p, y), fill=muted, width=max(3, int(6 * s)))
        n = len(info.items)
        for i, year in enumerate(info.items):
            x = x0 + (x1 - x0) * (0.5 if n == 1 else i / (n - 1))
            pop = ease_out_back(progress(t, 0.5 + 0.35 * i, 0.45))
            if pop <= 0:
                continue
            r = int(18 * s * pop)
            d.ellipse((x - r, y - r, x + r, y + r), fill=accent)
            w = big.getlength(year)
            d.text((x - w / 2, y - int(140 * s)), year, font=big, fill=text)
        return layer

    if info.kind == "list":
        rows = len(info.items)
        layer = panel(width, int((70 + 90 * rows) * s), s)
        d = ImageDraw.Draw(layer)
        for i, item in enumerate(info.items):
            p = ease_out_cubic(progress(t, 0.35 + 0.4 * i, 0.4))
            if p <= 0:
                continue
            y = int((40 + 90 * i) * s)
            x = int(40 * s) - int((1 - p) * 60 * s)
            c = int(22 * s)
            d.ellipse((x, y + int(8 * s), x + 2 * c, y + int(8 * s) + 2 * c), fill=accent)
            d.line(
                (
                    x + c * 0.55,
                    y + int(8 * s) + c * 1.05,
                    x + c * 0.9,
                    y + int(8 * s) + c * 1.4,
                    x + c * 1.5,
                    y + int(8 * s) + c * 0.6,
                ),
                fill=(20, 20, 30, 255),
                width=max(3, int(5 * s)),
            )
            d.text((x + 2 * c + int(24 * s), y), item, font=body, fill=(*text[:3], int(255 * p)))
        return layer

    if info.kind == "quote":
        qfont = theme.display_font(int(84 * s))
        lines = wrap(info.items[0], qfont, width - int(150 * s))[:3]
        line_h = int(sum(qfont.getmetrics()) * 0.95)
        layer = panel(width, int(130 * s) + line_h * len(lines), s)
        d = ImageDraw.Draw(layer)
        d.text((int(30 * s), int(-40 * s)), "“", font=theme.display_font(int(200 * s)), fill=accent)
        appear = ease_out_cubic(progress(t, 0.3, 0.6))
        for i, line in enumerate(lines):
            d.text(
                (int(110 * s), int(40 * s) + i * line_h),
                line,
                font=qfont,
                fill=(*text[:3], int(255 * appear)),
            )
        if source_label:
            label = "— " + source_label.removeprefix("Source: ")
            w = small.getlength(label)
            d.text(
                (width - w - int(40 * s), layer.height - int(70 * s)), label, font=small, fill=muted
            )
        return layer

    # pin: a location pin that drops in and pulses
    size = int(260 * s)
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    drop = ease_out_back(progress(t, 0.3, 0.6))
    if drop > 0:
        cx, cy = size / 2, size * 0.42 - (1 - drop) * 80 * s
        pulse = (t * 1.2) % 1.0
        rr = size * 0.18 + size * 0.25 * pulse
        d.ellipse(
            (cx - rr, size * 0.86 - rr * 0.35, cx + rr, size * 0.86 + rr * 0.35),
            outline=(*accent[:3], int(200 * (1 - pulse))),
            width=max(2, int(4 * s)),
        )
        r = size * 0.2
        d.polygon(
            [(cx - r * 0.8, cy + r * 0.5), (cx + r * 0.8, cy + r * 0.5), (cx, size * 0.86)],
            fill=accent,
        )
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=accent)
        d.ellipse((cx - r * 0.4, cy - r * 0.4, cx + r * 0.4, cy + r * 0.4), fill=(20, 20, 30, 255))
    return layer


def fade(layer: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 0.99:
        return layer
    out = layer.copy()
    out.putalpha(layer.getchannel("A").point(lambda a: int(a * max(0.0, alpha))))
    return out
