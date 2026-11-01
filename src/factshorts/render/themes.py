"""Themes: colors, fonts and animation speed, loaded from assets/themes/*.toml."""

from __future__ import annotations

import contextlib
import tomllib
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont
from pydantic import BaseModel, Field

from factshorts.config import ASSETS_DIR

RGB = tuple[int, int, int]


def hex_to_rgb(value: str) -> RGB:
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


class ThemeColors(BaseModel):
    background_top: str = "#0f1b3d"
    background_bottom: str = "#3a0f4d"
    background_accent: str = "#1d4f91"
    text: str = "#ffffff"
    accent: str = "#ffd23f"
    caption: str = "#ffffff"
    caption_stroke: str = "#000000"
    muted: str = "#c9c9e0"
    particle: str = "#ffffff"

    def rgb(self, name: str) -> RGB:
        return hex_to_rgb(getattr(self, name))


class ThemeFonts(BaseModel):
    display: str = "BebasNeue-Regular.ttf"
    display_weight: int = 0
    body: str = "Montserrat-Variable.ttf"
    body_weight: int = 800
    small_weight: int = 600


class ThemeAnimation(BaseModel):
    speed: float = 1.0
    particles: int = 36


class ThemeArt(BaseModel):
    style_name: str = "a flat 2D vector illustration"
    style: str = (
        "flat 2D vector illustration, bold clean shapes, soft gradients, vibrant colors, "
        "educational animation style, cel shaded, no text, no letters, no watermark"
    )


class Theme(BaseModel):
    name: str = "default"
    colors: ThemeColors = Field(default_factory=ThemeColors)
    fonts: ThemeFonts = Field(default_factory=ThemeFonts)
    animation: ThemeAnimation = Field(default_factory=ThemeAnimation)
    art: ThemeArt = Field(default_factory=ThemeArt)

    def display_font(self, size: int) -> ImageFont.FreeTypeFont:
        return load_font(self.fonts.display, size, self.fonts.display_weight)

    def body_font(self, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
        return load_font(self.fonts.body, size, weight or self.fonts.body_weight)


def themes_dir() -> Path:
    return ASSETS_DIR / "themes"


def list_themes() -> list[str]:
    return sorted(p.stem for p in themes_dir().glob("*.toml"))


def load_theme(name: str) -> Theme:
    path = themes_dir() / f"{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"theme '{name}' not found; available: {', '.join(list_themes())}")
    with path.open("rb") as fh:
        return Theme.model_validate(tomllib.load(fh))


@lru_cache(maxsize=64)
def load_font(file: str, size: int, weight: int = 0) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(ASSETS_DIR / "fonts" / file), size)
    if weight:
        with contextlib.suppress(OSError):  # not a variable font
            font.set_variation_by_axes([weight])
    return font
