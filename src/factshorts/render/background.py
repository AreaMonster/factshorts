"""Animated background: vertical gradient, drifting glow and floating particles (numpy)."""

from __future__ import annotations

import math

import numpy as np

from factshorts.render.themes import Theme


def _radial_sprite(radius: int, color: tuple[int, int, int], strength: float) -> np.ndarray:
    y, x = np.mgrid[-radius:radius, -radius:radius].astype(np.float32)
    d = np.sqrt(x * x + y * y) / radius
    falloff = np.clip(1 - d, 0, 1) ** 2 * strength
    return falloff[..., None] * np.array(color, dtype=np.float32)[None, None, :]


class Background:
    def __init__(self, theme: Theme, width: int, height: int, seed: int = 7) -> None:
        self.w, self.h = width, height
        top = np.array(theme.colors.rgb("background_top"), dtype=np.float32)
        bottom = np.array(theme.colors.rgb("background_bottom"), dtype=np.float32)
        ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
        self.base = top * (1 - ramp) + bottom * ramp
        self.base = np.broadcast_to(self.base, (height, width, 3)).copy()
        self.glow_r = int(width * 0.75)
        self.glow = _radial_sprite(self.glow_r, theme.colors.rgb("background_accent"), 0.85)
        self.particle_sprites = [
            _radial_sprite(r, theme.colors.rgb("particle"), 0.35) for r in (4, 6, 9, 13)
        ]
        rng = np.random.default_rng(seed)
        n = theme.animation.particles
        self.p_x = rng.uniform(0, width, n)
        self.p_y = rng.uniform(0, height, n)
        self.p_speed = rng.uniform(18, 60, n)  # px per second, upwards
        self.p_sway = rng.uniform(10, 40, n)
        self.p_phase = rng.uniform(0, 2 * math.pi, n)
        self.p_size = rng.integers(0, len(self.particle_sprites), n)

    def frame(self, t: float) -> np.ndarray:
        img = self.base.copy()
        # Glow drifts on a slow Lissajous path.
        cx = self.w * (0.5 + 0.35 * math.sin(t * 0.21))
        cy = self.h * (0.42 + 0.25 * math.sin(t * 0.13 + 1.0))
        self._add(img, self.glow, int(cx), int(cy))
        for i in range(len(self.p_x)):
            y = (self.p_y[i] - self.p_speed[i] * t) % (self.h + 40) - 20
            x = self.p_x[i] + self.p_sway[i] * math.sin(t * 0.8 + self.p_phase[i])
            self._add(img, self.particle_sprites[self.p_size[i]], int(x), int(y))
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _add(img: np.ndarray, sprite: np.ndarray, cx: int, cy: int) -> None:
        sh, sw = sprite.shape[:2]
        x0, y0 = cx - sw // 2, cy - sh // 2
        x1, y1 = x0 + sw, y0 + sh
        ix0, iy0 = max(0, x0), max(0, y0)
        ix1, iy1 = min(img.shape[1], x1), min(img.shape[0], y1)
        if ix0 >= ix1 or iy0 >= iy1:
            return
        img[iy0:iy1, ix0:ix1] += sprite[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0]
