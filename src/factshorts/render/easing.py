"""Easing curves on t in [0, 1]."""

from __future__ import annotations

import math


def clamp01(t: float) -> float:
    return 0.0 if t < 0 else 1.0 if t > 1 else t


def linear(t: float) -> float:
    return clamp01(t)


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_in_out_sine(t: float) -> float:
    t = clamp01(t)
    return -(math.cos(math.pi * t) - 1) / 2


def ease_out_back(t: float, overshoot: float = 1.4) -> float:
    t = clamp01(t)
    c3 = overshoot + 1
    return 1 + c3 * (t - 1) ** 3 + overshoot * (t - 1) ** 2


def progress(t: float, start: float, duration: float) -> float:
    if duration <= 0:
        return 1.0 if t >= start else 0.0
    return clamp01((t - start) / duration)
