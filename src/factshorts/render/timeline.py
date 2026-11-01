"""Frame ranges shared by the visuals step and the renderer."""

from __future__ import annotations

import math

from factshorts.models import AudioPlan


def total_frames(audio: AudioPlan, fps: int) -> int:
    return math.ceil(audio.duration * fps)


def segment_frame_ranges(audio: AudioPlan, fps: int) -> dict[str, tuple[int, int]]:
    """Frame range [start, end) per segment; contiguous and ending at the last frame."""
    total = total_frames(audio, fps)
    ranges: dict[str, tuple[int, int]] = {}
    for i, sa in enumerate(audio.segments):
        start = round(sa.start * fps)
        end = total if i == len(audio.segments) - 1 else round(sa.end * fps)
        ranges[sa.segment_id] = (start, end)
    return ranges
