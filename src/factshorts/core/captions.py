"""Word-level caption timing (proportional estimate; faster-whisper alignment comes later)."""

from __future__ import annotations

import re

from factshorts.core.textnorm import to_speakable
from factshorts.models import WordTiming

PUNCT_PAUSE = {",": 3, ";": 4, ":": 4, "-": 2}


def proportional_word_timings(
    sentence: str, start: float, duration: float, pronunciations: dict[str, str] | None = None
) -> list[WordTiming]:
    """Split `duration` across displayed words, weighted by the length of their spoken form."""
    words = sentence.split()
    if not words:
        return []
    weights: list[float] = []
    for w in words:
        core = re.sub(r"[^\w%°/.,'-]", "", w)
        spoken = to_speakable(core, pronunciations) if core else w
        weight = len(re.sub(r"[^a-zA-Z]", "", spoken)) + 2.0
        weight += PUNCT_PAUSE.get(w[-1], 0)
        weights.append(weight)
    total = sum(weights)
    out: list[WordTiming] = []
    t = start
    for w, weight in zip(words, weights, strict=True):
        d = duration * weight / total
        out.append(WordTiming(text=w, start=round(t, 4), end=round(t + d, 4)))
        t += d
    out[-1].end = round(start + duration, 4)
    return out


def caption_chunks(
    words: list[WordTiming], max_words: int = 4, max_chars: int = 22
) -> list[list[WordTiming]]:
    """Group words into short caption lines shown one at a time."""
    chunks: list[list[WordTiming]] = []
    current: list[WordTiming] = []
    for w in words:
        length = sum(len(x.text) + 1 for x in current) + len(w.text)
        if current and (len(current) >= max_words or length > max_chars):
            chunks.append(current)
            current = []
        current.append(w)
        if w.text[-1] in ".!?;:":
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks
