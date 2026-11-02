"""Word-level caption timing: engine timestamps when available, else a proportional estimate."""

from __future__ import annotations

import itertools
import re

from factshorts.core.textnorm import to_speakable
from factshorts.models import WordTiming

PUNCT_PAUSE = {",": 3, ";": 4, ":": 4, "-": 2}
_PIECE = re.compile(r"[A-Za-z0-9']+")


def _pieces(text: str) -> list[str]:
    return _PIECE.findall(text.replace("’", "'"))


def align_word_timings(
    sentence: str,
    spoken: list[tuple[str, float, float]],
    start: float,
    duration: float,
    pronunciations: dict[str, str] | None = None,
) -> list[WordTiming] | None:
    """Map engine word timestamps (spoken forms, e.g. 'eight', 'thousand') to display words.

    Each display word is expanded to its spoken form; the counts of word pieces must line up
    with the engine's tokens, otherwise None is returned and the caller falls back to an
    estimate. Times in `spoken` are relative to the sentence clip; output is absolute.
    """
    words = sentence.split()
    if not words:
        return []
    # Engine tokens → word pieces, each piece inheriting a share of its token's time span.
    timed: list[tuple[float, float]] = []
    for text, t0, t1 in spoken:
        parts = _pieces(text)
        if not parts:
            continue
        step = (t1 - t0) / len(parts)
        timed += [(t0 + i * step, t0 + (i + 1) * step) for i in range(len(parts))]
    # Spoken pieces per display word, counted on growing prefixes so that units stay with
    # their numbers ("-40 °C" → "minus forty degrees Celsius").
    totals = [
        len(_pieces(to_speakable(" ".join(words[: i + 1]), pronunciations)))
        for i in range(len(words))
    ]
    counts = [b - a for a, b in zip([0, *totals[:-1]], totals, strict=True)]
    if totals[-1] != len(timed) or any(c < 0 for c in counts):
        return None
    out: list[WordTiming] = []
    i = 0
    for w, c in zip(words, counts, strict=True):
        if c == 0:  # e.g. a lone dash: zero-length, keeps the previous word highlighted
            t = timed[min(i, len(timed) - 1)][0]
            out.append(WordTiming(text=w, start=round(start + t, 4), end=round(start + t, 4)))
            continue
        t0, t1 = timed[i][0], timed[i + c - 1][1]
        out.append(WordTiming(text=w, start=round(start + t0, 4), end=round(start + t1, 4)))
        i += c
    # Make highlights continuous: each word lasts until the next one starts.
    out[0].start = round(start, 4)
    for a, b in itertools.pairwise(out):
        a.end = max(a.start, b.start)
    out[-1].end = round(start + duration, 4)
    return out


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


def align_by_text(
    sentence: str,
    timed: list[tuple[str, float, float]],
    start: float,
    duration: float,
) -> list[WordTiming] | None:
    """Align a transcript that is written like the display text (e.g. Whisper: '8,849') by
    matching letters and digits; each display word takes the times of its matched letters."""
    from difflib import SequenceMatcher

    words = sentence.split()
    if not words or not timed:
        return None
    disp, owner = [], []
    for i, w in enumerate(words):
        for ch in re.sub(r"[^0-9a-z]", "", w.lower()):
            disp.append(ch)
            owner.append(i)
    heard: list[str] = []
    spans: list[tuple[float, float]] = []  # each heard character's time slot
    for text, t0, t1 in timed:
        chars = re.sub(r"[^0-9a-z]", "", text.lower())
        n = max(1, len(chars))
        for k, ch in enumerate(chars):
            heard.append(ch)
            spans.append((t0 + (t1 - t0) * k / n, t0 + (t1 - t0) * (k + 1) / n))
    matcher = SequenceMatcher(None, "".join(disp), "".join(heard), autojunk=False)
    times: dict[int, list[tuple[float, float]]] = {}
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            times.setdefault(owner[a + k], []).append(spans[b + k])
    if len(times) < len(words) * 0.6:
        return None  # too little of the sentence was recognized
    out: list[WordTiming] = []
    last = 0.0
    for i, w in enumerate(words):
        t0 = min(a for a, _ in times[i]) if i in times else last
        last = max(b for _, b in times[i]) if i in times else t0
        out.append(WordTiming(text=w, start=round(start + t0, 4), end=round(start + last, 4)))
    out[0].start = round(start, 4)
    for a, b in itertools.pairwise(out):
        b.start = max(b.start, a.start)
        a.end = b.start
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
