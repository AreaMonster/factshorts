"""Text normalization, sentence splitting and number/quantity parsing."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from factshorts.verify.units import match_unit

# --- character-level normalization -----------------------------------------

_QUOTES = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "′": "'",
    "`": "'",
    "´": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "″": '"',
    "«": '"',
    "»": '"',
}
_DASHES = dict.fromkeys(["‐", "‑", "‒", "–", "—", "―", "−", "﹣", "－"], "-")
_SPACES = dict.fromkeys([" ", " ", " ", " ", " ", " ", "　", "\t", "\r", "\n"], " ")
_TRANS = str.maketrans({**_QUOTES, **_DASHES, **_SPACES, "…": "...", "⁄": "/"})


def normalize_chars(text: str) -> str:
    """Unify quotes, dashes, spaces and ellipses; collapse whitespace. Keeps case and accents."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANS)
    text = text.replace("​", "").replace("­", "")
    return re.sub(r" {2,}", " ", text).strip()


def fold(text: str) -> str:
    """Case- and accent-insensitive form for name matching."""
    decomposed = unicodedata.normalize("NFKD", normalize_chars(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def strip_punct_keep_numbers(text: str) -> str:
    """Remove punctuation except separators inside numbers (8,848.86 stays intact)."""
    text = re.sub(r"(?<=\d)[.,](?=\d)", lambda m: "\x00" if m.group(0) == "." else "\x01", text)
    text = re.sub(r"[^\w\s\x00\x01%°]", " ", text)
    text = text.replace("\x00", ".").replace("\x01", ",")
    return re.sub(r"\s+", " ", text).strip()


# --- sentences ---------------------------------------------------------------

_ABBREV = {
    "mt",
    "st",
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "jr",
    "sr",
    "vs",
    "etc",
    "approx",
    "ca",
    "c",
    "e.g",
    "i.e",
    "u.s",
    "u.k",
    "no",
    "fig",
    "inc",
    "ltd",
    "co",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "est",
    "gen",
    "lt",
}
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9À-ɏ])")


def split_sentences(text: str) -> list[str]:
    text = normalize_chars(text)
    if not text:
        return []
    out: list[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        candidate = text[start : m.start()].strip()
        # The last word before the stop; "Earth's." is not an initial like "J.", and neither
        # is a unit: "104 km/h." and "6,412 m." end sentences (initials are capitals).
        last_word = re.search(r"(?<![°\w'’/])([\w.]+)[.!?]$", candidate)
        if last_word:
            raw = last_word.group(1).rstrip(".")
            if raw.lower() in _ABBREV or re.fullmatch(r"[A-Z]", raw):
                continue
        if candidate:
            out.append(candidate)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n") if p.strip()]


# --- number words ----------------------------------------------------------------

SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
NUMBER_WORDS = set(SMALL) | set(TENS) | set(SCALES) | {"hundred"}


def words_to_number(words: list[str]) -> float | None:
    """'eight thousand eight hundred forty-nine' → 8849. Returns None if not a number."""
    total = 0.0
    current = 0.0
    seen = False
    for raw in words:
        for w in raw.lower().split("-"):
            if w == "and":
                continue
            if w in SMALL:
                current += SMALL[w]
            elif w in TENS:
                current += TENS[w]
            elif w == "hundred":
                current = (current or 1) * 100
            elif w in SCALES:
                total += (current or 1) * SCALES[w]
                current = 0
            else:
                return None
            seen = True
    return total + current if seen else None


# --- quantities ----------------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str | None
    raw: str
    start: int
    end: int
    precision: float  # size of the last significant digit (8,800 → 100; 8.8 → 0.1)
    is_year: bool = False
    from_words: bool = False

    def __str__(self) -> str:
        return f"{self.raw}"


_DIGITS = re.compile(
    r"(?<![\w.,])"
    r"(?P<sign>[-−])?"
    r"(?P<num>\d{1,3}(?:(?:,|[   ]|\s(?=\d{3}(?!\d)))\d{3})+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?)"
    r"(?P<frac>\s?\+?\s?\d+[/⁄]\d+)?"
    r"(?P<suffix>st|nd|rd|th|s)?"
    r"(?!\d)"
)
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_MINUS_WORD = re.compile(r"(?<![A-Za-z])minus\s+$", re.I)
_MULT = re.compile(r"\s*(thousand|million|billion|trillion)\b", re.I)


def _precision(num: str) -> float:
    digits = re.sub(r"[^\d.]", "", num)
    if "." in digits:
        return 10.0 ** -len(digits.split(".")[1])
    stripped = digits.rstrip("0")
    zeros = len(digits) - len(stripped)
    return 10.0**zeros if stripped else 1.0


def parse_quantities(text: str, include_one: bool = False) -> list[Quantity]:
    """Find numbers (digits or words), with multipliers and units, in `text`.

    Text is character-normalized first; spans refer to the normalized text.
    `include_one` controls whether the word "one" counts (it is usually a pronoun).
    """
    text = normalize_chars(text)
    candidates: list[tuple[int, int, float, float, bool, str, bool]] = []
    for dm in _DIGITS.finditer(text):
        num = dm.group("num")
        value = float(re.sub(r"[^\d.]", "", num))
        if dm.group("frac"):
            n, d = re.split(r"/", re.sub(r"[\s+]", "", dm.group("frac")))
            if float(d):
                value += float(n) / float(d)
        if dm.group("sign") or _MINUS_WORD.search(text, max(0, dm.start() - 12), dm.start()):
            value = -value
        suffix = dm.group("suffix") or ""
        is_year = (
            re.fullmatch(r"\d{4}", num) is not None
            and 1000 <= abs(value) <= 2999
            and suffix in ("", "s")
            and not dm.group("sign")
        )
        candidates.append((dm.start(), dm.end(), value, _precision(num), is_year, suffix, False))
    for start, end, value in _number_word_spans(text, include_one):
        candidates.append((start, end, value, _word_precision(value), False, "", True))
    candidates.sort(key=lambda c: c[0])

    out: list[Quantity] = []
    last_end = -1
    for start, end, value, precision, is_year, suffix, from_words in candidates:
        if start < last_end:
            continue
        mult = _MULT.match(text, end)
        if mult:
            factor = SCALES[mult.group(1).lower()]
            value *= factor
            precision *= factor
            end = mult.end()
            is_year = False
        unit = None
        if suffix in ("", "s") and not (suffix == "s" and is_year):
            um = match_unit(text, end)
            if um is not None:
                unit, end = um
                is_year = False
        out.append(
            Quantity(
                value=value,
                unit=unit,
                raw=text[start:end],
                start=start,
                end=end,
                precision=precision,
                is_year=is_year,
                from_words=from_words,
            )
        )
        last_end = end
    return out


def _word_precision(value: float) -> float:
    if value == 0:
        return 1.0
    p = 1.0
    while value % (p * 10) == 0:
        p *= 10
    return p


_NEXT_WORD = re.compile(r"\s+(?:and\s+)?([A-Za-z]+(?:-[A-Za-z]+)*)")


def _is_number_word(word: str) -> bool:
    return all(p in NUMBER_WORDS for p in word.lower().split("-"))


def _number_word_spans(text: str, include_one: bool) -> list[tuple[int, int, float]]:
    spans: list[tuple[int, int, float]] = []
    pos = 0
    for m in _WORD.finditer(text):
        if m.start() < pos or not _is_number_word(m.group(0)):
            continue
        first = m.group(0).lower().split("-")[0]
        if first in SCALES or first == "hundred":
            continue
        end = m.end()
        words = [m.group(0)]
        while (n := _NEXT_WORD.match(text, end)) and _is_number_word(n.group(1)):
            words.append(n.group(1))
            end = n.end()
        value = words_to_number(words)
        if value is None or (not include_one and [w.lower() for w in words] == ["one"]):
            continue
        spans.append((m.start(), end, value))
        pos = end
    return spans


def round_to(value: float, precision: float) -> float:
    if precision <= 0:
        return value
    return math.floor(value / precision + 0.5) * precision


# --- keywords -------------------------------------------------------------------

STOPWORDS = {
    "a",
    "an",
    "the",
    "why",
    "how",
    "what",
    "which",
    "who",
    "when",
    "where",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "so",
    "of",
    "in",
    "on",
    "at",
    "to",
    "and",
    "or",
    "it",
    "its",
    "can",
    "really",
    "actually",
    "there",
    "that",
    "this",
    "be",
    "has",
    "have",
    "about",
    "for",
}


def normalize_title(text: str) -> str:
    """Lowercase, no punctuation, no filler words: 'Why is Venus so hot?' → 'venus hot'."""
    words = re.findall(r"[\wÀ-ɏ]+", text.lower())
    return " ".join(w for w in words if w not in STOPWORDS)
