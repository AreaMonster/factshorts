"""Extract checkable entities (numbers, years, dates, proper nouns) and match them to evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from factshorts.verify.normalize import (
    Quantity,
    fold,
    normalize_chars,
    parse_quantities,
    round_to,
)
from factshorts.verify.units import UNITS, convert, is_imperial, same_dimension

MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

# Capitalized words that are not names when they start a sentence or appear in title case.
COMMON_WORDS = {
    "a",
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "all",
    "almost",
    "along",
    "already",
    "also",
    "although",
    "always",
    "among",
    "an",
    "and",
    "another",
    "any",
    "are",
    "around",
    "as",
    "at",
    "back",
    "be",
    "because",
    "been",
    "before",
    "behind",
    "being",
    "below",
    "beneath",
    "best",
    "between",
    "beyond",
    "big",
    "biggest",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "down",
    "during",
    "each",
    "earth",
    "either",
    "enough",
    "even",
    "ever",
    "every",
    "fact",
    "facts",
    "far",
    "few",
    "find",
    "first",
    "follow",
    "for",
    "from",
    "fun",
    "get",
    "go",
    "guess",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "here's",
    "his",
    "how",
    "however",
    "i",
    "if",
    "imagine",
    "in",
    "inside",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "just",
    "last",
    "let",
    "let's",
    "like",
    "more",
    "most",
    "much",
    "my",
    "near",
    "nearly",
    "never",
    "new",
    "next",
    "no",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "picture",
    "ready",
    "really",
    "right",
    "roughly",
    "same",
    "see",
    "she",
    "since",
    "so",
    "some",
    "still",
    "stay",
    "such",
    "sure",
    "take",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "them",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "this",
    "those",
    "though",
    "through",
    "to",
    "today",
    "too",
    "top",
    "turns",
    "under",
    "until",
    "up",
    "us",
    "very",
    "wait",
    "was",
    "watch",
    "way",
    "we",
    "well",
    "were",
    "what",
    "what's",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "without",
    "wow",
    "yes",
    "yet",
    "you",
    "your",
    "you're",
    "who's",
    "think",
    "believe",
    "meet",
    "discover",
    "learn",
    "plus",
    "amazing",
    "incredible",
    "surprisingly",
    "actually",
    "officially",
    "together",
    "standing",
    "towering",
    "tallest",
    "highest",
    "largest",
    "longest",
    "deepest",
    "oldest",
    "smallest",
    "world",
    "world's",
    "thanks",
    "thank",
    "watching",
    "subscribe",
    "comment",
    "share",
    "remember",
    "time",
    "bye",
    "curious",
    "tuned",
    "mind",
    "blown",
    "keep",
    "exploring",
    "nature",
    "science",
    "history",
    "know",
    "secret",
    "secrets",
    "mystery",
    "truth",
    "answer",
    "question",
    "meanwhile",
    "finally",
    "second",
    "third",
    "later",
    "soon",
    "climbing",
    "climbers",
    "people",
    "scientists",
    "experts",
    "explorers",
    "measured",
}

_CAP_WORD = re.compile(r"(?<![\w'])([A-ZÀ-Þ][\wÀ-ɏ]*(?:['’]s)?)")


@dataclass
class EntityReport:
    numbers: list[Quantity] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    imperial_units: list[str] = field(default_factory=list)


@dataclass
class Vocab:
    """Words seen in the source and narration, used to tell names from ordinary words."""

    lower: set[str] = field(default_factory=set)  # words used in lowercase
    proper: set[str] = field(default_factory=set)  # words capitalized mid-sentence

    def is_ordinary(self, word: str) -> bool:
        low = word.lower()
        stems = {
            low,
            low.removesuffix("s"),
            low.removesuffix("es"),
            low.removesuffix("ed"),
            low.removesuffix("d"),
            low.removesuffix("ing"),
            low.removesuffix("ly"),
        }
        return any(s in self.lower for s in stems if len(s) > 2)


def build_vocab(*texts: str) -> Vocab:
    vocab = Vocab()
    for t in texts:
        t = normalize_chars(t)
        for m in re.finditer(r"[\wÀ-ɏ]+", t):
            w = m.group(0)
            if w.islower():
                vocab.lower.add(w)
            elif w[0].isupper() and not _is_sentence_start(t, m.start()):
                vocab.proper.add(w)
    return vocab


def extract_names(text: str, vocab: Vocab, titlecase: bool = False) -> list[str]:
    """Proper nouns in `text`.

    Mid-sentence capitalized words are names unless they are common words. Words at the start
    of a sentence (or every word, for title-case text like on-screen captions) are names if
    they are used capitalized mid-sentence elsewhere, or if they are unknown words; they are
    ordinary words if they appear in lowercase in the source or narration.
    """
    text = normalize_chars(text)
    names: list[str] = []
    for m in _CAP_WORD.finditer(text):
        word = re.sub(r"['’]s$", "", m.group(1))
        low = word.lower()
        if low in COMMON_WORDS or len(word) == 1:
            continue
        if low in MONTHS:
            names.append(word)
            continue
        ambiguous = titlecase or _is_sentence_start(text, m.start())
        if ambiguous and word not in vocab.proper and vocab.is_ordinary(word):
            continue
        names.append(word)
    return names


def _is_sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip(" \"'(")
    return not before or before[-1] in ".!?:;"


def extract_entities(text: str, vocab: Vocab, titlecase: bool = False) -> EntityReport:
    numbers = parse_quantities(text)
    imperial = [q.raw for q in numbers if q.unit and is_imperial(q.unit)]
    return EntityReport(
        numbers=numbers,
        names=extract_names(text, vocab, titlecase),
        imperial_units=imperial,
    )


# Short forms on screen that sources usually spell out (and vice versa).
NAME_ALIASES: dict[str, list[str]] = {
    "co2": ["carbon dioxide"],
    "h2o": ["water"],
    "o2": ["oxygen"],
    "n2": ["nitrogen"],
    "ch4": ["methane"],
    "h2so4": ["sulfuric acid", "sulphuric acid"],
    "so2": ["sulfur dioxide"],
    "us": ["united states"],
    "usa": ["united states"],
    "uk": ["united kingdom"],
    "un": ["united nations"],
    "eu": ["european union"],
    "ussr": ["soviet union"],
}


def _contains(term: str, evidence_folded: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(fold(term))}(?!\w)", evidence_folded) is not None


def name_in_evidence(name: str, evidence: str) -> bool:
    ev = fold(evidence)
    key = fold(name).replace("₂", "2")  # CO₂ → co2
    return _contains(name, ev) or any(_contains(a, ev) for a in NAME_ALIASES.get(key, []))


def quantity_matches(
    claim: Quantity,
    evidence: list[Quantity],
    max_rounding_error: float,
    conversion_tolerance: float,
) -> Quantity | None:
    """Return the evidence quantity that supports `claim`, or None."""
    for ev in evidence:
        if _matches(claim, ev, max_rounding_error, conversion_tolerance):
            return ev
    return None


def _matches(c: Quantity, e: Quantity, max_round: float, conv_tol: float) -> bool:
    if c.is_year or (e.is_year and c.unit is None):
        if c.raw.endswith("s") and c.is_year:  # decade: "1950s"
            return e.is_year and c.value <= e.value < c.value + 10
        return e.value == c.value
    target = e.value
    if c.unit and e.unit:
        if not same_dimension(c.unit, e.unit):
            return False
        target = convert(e.value, e.unit, c.unit)
    elif (c.unit and not e.unit) or (e.unit and not c.unit):
        # A bare number may match a number with a unit (or vice versa) only without conversion.
        if (c.unit or e.unit) and UNITS[c.unit or e.unit].dimension == "temperature":
            return False
    if target == c.value:
        return True
    scale = max(abs(target), 1e-12)
    converted = c.unit is not None and e.unit is not None and c.unit != e.unit
    if converted and abs(c.value - target) / scale <= conv_tol:
        return True
    # Rounding: "about 9,000 m" for 8,848.86 m is fine; "1 million" for 1.4 million is not.
    rounded = round_to(target, c.precision)
    close_enough = abs(c.value - target) / scale <= max_round
    return abs(rounded - c.value) <= c.precision * 1e-6 and close_enough
