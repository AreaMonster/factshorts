"""Turn display text into speakable text: numbers, years, units, abbreviations, symbols."""

from __future__ import annotations

import re

from num2words import num2words

from factshorts.verify.normalize import Quantity, normalize_chars, parse_quantities
from factshorts.verify.units import UNITS

ABBREVIATIONS = [
    (r"\bMt\.(?=\s+[A-Z])", "Mount"),
    (r"\bMts\.(?=\s+[A-Z])", "Mounts"),
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\bapprox\.", "approximately"),
    (r"\bca\.(?=\s*\d)", "about"),
    (r"\bc\.(?=\s*\d)", "circa"),
    (r"\bvs\.?(?=\s)", "versus"),
    (r"\betc\.", "et cetera"),
    (r"\bU\.S\.(?:A\.)?", "U S"),
    (r"\bU\.K\.", "U K"),
    (r"\bDr\.(?=\s+[A-Z])", "Doctor"),
    (r"\bSt\.(?=\s+[A-Z])", "Saint"),
    (r"\bNo\.(?=\s*\d)", "number"),
]
SYMBOLS = [
    ("&", " and "),
    ("~", "about "),
    ("≈", "about "),
    ("×", " times "),
    ("±", " plus or minus "),
    ("#", " number "),
    ("°", " degrees"),
    ("→", " to "),
]

_NUM_PART = re.compile(
    r"^-?(?:\d{1,3}(?:(?:,|[   ]|\s(?=\d{3}(?!\d)))\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?\+?\s?\d+/\d+)?(st|nd|rd|th|s)?"
)
_MULT_PART = re.compile(r"^\s*(thousand|million|billion|trillion)\b", re.I)


def number_words(value: float) -> str:
    """American style: no 'and', no commas ('eight thousand eight hundred forty-nine')."""
    if value == int(value) and abs(value) < 1e15:
        words = num2words(int(value), lang="en")
    else:
        words = num2words(value, lang="en")
    return words.replace(",", "").replace(" and ", " ")


def _decimal_words(raw_number: str) -> str:
    digits = re.sub(r"[^\d.\-]", "", raw_number)
    if "." in digits:
        whole, frac = digits.split(".", 1)
        whole_words = number_words(float(whole)) if whole not in ("", "-") else "zero"
        if whole.startswith("-") and whole_words and not whole_words.startswith("minus"):
            whole_words = f"minus {whole_words}"
        frac_words = " ".join(num2words(int(d), lang="en") for d in frac)
        return f"{whole_words} point {frac_words}"
    return number_words(float(digits))


def _plural(word: str) -> str:
    if word.endswith("y"):
        return word[:-1] + "ies"
    return word + "s"


def speak_quantity(q: Quantity, raw: str) -> str:
    if q.from_words:
        number_text = raw
        rest = ""
        m = re.search(r"\s*[^\w\s-].*$|\s+\S+$", raw) if q.unit else None
        if q.unit and m:
            number_text, rest = raw[: m.start()], raw[m.start() :]
        spoken_number = number_text
    else:
        nm = _NUM_PART.match(raw)
        number_text = nm.group(0) if nm else raw
        suffix = (nm.group(1) or "") if nm else ""
        rest = raw[len(number_text) :]
        bare = number_text[: len(number_text) - len(suffix)] if suffix else number_text
        if q.is_year:
            year = int(float(re.sub(r"[^\d]", "", bare)))
            spoken_number = num2words(year, lang="en", to="year").replace(",", "")
            spoken_number = spoken_number.replace(" and ", " ")
            if suffix == "s":  # decade: "1950s" → "nineteen fifties"
                words = spoken_number.split(" ")
                head, _, tail = words[-1].rpartition("-")
                words[-1] = f"{head}-{_plural(tail)}" if head else _plural(tail)
                spoken_number = " ".join(words)
        elif suffix in ("st", "nd", "rd", "th"):
            n = int(float(re.sub(r"[^\d]", "", bare)))
            spoken_number = num2words(n, lang="en", to="ordinal").replace(" and ", " ")
            spoken_number = spoken_number.replace(",", "")
        elif "/" in bare:
            whole, frac = re.split(r"\s?\+?\s?(?=\d+/\d+$)", bare.strip(), maxsplit=1)
            n, d = (int(x) for x in frac.split("/"))
            ordinal = num2words(d, lang="en", to="ordinal")
            singular, plural = {2: ("half", "halves"), 4: ("quarter", "quarters")}.get(
                d, (ordinal, f"{ordinal}s")
            )
            spoken_number = (
                f"{_decimal_words(whole)} and {number_words(n)} {singular if n == 1 else plural}"
            )
        else:
            spoken_number = _decimal_words(bare)
    mult = _MULT_PART.match(rest)
    if mult:
        spoken_number += f" {mult.group(1).lower()}"
        rest = rest[mult.end() :]
    if q.unit:
        unit = UNITS[q.unit]
        one = abs(q.value) == 1 and not mult
        spoken_unit = unit.spoken_singular if one else unit.spoken_plural
        return f"{spoken_number} {spoken_unit}"
    return spoken_number + rest


def to_speakable(text: str, pronunciations: dict[str, str] | None = None) -> str:
    text = normalize_chars(text)
    for word, spoken in (pronunciations or {}).items():
        text = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", spoken, text)
    for pattern, repl in ABBREVIATIONS:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"(\d)\+(?!\s?\d)", r"\1 plus", text)  # "200+" → "200 plus"
    quantities = parse_quantities(text, include_one=True)
    for q in sorted(quantities, key=lambda q: q.start, reverse=True):
        if q.from_words and not q.unit:
            continue
        raw = text[q.start : q.end]
        text = text[: q.start] + speak_quantity(q, raw) + text[q.end :]
    for symbol, repl in SYMBOLS:
        text = text.replace(symbol, repl)
    text = re.sub(r"(?<=\w)/(?=\w)", " per ", text)
    return re.sub(r"\s{2,}", " ", text).strip()
