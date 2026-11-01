"""Cross-check numbers and years about the article's subject against Wikidata statements."""

from __future__ import annotations

import re

from factshorts.config import VerificationConfig, WikidataProperty
from factshorts.models import CheckResult
from factshorts.sources.wikidata import WDValue
from factshorts.verify.entities import quantity_matches
from factshorts.verify.normalize import Quantity, normalize_chars, parse_quantities
from factshorts.verify.units import same_dimension

# Title words too generic to identify the subject on their own.
GENERIC_TITLE_WORDS = {
    "mount",
    "mountain",
    "river",
    "lake",
    "sea",
    "ocean",
    "city",
    "island",
    "national",
    "park",
    "the",
    "of",
    "and",
    "saint",
    "great",
    "new",
    "north",
    "south",
    "east",
    "west",
    "list",
}


def is_about_subject(sentence: str, title: str) -> bool:
    low = normalize_chars(sentence).lower()
    if re.match(r"^(it|its|it's|this)\b", low):
        return True
    words = [
        w
        for w in re.findall(r"[\wÀ-ɏ]+", title.lower())
        if w not in GENERIC_TITLE_WORDS and len(w) > 2
    ]
    return any(re.search(rf"\b{re.escape(w)}\b", low) for w in words)


def _evidence(q: Quantity, values: list[WDValue]) -> list[Quantity]:
    out = []
    for v in values:
        if q.unit and v.unit:
            if not same_dimension(q.unit, v.unit):
                continue
        elif q.unit or v.unit:
            continue  # a bare number vs. a measurement: not comparable
        out.append(Quantity(value=v.value, unit=v.unit, raw=v.text, start=0, end=0, precision=1.0))
    return out


def wikidata_check(
    text: str,
    title: str,
    values: dict[str, list[WDValue]],
    properties: dict[str, WikidataProperty],
    cfg: VerificationConfig,
) -> CheckResult:
    if not values:
        return CheckResult(check="wikidata", status="na", reason="no Wikidata values available")
    if not is_about_subject(text, title):
        return CheckResult(
            check="wikidata", status="na", reason="claim is not about the article's subject"
        )
    sentence = normalize_chars(text)
    low = sentence.lower()
    quantities = parse_quantities(sentence)
    matched: list[str] = []
    problems: list[str] = []
    for prop, spec in properties.items():
        vals = values.get(prop)
        if not vals or not any(re.search(rf"\b{re.escape(k)}", low) for k in spec.keywords):
            continue
        for q in quantities:
            if vals[0].kind == "time":
                if not q.is_year:
                    continue
                ok = any(int(v.value) == int(q.value) for v in vals)
            else:
                ev = _evidence(q, vals)
                if not ev:
                    continue
                ok = (
                    quantity_matches(q, ev, cfg.max_rounding_error, cfg.conversion_tolerance)
                    is not None
                )
            shown = ", ".join(v.text for v in vals)
            if ok:
                matched.append(f"'{q.raw}' matches Wikidata {prop} ({spec.label}: {shown})")
            else:
                problems.append(f"'{q.raw}' disagrees with Wikidata {prop} ({spec.label}: {shown})")
    if problems:
        return CheckResult(check="wikidata", status="fail", reason="; ".join(problems))
    if matched:
        return CheckResult(check="wikidata", status="pass", reason="; ".join(matched))
    return CheckResult(check="wikidata", status="na", reason="no matching Wikidata property")
