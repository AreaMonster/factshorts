"""Cheap deterministic checks: quote check and entity check."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from factshorts.config import VerificationConfig
from factshorts.models import CheckResult
from factshorts.verify.entities import (
    Vocab,
    extract_entities,
    name_in_evidence,
    quantity_matches,
)
from factshorts.verify.normalize import (
    normalize_chars,
    parse_quantities,
    split_paragraphs,
    strip_punct_keep_numbers,
)
from factshorts.verify.units import is_imperial

MIN_QUOTE_WORDS = 4


@dataclass
class QuoteMatch:
    result: CheckResult
    paragraph: str | None


def clean_quote(quote: str) -> str:
    q = normalize_chars(quote)
    q = re.sub(r"^\s*\[\d+\]\s*", "", q)  # sentence labels from the prompt
    return q.strip().strip("\"'").strip()


def quote_check(quote: str, source_text: str) -> QuoteMatch:
    """The quote must appear verbatim in the source; only punctuation may differ."""
    q = clean_quote(quote)
    if len(q.split()) < MIN_QUOTE_WORDS:
        return QuoteMatch(
            CheckResult(check="quote", status="fail", reason=f"quote too short: '{q}'"), None
        )
    q_norm = q.casefold()
    q_bare = strip_punct_keep_numbers(q).casefold()
    paragraphs = split_paragraphs(source_text)
    for para in paragraphs:
        p_norm = normalize_chars(para).casefold()
        if q_norm in p_norm:
            return QuoteMatch(CheckResult(check="quote", status="pass", score=100.0), para)
    for para in paragraphs:
        if q_bare in strip_punct_keep_numbers(normalize_chars(para)).casefold():
            return QuoteMatch(
                CheckResult(
                    check="quote",
                    status="pass",
                    score=99.0,
                    reason="matched ignoring punctuation",
                ),
                para,
            )
    best, best_para = 0.0, ""
    for para in paragraphs:
        score = fuzz.partial_ratio(q_norm, normalize_chars(para).casefold())
        if score > best:
            best, best_para = score, para
    hint = f" (closest paragraph {best:.0f}% similar)" if best_para else ""
    return QuoteMatch(
        CheckResult(
            check="quote",
            status="fail",
            score=round(best, 1),
            reason=f"quote not found verbatim in source{hint}",
        ),
        None,
    )


def entity_check(
    text: str,
    evidence: str,
    vocab: Vocab,
    cfg: VerificationConfig,
    allowed_names: set[str] | None = None,
    titlecase: bool = False,
    check_name: str = "entities",
) -> CheckResult:
    """Every number, year, date and proper noun in `text` must appear in `evidence`."""
    report = extract_entities(text, vocab, titlecase=titlecase)
    problems: list[str] = []
    for raw in report.imperial_units:
        problems.append(f"imperial unit in '{raw}' (metric only)")
    ev_numbers = parse_quantities(evidence, include_one=True)
    for q in report.numbers:
        if q.unit and is_imperial(q.unit):
            continue  # already reported as imperial
        if quantity_matches(q, ev_numbers, cfg.max_rounding_error, cfg.conversion_tolerance):
            continue
        problems.append(f"number '{q.raw}' not found in source paragraph")
    allowed = {a.casefold() for a in (allowed_names or set())}
    for name in report.names:
        if name.casefold() in allowed or name_in_evidence(name, evidence):
            continue
        problems.append(f"name '{name}' not found in cited quote or paragraph")
    checked = len(report.numbers) + len(report.names)
    if problems:
        return CheckResult(check=check_name, status="fail", reason="; ".join(problems))
    return CheckResult(
        check=check_name,
        status="pass",
        score=float(checked),
        reason=f"{checked} entities matched" if checked else "no entities to check",
    )


def has_checkable_entities(
    text: str, vocab: Vocab, allowed_names: set[str], titlecase: bool = False
) -> bool:
    report = extract_entities(text, vocab, titlecase=titlecase)
    allowed = {a.casefold() for a in allowed_names}
    names = [n for n in report.names if n.casefold() not in allowed]
    return bool(report.numbers or names or report.imperial_units)
