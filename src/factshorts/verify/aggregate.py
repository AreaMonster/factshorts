"""Run all enabled checks over a script and combine them: one failing check fails the claim."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from factshorts.config import VerificationConfig
from factshorts.models import (
    CheckResult,
    ClaimVerdict,
    Script,
    Segment,
    SentenceVerdict,
    SourceDoc,
    Verification,
)
from factshorts.verify.entities import Vocab, build_vocab
from factshorts.verify.normalize import normalize_chars, split_sentences
from factshorts.verify.programmatic import entity_check, has_checkable_entities, quote_check

# Checks implemented in later phases are reported as skipped, never silently passed.
LATER_CHECKS = {
    "wikidata": "wikidata_check",
    "nli": "nli_check",
    "llm_judge": "llm_judge",
}


def enabled_checks(cfg: VerificationConfig) -> list[str]:
    checks = []
    if cfg.quote_check:
        checks.append("quote")
    if cfg.entity_check:
        checks.append("entities")
    checks += [name for name, flag in LATER_CHECKS.items() if getattr(cfg, flag)]
    return checks


def subject_names(source: SourceDoc) -> set[str]:
    """Words of the article title (e.g. 'Mount', 'Everest') may be named anywhere."""
    return {w for w in re.findall(r"[\wÀ-ɏ]+", source.title) if len(w) > 2}


def _link(sentence: str, claim_text: str) -> bool:
    a = normalize_chars(sentence).casefold()
    b = normalize_chars(claim_text).casefold()
    return a == b or fuzz.ratio(a, b) >= 90 or (len(b) > 20 and fuzz.partial_ratio(b, a) >= 95)


def verify_script(
    script: Script, source: SourceDoc, cfg: VerificationConfig, attempt: int = 1
) -> Verification:
    checks = enabled_checks(cfg)
    vocab = build_vocab(
        source.text,
        *(s.narration for s in script.segments),
        *(s.on_screen_text for s in script.segments),
    )
    subject = subject_names(source)
    claim_verdicts: list[ClaimVerdict] = []
    sentence_verdicts: list[SentenceVerdict] = []
    failures: list[str] = []
    all_evidence: list[str] = []

    for seg in script.segments:
        evidence_by_claim: dict[str, str] = {}
        for claim in seg.claims:
            verdict = _verify_claim(
                seg, claim.id, claim.text, claim.source_quote, source, cfg, checks, vocab, subject
            )
            claim_verdicts.append(verdict)
            if verdict.passed and verdict.paragraph:
                evidence_by_claim[claim.id] = f"{claim.source_quote}\n{verdict.paragraph}"
            if not verdict.passed:
                failures += [
                    f"{seg.id} claim '{claim.text}': {c.check}: {c.reason}"
                    for c in verdict.checks
                    if c.status == "fail"
                ]
        all_evidence += evidence_by_claim.values()

        for sentence in split_sentences(seg.narration):
            sv = _verify_sentence(seg, sentence, evidence_by_claim, cfg, vocab, subject)
            sentence_verdicts.append(sv)
            if not sv.passed:
                failures += [
                    f"{seg.id} sentence '{sentence}': {c.reason}"
                    for c in sv.checks
                    if c.status == "fail"
                ]

        if cfg.entity_check and seg.on_screen_text.strip():
            ev = "\n".join(evidence_by_claim.values())
            res = entity_check(
                seg.on_screen_text,
                ev,
                vocab,
                cfg,
                allowed_names=subject,
                titlecase=True,
                check_name="on_screen_entities",
            )
            sentence_verdicts.append(
                SentenceVerdict(
                    segment_id=seg.id,
                    sentence=seg.on_screen_text,
                    claim_ids=list(evidence_by_claim),
                    kind="claim" if evidence_by_claim else "filler",
                    checks=[res],
                    passed=res.status != "fail",
                )
            )
            if res.status == "fail":
                failures.append(f"{seg.id} on-screen text '{seg.on_screen_text}': {res.reason}")

    if cfg.entity_check:
        res = entity_check(
            script.title,
            "\n".join(all_evidence),
            vocab,
            cfg,
            allowed_names=subject,
            titlecase=True,
            check_name="title_entities",
        )
        if res.status == "fail":
            failures.append(f"title '{script.title}': {res.reason}")
            sentence_verdicts.append(
                SentenceVerdict(
                    segment_id="title",
                    sentence=script.title,
                    claim_ids=[],
                    kind="unsupported",
                    checks=[res],
                    passed=False,
                )
            )

    return Verification(
        enabled_checks=checks,
        claims=claim_verdicts,
        sentences=sentence_verdicts,
        passed=not failures,
        failures=failures,
        attempt=attempt,
    )


def _verify_claim(
    seg: Segment,
    claim_id: str,
    text: str,
    quote: str,
    source: SourceDoc,
    cfg: VerificationConfig,
    checks: list[str],
    vocab: Vocab,
    subject: set[str],
) -> ClaimVerdict:
    results: list[CheckResult] = []
    paragraph: str | None = None
    qm = quote_check(quote, source.text)
    paragraph = qm.paragraph
    if cfg.quote_check:
        results.append(qm.result)
    if cfg.entity_check:
        if paragraph is None:
            results.append(
                CheckResult(
                    check="entities", status="fail", reason="no source paragraph (quote not found)"
                )
            )
        else:
            results.append(
                entity_check(text, f"{quote}\n{paragraph}", vocab, cfg, allowed_names=subject)
            )
    for name in checks:
        if name in LATER_CHECKS:
            results.append(
                CheckResult(check=name, status="skipped", reason="not implemented yet (phase 2)")
            )
    passed = paragraph is not None and all(r.status != "fail" for r in results)
    return ClaimVerdict(
        claim_id=claim_id,
        segment_id=seg.id,
        text=text,
        source_quote=quote,
        paragraph=paragraph,
        checks=results,
        passed=passed,
    )


def _verify_sentence(
    seg: Segment,
    sentence: str,
    evidence_by_claim: dict[str, str],
    cfg: VerificationConfig,
    vocab: Vocab,
    subject: set[str],
) -> SentenceVerdict:
    linked = [c.id for c in seg.claims if _link(sentence, c.text)]
    linked_ok = [cid for cid in linked if cid in evidence_by_claim]
    if linked:
        if not linked_ok:
            res = CheckResult(
                check="sentence_entities",
                status="fail",
                reason="all claims for this sentence failed verification",
            )
        elif cfg.entity_check:
            ev = "\n".join(evidence_by_claim[c] for c in linked_ok)
            res = entity_check(
                sentence, ev, vocab, cfg, allowed_names=subject, check_name="sentence_entities"
            )
        else:
            res = CheckResult(check="sentence_entities", status="skipped")
        return SentenceVerdict(
            segment_id=seg.id,
            sentence=sentence,
            claim_ids=linked,
            kind="claim",
            checks=[res],
            passed=res.status != "fail",
        )
    if has_checkable_entities(sentence, vocab, subject):
        res = CheckResult(
            check="sentence_link",
            status="fail",
            reason="sentence states numbers or names but is not linked to any claim",
        )
        return SentenceVerdict(
            segment_id=seg.id,
            sentence=sentence,
            claim_ids=[],
            kind="unsupported",
            checks=[res],
            passed=False,
        )
    res = CheckResult(
        check="filler",
        status="pass",
        reason="no numbers or names; LLM filler classification comes in phase 2",
    )
    return SentenceVerdict(
        segment_id=seg.id, sentence=sentence, claim_ids=[], kind="filler", checks=[res], passed=True
    )
