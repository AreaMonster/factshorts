"""Run all enabled checks over a script and combine them: one failing check fails the claim.

Order per claim, cheap first: quote → entities → Wikidata → NLI → LLM judge. Results are
cached per (claim text, quote), so after a targeted rewrite only changed claims are re-checked.
"""

from __future__ import annotations

import itertools
import re

from rapidfuzz import fuzz, process

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
from factshorts.sources.wikidata import WDValue
from factshorts.verify.entities import Vocab, build_vocab
from factshorts.verify.llm_judge import Judge
from factshorts.verify.nli import NLIModel, nli_check
from factshorts.verify.normalize import normalize_chars, split_paragraphs, split_sentences
from factshorts.verify.programmatic import entity_check, has_checkable_entities, quote_check
from factshorts.verify.wikidata_check import is_about_subject, wikidata_check

CHECK_FLAGS = {
    "quote": "quote_check",
    "entities": "entity_check",
    "wikidata": "wikidata_check",
    "nli": "nli_check",
    "llm_judge": "llm_judge",
}


def enabled_checks(cfg: VerificationConfig) -> list[str]:
    return [name for name, flag in CHECK_FLAGS.items() if getattr(cfg, flag)]


def subject_names(source: SourceDoc) -> set[str]:
    """Words of the article titles (e.g. 'Mount', 'Everest') may be named anywhere."""
    titles = [source.title, *(r.title for r in source.parts)]
    return {w for t in titles for w in re.findall(r"[\wÀ-ɏ]+", t) if len(w) > 2}


def _link(sentence: str, claim_text: str) -> bool:
    a = normalize_chars(sentence).casefold()
    b = normalize_chars(claim_text).casefold()
    return a == b or fuzz.ratio(a, b) >= 90 or (len(b) > 20 and fuzz.partial_ratio(b, a) >= 95)


def _not_run(check: str, reason: str) -> CheckResult:
    return CheckResult(check=check, status="na", reason=f"not run: {reason}")


class Verifier:
    def __init__(
        self,
        cfg: VerificationConfig,
        source: SourceDoc,
        nli: NLIModel | None = None,
        judge: Judge | None = None,
        # Wikidata values per document title (comparison ideas have two documents).
        wikidata_values: dict[str, dict[str, list[WDValue]]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.source = source
        self.nli = nli
        self.judge = judge
        self.wikidata_values = wikidata_values
        self.subject = subject_names(source)
        self._claims: dict[tuple[str, str], tuple[list[CheckResult], str | None]] = {}
        self._fillers: dict[str, CheckResult] = {}
        # Snap targets: single source sentences and adjacent pairs (writers often merge two).
        self._sentences = []
        for para in split_paragraphs(source.text):
            sents = split_sentences(para)
            self._sentences += sents
            self._sentences += [f"{a} {b}" for a, b in itertools.pairwise(sents)]
        self._snapped: dict[str, float] = {}  # exact source sentence → writer's similarity

    # --- public -------------------------------------------------------------------------

    def snap_quotes(self, script: Script) -> int:
        """Replace mis-copied quotes with the source sentence they were copied from."""
        if not self.cfg.snap_quote_min_ratio:
            return 0
        changed = 0
        for seg in script.segments:
            for claim in seg.claims:
                if quote_check(claim.source_quote, self.source.text).paragraph is not None:
                    continue
                best = process.extractOne(
                    normalize_chars(claim.source_quote),
                    self._sentences,
                    scorer=fuzz.ratio,
                    score_cutoff=self.cfg.snap_quote_min_ratio,
                )
                if best is not None:
                    self._snapped[normalize_chars(best[0])] = best[1]
                    claim.source_quote = best[0]
                    changed += 1
        return changed

    def verify(self, script: Script, attempt: int = 1) -> Verification:
        """Check every claim and sentence. Mis-copied quotes are snapped first (in place)."""
        self.snap_quotes(script)
        cfg = self.cfg
        vocab = build_vocab(
            self.source.text,
            *(s.narration for s in script.segments),
            *(s.on_screen_text for s in script.segments),
        )
        claim_verdicts: list[ClaimVerdict] = []
        sentence_verdicts: list[SentenceVerdict] = []
        failures: list[str] = []
        all_evidence: list[str] = []

        for seg in script.segments:
            evidence_by_claim: dict[str, str] = {}
            for claim in seg.claims:
                checks, paragraph = self._claim_checks(claim.text, claim.source_quote, vocab)
                passed = paragraph is not None and all(c.status != "fail" for c in checks)
                claim_verdicts.append(
                    ClaimVerdict(
                        claim_id=claim.id,
                        segment_id=seg.id,
                        text=claim.text,
                        source_quote=claim.source_quote,
                        paragraph=paragraph,
                        checks=checks,
                        passed=passed,
                    )
                )
                if passed and paragraph:
                    evidence_by_claim[claim.id] = f"{claim.source_quote}\n{paragraph}"
                else:
                    failures += [
                        f"{seg.id} claim '{claim.text}': {c.check}: {c.reason}"
                        for c in checks
                        if c.status == "fail"
                    ]
            all_evidence += evidence_by_claim.values()

            for sentence in split_sentences(seg.narration):
                sv = self._sentence(seg, sentence, evidence_by_claim, vocab)
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
                    allowed_names=self.subject,
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
                allowed_names=self.subject,
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
            enabled_checks=enabled_checks(cfg),
            claims=claim_verdicts,
            sentences=sentence_verdicts,
            passed=not failures,
            failures=failures,
            attempt=attempt,
        )

    # --- claims ------------------------------------------------------------------------

    def _claim_checks(
        self, text: str, quote: str, vocab: Vocab
    ) -> tuple[list[CheckResult], str | None]:
        key = (normalize_chars(text), normalize_chars(quote))
        if key in self._claims:
            return self._claims[key]
        cfg, title = self.cfg, self.source.title
        results: list[CheckResult] = []
        qm = quote_check(quote, self.source.text)
        paragraph = qm.paragraph
        similarity = self._snapped.get(normalize_chars(quote))
        if similarity is not None and qm.result.status == "pass":
            qm.result.reason = (
                f"exact source sentence; the writer's copy was "
                f"{similarity:.0f}% similar and was replaced"
            )
        if cfg.quote_check:
            results.append(qm.result)
        if cfg.entity_check:
            if paragraph is None:
                results.append(
                    CheckResult(
                        check="entities",
                        status="fail",
                        reason="no source paragraph (quote not found)",
                    )
                )
            else:
                results.append(
                    entity_check(
                        text, f"{quote}\n{paragraph}", vocab, cfg, allowed_names=self.subject
                    )
                )
        early_fail = paragraph is None or any(r.status == "fail" for r in results)
        if cfg.wikidata_check:
            if self.wikidata_values is None:
                results.append(
                    CheckResult(check="wikidata", status="skipped", reason="Wikidata unavailable")
                )
            else:
                results.append(self._wikidata(text))
        early_fail = early_fail or any(r.status == "fail" for r in results)
        if cfg.nli_check:
            if early_fail:
                results.append(_not_run("nli", "an earlier check failed"))
            elif self.nli is None:
                results.append(
                    CheckResult(check="nli", status="skipped", reason="NLI model unavailable")
                )
            else:
                results.append(
                    nli_check(self.nli, text, quote, paragraph, title, cfg.nli_threshold)
                )
        early_fail = early_fail or any(r.status == "fail" for r in results)
        if cfg.llm_judge:
            if early_fail:
                results.append(_not_run("llm_judge", "an earlier check failed"))
            elif self.judge is None:
                results.append(
                    CheckResult(check="llm_judge", status="skipped", reason="LLM judge unavailable")
                )
            else:
                results.append(self.judge.judge_claim(text, quote, paragraph, title))
        self._claims[key] = (results, paragraph)
        return results, paragraph

    def _wikidata(self, text: str) -> CheckResult:
        """Check against the document the claim is about; ambiguous claims are not checked."""
        assert self.wikidata_values is not None
        docs = self.wikidata_values
        if len(docs) > 1:
            docs = {t: v for t, v in docs.items() if is_about_subject(text, t, pronouns=False)}
        if len(docs) != 1:
            return CheckResult(
                check="wikidata",
                status="na",
                reason="claim is not clearly about one of the articles",
            )
        title, values = next(iter(docs.items()))
        return wikidata_check(text, title, values, self.cfg.wikidata_properties, self.cfg)

    # --- sentences ---------------------------------------------------------------------

    def _sentence(
        self, seg: Segment, sentence: str, evidence_by_claim: dict[str, str], vocab: Vocab
    ) -> SentenceVerdict:
        cfg = self.cfg
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
                    sentence,
                    ev,
                    vocab,
                    cfg,
                    allowed_names=self.subject,
                    check_name="sentence_entities",
                )
            else:
                res = CheckResult(check="sentence_entities", status="na")
            return SentenceVerdict(
                segment_id=seg.id,
                sentence=sentence,
                claim_ids=linked,
                kind="claim",
                checks=[res],
                passed=res.status != "fail",
            )
        if has_checkable_entities(sentence, vocab, self.subject):
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
        res = self._filler(sentence)
        kind = "unsupported" if res.status == "fail" else "filler"
        return SentenceVerdict(
            segment_id=seg.id,
            sentence=sentence,
            claim_ids=[],
            kind=kind,
            checks=[res],
            passed=res.status != "fail",
        )

    def _filler(self, sentence: str) -> CheckResult:
        """A sentence without a claim passes only if the judge finds no factual claim in it."""
        key = normalize_chars(sentence)
        if key not in self._fillers:
            if not self.cfg.llm_judge:
                res = CheckResult(
                    check="filler", status="na", reason="no numbers or names (LLM judge disabled)"
                )
            elif self.judge is None:
                res = CheckResult(
                    check="filler",
                    status="skipped",
                    reason="LLM judge unavailable to classify this sentence",
                )
            else:
                res = self.judge.classify_filler(sentence, self.source.title)
            self._fillers[key] = res
        return self._fillers[key]


def verify_script(
    script: Script, source: SourceDoc, cfg: VerificationConfig, attempt: int = 1
) -> Verification:
    """Programmatic checks only (no models); model-based checks are reported as skipped."""
    return Verifier(cfg, source).verify(script, attempt)
