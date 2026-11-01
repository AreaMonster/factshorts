"""Second, strict LLM pass (a different model than the writer): claim vs. source excerpt."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from factshorts.core.llm import LLM
from factshorts.models import CheckResult


class ClaimJudgement(BaseModel):
    reason: str = Field(description="One short sentence explaining the verdict")
    verdict: Literal["SUPPORTED", "NOT_SUPPORTED", "UNCLEAR"]


class FillerJudgement(BaseModel):
    reason: str = Field(description="One short sentence")
    states_a_fact: bool


CLAIM_PROMPT = """You are a strict fact-checker. Decide whether the CLAIM is fully supported by \
the SOURCE EXCERPT alone. Do not use outside knowledge, even if you know the claim is true.

Rules:
- Every part of the claim must be stated or directly implied by the excerpt: numbers, dates,
  names, places, comparisons, superlatives ("highest", "first"), causes and implications.
- Rounding is fine (8,848.86 m -> "8,849 meters"); unit conversion to metric is fine.
- A simpler paraphrase with the same meaning is SUPPORTED (e.g. "marks part of the
  China-Nepal border at its summit" supports "it lies on the China-Nepal border").
- If the claim adds anything the excerpt does not say, or says something stronger or different
  (e.g. "highest above sea level" does NOT support "tallest from base to summit"), answer
  NOT_SUPPORTED. Answer UNCLEAR only if the excerpt can honestly be read both ways.
- "It"/"its" in the claim refers to {title}.

SOURCE EXCERPT (from "{title}"):
{excerpt}

CLAIM: {claim}"""

FILLER_PROMPT = """A video about "{title}" has this narration sentence with no source attached:

"{sentence}"

Does the sentence assert any fact about the world (a number, date, name, place, event,
property, comparison, record or cause)? Rhetorical questions, transitions ("Here's the
twist."), teasers that state nothing ("But how tall is it really?") and calls to action
("Follow for more facts.") do NOT state a fact."""


class Judge:
    def __init__(self, llm: LLM, model: str) -> None:
        self.llm = llm
        self.model = model

    def judge_claim(self, claim: str, quote: str, paragraph: str | None, title: str) -> CheckResult:
        excerpt = quote if not paragraph else f"{paragraph}\n\n(Cited sentence: {quote})"
        result = self.llm.structured(
            [
                {
                    "role": "user",
                    "content": CLAIM_PROMPT.format(title=title, excerpt=excerpt, claim=claim),
                }
            ],
            ClaimJudgement,
            model=self.model,
            temperature=0.0,
        )
        status = "pass" if result.verdict == "SUPPORTED" else "fail"
        return CheckResult(
            check="llm_judge", status=status, reason=f"{result.verdict}: {result.reason}"
        )

    def classify_filler(self, sentence: str, title: str) -> CheckResult:
        result = self.llm.structured(
            [{"role": "user", "content": FILLER_PROMPT.format(title=title, sentence=sentence)}],
            FillerJudgement,
            model=self.model,
            temperature=0.0,
        )
        if result.states_a_fact:
            return CheckResult(
                check="filler",
                status="fail",
                reason=f"states a fact without a source: {result.reason}",
            )
        return CheckResult(
            check="filler", status="pass", reason=f"no factual claim: {result.reason}"
        )
