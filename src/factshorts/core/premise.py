"""False-premise check: does the source contradict something the idea takes for granted?"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from factshorts.core.llm import LLM
from factshorts.models import Idea, SourceDoc
from factshorts.verify.normalize import normalize_chars, split_paragraphs

log = logging.getLogger(__name__)


class Presuppositions(BaseModel):
    statements: list[str] = Field(
        description="Factual claims the idea takes for granted, as short standalone statements",
        max_length=3,
    )


class PremiseJudgement(BaseModel):
    reason: str = Field(description="One short sentence")
    verdict: Literal["SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"]
    source_sentence: str = Field(
        description="The deciding source sentence, copied exactly; empty if not mentioned"
    )


class PremiseItem(BaseModel):
    statement: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"]
    reason: str
    source_sentence: str = ""


class PremiseResult(BaseModel):
    checked: bool = True
    items: list[PremiseItem] = Field(default_factory=list)
    # Set when the user answered a needs_input question.
    resolution: Literal["corrected", "cancelled"] | None = None
    correction: str | None = None

    @property
    def contradicted(self) -> list[PremiseItem]:
        return [i for i in self.items if i.verdict == "CONTRADICTED"]

    def question(self) -> str:
        lines = ["The idea assumes something the source contradicts:"]
        for i in self.contradicted:
            lines.append(f'- Idea assumes: "{i.statement}"')
            if i.source_sentence:
                lines.append(f'  Source says: "{i.source_sentence}"')
            lines.append(f"  ({i.reason})")
        return "\n".join(lines)

    def correction_note(self) -> str:
        parts = []
        for i in self.contradicted:
            said = f' The source says: "{i.source_sentence}"' if i.source_sentence else ""
            parts.append(f'The idea wrongly assumes "{i.statement}".{said}')
        return " ".join(parts)


EXTRACT_PROMPT = """List the factual claims this video idea takes for granted: things it
presents as true without asking about them. Write each as a short standalone statement.
Asking is not assuming: "Why is Venus hotter than Mercury?" assumes "Venus is hotter than
Mercury."; "How tall is Mount Everest?" assumes nothing. Return an empty list if there are none.

IDEA: {idea}"""

CHECK_PROMPT = """Does the SOURCE EXCERPT support, contradict, or not mention the STATEMENT?
Use only the excerpt, not outside knowledge. CONTRADICTED means the excerpt clearly says the
statement is false. Copy the one sentence from the excerpt that decides it, or leave it empty
if the excerpt does not mention it.

STATEMENT: {statement}

SOURCE EXCERPT:
{excerpt}"""


def relevant_passages(statement: str, source: SourceDoc, k: int = 3, max_words: int = 900) -> str:
    """The paragraphs that best match the statement's words (simple lexical retrieval)."""
    paragraphs = split_paragraphs(source.text)
    ranked = sorted(
        paragraphs,
        key=lambda p: fuzz.token_set_ratio(
            normalize_chars(statement).lower(), normalize_chars(p).lower()
        ),
        reverse=True,
    )
    out: list[str] = []
    words = 0
    for p in ranked[:k]:
        words += len(p.split())
        if out and words > max_words:
            break
        out.append(p)
    return "\n\n".join(out)


def check_premise(
    idea: Idea, source: SourceDoc, llm: LLM, writer_model: str, judge_model: str
) -> PremiseResult:
    found = llm.structured(
        [{"role": "user", "content": EXTRACT_PROMPT.format(idea=idea.text)}],
        Presuppositions,
        model=writer_model,
        temperature=0.0,
    )
    items: list[PremiseItem] = []
    for statement in found.statements:
        statement = statement.strip()
        if not statement:
            continue
        excerpt = relevant_passages(statement, source)
        j = llm.structured(
            [
                {
                    "role": "user",
                    "content": CHECK_PROMPT.format(statement=statement, excerpt=excerpt),
                }
            ],
            PremiseJudgement,
            model=judge_model,
            temperature=0.0,
        )
        log.info("premise '%s': %s (%s)", statement, j.verdict, j.reason)
        items.append(
            PremiseItem(
                statement=statement,
                verdict=j.verdict,
                reason=j.reason,
                source_sentence=j.source_sentence.strip(),
            )
        )
    return PremiseResult(items=items)
