"""Resolve an idea to a source document (Phase 1: Wikipedia only)."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from factshorts.core.llm import LLM
from factshorts.models import Idea, SourceDoc
from factshorts.sources.wikipedia import ArticleNotFound, Wikipedia

log = logging.getLogger(__name__)

MIN_ARTICLE_WORDS = 120  # shorter articles are stubs with too little to verify against


class SourcingError(RuntimeError):
    """The idea has no reliable source; the user must provide one."""


class ArticleGuess(BaseModel):
    article_title: str = Field(description="Exact title of the best English Wikipedia article")
    search_query: str = Field(description="Short Wikipedia search query for the topic")


GUESS_PROMPT = """Which single English Wikipedia article is the best source to answer this idea \
for a short fact video? Give its exact article title and a short search query.

IDEA: {idea}"""


def resolve_wikipedia(idea: Idea, wiki: Wikipedia, llm: LLM, model: str) -> SourceDoc:
    guess = llm.structured(
        [{"role": "user", "content": GUESS_PROMPT.format(idea=idea.text)}],
        ArticleGuess,
        model=model,
        temperature=0.0,
    )
    log.info("article guess: %s (query: %s)", guess.article_title, guess.search_query)

    candidates: dict[str, float] = {}
    for query in dict.fromkeys([guess.article_title, guess.search_query, idea.text]):
        for rank, hit in enumerate(wiki.search(query)):
            score = fuzz.token_set_ratio(guess.article_title.lower(), hit.title.lower()) - rank * 2
            candidates[hit.title] = max(candidates.get(hit.title, 0.0), score)
    ordered = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    log.info("article candidates: %s", ordered[:5])

    # Prefer the guessed title if it exists; otherwise the best-matching search hit.
    tried: list[str] = []
    for title in [guess.article_title, *[t for t, s in ordered if s >= 70]]:
        if title in tried:
            continue
        tried.append(title)
        try:
            doc = wiki.fetch(title)
        except ArticleNotFound as exc:
            log.info("skip %s: %s", title, exc)
            continue
        if len(doc.text.split()) < MIN_ARTICLE_WORDS:
            log.info("skip %s: article too short", doc.title)
            continue
        return doc
    raise SourcingError(
        f"No reliable Wikipedia article found for '{idea.text}' (tried: {', '.join(tried)}). "
        "Please provide a source URL, text or file."
    )
