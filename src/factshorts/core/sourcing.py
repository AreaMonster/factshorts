"""Resolve an idea to a source: user source → Wikipedia (one or two articles) → reject."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from factshorts.core.llm import LLM
from factshorts.models import Idea, Section, SourceDoc
from factshorts.sources import user_source
from factshorts.sources.wikipedia import ArticleNotFound, Wikipedia

log = logging.getLogger(__name__)

MIN_ARTICLE_WORDS = 120  # shorter articles are stubs with too little to verify against


class SourcingError(RuntimeError):
    """The idea has no reliable source; the user must provide one."""


class ArticleGuess(BaseModel):
    article_title: str = Field(description="Exact title of the best English Wikipedia article")
    search_query: str = Field(description="Short Wikipedia search query for the topic")
    second_article: str = Field(
        default="",
        description="Only if the idea compares two subjects: exact title of the article about "
        "the second subject; otherwise empty",
    )


GUESS_PROMPT = """Which English Wikipedia article is the best source to answer this idea for a \
short fact video? Give its exact article title and a short search query. If the idea compares
two different subjects (e.g. "Why is Venus hotter than Mercury?"), also give the exact title of
the article about the second subject; otherwise leave second_article empty.

IDEA: {idea}"""


def resolve_source(
    idea: Idea, wiki: Wikipedia, llm: LLM, model: str, http: Any = None
) -> SourceDoc:
    """Source resolution order: user source → Wikipedia → reject with a clear message."""
    try:
        if idea.source_text:
            return user_source.from_text(idea.source_text)
        if idea.source_file:
            return user_source.from_file(Path(idea.source_file))
        if idea.source_url:
            if http is None:
                raise SourcingError("cannot fetch a source URL without an HTTP client")
            return user_source.from_url(idea.source_url, http)
    except user_source.UserSourceError as exc:
        raise SourcingError(f"Your source could not be used: {exc}") from exc
    if idea.article:
        return _fetch_ok(idea.article, wiki, idea.text) or resolve_wikipedia(idea, wiki, llm, model)
    return resolve_wikipedia(idea, wiki, llm, model)


def _fetch_ok(title: str, wiki: Wikipedia, focus: str | None = None) -> SourceDoc | None:
    try:
        doc = wiki.fetch(title, focus=focus)
    except ArticleNotFound as exc:
        log.info("skip %s: %s", title, exc)
        return None
    if len(doc.text.split()) < MIN_ARTICLE_WORDS:
        log.info("skip %s: article too short", doc.title)
        return None
    return doc


def resolve_wikipedia(idea: Idea, wiki: Wikipedia, llm: LLM, model: str) -> SourceDoc:
    guess = llm.structured(
        [{"role": "user", "content": GUESS_PROMPT.format(idea=idea.text)}],
        ArticleGuess,
        model=model,
        temperature=0.0,
    )
    log.info(
        "article guess: %s (query: %s, second: %s)",
        guess.article_title,
        guess.search_query,
        guess.second_article or "-",
    )

    candidates: dict[str, float] = {}
    for query in dict.fromkeys([guess.article_title, guess.search_query, idea.text]):
        for rank, hit in enumerate(wiki.search(query)):
            score = fuzz.token_set_ratio(guess.article_title.lower(), hit.title.lower()) - rank * 2
            candidates[hit.title] = max(candidates.get(hit.title, 0.0), score)
    ordered = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    log.info("article candidates: %s", ordered[:5])

    # Prefer the guessed title if it exists; otherwise the best-matching search hit.
    tried: list[str] = []
    main: SourceDoc | None = None
    for title in [guess.article_title, *[t for t, s in ordered if s >= 70]]:
        if title in tried:
            continue
        tried.append(title)
        main = _fetch_ok(title, wiki, idea.text)
        if main is not None:
            break
    if main is None:
        raise SourcingError(
            f"No reliable Wikipedia article found for '{idea.text}' (tried: {', '.join(tried)}). "
            "Please provide a source URL, text or file."
        )
    second_title = guess.second_article.strip()
    if second_title and second_title.casefold() != main.title.casefold():
        second = _fetch_ok(second_title, wiki, idea.text)
        if second is not None and second.title != main.title:
            log.info("comparison idea: also using '%s'", second.title)
            return merge_sources(main, second)
    return main


def merge_sources(a: SourceDoc, b: SourceDoc) -> SourceDoc:
    """One source from two articles. Sections alternate (a's lead, b's lead, ...) so both
    survive when the text given to the writer is cut to its word limit."""
    sections: list[Section] = []
    for i in range(max(len(a.sections), len(b.sections))):
        for doc in (a, b):
            if i < len(doc.sections):
                sec = doc.sections[i]
                sections.append(
                    Section(index=len(sections), title=f"{doc.title}: {sec.title}", text=sec.text)
                )
    text = "\n\n".join(s.text for s in sections)
    return SourceDoc(
        kind=a.kind,
        title=f"{a.title} and {b.title}",
        url=a.url,
        language=a.language,
        revision_id=a.revision_id,
        page_id=a.page_id,
        wikidata_id=a.wikidata_id,
        sections=sections,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        parts=[*a.refs, *b.refs],
    )
