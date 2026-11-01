"""Ideas: parsing custom ideas, the idea queue with dedupe, and automatic idea generation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from rapidfuzz import fuzz

from factshorts.core.llm import LLM
from factshorts.db import Database
from factshorts.models import Idea
from factshorts.verify.normalize import normalize_title

log = logging.getLogger(__name__)

SETTING_KEYS = {"length", "tone", "factcheck", "force", "source_url", "source_file", "topic"}


class DuplicateIdea(ValueError):
    pass


# --- custom ideas --------------------------------------------------------------------------


def parse_idea_line(line: str, defaults: dict[str, Any]) -> Idea:
    """'Why is Venus hotter than Mercury? | length=45 | tone=funny' → Idea."""
    text, *parts = [p.strip() for p in line.split("|")]
    settings: dict[str, Any] = dict(defaults)
    for part in parts:
        if "=" not in part:
            raise ValueError(f"setting '{part}' should look like key=value")
        key, value = (x.strip() for x in part.split("=", 1))
        key = key.lower().replace("-", "_")
        if key not in SETTING_KEYS:
            raise ValueError(
                f"unknown setting '{key}' (allowed: {', '.join(sorted(SETTING_KEYS))})"
            )
        if key in ("factcheck", "force"):
            settings[key] = value.lower() in ("1", "true", "yes", "on")
        elif key == "length":
            settings[key] = int(value.rstrip("s"))
        else:
            settings[key] = value
    try:
        return Idea(text=text, **settings)
    except ValidationError as exc:
        raise ValueError(f"invalid idea '{text}': {exc.errors()[0]['msg']}") from exc


def parse_ideas_text(text: str, defaults: dict[str, Any]) -> list[Idea]:
    """One idea per line; blank lines and lines starting with # are ignored."""
    ideas = []
    for n, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            ideas.append(parse_idea_line(line, defaults))
        except ValueError as exc:
            raise ValueError(f"line {n}: {exc}") from exc
    return ideas


# --- queue + dedupe -------------------------------------------------------------------------


@dataclass
class StoredIdea:
    id: int
    idea: Idea
    status: str
    article: str | None
    angle: str | None


class IdeaStore:
    """Ideas in SQLite. Status: queued → processing → done | rejected | cancelled."""

    def __init__(self, db: Database, text_threshold: float = 90, angle_threshold: float = 75):
        self.db = db
        self.text_threshold = text_threshold
        self.angle_threshold = angle_threshold
        self._backfill()

    def _backfill(self) -> None:
        """Jobs made before the idea store existed still count for duplicate detection."""
        state_to_status = {
            "done": "done",
            "rejected": "rejected",
            "cancelled": "cancelled",
            "needs_input": "needs_input",
        }
        for row in self.db.query(
            "SELECT id, idea, state, created_at FROM jobs WHERE idea_id IS NULL"
        ):
            idea = Idea.model_validate_json(row["idea"])
            idea_id = self.db.execute(
                "INSERT INTO ideas (text, norm_title, article, angle, origin, topic, status,"
                " settings, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    idea.text,
                    normalize_title(idea.text),
                    idea.article,
                    idea.angle,
                    idea.origin,
                    idea.topic,
                    state_to_status.get(row["state"], "processing"),
                    idea.model_dump_json(),
                    row["created_at"],
                ),
            )
            self.db.execute("UPDATE jobs SET idea_id = ? WHERE id = ?", (idea_id, row["id"]))

    def find_duplicate(
        self, idea: Idea, article: str | None = None, angle: str | None = None
    ) -> str | None:
        """Reason string if a similar idea exists (same question, or same article + angle)."""
        norm = normalize_title(idea.text)
        for row in self.db.query(
            "SELECT id, text, norm_title, article, angle, status FROM ideas"
            " WHERE status != 'cancelled'"
        ):
            if norm and fuzz.token_set_ratio(norm, row["norm_title"] or "") >= self.text_threshold:
                return f"same idea as #{row['id']} '{row['text']}' ({row['status']})"
            if article and row["article"] and article.casefold() == row["article"].casefold():
                a = normalize_title(angle or idea.text)
                b = normalize_title(row["angle"] or row["text"])
                if fuzz.token_set_ratio(a, b) >= self.angle_threshold:
                    return (
                        f"same article '{article}' and angle as #{row['id']} "
                        f"'{row['text']}' ({row['status']})"
                    )
        return None

    def add(
        self,
        idea: Idea,
        article: str | None = None,
        angle: str | None = None,
        status: str = "queued",
    ) -> int:
        if not idea.force:
            reason = self.find_duplicate(idea, article, angle)
            if reason:
                raise DuplicateIdea(
                    f"'{idea.text}' looks like a duplicate: {reason}. Use force to make it anyway."
                )
        return self.db.execute(
            "INSERT INTO ideas (text, norm_title, article, angle, origin, topic, status, settings,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                idea.text,
                normalize_title(idea.text),
                article,
                angle,
                idea.origin,
                idea.topic,
                status,
                idea.model_dump_json(),
                time.time(),
            ),
        )

    def _row(self, row: Any) -> StoredIdea:
        return StoredIdea(
            id=row["id"],
            idea=Idea.model_validate_json(row["settings"]),
            status=row["status"],
            article=row["article"],
            angle=row["angle"],
        )

    def list(self, status: str | None = None) -> list[StoredIdea]:
        if status:
            rows = self.db.query("SELECT * FROM ideas WHERE status = ? ORDER BY id", (status,))
        else:
            rows = self.db.query("SELECT * FROM ideas ORDER BY id")
        return [self._row(r) for r in rows]

    def queued(self) -> list[StoredIdea]:
        """The inbox: custom ideas first, then auto ideas, oldest first."""
        rows = self.db.query(
            "SELECT * FROM ideas WHERE status = 'queued' ORDER BY origin = 'auto', id"
        )
        return [self._row(r) for r in rows]

    def set_status(self, idea_id: int, status: str) -> None:
        self.db.execute("UPDATE ideas SET status = ? WHERE id = ?", (status, idea_id))

    def clear(self) -> int:
        n = len(self.db.query("SELECT id FROM ideas WHERE status = 'queued'"))
        self.db.execute("UPDATE ideas SET status = 'cancelled' WHERE status = 'queued'")
        return n


# --- automatic ideas ------------------------------------------------------------------------


class ProposedIdea(BaseModel):
    question: str = Field(description="The video idea as a short, intriguing question")
    article_title: str = Field(description="Exact title of the English Wikipedia article")
    angle: str = Field(description="The specific aspect the video covers, a few words")


class ProposedIdeas(BaseModel):
    ideas: list[ProposedIdea]


IDEAS_PROMPT = """Propose {n} ideas for short, fact-checked vertical videos (about 75 seconds)
on the topic "{topic}".

Each idea:
- is one short, intriguing question a curious viewer wants answered;
- can be answered completely from ONE specific English Wikipedia article: give its exact
  title, and the angle (the specific aspect, a few words);
- is about surprising but well-documented facts (records, numbers, how something works,
  history), not opinions or rumors;
- is clearly different from the others and from these existing ideas:
{existing}
Mix well-known subjects with lesser-known ones."""


@dataclass
class ResolvedIdea:
    idea: Idea
    article: str
    angle: str


def generate_ideas(
    topic: str,
    count: int,
    llm: LLM,
    model: str,
    wiki: Any,
    store: IdeaStore,
    defaults: dict[str, Any] | None = None,
) -> list[ResolvedIdea]:
    """Ask the LLM for candidates, keep those with a real Wikipedia article and no duplicate."""
    existing = [s.idea.text for s in store.list() if s.idea.topic == topic][-30:]
    shown = "\n".join(f"- {t}" for t in existing) or "- (none yet)"
    result = llm.structured(
        [
            {
                "role": "user",
                "content": IDEAS_PROMPT.format(n=count * 2, topic=topic, existing=shown),
            }
        ],
        ProposedIdeas,
        model=model,
        temperature=0.9,
    )
    accepted: list[ResolvedIdea] = []
    for p in result.ideas:
        if len(accepted) >= count:
            break
        article = resolve_article(p.article_title, wiki)
        if article is None:
            log.info("idea '%s': no Wikipedia article '%s'", p.question, p.article_title)
            continue
        idea = Idea(text=p.question.strip(), origin="auto", topic=topic, **(defaults or {}))
        reason = store.find_duplicate(idea, article, p.angle)
        if reason is None:
            reason = next(
                (
                    f"same as '{a.idea.text}' in this batch"
                    for a in accepted
                    if a.article.casefold() == article.casefold()
                    or fuzz.token_set_ratio(
                        normalize_title(a.idea.text), normalize_title(idea.text)
                    )
                    >= 90
                ),
                None,
            )
        if reason:
            log.info("idea '%s' skipped: %s", p.question, reason)
            continue
        accepted.append(ResolvedIdea(idea=idea, article=article, angle=p.angle))
    return accepted


def resolve_article(title: str, wiki: Any) -> str | None:
    """Exact (or near-exact) title of an existing article, via Wikipedia search."""
    hits = wiki.search(title, limit=3)
    for hit in hits:
        if fuzz.ratio(hit.title.casefold(), title.casefold()) >= 90:
            return hit.title
    return None


def ideas_to_json(items: list[StoredIdea]) -> str:
    return json.dumps(
        [
            {
                "id": s.id,
                "text": s.idea.text,
                "status": s.status,
                "article": s.article,
                "origin": s.idea.origin,
            }
            for s in items
        ],
        indent=2,
    )
