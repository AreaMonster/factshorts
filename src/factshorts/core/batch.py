"""Batch runs: work through the idea inbox first, then fill up with automatic ideas."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from factshorts.config import Config
from factshorts.core.ideas import DuplicateIdea, IdeaStore, generate_ideas
from factshorts.core.pipeline import Pipeline
from factshorts.models import Idea

log = logging.getLogger(__name__)


@dataclass
class BatchReport:
    done: list[str] = field(default_factory=list)  # output folders
    rejected: list[str] = field(default_factory=list)  # "idea: reason"
    needs_input: list[str] = field(default_factory=list)  # job ids
    failed: list[str] = field(default_factory=list)


def idea_defaults(cfg: Config) -> dict:
    return {"length": cfg.script.default_length, "tone": cfg.script.default_tone}


def queue_auto_ideas(pipeline: Pipeline, topic: str, count: int) -> list[int]:
    """Generate `count` new ideas for `topic` and put them in the queue."""
    services, cfg = pipeline.services, pipeline.cfg
    assert services.llm is not None and services.wiki is not None
    store = IdeaStore(pipeline.db)
    found = generate_ideas(
        topic,
        count,
        services.llm,
        cfg.ollama.writer_model,
        services.wiki,
        store,
        idea_defaults(cfg),
    )
    ids = []
    for r in found:
        r.idea.article, r.idea.angle = r.article, r.angle
        try:
            ids.append(store.add(r.idea, article=r.article, angle=r.angle))
        except DuplicateIdea as exc:  # a race with another run; skip
            log.info("%s", exc)
    return ids


def run_batch(
    pipeline: Pipeline,
    per_topic: int,
    topics: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
    step_progress: Callable[[str, str], None] | None = None,
) -> BatchReport:
    say = progress or (lambda m: None)
    store = IdeaStore(pipeline.db)
    report = BatchReport()

    def run_one(idea_id: int, idea: Idea) -> None:
        job_id = pipeline.create_job(idea, idea_id=idea_id)
        say(f"job {job_id}: {idea.text}")
        try:
            result = pipeline.run(job_id, progress=step_progress)
        except Exception as exc:  # one broken job must not stop the batch
            report.failed.append(f"{idea.text}: {exc}")
            store.set_status(idea_id, "failed")
            return
        row = pipeline.db.get_job(job_id)
        if result is not None:
            report.done.append(result.folder)
        elif row and row["state"] == "needs_input":
            report.needs_input.append(job_id)
        else:
            report.rejected.append(f"{idea.text}: {row['error'] if row else 'unknown'}")

    inbox = store.queued()
    say(f"inbox: {len(inbox)} queued idea(s)")
    for item in inbox:
        run_one(item.id, item.idea)
    for topic in topics if topics is not None else pipeline.cfg.ideas.topics:
        say(f"topic '{topic}': generating {per_topic} new idea(s)")
        for idea_id in queue_auto_ideas(pipeline, topic, per_topic):
            item = next(s for s in store.list("queued") if s.id == idea_id)
            run_one(item.id, item.idea)
    return report
