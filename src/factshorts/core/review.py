"""Approving and rejecting finished videos; shared by the Web UI and `factshorts review`."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from factshorts.config import Config
from factshorts.db import Database
from factshorts.models import Script, SourceDoc, Verification

STATUSES = ("pending", "approved", "rejected")


def set_video_status(cfg: Config, db: Database, job_id: str, status: str) -> Path:
    """Move the video's folder to output/<status>/ and record it. Returns the new folder."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    video = db.get_video(job_id)
    if video is None:
        raise ValueError(f"no video for job {job_id}")
    folder = Path(video["folder"])
    target = cfg.output_dir / status / folder.name
    if folder.is_dir() and folder != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(target))
    db.update_video(job_id, status=status, folder=str(target))
    return target


@dataclass
class ReviewItem:
    job_id: str
    title: str
    label: str
    folder: Path
    video: Path | None
    sources: list[str]
    lines: list[str]  # narration per segment with its claims, for the terminal


def review_item(cfg: Config, db: Database, job_id: str) -> ReviewItem:
    video = db.get_video(job_id)
    if video is None:
        raise ValueError(f"no video for job {job_id}")
    wd = cfg.work_dir / job_id
    script = Script.model_validate_json((wd / "script.json").read_text("utf-8"))
    source = SourceDoc.model_validate_json((wd / "source.json").read_text("utf-8"))
    verification = Verification.model_validate_json((wd / "verification.json").read_text("utf-8"))
    lines = []
    for seg in script.segments:
        lines.append(f"[{seg.id}] {seg.narration}")
        for c in verification.claims:
            if c.segment_id != seg.id:
                continue
            checks = " ".join(f"{r.check}:{r.status}" for r in c.checks)
            lines.append(f'    {"ok " if c.passed else "BAD"} "{c.source_quote[:110]}"')
            lines.append(f"        {checks}")
    folder = Path(video["folder"])
    mp4 = next(iter(sorted(folder.glob("*.mp4"))), None) if folder.is_dir() else None
    return ReviewItem(
        job_id=job_id,
        title=script.title,
        label=verification.label,
        folder=folder,
        video=mp4,
        sources=[r.revision_url or r.title for r in source.refs],
        lines=lines,
    )
