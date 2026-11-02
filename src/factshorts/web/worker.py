"""Background worker for the Web UI. Tasks live in SQLite, so a restart loses nothing:
interrupted tasks run again and their jobs resume from the last finished step."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from factshorts.core.batch import queue_auto_ideas
from factshorts.core.ideas import DuplicateIdea, IdeaStore
from factshorts.core.pipeline import Cancelled, Pipeline
from factshorts.db import Database

log = logging.getLogger(__name__)

TASK_KINDS = ("run", "edit", "regenerate", "rerender", "answer", "batch")


def progress_path(pipeline: Pipeline, job_id: str):
    return pipeline.workdir(job_id).path("progress.log")


class Worker:
    def __init__(
        self,
        db: Database,
        make_pipeline: Callable[[], Pipeline],
        threads: int = 1,
        poll_s: float = 1.0,
    ) -> None:
        self.db = db
        self.make_pipeline = make_pipeline
        self.threads = max(1, threads)
        self.poll_s = poll_s
        self.settings_version = 0  # bumped when settings change: pipelines are rebuilt
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._local = threading.local()

    # --- lifecycle ------------------------------------------------------------------

    def start(self) -> None:
        n = self.db.requeue_interrupted()
        if n:
            log.info("resuming %d interrupted task(s)", n)
        for i in range(self.threads):
            t = threading.Thread(target=self._loop, name=f"factshorts-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                busy = self.process_next()
            except Exception:  # never let the worker thread die
                log.exception("worker error")
                busy = False
            if not busy:
                self._stop.wait(self.poll_s)

    def pipeline(self) -> Pipeline:
        """One pipeline per thread (keeps models loaded), rebuilt after settings changes."""
        current = getattr(self._local, "pipeline", None)
        if current is None or self._local.version != self.settings_version:
            self._local.pipeline = self.make_pipeline()
            self._local.version = self.settings_version
        return self._local.pipeline

    # --- tasks ------------------------------------------------------------------------

    def process_next(self) -> bool:
        """Run the next queued task, if any. Returns whether there was one."""
        task = self.db.claim_task()
        if task is None:
            return False
        task_id, job_id, kind = task["id"], task["job_id"], task["kind"]
        payload: dict[str, Any] = json.loads(task["payload"])
        pipeline = self.pipeline()
        log.info("task %d: %s %s", task_id, kind, job_id or "")
        try:
            state, message = getattr(self, f"_do_{kind}")(pipeline, task_id, job_id, payload)
        except Cancelled:
            state, message = "cancelled", "cancelled"
            if job_id:
                self.db.update_job(job_id, state="cancelled", error="cancelled by the user")
        except Exception as exc:  # shown in the UI; the traceback is in the log
            log.exception("task %d failed", task_id)
            state, message = "failed", f"{type(exc).__name__}: {exc}"
        self.db.update_task(task_id, state=state, message=message)
        self._release_gpu(pipeline)
        return True

    @staticmethod
    def _release_gpu(pipeline: Pipeline) -> None:
        """Between tasks the UI process holds no GPU memory (voice, Whisper, cached blocks),
        so a job started from the command line meanwhile gets the whole card."""
        pipeline.services.release_synth()
        pipeline._whisper = None
        from factshorts.render.aigen import free_gpu

        free_gpu()

    def _progress(self, pipeline: Pipeline, task_id: int, job_id: str):
        path = progress_path(pipeline, job_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        def notify(step: str, message: str) -> None:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now():%H:%M:%S} [{step}] {message}\n")
            row = self.db.get_task(task_id)
            if row is not None and row["state"] == "cancelling":
                raise Cancelled()

        return notify

    def _run_job(self, pipeline: Pipeline, task_id: int, job_id: str) -> tuple[str, str]:
        result = pipeline.run(job_id, progress=self._progress(pipeline, task_id, job_id))
        row = self.db.get_job(job_id)
        if result is not None:
            return "done", f"video ready: {result.folder}"
        state = row["state"] if row else "failed"
        if state == "cancelled":
            return "cancelled", "cancelled"
        return "done" if state == "needs_input" else "failed", (
            row["error"] if row else ""
        ) or state

    def _do_run(self, pipeline: Pipeline, task_id: int, job_id: str, payload: dict):
        return self._run_job(pipeline, task_id, job_id)

    def _edit_then_run(
        self, pipeline: Pipeline, task_id: int, job_id: str, problems: list[str]
    ) -> tuple[str, str]:
        if problems:
            return "failed", "Not changed: the new text did not pass the fact-check.\n" + "\n".join(
                f"- {p}" for p in problems[:8]
            )
        return self._run_job(pipeline, task_id, job_id)

    def _do_edit(self, pipeline: Pipeline, task_id: int, job_id: str, payload: dict):
        notify = self._progress(pipeline, task_id, job_id)
        problems = pipeline.edit_segment(
            job_id,
            payload["segment"],
            payload["narration"],
            payload.get("on_screen_text"),
            notify=notify,
        )
        return self._edit_then_run(pipeline, task_id, job_id, problems)

    def _do_regenerate(self, pipeline: Pipeline, task_id: int, job_id: str, payload: dict):
        notify = self._progress(pipeline, task_id, job_id)
        problems = pipeline.regenerate_segment(job_id, payload["segment"], notify=notify)
        return self._edit_then_run(pipeline, task_id, job_id, problems)

    def _do_rerender(self, pipeline: Pipeline, task_id: int, job_id: str, payload: dict):
        pipeline.rerender(
            job_id,
            theme=payload.get("theme"),
            voice=payload.get("voice"),
            music=payload.get("music"),
        )
        return self._run_job(pipeline, task_id, job_id)

    def _do_answer(self, pipeline: Pipeline, task_id: int, job_id: str, payload: dict):
        pipeline.answer(job_id, payload["choice"])
        if payload["choice"] == "cancel":
            return "done", "cancelled: the idea's premise was false"
        return self._run_job(pipeline, task_id, job_id)

    def _do_batch(self, pipeline: Pipeline, task_id: int, job_id: str | None, payload: dict):
        """Inbox first, then new auto ideas per topic; every idea becomes its own run task."""
        store = IdeaStore(self.db)
        made = 0
        for item in store.queued():
            self.queue_job(pipeline, item.idea, item.id)
            made += 1
        skipped = 0
        for topic in payload.get("topics") or pipeline.cfg.ideas.topics:
            for idea_id in queue_auto_ideas(pipeline, topic, int(payload.get("per_topic", 1))):
                item = next(s for s in store.list("queued") if s.id == idea_id)
                try:
                    self.queue_job(pipeline, item.idea, item.id)
                    made += 1
                except DuplicateIdea:
                    skipped += 1
        return "done", f"{made} job(s) queued" + (
            f", {skipped} duplicate(s) skipped" if skipped else ""
        )

    # --- used by the web routes ------------------------------------------------------

    def queue_job(self, pipeline: Pipeline, idea, idea_id: int | None = None) -> str:
        job_id = pipeline.create_job(idea, idea_id=idea_id)
        self.db.add_task("run", job_id)
        return job_id

    def busy(self, job_id: str) -> bool:
        return bool(self.db.job_tasks(job_id, active_only=True))

    def cancel(self, job_id: str) -> None:
        """Queued work is dropped now; a running job stops at its next checkpoint."""
        running = False
        for t in self.db.job_tasks(job_id, active_only=True):
            if t["state"] == "queued":
                self.db.update_task(t["id"], state="cancelled", message="cancelled")
            else:
                self.db.update_task(t["id"], state="cancelling")
                running = True
        if not running:
            row = self.db.get_job(job_id)
            if row is not None and row["state"] in ("queued", "failed", "needs_input"):
                self.db.update_job(job_id, state="cancelled", error="cancelled by the user")
                if row["idea_id"] is not None:
                    IdeaStore(self.db).set_status(row["idea_id"], "cancelled")

    def wait_idle(self, timeout: float = 30.0) -> None:
        """Tests: wait until no task is queued or running."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rows = self.db.query(
                "SELECT 1 FROM tasks WHERE state IN ('queued', 'running', 'cancelling')"
            )
            if not rows:
                return
            time.sleep(0.05)
        raise TimeoutError("worker still busy")
