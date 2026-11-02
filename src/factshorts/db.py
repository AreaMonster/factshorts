"""SQLite storage: ideas, jobs, steps, videos, rejections, settings, tasks, HTTP cache."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    norm_title TEXT,
    article TEXT,
    angle TEXT,
    origin TEXT NOT NULL DEFAULT 'custom',
    topic TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    settings TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    idea_id INTEGER REFERENCES ideas(id),
    idea TEXT NOT NULL,
    state TEXT NOT NULL,
    current_step TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS job_steps (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    input_hash TEXT,
    started_at REAL,
    finished_at REAL,
    error TEXT,
    PRIMARY KEY (job_id, step)
);
CREATE TABLE IF NOT EXISTS videos (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id),
    title TEXT,
    topic TEXT,
    folder TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    verified INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    idea TEXT NOT NULL,
    step TEXT,
    reason TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'queued',
    message TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS http_cache (
    key TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    body TEXT NOT NULL,
    fetched_at REAL NOT NULL
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params))

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.lastrowid or 0

    # --- jobs ---------------------------------------------------------------

    def create_job(self, job_id: str, idea: dict[str, Any], idea_id: int | None = None) -> None:
        now = time.time()
        self.execute(
            "INSERT INTO jobs (id, idea_id, idea, state, created_at, updated_at)"
            " VALUES (?, ?, ?, 'queued', ?, ?)",
            (job_id, idea_id, json.dumps(idea), now, now),
        )

    def get_job(self, job_id: str) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return rows[0] if rows else None

    def update_job(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def step_status(self, job_id: str, step: str) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM job_steps WHERE job_id = ? AND step = ?", (job_id, step))
        return rows[0] if rows else None

    def set_step(
        self,
        job_id: str,
        step: str,
        status: str,
        input_hash: str | None = None,
        error: str | None = None,
    ) -> None:
        now = time.time()
        if status == "running":
            self.execute(
                "INSERT INTO job_steps (job_id, step, status, input_hash, started_at)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT(job_id, step) DO UPDATE SET"
                " status=excluded.status, input_hash=excluded.input_hash,"
                " started_at=excluded.started_at, finished_at=NULL, error=NULL",
                (job_id, step, status, input_hash, now),
            )
        else:
            self.execute(
                "UPDATE job_steps SET status = ?, finished_at = ?, error = ?"
                " WHERE job_id = ? AND step = ?",
                (status, now, error, job_id, step),
            )

    def set_step_hash(self, job_id: str, step: str, input_hash: str) -> None:
        self.execute(
            "UPDATE job_steps SET input_hash = ? WHERE job_id = ? AND step = ?",
            (input_hash, job_id, step),
        )

    def add_rejection(self, idea: str, reason: str, job_id: str | None, step: str | None) -> None:
        self.execute(
            "INSERT INTO rejections (job_id, idea, step, reason, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (job_id, idea, step, reason, time.time()),
        )

    def add_video(
        self, job_id: str, title: str, topic: str | None, folder: str, verified: bool
    ) -> None:
        self.execute(
            "INSERT OR REPLACE INTO videos (job_id, title, topic, folder, status, verified,"
            " created_at) VALUES (?, ?, ?, ?, 'pending', ?, ?)",
            (job_id, title, topic, folder, int(verified), time.time()),
        )

    def get_video(self, job_id: str) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM videos WHERE job_id = ?", (job_id,))
        return rows[0] if rows else None

    def update_video(self, job_id: str, **fields: Any) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.execute(f"UPDATE videos SET {cols} WHERE job_id = ?", (*fields.values(), job_id))

    def delete_video(self, job_id: str) -> None:
        self.execute("DELETE FROM videos WHERE job_id = ?", (job_id,))

    # --- settings (changes made in the Web UI, layered over config.toml) ------------

    def get_settings(self) -> dict[str, Any]:
        return {r["key"]: json.loads(r["value"]) for r in self.query("SELECT * FROM settings")}

    def set_setting(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, json.dumps(value))
        )

    def delete_setting(self, key: str) -> None:
        self.execute("DELETE FROM settings WHERE key = ?", (key,))

    # --- tasks (the Web UI's work queue) ------------------------------------------

    def add_task(self, kind: str, job_id: str | None = None, payload: dict | None = None) -> int:
        now = time.time()
        return self.execute(
            "INSERT INTO tasks (job_id, kind, payload, state, created_at, updated_at)"
            " VALUES (?, ?, ?, 'queued', ?, ?)",
            (job_id, kind, json.dumps(payload or {}), now, now),
        )

    def get_task(self, task_id: int) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return rows[0] if rows else None

    def claim_task(self) -> sqlite3.Row | None:
        """The oldest queued task whose job has no other task running; marked running."""
        with self.tx() as conn:
            row = conn.execute(
                "SELECT * FROM tasks t WHERE state = 'queued' AND (job_id IS NULL OR NOT EXISTS"
                " (SELECT 1 FROM tasks r WHERE r.job_id = t.job_id AND r.state IN"
                " ('running', 'cancelling'))) ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE tasks SET state = 'running', updated_at = ? WHERE id = ?",
                (time.time(), row["id"]),
            )
        return self.get_task(row["id"])

    def update_task(self, task_id: int, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.execute(f"UPDATE tasks SET {cols} WHERE id = ?", (*fields.values(), task_id))

    def job_tasks(self, job_id: str, active_only: bool = False) -> list[sqlite3.Row]:
        extra = " AND state IN ('queued', 'running', 'cancelling')" if active_only else ""
        return self.query(f"SELECT * FROM tasks WHERE job_id = ?{extra} ORDER BY id", (job_id,))

    def requeue_interrupted(self) -> int:
        """After a restart, tasks that were running start again (jobs resume)."""
        rows = self.query("SELECT id FROM tasks WHERE state IN ('running', 'cancelling')")
        self.execute(
            "UPDATE tasks SET state = 'queued', updated_at = ? WHERE state = 'running'",
            (time.time(),),
        )
        self.execute(
            "UPDATE tasks SET state = 'cancelled', updated_at = ? WHERE state = 'cancelling'",
            (time.time(),),
        )
        return len(rows)

    # --- http cache -----------------------------------------------------------

    def cache_get(self, key: str, max_age_s: float) -> str | None:
        rows = self.query("SELECT body, fetched_at FROM http_cache WHERE key = ?", (key,))
        if not rows or time.time() - rows[0]["fetched_at"] > max_age_s:
            return None
        return rows[0]["body"]

    def cache_put(self, key: str, url: str, body: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO http_cache (key, url, body, fetched_at) VALUES (?, ?, ?, ?)",
            (key, url, body, time.time()),
        )
