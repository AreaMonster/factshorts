"""Rotating log file + console, and per-job log files."""

from __future__ import annotations

import logging
import logging.handlers
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_configured = False


def setup_logging(logs_dir: Path, level: str = "INFO", console: bool = True) -> None:
    global _configured
    if _configured:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("factshorts")
    root.setLevel(level.upper())
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "factshorts.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        root.addHandler(console_handler)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


@contextmanager
def job_log(log_file: Path) -> Iterator[logging.Handler]:
    """Mirror everything logged under 'factshorts' into a job's own log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT))
    # Only this thread's records: the Web UI may run several jobs at once.
    thread = threading.get_ident()
    handler.addFilter(lambda record: record.thread == thread)
    root = logging.getLogger("factshorts")
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        handler.close()
