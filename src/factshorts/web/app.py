"""FastAPI app: Create, Jobs, Review, Library and Settings pages over the same core library
the CLI uses. Local only by default; a password is required for any other host."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from factshorts.config import ASSETS_DIR, Config, apply_overrides, load_config
from factshorts.core.pipeline import STEPS, Pipeline, Services, default_services
from factshorts.core.premise import PremiseResult
from factshorts.db import Database
from factshorts.models import Idea, Script, SourceDoc, Verification
from factshorts.web.worker import Worker

WEB_DIR = Path(__file__).parent
CHECK_NAMES = ["quote", "entities", "wikidata", "nli", "llm_judge"]
LENGTHS = [30, 45, 60, 75, 90, 120]
TONES = ["neutral", "funny", "dramatic", "kid-friendly"]
SOURCE_SUFFIXES = {".txt", ".md", ".pdf"}
OPEN_PATHS = ("/login", "/static/", "/fonts/")


class App:
    """Everything the routes share: config (config.toml + UI settings), db, worker."""

    def __init__(
        self,
        cfg: Config,
        db: Database,
        services: Services | Callable[[Config], Services] | None = None,
    ) -> None:
        self.base_cfg = cfg
        self.db = db
        self._services = services
        self.worker = Worker(db, self.pipeline, threads=self.cfg.web.workers)
        self.templates = Jinja2Templates(directory=WEB_DIR / "templates")
        self.templates.env.filters["when"] = _when
        self.templates.env.filters["duration"] = _duration
        # Changes with the stylesheet, so browsers never keep an old copy after an update.
        css = WEB_DIR / "static" / "app.css"
        version = int(css.stat().st_mtime) if css.is_file() else 0
        self.templates.env.globals["asset_version"] = version
        self.templates.env.filters["label_class"] = _label_class

    @property
    def cfg(self) -> Config:
        return apply_overrides(self.base_cfg, self.db.get_settings())

    def pipeline(self) -> Pipeline:
        cfg = self.cfg
        services = self._services
        if callable(services):
            services = services(cfg)
        return Pipeline(cfg, self.db, services or default_services(cfg, self.db))

    def render(self, request: Request, name: str, **ctx: Any) -> HTMLResponse:
        cfg = self.cfg
        ctx.setdefault("nav", name.split(".")[0].split("_")[0])
        counts = {}
        if not cfg.web.password_hash or request.session.get("auth"):
            counts = self.counts()
        return self.templates.TemplateResponse(
            request,
            name,
            {"cfg": cfg, "login": bool(cfg.web.password_hash), "counts": counts, **ctx},
        )

    def counts(self) -> dict[str, int]:
        """Numbers shown next to the navigation: videos to review, tasks working or queued."""
        review = self.db.query("SELECT COUNT(*) AS n FROM videos WHERE status = 'pending'")
        active = self.db.query(
            "SELECT COUNT(*) AS n FROM tasks WHERE state IN ('queued', 'running', 'cancelling')"
        )
        return {"review": review[0]["n"], "active": active[0]["n"]}

    def partial(self, name: str, **ctx: Any) -> str:
        return self.templates.get_template(name).render(cfg=self.cfg, **ctx)


def _when(ts: float | None) -> str:
    if not ts:
        return ""
    d = datetime.fromtimestamp(ts)
    return d.strftime("%H:%M") if d.date() == datetime.now().date() else d.strftime("%b %d, %H:%M")


def _duration(seconds: float | None) -> str:
    if not seconds:
        return ""
    s = round(seconds)
    return f"{s // 60}:{s % 60:02d}"


def _label_class(label: str | None) -> str:
    return {"VERIFIED": "ok", "PARTIALLY_VERIFIED": "warn"}.get(label or "", "bad")


def _secret_key(cfg: Config) -> str:
    path = cfg.data_dir / "web_secret.key"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def create_app(
    cfg: Config | None = None,
    db: Database | None = None,
    services: Services | Callable[[Config], Services] | None = None,
    start_worker: bool = True,
) -> FastAPI:
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    db = db or Database(cfg.db_path)
    state = App(cfg, db, services)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if start_worker:
            state.worker.start()
        yield
        state.worker.stop()

    app = FastAPI(
        title="factshorts", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.fs = state
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    # The UI uses the same bundled (OFL) display font as the videos.
    app.mount("/fonts", StaticFiles(directory=ASSETS_DIR / "fonts"), name="fonts")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Cross-site form posts (another web page posting to this local server) are refused.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return PlainTextResponse("cross-site request refused", status_code=403)
        locked = state.cfg.web.password_hash and not request.url.path.startswith(OPEN_PATHS)
        if locked and not request.session.get("auth"):
            if request.headers.get("hx-request"):
                return Response(status_code=401, headers={"HX-Redirect": "/login"})
            return RedirectResponse("/login", status_code=303)
        return await call_next(request)

    app.add_middleware(
        SessionMiddleware,
        secret_key=_secret_key(cfg),
        same_site="strict",
        session_cookie="factshorts_session",
        max_age=14 * 24 * 3600,
    )

    from factshorts.web import routes

    routes.register(app, state)
    return app


# --- data shaped for the templates ------------------------------------------------------


def job_view(state: App, row: Any) -> dict[str, Any]:
    idea = Idea.model_validate_json(row["idea"])
    steps = {
        r["step"]: r["status"]
        for r in state.db.query("SELECT step, status FROM job_steps WHERE job_id = ?", (row["id"],))
    }
    video = state.db.get_video(row["id"])
    tasks = state.db.job_tasks(row["id"])
    busy = any(t["state"] in ("queued", "running", "cancelling") for t in tasks)
    return {
        "id": row["id"],
        "idea": idea,
        "state": "cancelling"
        if any(t["state"] == "cancelling" for t in tasks)
        else "working"
        if busy and row["state"] != "running"
        else row["state"],
        "raw_state": row["state"],
        "step": row["current_step"],
        "error": row["error"],
        "created": row["created_at"],
        "updated": row["updated_at"],
        "steps": [(s, steps.get(s, "")) for s in STEPS],
        "done_steps": sum(1 for s in STEPS if steps.get(s) == "done"),
        "video": video,
        "thumb": video is not None and (Path(video["folder"]) / "thumbnail.jpg").is_file(),
        "busy": busy,
        "last_task": tasks[-1] if tasks else None,
    }


def jobs_list(state: App, limit: int = 40) -> list[dict[str, Any]]:
    rows = state.db.query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
    return [job_view(state, r) for r in rows]


def progress_lines(state: App, pipeline_cfg: Config, job_id: str, n: int = 60) -> list[str]:
    path = pipeline_cfg.work_dir / job_id / "progress.log"
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


def premise_view(cfg: Config, job_id: str) -> dict[str, Any] | None:
    wd = cfg.work_dir / job_id
    if not (wd / "premise.json").is_file():
        return None
    premise = PremiseResult.model_validate_json((wd / "premise.json").read_text("utf-8"))
    source = SourceDoc.model_validate_json((wd / "source.json").read_text("utf-8"))
    return {"items": premise.contradicted, "refs": source.refs, "note": premise.correction_note()}


def review_view(cfg: Config, db: Database, job_id: str) -> dict[str, Any]:
    wd = cfg.work_dir / job_id
    if not (wd / "script.json").is_file():
        raise HTTPException(404, "this job has no script yet")
    script = Script.model_validate_json((wd / "script.json").read_text("utf-8"))
    source = SourceDoc.model_validate_json((wd / "source.json").read_text("utf-8"))
    verification = None
    if (wd / "verification.json").is_file():
        verification = Verification.model_validate_json(
            (wd / "verification.json").read_text("utf-8")
        )
    idea = Idea.model_validate_json((wd / "idea.json").read_text("utf-8"))
    segments = []
    for seg in script.segments:
        claims, sentences = [], []
        if verification is not None:
            claims = [c for c in verification.claims if c.segment_id == seg.id]
            sentences = [
                s for s in verification.sentences if s.segment_id == seg.id and s.kind != "claim"
            ]
        segments.append({"seg": seg, "claims": claims, "sentences": sentences})
    checked = [c.passed for s in segments for c in s["claims"]]
    checked += [x.passed for s in segments for x in s["sentences"]]
    video = db.get_video(job_id)
    files = video_files(Path(video["folder"])) if video is not None else {}
    folder = Path(video["folder"]) if video is not None else None

    def text(key: str) -> str:
        return (folder / files[key]).read_text("utf-8") if folder and key in files else ""

    music_default = True
    if "metadata" in files:
        music_default = json.loads(text("metadata")).get("music_default", True)
    return {
        "description": text("description"),
        "caption": text("tiktok"),
        "music_default": music_default,
        "duration": json.loads(text("metadata")).get("duration_s") if "metadata" in files else None,
        "summary": {"total": len(checked), "failed": checked.count(False)},
        "script": script,
        "source": source,
        "verification": verification,
        "label": verification.label if verification else "UNVERIFIED",
        "idea": idea,
        "segments": segments,
        "video": video,
        "files": files,
        "check_names": CHECK_NAMES,
    }


def video_card(row: Any) -> dict[str, Any]:
    """A video for the card grids: its row, its files by role and its length."""
    folder = Path(row["folder"])
    files = video_files(folder)
    duration = None
    if "metadata" in files:
        with suppress(OSError, ValueError):
            duration = json.loads((folder / files["metadata"]).read_text("utf-8")).get("duration_s")
    return {"v": row, "duration": duration, **files}


def video_files(folder: Path) -> dict[str, str]:
    """The deliverables in a video folder by role: mp4 (with music, or the only version),
    mp4_plain (without music), description, tiktok, sources, metadata, srt, jpg."""
    files: dict[str, str] = {}
    if not folder.is_dir():
        return files
    for f in sorted(folder.iterdir()):
        if f.suffix == ".mp4":
            files["mp4_plain" if f.stem.endswith("_no-music") else "mp4"] = f.name
        elif f.name in ("description.txt", "tiktok.txt", "sources.json", "metadata.json"):
            files[f.stem] = f.name
        elif f.suffix in (".srt", ".jpg"):
            files[f.suffix.lstrip(".")] = f.name
    return files


async def sse_stream(
    request: Request, event: str, render: Callable[[], str], once: bool = False
) -> AsyncIterator[str]:
    """Send `render()` as an SSE event whenever it changes (checked every second)."""
    last = None
    quiet = 0.0
    while True:
        html = await asyncio.to_thread(render)
        digest = hashlib.sha1(html.encode()).hexdigest()
        if digest != last:
            last = digest
            data = "\n".join(f"data: {line}" for line in html.splitlines() or [""])
            yield f"event: {event}\n{data}\n\n"
            quiet = 0.0
        if once or await request.is_disconnected():
            return
        await asyncio.sleep(1.0)
        quiet += 1.0
        if quiet >= 15:
            yield ": keep-alive\n\n"
            quiet = 0.0


def sse_response(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- shared request helpers -------------------------------------------------------------


def save_upload(cfg: Config, upload: UploadFile | None) -> str | None:
    if upload is None or not upload.filename:
        return None
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in SOURCE_SUFFIXES:
        raise ValueError(f"source files must be .txt, .md or .pdf, not {suffix or 'no extension'}")
    data = upload.file.read()
    if not data:
        return None
    folder = cfg.data_dir / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(upload.filename).name)
    path = folder / f"{hashlib.sha256(data).hexdigest()[:10]}_{safe}"
    path.write_bytes(data)
    return str(path)


def idea_settings(
    cfg: Config,
    length: int,
    tone: str,
    voice: str,
    theme: str,
    music: bool,
    factcheck: bool,
    force: bool,
) -> dict[str, Any]:
    return {
        "length": length or cfg.script.default_length,
        "tone": tone or cfg.script.default_tone,
        "voice": voice if voice and voice != cfg.tts.voice else None,
        "theme": theme if theme and theme != cfg.render.theme else None,
        "music": music if music != cfg.audio.music else None,
        "factcheck": factcheck,
        "force": force,
    }


def move_video(cfg: Config, db: Database, job_id: str, status: str) -> None:
    from factshorts.core.review import set_video_status

    try:
        set_video_status(cfg, db, job_id, status)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
