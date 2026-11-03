"""Routes for the five pages, live updates (SSE), media and a small JSON API."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)

from factshorts.core.batch import idea_defaults
from factshorts.core.doctor import run_checks
from factshorts.core.ideas import DuplicateIdea, IdeaStore, generate_ideas, parse_ideas_text
from factshorts.core.tts import KOKORO_VOICES
from factshorts.models import Idea
from factshorts.render.themes import list_themes
from factshorts.web.app import (
    LENGTHS,
    TONES,
    App,
    idea_settings,
    job_view,
    jobs_list,
    move_video,
    premise_view,
    progress_lines,
    review_view,
    save_upload,
    sse_response,
    sse_stream,
    video_card,
)

Text = Annotated[str, Form()]
OptText = Annotated[str, Form()]


def _back(url: str, request: Request) -> Response:
    """Redirect after a form post (HTMX follows HX-Redirect, plain forms a 303)."""
    if request.headers.get("hx-request"):
        return Response(status_code=204, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=303)


def _choices(state: App) -> dict[str, Any]:
    cfg = state.cfg
    return {
        "lengths": LENGTHS,
        "tones": TONES,
        "themes": list_themes(),
        "voices": KOKORO_VOICES if cfg.tts.engine == "kokoro" else [cfg.tts.piper_voice],
    }


def register(app: FastAPI, state: App) -> None:
    db = state.db

    def job_or_404(job_id: str) -> Any:
        row = db.get_job(job_id)
        if row is None:
            raise HTTPException(404, f"no job {job_id}")
        return row

    def not_busy(job_id: str) -> None:
        if state.worker.busy(job_id):
            raise HTTPException(409, "this job is still working; wait until it has finished")

    # --- login -------------------------------------------------------------------

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, error: str = ""):
        return state.render(request, "login.html", error=error)

    @app.post("/login")
    def login(request: Request, password: Text):
        from argon2 import PasswordHasher
        from argon2.exceptions import Argon2Error

        hashed = state.cfg.web.password_hash
        try:
            ok = bool(hashed) and PasswordHasher().verify(hashed, password)
        except Argon2Error:
            ok = False
        if not ok:
            return RedirectResponse("/login?error=Wrong+password", status_code=303)
        request.session["auth"] = True
        return RedirectResponse("/", status_code=303)

    @app.post("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    # --- create --------------------------------------------------------------------

    @app.get("/")
    def home():
        return RedirectResponse("/create")

    def create_ctx() -> dict[str, Any]:
        return {**_choices(state), "inbox": IdeaStore(db).queued()}

    @app.get("/create", response_class=HTMLResponse)
    def create_page(request: Request, message: str = "", error: str = ""):
        return state.render(request, "create.html", message=message, error=error, **create_ctx())

    @app.post("/create")
    def create(
        request: Request,
        ideas: Text = "",
        source_url: OptText = "",
        source_text: OptText = "",
        source_file: UploadFile | None = None,
        length: Annotated[int, Form()] = 0,
        tone: OptText = "",
        voice: OptText = "",
        theme: OptText = "",
        music: Annotated[bool, Form()] = False,
        factcheck: Annotated[bool, Form()] = False,
        force: Annotated[bool, Form()] = False,
        action: OptText = "make",
    ):
        cfg = state.cfg
        settings = idea_settings(cfg, length, tone, voice, theme, music, factcheck, force)
        try:
            path = save_upload(cfg, source_file)
            if path:
                settings["source_file"] = path
            if source_url.strip():
                settings["source_url"] = source_url.strip()
            if source_text.strip():
                settings["source_text"] = source_text.strip()
            items = parse_ideas_text(ideas, settings)
        except ValueError as exc:
            return state.render(request, "create.html", error=str(exc), **create_ctx())
        if not items:
            return state.render(
                request, "create.html", error="Write at least one idea.", **create_ctx()
            )
        made, skipped = _make_or_queue(state, items, action)
        message = (
            f"{len(made)} video(s) started." if action == "make" else f"{len(made)} idea(s) queued."
        )
        if skipped:
            message += " Skipped: " + " ".join(skipped)
        if action == "make" and made:
            return RedirectResponse("/jobs", status_code=303)
        return state.render(request, "create.html", message=message, **create_ctx())

    @app.post("/ideas/generate", response_class=HTMLResponse)
    def ideas_generate(
        request: Request,
        topics: Annotated[list[str] | None, Form()] = None,
        custom_topic: OptText = "",
        count: Annotated[int, Form()] = 3,
    ):
        pipeline = state.pipeline()
        wanted = [t for t in (topics or []) if t] + (
            [custom_topic.strip()] if custom_topic.strip() else []
        )
        if not wanted:
            return HTMLResponse('<p class="error">Pick at least one topic.</p>')
        store = IdeaStore(db)
        proposals, errors = [], []
        for topic in wanted:
            try:
                found = generate_ideas(
                    topic,
                    max(1, min(count, 10)),
                    pipeline.services.llm,
                    pipeline.cfg.ollama.writer_model,
                    pipeline.services.wiki,
                    store,
                    idea_defaults(pipeline.cfg),
                )
            except Exception as exc:
                errors.append(f"{topic}: {exc}")
                continue
            proposals += [
                {"text": r.idea.text, "article": r.article, "angle": r.angle, "topic": topic}
                for r in found
            ]
        return HTMLResponse(state.partial("_proposals.html", proposals=proposals, errors=errors))

    @app.post("/ideas/add")
    async def ideas_add(request: Request):
        form = await request.form()
        action = str(form.get("action", "queue"))
        cfg = state.cfg
        items: list[Idea] = []
        for key in form:
            if not (key.startswith("p-") and key.endswith("-pick")):
                continue
            n = key[2:-5]
            text = str(form.get(f"p-{n}-text", "")).strip()
            if not text:
                continue
            items.append(
                Idea(
                    text=text,
                    origin="auto",
                    topic=str(form.get(f"p-{n}-topic", "")) or None,
                    article=str(form.get(f"p-{n}-article", "")) or None,
                    angle=str(form.get(f"p-{n}-angle", "")) or None,
                    **idea_defaults(cfg),
                )
            )
        made, skipped = _make_or_queue(state, items, action)
        if action == "make":
            return _back("/jobs", request)
        msg = f"{len(made)} idea(s) queued." + (" Skipped: " + " ".join(skipped) if skipped else "")
        return _back(f"/create?message={msg}", request)

    # --- jobs ------------------------------------------------------------------------

    def jobs_partial() -> str:
        return state.partial("_jobs_list.html", jobs=jobs_list(state))

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request):
        rejections = db.query("SELECT * FROM rejections ORDER BY created_at DESC LIMIT 30")
        return state.render(
            request,
            "jobs.html",
            jobs_html=jobs_partial(),
            rejections=rejections,
        )

    @app.get("/jobs/stream")
    async def jobs_stream(request: Request, once: bool = False):
        return sse_response(sse_stream(request, "jobs", jobs_partial, once))

    def job_partial(job_id: str) -> str:
        row = job_or_404(job_id)
        job = job_view(state, row)
        cfg = state.cfg
        return state.partial(
            "_job_live.html",
            job=job,
            lines=progress_lines(state, cfg, job_id),
            premise=premise_view(cfg, job_id) if row["state"] == "needs_input" else None,
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, job_id: str):
        row = job_or_404(job_id)
        return state.render(
            request, "job.html", job=job_view(state, row), live_html=job_partial(job_id)
        )

    @app.get("/jobs/{job_id}/stream")
    async def job_stream(request: Request, job_id: str, once: bool = False):
        job_or_404(job_id)
        return sse_response(sse_stream(request, "job", lambda: job_partial(job_id), once))

    @app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str):
        job_or_404(job_id)
        path = state.cfg.work_dir / job_id / "job.log"
        return path.read_text("utf-8", errors="replace") if path.is_file() else "(no log yet)"

    @app.post("/jobs/{job_id}/cancel")
    def job_cancel(request: Request, job_id: str):
        job_or_404(job_id)
        state.worker.cancel(job_id)
        return _back(f"/jobs/{job_id}", request)

    @app.post("/jobs/{job_id}/retry")
    def job_retry(request: Request, job_id: str):
        row = job_or_404(job_id)
        not_busy(job_id)
        if row["state"] == "done":
            raise HTTPException(409, "this job is already done")
        db.update_job(job_id, state="queued", error=None)
        db.add_task("run", job_id)
        return _back(f"/jobs/{job_id}", request)

    @app.post("/jobs/{job_id}/answer")
    def job_answer(request: Request, job_id: str, choice: Text):
        row = job_or_404(job_id)
        if row["state"] != "needs_input" or choice not in ("corrected", "cancel"):
            raise HTTPException(409, "this job is not waiting for an answer")
        not_busy(job_id)
        db.add_task("answer", job_id, {"choice": choice})
        return _back(f"/jobs/{job_id}", request)

    @app.post("/inbox/{idea_id}/make")
    def inbox_make(request: Request, idea_id: int):
        store = IdeaStore(db)
        item = next((s for s in store.queued() if s.id == idea_id), None)
        if item is None:
            raise HTTPException(404, "this idea is not in the inbox")
        job_id = state.worker.queue_job(state.pipeline(), item.idea, item.id)
        return _back(f"/jobs/{job_id}", request)

    @app.post("/inbox/{idea_id}/remove")
    def inbox_remove(request: Request, idea_id: int):
        IdeaStore(db).set_status(idea_id, "cancelled")
        return _back("/create", request)

    @app.post("/batch")
    def batch(
        request: Request,
        per_topic: Annotated[int, Form()] = 1,
        topics: Annotated[list[str] | None, Form()] = None,
    ):
        db.add_task("batch", None, {"per_topic": max(0, per_topic), "topics": topics or []})
        return _back("/jobs", request)

    # --- review ----------------------------------------------------------------------

    @app.get("/review", response_class=HTMLResponse)
    def review_list(request: Request):
        videos = db.query("SELECT * FROM videos WHERE status = 'pending' ORDER BY created_at DESC")
        return state.render(request, "review_list.html", items=[video_card(v) for v in videos])

    @app.get("/review/{job_id}", response_class=HTMLResponse)
    def review_page(request: Request, job_id: str, message: str = ""):
        row = job_or_404(job_id)
        cfg = state.cfg
        return state.render(
            request,
            "review.html",
            nav="review",
            job=job_view(state, row),
            message=message,
            **review_view(cfg, db, job_id),
            **_choices(state),
        )

    @app.post("/review/{job_id}/approve")
    def approve(request: Request, job_id: str):
        job_or_404(job_id)
        not_busy(job_id)
        move_video(state.cfg, db, job_id, "approved")
        return _back("/review", request)

    @app.post("/review/{job_id}/reject")
    def reject(request: Request, job_id: str):
        job_or_404(job_id)
        not_busy(job_id)
        move_video(state.cfg, db, job_id, "rejected")
        return _back("/review", request)

    @app.post("/review/{job_id}/edit")
    def edit(
        request: Request,
        job_id: str,
        segment: Text,
        narration: Text,
        on_screen_text: OptText = "",
    ):
        job_or_404(job_id)
        not_busy(job_id)
        if not narration.strip():
            raise HTTPException(400, "the narration cannot be empty")
        db.add_task(
            "edit",
            job_id,
            {"segment": segment, "narration": narration, "on_screen_text": on_screen_text.strip()},
        )
        return _back(f"/jobs/{job_id}", request)

    @app.post("/review/{job_id}/regenerate")
    def regenerate(request: Request, job_id: str, segment: Text):
        job_or_404(job_id)
        not_busy(job_id)
        db.add_task("regenerate", job_id, {"segment": segment})
        return _back(f"/jobs/{job_id}", request)

    @app.post("/review/{job_id}/rerender")
    def rerender(
        request: Request,
        job_id: str,
        theme: OptText = "",
        voice: OptText = "",
        music: Annotated[bool, Form()] = False,
    ):
        job_or_404(job_id)
        not_busy(job_id)
        db.add_task(
            "rerender",
            job_id,
            {"theme": theme or None, "voice": voice or None, "music": music},
        )
        return _back(f"/jobs/{job_id}", request)

    @app.get("/media/{job_id}/{name}")
    def media(job_id: str, name: str, download: bool = False):
        video = db.get_video(job_id)
        if video is None:
            raise HTTPException(404, "no video")
        folder = Path(video["folder"])
        path = folder / name
        # Only files that are really in the video's folder (no ../ tricks).
        if not folder.is_dir() or name not in {p.name for p in folder.iterdir()}:
            raise HTTPException(404, "no such file")
        return FileResponse(path, filename=name if download else None)

    # --- library ------------------------------------------------------------------

    @app.get("/library", response_class=HTMLResponse)
    def library(
        request: Request,
        q: str = "",
        topic: str = "",
        verified: str = "",
        since: str = "",
        status: str = "approved",
    ):
        sql = "SELECT * FROM videos WHERE status = ?"
        params: list[Any] = [
            status if status in ("approved", "rejected", "pending") else "approved"
        ]
        if q:
            sql += " AND title LIKE ?"
            params.append(f"%{q}%")
        if topic:
            sql += " AND topic = ?"
            params.append(topic)
        if verified in ("yes", "no"):
            sql += " AND verified = ?"
            params.append(1 if verified == "yes" else 0)
        if since:
            from datetime import datetime

            try:
                sql += " AND created_at >= ?"
                params.append(datetime.fromisoformat(since).timestamp())
            except ValueError:
                sql = sql.removesuffix(" AND created_at >= ?")
        videos = db.query(sql + " ORDER BY created_at DESC", tuple(params))
        topics = [
            r["topic"]
            for r in db.query(
                "SELECT DISTINCT topic FROM videos WHERE topic IS NOT NULL ORDER BY topic"
            )
        ]
        items = [video_card(v) for v in videos]
        by_status = {
            r["status"]: r["n"]
            for r in db.query("SELECT status, COUNT(*) AS n FROM videos GROUP BY status")
        }
        return state.render(
            request,
            "library.html",
            items=items,
            topics=topics,
            q=q,
            topic=topic,
            verified=verified,
            since=since,
            status=status,
            by_status=by_status,
        )

    @app.post("/library/{job_id}/delete")
    def library_delete(request: Request, job_id: str):
        video = db.get_video(job_id)
        if video is None:
            raise HTTPException(404, "no video")
        not_busy(job_id)
        folder = Path(video["folder"])
        if folder.is_dir() and folder.resolve().is_relative_to(state.cfg.output_dir.resolve()):
            shutil.rmtree(folder)
        db.delete_video(job_id)
        return _back("/library", request)

    # --- settings -----------------------------------------------------------------

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request, message: str = "", error: str = ""):
        from factshorts.core.llm import OllamaLLM

        cfg = state.cfg
        try:
            models = OllamaLLM(cfg.ollama).list_models()
        except Exception:
            models = []
        piper_dir = cfg.models_dir / "piper"
        piper = sorted(p.stem for p in piper_dir.rglob("*.onnx")) if piper_dir.is_dir() else []
        return state.render(
            request,
            "settings.html",
            models=models,
            piper_voices=piper,
            overrides=db.get_settings(),
            message=message,
            error=error,
            kokoro_voices=KOKORO_VOICES,
            **_choices(state),
        )

    @app.post("/settings")
    async def settings_save(request: Request):
        from factshorts.config import apply_overrides

        form = await request.form()
        bools = [k[len("bool:") :] for k in form if k.startswith("bool:")]
        changes: dict[str, Any] = {}
        for key, value in form.multi_items():
            if key.startswith("bool:") or key.startswith("_"):
                continue
            value = str(value)
            if key == "ideas.topics":
                changes[key] = [t.strip() for t in value.splitlines() if t.strip()]
            else:
                changes[key] = value.strip()
        for key in bools:
            changes[key] = key in form
        current = state.cfg.model_dump()
        try:
            for key, value in list(changes.items()):
                section, _, name = key.partition(".")
                old = current.get(section, {}).get(name)
                if isinstance(old, bool) and not isinstance(value, bool):
                    value = value in ("on", "true", "1")
                elif isinstance(old, int) and not isinstance(old, bool):
                    value = int(value)
                elif isinstance(old, float):
                    value = float(value)
                changes[key] = value
            apply_overrides(state.base_cfg, {**db.get_settings(), **changes})
        except Exception as exc:  # pydantic tells which value is wrong
            return _back(f"/settings?error={str(exc).splitlines()[0][:200]}", request)
        base = state.base_cfg.model_dump()
        for key, value in changes.items():
            section, _, name = key.partition(".")
            if base.get(section, {}).get(name) == value:
                db.delete_setting(key)  # back to the config.toml value
            else:
                db.set_setting(key, value)
        state.worker.settings_version += 1
        return _back("/settings?message=Saved", request)

    @app.post("/settings/reset")
    def settings_reset(request: Request):
        for key in list(db.get_settings()):
            if key != "web.password_hash":
                db.delete_setting(key)
        state.worker.settings_version += 1
        return _back("/settings?message=Back+to+the+values+in+config.toml", request)

    @app.post("/settings/password")
    def settings_password(request: Request, password: OptText = "", confirm: OptText = ""):
        if not password:
            db.delete_setting("web.password_hash")
            return _back("/settings?message=Password+removed+(local+use+only)", request)
        if password != confirm:
            return _back("/settings?error=The+passwords+do+not+match", request)
        if len(password) < 8:
            return _back("/settings?error=Use+at+least+8+characters", request)
        from argon2 import PasswordHasher

        db.set_setting("web.password_hash", PasswordHasher().hash(password))
        request.session["auth"] = True
        return _back("/settings?message=Password+set", request)

    @app.get("/settings/health", response_class=HTMLResponse)
    def settings_health():
        return HTMLResponse(state.partial("_health.html", checks=run_checks(state.cfg)))

    @app.get("/settings/voice-preview")
    def voice_preview(engine: str = "kokoro", voice: str = ""):
        import wave

        from factshorts.core.tts import make_synth

        cfg = state.cfg
        if engine == "kokoro" and voice not in KOKORO_VOICES:
            raise HTTPException(400, "unknown Kokoro voice")
        folder = cfg.cache_dir / "voice_previews"
        folder.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in voice if c.isalnum() or c in "_-")
        path = folder / f"{engine}_{safe}.wav"
        if not path.is_file():
            update = {"engine": engine, "voice" if engine == "kokoro" else "piper_voice": voice}
            try:
                synth = make_synth(cfg.tts.model_copy(update=update), cfg.models_dir)
            except Exception as exc:
                raise HTTPException(400, f"cannot load voice {voice}: {exc}") from exc
            result = synth.synthesize(
                "Mount Everest is 8,849 meters tall. That's almost nine kilometers!"
            )
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(synth.sample_rate)
                wf.writeframes(result.samples.astype("<i2").tobytes())
            del synth
        return FileResponse(path, media_type="audio/wav")

    @app.post("/settings/piper-download")
    def piper_download(request: Request, voice: Text):
        from factshorts.core.tts import ensure_piper_voice

        try:
            ensure_piper_voice(voice.strip(), state.cfg.models_dir)
        except Exception as exc:
            return _back(f"/settings?error=Download failed: {exc}"[:300], request)
        return _back(f"/settings?message=Downloaded {voice}", request)

    # --- JSON API (for scripts and tests) -------------------------------------------

    @app.get("/api/jobs")
    def api_jobs():
        return JSONResponse(
            [
                {
                    "id": j["id"],
                    "idea": j["idea"].text,
                    "state": j["raw_state"],
                    "step": j["step"],
                    "error": j["error"],
                    "busy": j["busy"],
                }
                for j in jobs_list(state, 200)
            ]
        )

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        j = job_view(state, job_or_404(job_id))
        return JSONResponse(
            {
                "id": j["id"],
                "idea": j["idea"].model_dump(),
                "state": j["raw_state"],
                "step": j["step"],
                "error": j["error"],
                "steps": dict(j["steps"]),
                "busy": j["busy"],
                "tasks": [
                    {"kind": t["kind"], "state": t["state"], "message": t["message"]}
                    for t in db.job_tasks(job_id)
                ],
                "video": dict(j["video"]) if j["video"] else None,
            }
        )

    @app.get("/api/health")
    def api_health():
        return JSONResponse([c.__dict__ for c in run_checks(state.cfg)])


def _make_or_queue(state: App, items: list[Idea], action: str) -> tuple[list[str], list[str]]:
    """Start jobs now ("make") or put the ideas in the inbox; duplicates are reported."""
    store = IdeaStore(state.db)
    pipeline = state.pipeline() if action == "make" else None
    made, skipped = [], []
    for idea in items:
        try:
            if pipeline is not None:
                made.append(state.worker.queue_job(pipeline, idea))
            else:
                made.append(str(store.add(idea, article=idea.article, angle=idea.angle)))
        except DuplicateIdea as exc:
            skipped.append(str(exc))
    return made, skipped
