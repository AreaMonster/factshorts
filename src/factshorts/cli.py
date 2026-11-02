"""Command line interface. Thin layer over the core pipeline."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Annotated, Any

import typer

from factshorts.config import Config, apply_overrides, load_config
from factshorts.db import Database
from factshorts.logging_setup import setup_logging
from factshorts.models import Idea

app = typer.Typer(
    help="factshorts: fact-checked vertical short videos, made locally.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
queue_app = typer.Typer(
    help="The idea inbox: custom ideas wait here for `batch`.", no_args_is_help=True
)
app.add_typer(queue_app, name="queue")


def _bootstrap(config: Path | None) -> tuple[Config, Database]:
    """config.toml plus the settings changed in the Web UI, so both behave the same."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # Library deprecation notices (e.g. torch.jit inside Kokoro) are not for the user.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\..*")
    base = load_config(config)
    base.ensure_dirs()
    db = Database(base.db_path)
    cfg = apply_overrides(base, db.get_settings())
    cfg.ensure_dirs()
    setup_logging(cfg.logs_dir, cfg.general.log_level, console=False)
    return cfg, db


def _progress(step: str, message: str) -> None:
    typer.echo(f"  [{step:>12}] {message}")


def _fail(message: str, code: int = 1) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code)


ConfigOpt = Annotated[Path | None, typer.Option("--config", help="Path to config.toml")]
LengthOpt = Annotated[int | None, typer.Option(help="Target length in seconds (20-180)")]
ToneOpt = Annotated[str | None, typer.Option(help="neutral, funny, dramatic, kid-friendly")]
SourceUrlOpt = Annotated[
    str | None, typer.Option("--source-url", help="Use this web page as source")
]
SourceFileOpt = Annotated[
    Path | None, typer.Option("--source-file", help="Use a .txt, .md or .pdf file as source")
]
SourceTextOpt = Annotated[str | None, typer.Option("--source-text", help="Use this text as source")]
NoFactcheckOpt = Annotated[
    bool, typer.Option("--no-factcheck", help="Opinion/creative video; labeled UNVERIFIED")
]
ThemeOpt = Annotated[str | None, typer.Option(help="Theme: default, neon, papercut, watercolor")]
VoiceOpt = Annotated[str | None, typer.Option(help="TTS voice, e.g. am_michael, af_heart")]
MusicOpt = Annotated[bool | None, typer.Option("--music/--no-music", help="Background music")]
ForceOpt = Annotated[
    bool, typer.Option("--force", help="Make it even if it looks like a duplicate")
]


def _settings(
    cfg: Config,
    length: int | None,
    tone: str | None,
    source_url: str | None,
    source_file: Path | None,
    source_text: str | None,
    no_factcheck: bool,
    force: bool,
    theme: str | None = None,
    voice: str | None = None,
    music: bool | None = None,
) -> dict[str, Any]:
    s: dict[str, Any] = {
        "length": length or cfg.script.default_length,
        "tone": tone or cfg.script.default_tone,
        "factcheck": not no_factcheck,
        "force": force,
        "theme": theme,
        "voice": voice,
        "music": music,
    }
    if source_url:
        s["source_url"] = source_url
    if source_file:
        s["source_file"] = str(source_file.resolve())
    if source_text:
        s["source_text"] = source_text
    return s


@app.command()
def make(
    idea: Annotated[
        list[str] | None, typer.Option("--idea", help="A video idea (repeatable)")
    ] = None,
    ideas_file: Annotated[
        Path | None, typer.Option("--ideas-file", help="One idea per line: 'idea | length=45'")
    ] = None,
    topic: Annotated[str | None, typer.Option(help="Generate ideas for this topic")] = None,
    count: Annotated[int, typer.Option(help="How many ideas to generate for --topic")] = 1,
    length: LengthOpt = None,
    tone: ToneOpt = None,
    source_url: SourceUrlOpt = None,
    source_file: SourceFileOpt = None,
    source_text: SourceTextOpt = None,
    no_factcheck: NoFactcheckOpt = False,
    force: ForceOpt = False,
    theme: ThemeOpt = None,
    voice: VoiceOpt = None,
    music: MusicOpt = None,
    config: ConfigOpt = None,
) -> None:
    """Make videos now: from --idea, an --ideas-file, or ideas generated for a --topic."""
    from factshorts.core.batch import queue_auto_ideas
    from factshorts.core.ideas import DuplicateIdea, IdeaStore, parse_idea_line, parse_ideas_text
    from factshorts.core.pipeline import Pipeline

    cfg, db = _bootstrap(config)
    defaults = _settings(
        cfg,
        length,
        tone,
        source_url,
        source_file,
        source_text,
        no_factcheck,
        force,
        theme,
        voice,
        music,
    )
    pipeline = Pipeline(cfg, db)
    todo: list[tuple[Idea, int | None]] = []
    try:
        todo += [(parse_idea_line(i, defaults), None) for i in idea or []]
        if ideas_file:
            text = ideas_file.read_text(encoding="utf-8")
            todo += [(i, None) for i in parse_ideas_text(text, defaults)]
    except (ValueError, OSError) as exc:
        _fail(f"Could not read the ideas: {exc}")
    if topic:
        typer.echo(f"Generating {count} idea(s) for '{topic}'...")
        store = IdeaStore(db)
        for idea_id in queue_auto_ideas(pipeline, topic, count):
            item = next(s for s in store.list() if s.id == idea_id)
            todo.append((item.idea, item.id))
    if not todo:
        _fail("Nothing to make: give --idea, --ideas-file or --topic.")
    exit_code = 0
    for n, (item, idea_id) in enumerate(todo, start=1):
        try:
            job_id = pipeline.create_job(item, idea_id=idea_id)
        except DuplicateIdea as exc:
            typer.echo(f"\nSkipped: {exc}")
            continue
        typer.echo(f"\n[{n}/{len(todo)}] Job {job_id}: {item.text}")
        exit_code = max(exit_code, _run(pipeline, job_id))
    raise typer.Exit(exit_code)


@app.command()
def resume(
    job_id: str,
    from_step: Annotated[
        str | None,
        typer.Option("--from", help="Redo from this step, e.g. verification or voice"),
    ] = None,
    config: ConfigOpt = None,
) -> None:
    """Resume a job from its last completed step (or redo it from --from STEP)."""
    from factshorts.core.pipeline import Pipeline

    cfg, db = _bootstrap(config)
    if db.get_job(job_id) is None:
        _fail(f"No job {job_id}")
    pipeline = Pipeline(cfg, db)
    if from_step:
        try:
            pipeline.redo_from(job_id, from_step)
        except ValueError as exc:
            _fail(str(exc))
    raise typer.Exit(_run(pipeline, job_id))


@app.command()
def answer(
    job_id: str,
    choice: Annotated[str, typer.Argument(help="corrected or cancel")],
    config: ConfigOpt = None,
) -> None:
    """Answer a job waiting for input (false premise): make a corrected video, or cancel."""
    from factshorts.core.pipeline import Pipeline

    cfg, db = _bootstrap(config)
    pipeline = Pipeline(cfg, db)
    try:
        pipeline.answer(job_id, choice)
    except ValueError as exc:
        _fail(str(exc))
    if choice == "cancel":
        typer.echo(f"Job {job_id} cancelled.")
        return
    typer.echo(f"Making a corrected video for job {job_id}...")
    raise typer.Exit(_run(pipeline, job_id))


@app.command()
def rerender(
    job_id: str,
    theme: ThemeOpt = None,
    voice: VoiceOpt = None,
    music: MusicOpt = None,
    config: ConfigOpt = None,
) -> None:
    """Render a job again with another theme, voice or music; script and checks are kept."""
    from factshorts.core.pipeline import Pipeline

    cfg, db = _bootstrap(config)
    pipeline = Pipeline(cfg, db)
    try:
        pipeline.rerender(job_id, theme=theme, voice=voice, music=music)
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    raise typer.Exit(_run(pipeline, job_id))


@app.command()
def jobs(
    limit: Annotated[int, typer.Option(help="How many recent jobs to show")] = 20,
    config: ConfigOpt = None,
) -> None:
    """Recent jobs and their state."""
    _, db = _bootstrap(config)
    rows = db.query(
        "SELECT id, state, current_step, idea, error FROM jobs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    for r in rows:
        idea = Idea.model_validate_json(r["idea"]).text
        extra = f" ({r['current_step']})" if r["state"] in ("running", "failed") else ""
        typer.echo(f"{r['id']}  {r['state']:<12}{extra}  {idea}")
        if r["state"] in ("rejected", "needs_input", "failed", "cancelled") and r["error"]:
            typer.echo(f"    {r['error'].splitlines()[0][:150]}")


@app.command()
def batch(
    per_topic: Annotated[int | None, typer.Option(help="Auto ideas per topic")] = None,
    topic: Annotated[list[str] | None, typer.Option(help="Only these topics (repeatable)")] = None,
    config: ConfigOpt = None,
) -> None:
    """Make videos for every queued idea, then for new auto ideas per topic."""
    from factshorts.core.batch import run_batch
    from factshorts.core.pipeline import Pipeline

    cfg, db = _bootstrap(config)
    report = run_batch(
        Pipeline(cfg, db),
        per_topic if per_topic is not None else cfg.ideas.per_topic,
        topic or None,
        progress=lambda m: typer.echo(f"\n== {m}"),
        step_progress=_progress,
    )
    typer.echo(
        f"\nBatch finished: {len(report.done)} made, {len(report.rejected)} rejected, "
        f"{len(report.needs_input)} need your answer, {len(report.failed)} failed."
    )
    for folder in report.done:
        typer.echo(f"  made: {folder}")
    for line in report.rejected:
        typer.echo(f"  rejected: {line[:200]}")
    for job_id in report.needs_input:
        typer.echo(f"  needs input: factshorts answer {job_id} corrected|cancel")
    for line in report.failed:
        typer.echo(f"  failed: {line[:200]}")


# --- queue -------------------------------------------------------------------------------


@queue_app.command("add")
def queue_add(
    idea: Annotated[
        list[str] | None, typer.Argument(help="Idea(s); 'idea | length=45' works")
    ] = None,
    ideas_file: Annotated[Path | None, typer.Option("--ideas-file")] = None,
    length: LengthOpt = None,
    tone: ToneOpt = None,
    source_url: SourceUrlOpt = None,
    source_file: SourceFileOpt = None,
    source_text: SourceTextOpt = None,
    no_factcheck: NoFactcheckOpt = False,
    force: ForceOpt = False,
    theme: ThemeOpt = None,
    voice: VoiceOpt = None,
    music: MusicOpt = None,
    config: ConfigOpt = None,
) -> None:
    """Add custom ideas to the inbox."""
    from factshorts.core.ideas import DuplicateIdea, IdeaStore, parse_idea_line, parse_ideas_text

    cfg, db = _bootstrap(config)
    defaults = _settings(
        cfg,
        length,
        tone,
        source_url,
        source_file,
        source_text,
        no_factcheck,
        force,
        theme,
        voice,
        music,
    )
    try:
        ideas = [parse_idea_line(i, defaults) for i in idea or []]
        if ideas_file:
            ideas += parse_ideas_text(ideas_file.read_text(encoding="utf-8"), defaults)
    except (ValueError, OSError) as exc:
        _fail(f"Could not read the ideas: {exc}")
    store = IdeaStore(db)
    for item in ideas:
        try:
            idea_id = store.add(item)
            typer.echo(f"queued #{idea_id}: {item.text}")
        except DuplicateIdea as exc:
            typer.echo(f"skipped: {exc}")


@queue_app.command("generate")
def queue_generate(
    topic: Annotated[str, typer.Option(help="Topic for the new ideas")],
    count: Annotated[int, typer.Option(help="How many ideas")] = 5,
    config: ConfigOpt = None,
) -> None:
    """Generate ideas for a topic (each with its Wikipedia article) and queue them."""
    from factshorts.core.batch import queue_auto_ideas
    from factshorts.core.ideas import IdeaStore
    from factshorts.core.pipeline import Pipeline

    cfg, db = _bootstrap(config)
    ids = queue_auto_ideas(Pipeline(cfg, db), topic, count)
    store = IdeaStore(db)
    for s in store.list():
        if s.id in ids:
            typer.echo(f"queued #{s.id}: {s.idea.text}   [Wikipedia: {s.article}]")
    if not ids:
        typer.echo("No new ideas (all candidates were duplicates or had no Wikipedia article).")


@queue_app.command("list")
def queue_list(
    all_: Annotated[bool, typer.Option("--all", help="Show every idea, not only queued")] = False,
    config: ConfigOpt = None,
) -> None:
    """Show the queued ideas (custom first, then auto)."""
    from factshorts.core.ideas import IdeaStore

    _, db = _bootstrap(config)
    store = IdeaStore(db)
    items = store.list() if all_ else store.queued()
    if not items:
        typer.echo("The queue is empty.")
    for s in items:
        source = f"  [Wikipedia: {s.article}]" if s.article else ""
        flag = "" if s.idea.factcheck else "  [no fact-check]"
        typer.echo(f"#{s.id:<4} {s.status:<11} {s.idea.origin:<6} {s.idea.text}{source}{flag}")


@queue_app.command("clear")
def queue_clear(config: ConfigOpt = None) -> None:
    """Remove all queued ideas (they are marked cancelled, not deleted)."""
    from factshorts.core.ideas import IdeaStore

    _, db = _bootstrap(config)
    typer.echo(f"Cleared {IdeaStore(db).clear()} queued idea(s).")


@app.command()
def review(config: ConfigOpt = None) -> None:
    """Review pending videos in the terminal: approve, reject or skip each one."""
    from factshorts.core.review import review_item, set_video_status

    cfg, db = _bootstrap(config)
    rows = db.query("SELECT job_id FROM videos WHERE status = 'pending' ORDER BY created_at")
    if not rows:
        typer.echo("Nothing to review.")
        return
    typer.echo(
        "Automatic checks reduce mistakes but cannot rule them out: watch each video and read "
        "its claims before approving."
    )
    for n, row in enumerate(rows, start=1):
        try:
            item = review_item(cfg, db, row["job_id"])
        except (ValueError, OSError) as exc:
            typer.echo(f"\n[{n}/{len(rows)}] {row['job_id']}: cannot show it ({exc})")
            continue
        typer.echo(f"\n[{n}/{len(rows)}] {item.title}  [{item.label}]")
        typer.echo(f"  video:   {item.video or '(missing)'}")
        typer.echo(f"  sources: {', '.join(item.sources)}")
        for line in item.lines:
            typer.echo(f"  {line}")
        choice = typer.prompt("  [a]pprove, [r]eject, [s]kip, [q]uit", default="s").strip().lower()
        if choice.startswith("q"):
            break
        if choice[:1] in ("a", "r"):
            status = "approved" if choice.startswith("a") else "rejected"
            folder = set_video_status(cfg, db, item.job_id, status)
            typer.echo(f"  {status}: {folder}")


@app.command()
def ui(
    host: Annotated[str | None, typer.Option(help="Address to listen on (default: config)")] = None,
    port: Annotated[int | None, typer.Option(help="Port (default from config)")] = None,
    config: ConfigOpt = None,
) -> None:
    """Start the Web UI (http://127.0.0.1:8000 by default)."""
    import uvicorn

    from factshorts.core.doctor import is_local_host
    from factshorts.web.app import create_app

    cfg, db = _bootstrap(config)
    host = host or cfg.web.host
    port = port or cfg.web.port
    if not is_local_host(host) and not cfg.web.password_hash:
        _fail(
            f"Refusing to listen on {host}: other devices could reach the UI, so a password is "
            "required. Set one with `factshorts password` (or on the Settings page)."
        )
    base = load_config(config)
    app_ = create_app(base, db)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    typer.echo(f"factshorts UI on http://{shown}:{port}  (Ctrl+C to stop)")
    uvicorn.run(app_, host=host, port=port, log_level="warning")


@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """Check the setup: Python, ffmpeg, Ollama and models, voice, NLI model, folders, config."""
    from factshorts.core.doctor import run_checks

    cfg, _ = _bootstrap(config)
    checks = run_checks(cfg)
    marks = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
    for c in checks:
        typer.echo(f"[{marks[c.status]}] {c.name:<17} {c.detail}")
    failed = [c for c in checks if c.status == "fail"]
    typer.echo(f"\n{len(failed)} problem(s)." if failed else "\nAll good.")
    raise typer.Exit(1 if failed else 0)


@app.command()
def password(
    remove: Annotated[bool, typer.Option("--remove", help="Remove the password")] = False,
    config: ConfigOpt = None,
) -> None:
    """Set the Web UI password (needed to open the UI from another device)."""
    from argon2 import PasswordHasher

    _, db = _bootstrap(config)
    if remove:
        db.delete_setting("web.password_hash")
        typer.echo("Password removed. The UI now only works on this computer.")
        return
    pw = typer.prompt("New password", hide_input=True, confirmation_prompt=True)
    if len(pw) < 8:
        _fail("Use at least 8 characters.")
    db.set_setting("web.password_hash", PasswordHasher().hash(pw))
    typer.echo("Password set.")


def _run(pipeline: Any, job_id: str) -> int:
    try:
        result = pipeline.run(job_id, progress=_progress)
    except Exception as exc:  # the traceback is in the log, not on screen
        typer.echo(
            f"\nFailed: {exc}\nDetails: {pipeline.workdir(job_id).path('job.log')}", err=True
        )
        return 1
    if result is None:
        row = pipeline.db.get_job(job_id)
        if row and row["state"] == "needs_input":
            typer.echo(
                f"\nWaiting for your answer:\n  factshorts answer {job_id} corrected"
                f"\n  factshorts answer {job_id} cancel"
            )
            return 3
        typer.echo("\nRejected. See the reason above.")
        return 2
    typer.echo(f"\nDone. Review the video before publishing:\n  {result.video}")
    return 0


if __name__ == "__main__":
    app()
