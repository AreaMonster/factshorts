"""Command line interface. Thin layer over the core pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from factshorts.config import Config, load_config
from factshorts.db import Database
from factshorts.logging_setup import setup_logging
from factshorts.models import Idea

app = typer.Typer(
    help="factshorts: fact-checked vertical short videos, made locally.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _bootstrap(config: Path | None) -> tuple[Config, Database]:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(config)
    cfg.ensure_dirs()
    setup_logging(cfg.logs_dir, cfg.general.log_level, console=False)
    return cfg, Database(cfg.db_path)


def _progress(step: str, message: str) -> None:
    typer.echo(f"  [{step:>12}] {message}")


ConfigOpt = Annotated[Path | None, typer.Option("--config", help="Path to config.toml")]


@app.command()
def make(
    idea: Annotated[str, typer.Option("--idea", help="The video idea, e.g. a question")],
    length: Annotated[
        int | None, typer.Option(help="Target length in seconds (20-180, default 75)")
    ] = None,
    tone: Annotated[str | None, typer.Option(help="neutral, funny, dramatic, kid-friendly")] = None,
    config: ConfigOpt = None,
) -> None:
    """Make one video from a custom idea."""
    from factshorts.core.pipeline import Pipeline

    cfg, db = _bootstrap(config)
    idea_model = Idea(
        text=idea.strip(),
        length=length or cfg.script.default_length,
        tone=tone or cfg.script.default_tone,  # type: ignore[arg-type]
    )
    pipeline = Pipeline(cfg, db)
    job_id = pipeline.create_job(idea_model)
    typer.echo(f"Job {job_id}: {idea_model.text}")
    _run(pipeline, job_id)


@app.command()
def resume(job_id: str, config: ConfigOpt = None) -> None:
    """Resume a job from its last completed step."""
    from factshorts.core.pipeline import Pipeline

    cfg, db = _bootstrap(config)
    if db.get_job(job_id) is None:
        typer.echo(f"No job {job_id}", err=True)
        raise typer.Exit(1)
    _run(Pipeline(cfg, db), job_id)


def _run(pipeline, job_id: str) -> None:
    try:
        result = pipeline.run(job_id, progress=_progress)
    except Exception as exc:  # the traceback is in the log, not on screen
        typer.echo(
            f"\nFailed: {exc}\nDetails: {pipeline.workdir(job_id).path('job.log')}", err=True
        )
        raise typer.Exit(1) from exc
    if result is None:
        typer.echo("\nRejected. See the reason above.")
        raise typer.Exit(2)
    typer.echo(f"\nDone. Review the video before publishing:\n  {result.video}")


if __name__ == "__main__":
    app()
