"""Job runner: idea → sourcing → script → verification → voice → render → export, resumable."""

from __future__ import annotations

import hashlib
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from factshorts.config import Config
from factshorts.core.export import export, source_label
from factshorts.core.http import CachedHttp
from factshorts.core.llm import LLM, OllamaLLM
from factshorts.core.script import ScriptWriter, generate_script
from factshorts.core.sourcing import SourcingError, resolve_wikipedia
from factshorts.core.tts import Synthesizer, build_audio, make_synth
from factshorts.core.visuals import VisualBuilder
from factshorts.core.workdir import WorkDir
from factshorts.db import Database
from factshorts.logging_setup import job_log
from factshorts.models import (
    AudioPlan,
    ExportResult,
    Idea,
    RenderResult,
    Script,
    SourceDoc,
    Verification,
    VisualsPlan,
)
from factshorts.render import ffmpeg
from factshorts.render.renderer import render_video
from factshorts.render.themes import Theme, load_theme
from factshorts.sources.wikipedia import Wikipedia
from factshorts.verify.aggregate import verify_script

log = logging.getLogger(__name__)

STEPS = ["sourcing", "script", "verification", "voice", "visuals", "render", "export"]
STEP_OUTPUT = {
    "sourcing": "source.json",
    "script": "script.json",
    "verification": "verification.json",
    "voice": "audio/audio.json",
    "visuals": "visuals/visuals.json",
    "render": "render/render.json",
    "export": "export.json",
}

ProgressFn = Callable[[str, str], None]


class Rejected(RuntimeError):
    """The idea cannot be turned into a verified video; reason is user-facing."""


@dataclass
class Services:
    llm: LLM | None = None
    wiki: Wikipedia | None = None
    synth_factory: Callable[[], Synthesizer] | None = None
    visuals_factory: Callable[[Theme], VisualBuilder] | None = None
    _synth: Synthesizer | None = field(default=None, init=False)

    def release_synth(self) -> None:
        if self._synth is not None:
            self._synth = None
            from factshorts.render.aigen import free_gpu

            free_gpu()

    def synth(self) -> Synthesizer:
        if self._synth is None:
            assert self.synth_factory is not None
            self._synth = self.synth_factory()
        return self._synth


def new_job_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def default_services(cfg: Config, db: Database) -> Services:
    http = CachedHttp(cfg.wikipedia.user_agent, db, cfg.cache.http_ttl_hours)
    return Services(
        llm=OllamaLLM(cfg.ollama),
        wiki=Wikipedia(http, cfg.wikipedia),
        synth_factory=lambda: make_synth(cfg.tts, cfg.models_dir),
        visuals_factory=lambda theme: VisualBuilder(cfg, theme),
    )


class Pipeline:
    def __init__(self, cfg: Config, db: Database, services: Services | None = None) -> None:
        self.cfg = cfg
        self.db = db
        self.services = services or default_services(cfg, db)

    def create_job(self, idea: Idea) -> str:
        job_id = new_job_id()
        self.db.create_job(job_id, idea.model_dump())
        wd = self.workdir(job_id)
        wd.write_model("idea.json", idea)
        return job_id

    def workdir(self, job_id: str) -> WorkDir:
        return WorkDir(self.cfg.work_dir / job_id)

    def run(self, job_id: str, progress: ProgressFn | None = None) -> ExportResult | None:
        wd = self.workdir(job_id)
        idea = wd.read_model("idea.json", Idea)
        notify = progress or (lambda step, msg: None)
        with job_log(wd.path("job.log")):
            self.db.update_job(job_id, state="running", error=None)
            try:
                rerun = False  # once a step re-runs, everything after it must re-run too
                for step in STEPS:
                    if not rerun and self._is_done(job_id, step, wd):
                        notify(step, "already done, skipping")
                        continue
                    rerun = True
                    self.db.update_job(job_id, current_step=step)
                    self.db.set_step(job_id, step, "running", self._input_hash(wd, step))
                    notify(step, "started")
                    log.info("[%s] step %s started", job_id, step)
                    getattr(self, f"_step_{step}")(job_id, idea, wd, notify)
                    self.db.set_step(job_id, step, "done")
                    log.info("[%s] step %s done", job_id, step)
            except (Rejected, SourcingError) as exc:
                reason = str(exc)
                log.warning("[%s] rejected: %s", job_id, reason)
                self.db.set_step(job_id, self._current(job_id), "failed", error=reason)
                self.db.update_job(job_id, state="rejected", error=reason)
                self.db.add_rejection(idea.text, reason, job_id, self._current(job_id))
                notify("rejected", reason)
                return None
            except Exception as exc:
                log.exception("[%s] failed", job_id)
                message = f"{type(exc).__name__}: {exc}"
                self.db.set_step(job_id, self._current(job_id), "failed", error=message)
                self.db.update_job(job_id, state="failed", error=message)
                notify("failed", message)
                raise
            self.db.update_job(job_id, state="done", current_step=None)
            return wd.read_model("export.json", ExportResult)

    # --- resume support ------------------------------------------------------

    def _current(self, job_id: str) -> str:
        row = self.db.get_job(job_id)
        return (row["current_step"] if row else None) or "idea"

    def _input_hash(self, wd: WorkDir, step: str) -> str:
        idx = STEPS.index(step)
        files = ["idea.json"] + [STEP_OUTPUT[s] for s in STEPS[:idx]]
        h = hashlib.sha256()
        for name in files:
            p = wd.root / name
            if p.is_file():
                h.update(p.read_bytes())
        return h.hexdigest()

    def _is_done(self, job_id: str, step: str, wd: WorkDir) -> bool:
        row = self.db.step_status(job_id, step)
        return (
            row is not None
            and row["status"] == "done"
            and wd.exists(STEP_OUTPUT[step])
            and row["input_hash"] == self._input_hash(wd, step)
        )

    # --- steps -----------------------------------------------------------------

    def _step_sourcing(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        assert self.services.wiki is not None and self.services.llm is not None
        source = resolve_wikipedia(
            idea, self.services.wiki, self.services.llm, self.cfg.ollama.writer_model
        )
        notify("sourcing", f"Wikipedia: {source.title} (revision {source.revision_id})")
        wd.write_model("source.json", source)

    def _step_script(
        self,
        job_id: str,
        idea: Idea,
        wd: WorkDir,
        notify: ProgressFn,
        feedback: list[str] | None = None,
    ) -> Script:
        assert self.services.llm is not None
        source = wd.read_model("source.json", SourceDoc)
        script = generate_script(
            idea,
            source,
            self.services.llm,
            self.cfg.ollama.writer_model,
            self.cfg.script,
            self.cfg.tts,
            feedback=feedback,
        )
        notify(
            "script", f"{script.title!r}: {len(script.scenes)} scenes, {script.word_count()} words"
        )
        wd.write_model("script.json", script)
        return script

    def _step_verification(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        assert self.services.llm is not None
        source = wd.read_model("source.json", SourceDoc)
        script = wd.read_model("script.json", Script)
        writer = ScriptWriter(
            idea,
            source,
            self.services.llm,
            self.cfg.ollama.writer_model,
            self.cfg.script,
            self.cfg.tts,
        )
        attempts = self.cfg.verification.max_script_attempts
        for attempt in range(1, attempts + 1):
            result = verify_script(script, source, self.cfg.verification, attempt=attempt)
            wd.write_model(f"verification_attempt{attempt}.json", result)
            if result.passed:
                wd.write_model("verification.json", result)
                notify(
                    "verification",
                    f"passed: {len(result.claims)} claims, "
                    f"{script.word_count()} words, attempt {attempt}",
                )
                return
            notify("verification", f"attempt {attempt} failed: {len(result.failures)} problems")
            for f in result.failures:
                log.info("  - %s", f)
            if attempt == attempts:
                break
            failed = failed_segments(result)
            scene_ids = {s.id for s in script.scenes}
            keep = [s for s in script.scenes if s.id not in failed]
            if failed <= scene_ids and len(keep) >= self.cfg.script.min_scenes:
                # Only scenes failed: drop them and add new scenes with other facts.
                avoid = [s.narration for s in script.scenes if s.id in failed]
                script.scenes = keep
                script.assign_ids()
                notify("verification", f"dropped {len(failed)} failing scene(s), adding new ones")
                script = writer.top_up(script, avoid=avoid)
            else:
                notify("verification", "regenerating the script with the failures as feedback")
                script = writer.draft(feedback=result.failures)
            wd.write_model("script.json", script)
        raise Rejected(
            f"fact-check failed after {attempts} attempts: " + "; ".join(result.failures[:5])
        )

    def _step_voice(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        script = wd.read_model("script.json", Script)
        audio = build_audio(
            script, self.services.synth(), self.cfg.tts, wd.subdir("audio"), self.cfg.cache_dir
        )
        need = self.cfg.script.min_duration_s if idea.length >= 60 else 0.0
        for _ in range(self.cfg.script.max_topups):
            if audio.duration >= need:
                break
            notify("voice", f"{audio.duration:.1f} s is under {need:.0f} s; adding scenes")
            script = self._extend(idea, wd, script, need - audio.duration + 3.0, notify)
            audio = build_audio(
                script,
                self.services.synth(),
                self.cfg.tts,
                wd.subdir("audio"),
                self.cfg.cache_dir,
            )
        if audio.duration < need:
            notify(
                "voice",
                f"warning: only {audio.duration:.1f} s (target {need:.0f} s); "
                "the source may not have enough verifiable facts",
            )
        notify("voice", f"{audio.duration:.1f} s narration ({audio.voice})")
        wd.write_model("audio/audio.json", audio)

    def _extend(
        self, idea: Idea, wd: WorkDir, script: Script, seconds: float, notify: ProgressFn
    ) -> Script:
        """Add verified scenes worth about `seconds` of speech; unverifiable ones are dropped."""
        assert self.services.llm is not None
        source = wd.read_model("source.json", SourceDoc)
        writer = ScriptWriter(
            idea,
            source,
            self.services.llm,
            self.cfg.ollama.writer_model,
            self.cfg.script,
            self.cfg.tts,
        )
        before = {s.id for s in script.scenes}
        extra_words = int(seconds * self.cfg.script.words_per_minute / 60)
        script = writer.top_up(script, target_words=script.word_count() + extra_words)
        result = verify_script(script, source, self.cfg.verification)
        failed = failed_segments(result)
        if failed:
            kept = [s for s in script.scenes if s.id not in failed or s.id in before]
            notify(
                "voice",
                f"dropped {len(script.scenes) - len(kept)} new scene(s) that failed fact-checking",
            )
            script.scenes = kept
            script.assign_ids()
            result = verify_script(script, source, self.cfg.verification)
        if not result.passed:
            raise Rejected(
                "extended script failed fact-checking: " + "; ".join(result.failures[:5])
            )
        wd.write_model("script.json", script)
        wd.write_model("verification.json", result)
        return script

    def _step_visuals(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        script = wd.read_model("script.json", Script)
        audio = wd.read_model("audio/audio.json", AudioPlan)
        self.services.release_synth()  # free the TTS model's GPU memory
        if self.cfg.visuals.mode == "off" or self.services.visuals_factory is None:
            plan = VisualsPlan(mode="off")
        else:
            builder = self.services.visuals_factory(load_theme(self.cfg.render.theme))
            llm = self.services.llm
            plan = builder.build(
                job_id,
                script,
                audio,
                wd.subdir("visuals"),
                llm,
                unload_llm=getattr(llm, "unload_all", lambda: None),
                notify=lambda msg: notify("visuals", msg),
            )
            kinds = [s.kind for s in plan.scenes]
            notify(
                "visuals",
                f"{kinds.count('wan')} animated, {len(kinds) - kinds.count('wan')} parallax scenes",
            )
        wd.write_model("visuals/visuals.json", plan)

    def _step_render(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        script = wd.read_model("script.json", Script)
        audio = wd.read_model("audio/audio.json", AudioPlan)
        source = wd.read_model("source.json", SourceDoc)
        visuals = wd.read_model("visuals/visuals.json", VisualsPlan)
        result = render_video(
            script,
            audio,
            wd.subdir("audio"),
            wd.subdir("render"),
            self.cfg.render,
            source_label(source),
            backgrounds={
                sc.segment_id: str(wd.root / "visuals" / sc.clip)
                for sc in visuals.scenes
                if sc.clip
            },
        )
        sync = check_sync(Path(result.video))
        notify(
            "render",
            f"{result.frames} frames with {result.encoder}; "
            f"audio/video diff {sync['diff_ms']:.0f} ms",
        )
        wd.write_json("render/sync.json", sync)
        wd.write_model("render/render.json", result)

    def _step_export(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        verification = wd.read_model("verification.json", Verification)
        result = export(
            job_id=job_id,
            idea=idea,
            source=wd.read_model("source.json", SourceDoc),
            script=wd.read_model("script.json", Script),
            verification=verification,
            audio=wd.read_model("audio/audio.json", AudioPlan),
            render=wd.read_model("render/render.json", RenderResult),
            output_dir=self.cfg.output_dir,
            verified=verification.passed,
            sync=_read_json(wd.root / "render" / "sync.json"),
            visuals=wd.read_model("visuals/visuals.json", VisualsPlan),
        )
        self.db.add_video(
            job_id,
            wd.read_model("script.json", Script).title,
            idea.topic,
            result.folder,
            result.verified,
        )
        notify("export", f"{verification.label}: {result.folder}")
        wd.write_model("export.json", result)


def failed_segments(result: Verification) -> set[str]:
    ids = {c.segment_id for c in result.claims if not c.passed}
    ids |= {s.segment_id for s in result.sentences if not s.passed}
    return ids


def check_sync(video: Path) -> dict[str, float]:
    d = ffmpeg.stream_durations(video)
    diff = abs(d.get("video", 0.0) - d.get("audio", 0.0)) * 1000
    return {**d, "diff_ms": diff}


def _read_json(path: Path) -> dict | None:
    import json

    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
