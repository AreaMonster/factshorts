"""Job runner: idea → sourcing → script → verification → voice → render → export, resumable."""

from __future__ import annotations

import hashlib
import logging
import secrets
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from factshorts.config import Config
from factshorts.core.art import ImageReviewer
from factshorts.core.export import export, source_label
from factshorts.core.http import CachedHttp
from factshorts.core.ideas import IdeaStore
from factshorts.core.llm import LLM, OllamaLLM
from factshorts.core.premise import PremiseResult, check_premise
from factshorts.core.script import (
    ScriptWriter,
    dedupe_script,
    replace_segment,
)
from factshorts.core.sourcing import SourcingError, resolve_source
from factshorts.core.tts import Synthesizer, build_audio, make_synth
from factshorts.core.visuals import VisualBuilder, seed_for
from factshorts.core.whisper_align import WhisperAligner
from factshorts.core.workdir import WorkDir
from factshorts.db import Database
from factshorts.logging_setup import job_log
from factshorts.models import (
    AudioPlan,
    ExportResult,
    Idea,
    RenderResult,
    Script,
    Segment,
    SourceDoc,
    Verification,
    VisualsPlan,
)
from factshorts.render import ffmpeg
from factshorts.render.renderer import render_video
from factshorts.render.themes import Theme, load_theme
from factshorts.sources.wikidata import Wikidata
from factshorts.sources.wikipedia import Wikipedia
from factshorts.verify.aggregate import Verifier, subject_names
from factshorts.verify.llm_judge import Judge
from factshorts.verify.nli import NLIModel, TransformersNLI
from factshorts.verify.normalize import normalize_chars

log = logging.getLogger(__name__)

PRESENTATION_FIELDS = ("theme", "voice", "music")
STEPS = ["sourcing", "premise", "script", "verification", "voice", "visuals", "render", "export"]
STEP_OUTPUT = {
    "sourcing": "source.json",
    "premise": "premise.json",
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


class NeedsInput(RuntimeError):
    """The job waits for the user (e.g. the idea's premise is false)."""


class Cancelled(RuntimeError):
    """The user cancelled the job; raised from the progress callback between stages."""


@dataclass
class Services:
    llm: LLM | None = None
    wiki: Wikipedia | None = None
    synth_factory: Callable[[], Synthesizer] | None = None
    visuals_factory: Callable[[Theme], VisualBuilder] | None = None
    nli_factory: Callable[[], NLIModel] | None = None
    http: CachedHttp | None = None
    wikidata: Wikidata | None = None
    voice_factory: Callable[[str], Synthesizer] | None = None
    _synth: Synthesizer | None = field(default=None, init=False)
    _voice: tuple[str, Synthesizer] | None = field(default=None, init=False)
    _nli: NLIModel | None = field(default=None, init=False)

    def nli(self) -> NLIModel | None:
        if self._nli is None and self.nli_factory is not None:
            self._nli = self.nli_factory()
        return self._nli

    def release_synth(self) -> None:
        if self._synth is not None or self._voice is not None:
            self._synth = self._voice = None
            from factshorts.render.aigen import free_gpu

            free_gpu()

    def synth(self, voice: str | None = None) -> Synthesizer:
        """The default voice, or another one when an idea asks for it."""
        if voice is not None and self.voice_factory is not None:
            if self._voice is None or self._voice[0] != voice:
                self._voice = (voice, self.voice_factory(voice))
            return self._voice[1]
        if self._synth is None:
            assert self.synth_factory is not None
            self._synth = self.synth_factory()
        return self._synth


def new_job_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def default_services(cfg: Config, db: Database) -> Services:
    http = CachedHttp(cfg.wikipedia.user_agent, db, cfg.cache.http_ttl_hours)
    llm = OllamaLLM(cfg.ollama)
    reviewer = ImageReviewer(llm, cfg.visuals.review_model or cfg.ollama.judge_model)
    return Services(
        llm=llm,
        wiki=Wikipedia(http, cfg.wikipedia),
        synth_factory=lambda: make_synth(cfg.tts, cfg.models_dir),
        voice_factory=lambda v: make_synth(cfg.tts.model_copy(update={"voice": v}), cfg.models_dir),
        visuals_factory=lambda theme: VisualBuilder(cfg, theme, reviewer=reviewer),
        nli_factory=lambda: TransformersNLI(
            cfg.verification.nli_model, cfg.verification.nli_device
        ),
        wikidata=Wikidata(http),
        http=http,
    )


class Pipeline:
    def __init__(self, cfg: Config, db: Database, services: Services | None = None) -> None:
        self.cfg = cfg
        self.db = db
        self.services = services or default_services(cfg, db)
        self._verifier_for: Path | None = None
        self._verifier_cache: Verifier | None = None
        self._whisper: WhisperAligner | None = None

    def create_job(self, idea: Idea, idea_id: int | None = None) -> str:
        """Start a job. The idea is registered in the idea store (raises DuplicateIdea for a
        repeat unless idea.force), or `idea_id` points at an idea already queued there."""
        store = IdeaStore(self.db)
        if idea_id is None:
            idea_id = store.add(idea, article=idea.article, angle=idea.angle, status="processing")
        else:
            store.set_status(idea_id, "processing")
        job_id = new_job_id()
        self.db.create_job(job_id, idea.model_dump(), idea_id)
        wd = self.workdir(job_id)
        wd.write_model("idea.json", idea)
        return job_id

    def _set_idea_status(self, job_id: str, status: str) -> None:
        row = self.db.get_job(job_id)
        if row is not None and row["idea_id"] is not None:
            IdeaStore(self.db).set_status(row["idea_id"], status)

    def answer(self, job_id: str, choice: str) -> None:
        """Answer a job waiting in needs_input: 'corrected' or 'cancel'."""
        row = self.db.get_job(job_id)
        if row is None or row["state"] != "needs_input":
            raise ValueError(f"job {job_id} is not waiting for an answer")
        wd = self.workdir(job_id)
        premise = wd.read_model("premise.json", PremiseResult)
        if choice == "cancel":
            premise.resolution = "cancelled"
            wd.write_model("premise.json", premise)
            reason = "false premise: " + premise.correction_note()
            self.db.update_job(job_id, state="cancelled", error=reason)
            self.db.add_rejection(wd.read_model("idea.json", Idea).text, reason, job_id, "premise")
            self._set_idea_status(job_id, "cancelled")
            return
        if choice != "corrected":
            raise ValueError("answer must be 'corrected' or 'cancel'")
        premise.resolution = "corrected"
        premise.correction = premise.correction_note()
        wd.write_model("premise.json", premise)
        self.db.set_step(job_id, "premise", "done")
        self.db.set_step_hash(job_id, "premise", self._input_hash(wd, "premise"))
        self.db.update_job(job_id, state="queued", error=None)

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
                    # Steps may rewrite their own inputs (verification rewrites script.json),
                    # so the fingerprint that counts for resuming is the one after the step.
                    self.db.set_step_hash(job_id, step, self._input_hash(wd, step))
                    log.info("[%s] step %s done", job_id, step)
            except Cancelled:
                log.info("[%s] cancelled by the user", job_id)
                self.db.set_step(job_id, self._current(job_id), "cancelled", error="cancelled")
                self.db.update_job(job_id, state="cancelled", error="cancelled by the user")
                self._set_idea_status(job_id, "cancelled")
                return None
            except NeedsInput as exc:
                message = str(exc)
                log.info("[%s] needs input: %s", job_id, message)
                self.db.set_step(job_id, self._current(job_id), "needs_input", error=message)
                self.db.update_job(job_id, state="needs_input", error=message)
                self._set_idea_status(job_id, "needs_input")
                notify("needs_input", message)
                return None
            except (Rejected, SourcingError) as exc:
                reason = str(exc)
                log.warning("[%s] rejected: %s", job_id, reason)
                self.db.set_step(job_id, self._current(job_id), "failed", error=reason)
                self.db.update_job(job_id, state="rejected", error=reason)
                self.db.add_rejection(idea.text, reason, job_id, self._current(job_id))
                self._set_idea_status(job_id, "rejected")
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
            self._set_idea_status(job_id, "done")
            return wd.read_model("export.json", ExportResult)

    # --- resume support ------------------------------------------------------

    def _current(self, job_id: str) -> str:
        row = self.db.get_job(job_id)
        return (row["current_step"] if row else None) or "idea"

    def _input_hash(self, wd: WorkDir, step: str) -> str:
        idx = STEPS.index(step)
        files = [STEP_OUTPUT[s] for s in STEPS[:idx]]
        h = hashlib.sha256()
        idea = wd.read_model("idea.json", Idea)
        if idx < STEPS.index("voice"):
            # Theme, voice and music do not change sources, script or fact-checks, so a
            # rerender with other presentation settings starts at the voice step.
            idea = idea.model_copy(update=dict.fromkeys(PRESENTATION_FIELDS))
        h.update(idea.model_dump_json().encode())
        for name in files:
            p = wd.root / name
            if p.is_file():
                h.update(p.read_bytes())
        return h.hexdigest()

    def rerender(
        self,
        job_id: str,
        theme: str | None = None,
        voice: str | None = None,
        music: bool | None = None,
    ) -> None:
        """Change theme, voice or music of a job; `run` then redoes only voice onward."""
        if self.db.get_job(job_id) is None:
            raise ValueError(f"no job {job_id}")
        if theme is not None:
            load_theme(theme)  # fails early with the list of available themes
        wd = self.workdir(job_id)
        # Jobs made before Phase 4 stored a fingerprint of the raw idea.json, which the change
        # below breaks; restamp the finished steps first so sources, script and checks are kept.
        for step in STEPS[: STEPS.index("voice")]:
            if self._is_done(job_id, step, wd):
                self.db.set_step_hash(job_id, step, self._input_hash(wd, step))
        idea = wd.read_model("idea.json", Idea)
        updates = {
            k: v for k, v in (("theme", theme), ("voice", voice), ("music", music)) if v is not None
        }
        wd.write_model("idea.json", idea.model_copy(update=updates))
        self.db.update_job(job_id, state="queued", error=None)

    def edit_segment(
        self,
        job_id: str,
        seg_id: str,
        narration: str,
        on_screen_text: str | None = None,
        notify: ProgressFn | None = None,
    ) -> list[str]:
        """Replace one segment's narration with the user's wording. The segment alone is
        fact-checked; if it passes, the change is saved and `run` redoes voice onward (unchanged
        scenes reuse their cached audio and pictures). Returns the problems if it fails."""
        notify = notify or (lambda step, msg: None)
        wd, idea, script = self._editable(job_id)
        old = next(s for s in script.segments if s.id == seg_id)
        if idea.factcheck:
            notify("edit", f"finding source quotes for the new {seg_id} narration")
            new = self._writer(idea, wd).annotate(script, seg_id, narration)
        else:
            new = Segment(narration=" ".join(narration.split()), on_screen_text="")
        new = new.model_copy(
            update={
                "on_screen_text": old.on_screen_text if on_screen_text is None else on_screen_text,
                "visual": old.visual,
                "image_prompt": old.image_prompt,
                "motion": old.motion,
            }
        )
        candidate = replace_segment(script.model_copy(deep=True), seg_id, new)
        return self._apply_edit(job_id, wd, idea, candidate, seg_id, notify)

    def regenerate_segment(
        self, job_id: str, seg_id: str, notify: ProgressFn | None = None
    ) -> list[str]:
        """Let the writer make a fresh, fact-checked version of one segment (up to
        max_rewrite_rounds attempts). Returns the problems if no version passed."""
        notify = notify or (lambda step, msg: None)
        wd, idea, script = self._editable(job_id)
        writer = self._writer(idea, wd)
        header = (
            f"The user wants a new version of segment {seg_id}: different wording, and a "
            "different fact from SOURCE if one fits the story better. Current version:"
        )
        problems = [next(s for s in script.segments if s.id == seg_id).narration]
        for attempt in range(1, self.cfg.verification.max_rewrite_rounds + 1):
            notify("edit", f"writing a new {seg_id} (attempt {attempt})")
            candidate = writer.rewrite_segment(
                script.model_copy(deep=True), seg_id, problems, header
            )
            problems = self._apply_edit(job_id, wd, idea, candidate, seg_id, notify)
            if not problems:
                return []
            header = f"Segment {seg_id} of the script failed fact-checking:"
        return problems

    def _editable(self, job_id: str) -> tuple[WorkDir, Idea, Script]:
        row = self.db.get_job(job_id)
        if row is None:
            raise ValueError(f"no job {job_id}")
        wd = self.workdir(job_id)
        if not (wd.exists("script.json") and self._is_done(job_id, "verification", wd)):
            raise ValueError("the script of this job is not finished yet")
        return wd, wd.read_model("idea.json", Idea), wd.read_model("script.json", Script)

    def _apply_edit(
        self,
        job_id: str,
        wd: WorkDir,
        idea: Idea,
        script: Script,
        seg_id: str,
        notify: ProgressFn,
    ) -> list[str]:
        verification = wd.read_model("verification.json", Verification)
        if idea.factcheck:
            notify("edit", f"fact-checking {seg_id}")
            check = self._verifier(wd).verify(script, only={seg_id})
            if not check.passed:
                for f in check.failures:
                    log.info("  - %s", f)
                return check.failures
            verification = merge_verification(verification, check, script, seg_id)
        new = next(s for s in script.segments if s.id == seg_id)
        wd.write_model("script.json", script)
        wd.write_model("verification.json", verification)
        # The script is checked: keep the verification step from re-running its rewrites.
        self.db.set_step_hash(job_id, "verification", self._input_hash(wd, "verification"))
        if wd.exists("visuals/visuals.json"):
            # The scene keeps its picture; the vision check looks at it again for the new text.
            plan = wd.read_model("visuals/visuals.json", VisualsPlan)
            for sc in plan.scenes:
                if sc.segment_id == seg_id:
                    sc.narration, sc.review = new.narration, ""
            wd.write_model("visuals/visuals.json", plan)
        self.db.update_job(job_id, state="queued", error=None)
        log.info("[%s] %s changed: %s", job_id, seg_id, new.narration)
        notify("edit", f"{seg_id} passed the checks: {new.narration}")
        return []

    def redo_from(self, job_id: str, step: str) -> None:
        """Make `run` start again at `step` (e.g. after an improvement to the checks).
        Earlier steps are kept; the script itself is kept too when redoing verification."""
        if step not in STEPS:
            raise ValueError(f"unknown step '{step}' (one of: {', '.join(STEPS)})")
        if self.db.get_job(job_id) is None:
            raise ValueError(f"no job {job_id}")
        for later in STEPS[STEPS.index(step) :]:
            self.db.set_step(job_id, later, "stale")
        if step == "visuals":
            # Redoing the pictures means new pictures: forget the checked ones it would reuse.
            self.workdir(job_id).path("visuals", "visuals.json").unlink(missing_ok=True)
        self.db.update_job(job_id, state="queued", error=None)

    def _is_done(self, job_id: str, step: str, wd: WorkDir) -> bool:
        row = self.db.step_status(job_id, step)
        return (
            row is not None
            and row["status"] == "done"
            and wd.exists(STEP_OUTPUT[step])
            and row["input_hash"] in (self._input_hash(wd, step), self._legacy_hash(wd, step))
        )

    def _legacy_hash(self, wd: WorkDir, step: str) -> str:
        """Fingerprint used before theme/voice rerenders existed (jobs made until Phase 4)."""
        h = hashlib.sha256()
        for name in ["idea.json"] + [STEP_OUTPUT[s] for s in STEPS[: STEPS.index(step)]]:
            p = wd.root / name
            if p.is_file():
                h.update(p.read_bytes())
        return h.hexdigest()

    # --- steps -----------------------------------------------------------------

    def _step_sourcing(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        assert self.services.wiki is not None and self.services.llm is not None
        try:
            source = resolve_source(
                idea,
                self.services.wiki,
                self.services.llm,
                self.cfg.ollama.writer_model,
                self.services.http,
            )
        except SourcingError:
            if idea.factcheck:
                raise
            source = SourceDoc(kind="none", title="", sections=[], sha256="")
            notify("sourcing", "no source found; opinion video without fact-check (UNVERIFIED)")
            wd.write_model("source.json", source)
            return
        if source.kind == "user":
            notify("sourcing", f"your source: {source.title} ({len(source.text.split())} words)")
        else:
            refs = ", ".join(f"{r.title} (revision {r.revision_id})" for r in source.refs)
            notify("sourcing", f"Wikipedia: {refs}")
        wd.write_model("source.json", source)

    def _step_premise(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        source = wd.read_model("source.json", SourceDoc)
        if not idea.factcheck or source.kind == "none" or not self.cfg.verification.premise_check:
            wd.write_model("premise.json", PremiseResult(checked=False))
            return
        assert self.services.llm is not None
        result = check_premise(
            idea,
            source,
            self.services.llm,
            self.cfg.ollama.writer_model,
            self.cfg.ollama.judge_model,
        )
        wd.write_model("premise.json", result)
        if result.contradicted:
            raise NeedsInput(
                result.question() + "\nAnswer with: factshorts answer "
                f"{job_id} corrected   (or: cancel)"
            )
        notify("premise", f"{len(result.items)} assumption(s) checked, none contradicted")

    def _aligner(self) -> WhisperAligner | None:
        if not self.cfg.tts.use_whisper:
            return None
        if self._whisper is None:
            self._whisper = WhisperAligner(self.cfg.tts.whisper_model)
        return self._whisper

    def _writer(self, idea: Idea, wd: WorkDir) -> ScriptWriter:
        assert self.services.llm is not None
        premise = (
            wd.read_model("premise.json", PremiseResult) if wd.exists("premise.json") else None
        )
        note = premise.correction if premise and premise.resolution == "corrected" else None
        return ScriptWriter(
            idea,
            wd.read_model("source.json", SourceDoc),
            self.services.llm,
            self.cfg.ollama.writer_model,
            self.cfg.script,
            self.cfg.tts,
            note,
        )

    def _step_script(
        self,
        job_id: str,
        idea: Idea,
        wd: WorkDir,
        notify: ProgressFn,
        feedback: list[str] | None = None,
    ) -> Script:
        script = self._writer(idea, wd).draft(feedback)
        notify(
            "script", f"{script.title!r}: {len(script.scenes)} scenes, {script.word_count()} words"
        )
        wd.write_model("script.json", script)
        return script

    # --- verification ----------------------------------------------------------

    def _verifier(self, wd: WorkDir) -> Verifier:
        """One verifier per job run, so claim results are cached across rewrites."""
        if self._verifier_for == wd.root and self._verifier_cache is not None:
            return self._verifier_cache
        source = wd.read_model("source.json", SourceDoc)
        vcfg = self.cfg.verification
        values: dict | None = {}  # no Wikidata items: the check does not apply
        refs = [r for r in source.refs if r.wikidata_id]
        if vcfg.wikidata_check and refs:
            if self.services.wikidata is None:
                values = None
            else:
                try:
                    values = {
                        r.title: self.services.wikidata.statements(
                            r.wikidata_id, list(vcfg.wikidata_properties)
                        )
                        for r in refs
                    }
                except Exception as exc:  # network problems must not pass silently
                    log.warning("Wikidata unavailable: %s", exc)
                    values = None
        judge = None
        if vcfg.llm_judge and self.services.llm is not None:
            judge = Judge(self.services.llm, self.cfg.ollama.judge_model)
        nli = self.services.nli() if vcfg.nli_check else None
        self._verifier_cache = Verifier(vcfg, source, nli=nli, judge=judge, wikidata_values=values)
        self._verifier_for = wd.root
        return self._verifier_cache

    def _step_verification(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        assert self.services.llm is not None
        source = wd.read_model("source.json", SourceDoc)
        script = wd.read_model("script.json", Script)
        if not idea.factcheck:
            wd.write_model(
                "verification.json",
                Verification(
                    enabled_checks=[], claims=[], sentences=[], passed=True, factcheck_off=True
                ),
            )
            notify("verification", "fact-check turned off: the video is labeled UNVERIFIED")
            return
        writer = self._writer(idea, wd)
        verifier = self._verifier(wd)
        n = 1
        result = verifier.verify(script, attempt=n)
        wd.write_model(f"verification_attempt{n}.json", result)
        title_fallbacks = [idea.text, source.title]

        # 1. Targeted rewrites of failing segments (max_rewrite_rounds).
        for rnd in range(1, self.cfg.verification.max_rewrite_rounds + 1):
            if result.passed:
                break
            failed = failed_segments(result)
            notify(
                "verification",
                f"round {rnd}: {len(result.failures)} problem(s); "
                f"rewriting {', '.join(sorted(failed))}",
            )
            for f in result.failures:
                log.info("  - %s", f)
            if "title" in failed and title_fallbacks:
                script.title = title_fallbacks.pop(0)
            for seg_id in sorted(failed - {"title"}):
                problems = [f for f in result.failures if f.startswith(f"{seg_id} ")]
                script = writer.rewrite_segment(script, seg_id, problems)
            n += 1
            result = verifier.verify(script, attempt=n)
            wd.write_model(f"verification_attempt{n}.json", result)

        # 2. Scenes that still fail are dropped (and replaced with fresh, checked scenes).
        if not result.passed:
            script, result, n = self._drop_failing(script, result, verifier, wd, n, notify, writer)

        # 3. Top up if dropping made the video too short; new scenes must pass as well.
        if script.word_count() < writer.budget * 0.9 and self.cfg.script.max_topups:
            before = len(script.scenes)
            script = writer.top_up(script)
            if len(script.scenes) > before:
                n += 1
                result = verifier.verify(script, attempt=n)
                if not result.passed:
                    script, result, n = self._drop_failing(script, result, verifier, wd, n, notify)

        # 4. Editing pass for flow; edited segments are kept only if they still pass.
        if self.cfg.script.polish:
            script, result, n = self._polish(script, result, writer, verifier, wd, n, notify)

        # 5. Remove repeated sentences and scenes left without a fact.
        nli = self.services.nli() if self.cfg.verification.nli_check else None
        entails = None
        if nli is not None and self.cfg.script.repeat_entailment:
            entails = lambda p, h: nli.scores(p, h).get("entailment", 0.0)  # noqa: E731
        names = {w.lower() for w in subject_names(source)}
        # First with the NLI model (reworded repeats too), then only near-identical text, which
        # removes less, if the first result does not pass the checks again.
        for how, ent in (("reworded", entails), ("exact", None)):
            if how == "exact" and entails is None:
                break
            deduped, removed = dedupe_script(
                script, self.cfg.script.min_scenes, ent, self.cfg.script.repeat_entailment, names
            )
            if not removed and len(deduped.scenes) == len(script.scenes):
                break
            n += 1
            check = verifier.verify(deduped, attempt=n)
            wd.write_model(f"verification_attempt{n}.json", check)
            if check.passed:
                notify(
                    "verification",
                    f"removed {removed} repeated sentence(s); {len(deduped.scenes)} scenes",
                )
                script, result = deduped, check
                break
            notify(
                "verification",
                f"removing {removed} repeat(s) ({how}) broke the checks; "
                + ("trying exact repeats only" if how == "reworded" else "keeping them"),
            )
            for f in check.failures:
                log.info("  - %s", f)
        wd.write_model("script.json", script)
        wd.write_model("verification.json", result)
        notify(
            "verification",
            f"passed: {len(result.claims)} claims, {script.word_count()} words ({result.label})",
        )

    def _polish(
        self,
        script: Script,
        result: Verification,
        writer: ScriptWriter,
        verifier: Verifier,
        wd: WorkDir,
        n: int,
        notify: ProgressFn,
    ) -> tuple[Script, Verification, int]:
        polished = writer.polish(script.model_copy(deep=True))
        if polished is script or len(polished.scenes) != len(script.scenes):
            return script, result, n
        check = verifier.verify(polished, attempt=n + 1)
        bad = failed_segments(check)
        merged = script.model_copy(deep=True)
        kept = 0
        for old_seg, new_seg in zip(script.segments, polished.segments, strict=True):
            # Polishing may change wording, not move facts: the segment must cite exactly the
            # same source sentences as before, or content would get duplicated or lost.
            if new_seg.id not in bad and _quotes(new_seg) == _quotes(old_seg):
                merged = replace_segment(merged, old_seg.id, new_seg)
                kept += 1
        if "title" not in bad:
            merged.title = polished.title
        n += 1
        final = verifier.verify(merged, attempt=n)
        wd.write_model(f"verification_attempt{n}.json", final)
        if not final.passed:  # segments that pass alone can still clash; keep the original
            notify("verification", "editing pass rejected; keeping the checked draft")
            return script, result, n
        notify(
            "verification",
            f"editing pass: {kept}/{len(script.segments)} segments rewritten for flow",
        )
        return merged, final, n

    def _drop_failing(
        self,
        script: Script,
        result: Verification,
        verifier: Verifier,
        wd: WorkDir,
        n: int,
        notify: ProgressFn,
        writer: ScriptWriter | None = None,
    ) -> tuple[Script, Verification, int]:
        """Drop segments that still fail. With a writer, top up to refill the story first;
        reject only if fewer than min_scenes checked scenes remain."""
        failed = failed_segments(result)
        original_failures = result.failures
        if "outro" in failed:
            # A plain sign-off states no fact, so it cannot fail the checks.
            script.outro = Segment(
                narration="Follow for more facts.", on_screen_text="Follow for more", visual="title"
            )
            failed.discard("outro")
            notify("verification", "outro still failed; using a plain sign-off")
        scene_ids = {s.id for s in script.scenes}
        keep = [s for s in script.scenes if s.id not in failed]
        if "hook" in failed and len(keep) > self.cfg.script.min_scenes:
            # The first good scene becomes the opening; the story still starts coherently.
            first = keep.pop(0)
            script.hook = first.model_copy(update={"visual": "title"})
            failed.discard("hook")
            notify("verification", f"hook still failed; scene {first.id} now opens the video")
        if failed <= scene_ids and len(keep) < self.cfg.script.min_scenes and writer:
            avoid = [x.narration for x in script.scenes if x.id in failed]
            script.scenes = keep
            script.assign_ids()
            notify("verification", f"dropped {len(failed)} failing scene(s); writing new ones")
            before = {x.narration for x in script.scenes}
            script = writer.top_up(script, avoid=avoid)
            n += 1
            result = verifier.verify(script, attempt=n)
            wd.write_model(f"verification_attempt{n}.json", result)
            if result.passed and len(script.scenes) >= self.cfg.script.min_scenes:
                return script, result, n
            # Keep the new scenes that passed; the old ones passed before.
            new_failed = {
                x.id
                for x in script.scenes
                if x.narration not in before and x.id in failed_segments(result)
            }
            failed = new_failed | (failed_segments(result) - {x.id for x in script.scenes})
            keep = [x for x in script.scenes if x.id not in failed]
            scene_ids = {x.id for x in script.scenes}
        if not failed <= scene_ids or len(keep) < self.cfg.script.min_scenes:
            wd.write_model("script.json", script)
            reasons = result.failures or original_failures
            raise Rejected(
                f"fact-check failed after targeted rewrites ({len(keep)} checked scene(s) left, "
                f"minimum {self.cfg.script.min_scenes}): " + "; ".join(reasons[:5])
            )
        if failed:
            notify(
                "verification",
                f"dropping {len(failed)} scene(s) that still fail: {', '.join(sorted(failed))}",
            )
        script.scenes = keep
        script.assign_ids()
        n += 1
        result = verifier.verify(script, attempt=n)
        wd.write_model(f"verification_attempt{n}.json", result)
        if not result.passed:
            raise Rejected("fact-check failed: " + "; ".join(result.failures[:5]))
        return script, result, n

    def _step_voice(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        script = wd.read_model("script.json", Script)
        audio = build_audio(
            script,
            self.services.synth(idea.voice),
            self.cfg.tts,
            wd.subdir("audio"),
            self.cfg.cache_dir,
            self._aligner(),
        )
        need = self.cfg.script.min_duration_s if idea.length >= 60 and idea.factcheck else 0.0
        for _ in range(self.cfg.script.max_topups):
            if audio.duration >= need:
                break
            notify("voice", f"{audio.duration:.1f} s is under {need:.0f} s; adding scenes")
            script = self._extend(idea, wd, script, need - audio.duration + 3.0, notify)
            audio = build_audio(
                script,
                self.services.synth(idea.voice),
                self.cfg.tts,
                wd.subdir("audio"),
                self.cfg.cache_dir,
                self._aligner(),
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
        writer = self._writer(idea, wd)
        # New scenes are inserted mid-story and renumber the ids, so tell them apart by text.
        before = {s.narration for s in script.scenes}
        extra_words = int(seconds * self.cfg.script.words_per_minute / 60)
        script = writer.top_up(script, target_words=script.word_count() + extra_words)
        verifier = self._verifier(wd)
        result = verifier.verify(script)
        failed = failed_segments(result)
        if failed:
            kept = [s for s in script.scenes if s.id not in failed or s.narration in before]
            notify(
                "voice",
                f"dropped {len(script.scenes) - len(kept)} new scene(s) that failed fact-checking",
            )
            script.scenes = kept
            script.assign_ids()
            result = verifier.verify(script)
        if not result.passed:
            raise Rejected(
                "extended script failed fact-checking: " + "; ".join(result.failures[:5])
            )
        wd.write_model("script.json", script)
        wd.write_model("verification.json", result)
        # Verification now reflects the extended script; keep it from re-running on resume.
        job_id = wd.root.name
        self.db.set_step_hash(job_id, "verification", self._input_hash(wd, "verification"))
        return script

    def _step_visuals(self, job_id: str, idea: Idea, wd: WorkDir, notify: ProgressFn) -> None:
        script = wd.read_model("script.json", Script)
        audio = wd.read_model("audio/audio.json", AudioPlan)
        self.services.release_synth()  # free the TTS model's GPU memory
        if self.cfg.visuals.mode == "off" or self.services.visuals_factory is None:
            plan = VisualsPlan(mode="off")
        else:
            builder = self.services.visuals_factory(load_theme(idea.theme or self.cfg.render.theme))
            llm = self.services.llm
            plan = builder.build(
                job_id,
                script,
                audio,
                wd.subdir("visuals"),
                llm,
                unload_llm=getattr(llm, "unload_all", lambda: None),
                notify=lambda msg: notify("visuals", msg),
                source=wd.read_model("source.json", SourceDoc),
                previous=wd.read_model("visuals/visuals.json", VisualsPlan)
                if wd.exists("visuals/visuals.json")
                else None,
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
        music = self.cfg.audio.music if idea.music is None else idea.music
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
            audio_cfg=self.cfg.audio,
            music_seed=seed_for(job_id, "music"),  # always made, so it can be toggled later
            music_default=music,
            theme=idea.theme,
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
        visuals = wd.read_model("visuals/visuals.json", VisualsPlan)
        first = visuals.scenes[0].image if visuals.scenes else None
        thumb_base = wd.root / "visuals" / first if first else None
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
            visuals=visuals,
            thumbnail_base=thumb_base,
            theme=idea.theme or self.cfg.render.theme,
            llm=self.services.llm,
            caption_model=self.cfg.ollama.writer_model,
        )
        old = self.db.get_video(job_id)
        if old is not None and Path(old["folder"]) != Path(result.folder):
            # A re-render replaces the job's earlier output; the new one waits for review.
            folder = Path(old["folder"])
            if folder.is_dir() and folder.resolve().is_relative_to(self.cfg.output_dir.resolve()):
                shutil.rmtree(folder, ignore_errors=True)
                notify("export", f"replaced the earlier version ({folder.name})")
        self.db.add_video(
            job_id,
            wd.read_model("script.json", Script).title,
            idea.topic,
            result.folder,
            result.verified,
        )
        notify("export", f"{verification.label}: {result.folder}")
        wd.write_model("export.json", result)


def _quotes(seg: Segment) -> set[str]:
    return {normalize_chars(c.source_quote).casefold() for c in seg.claims}


def merge_verification(
    old: Verification, new: Verification, script: Script, seg_id: str
) -> Verification:
    """`old` with the results for `seg_id` replaced by `new` (a check of that segment only)."""
    order = {s.id: i for i, s in enumerate(script.segments)}
    order["title"] = len(order)
    claims = [c for c in old.claims if c.segment_id != seg_id] + new.claims
    sentences = [s for s in old.sentences if s.segment_id != seg_id] + new.sentences
    claims.sort(key=lambda c: order.get(c.segment_id, 99))
    sentences.sort(key=lambda s: order.get(s.segment_id, 99))
    failures = [f for f in old.failures if not f.startswith(f"{seg_id} ")] + new.failures
    return old.model_copy(
        update={
            "claims": claims,
            "sentences": sentences,
            "failures": failures,
            "passed": not failures,
        }
    )


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
