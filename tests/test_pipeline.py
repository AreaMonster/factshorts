"""End-to-end pipeline with mocked LLM, TTS and Wikipedia; includes the golden render test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import (
    FakeLLM,
    FakeNLI,
    FakeSynth,
    FakeWiki,
    FakeWikidata,
    needs_ffmpeg,
    script_to_llm,
)
from factshorts.config import Config
from factshorts.core.pipeline import Pipeline, Services
from factshorts.core.sourcing import ArticleGuess
from factshorts.db import Database
from factshorts.models import Idea, Script, SourceDoc
from factshorts.render.ffmpeg import probe


def make_pipeline(
    cfg: Config, db: Database, source: SourceDoc, script: Script, bad_script: Script | None = None
) -> tuple[Pipeline, FakeLLM, FakeSynth]:
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(bad_script or script)],
        }
    )
    synth = FakeSynth()
    services = Services(
        llm=llm,
        wiki=FakeWiki(source),
        synth_factory=lambda: synth,
        nli_factory=FakeNLI,
        wikidata=FakeWikidata(),
    )
    return Pipeline(cfg, db, services), llm, synth


@needs_ffmpeg
def test_golden_render(cfg: Config, db: Database, source: SourceDoc, script: Script) -> None:
    pipeline, _, synth = make_pipeline(cfg, db, source, script)
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    steps: list[str] = []
    result = pipeline.run(job_id, progress=lambda step, msg: steps.append(step))
    assert result is not None
    video = Path(result.video)
    assert video.is_file()

    info = probe(video)
    streams = {s["codec_type"]: s for s in info["streams"]}
    assert set(streams) == {"video", "audio"}
    v, a = streams["video"], streams["audio"]
    assert (v["width"], v["height"]) == (1080, 1920)
    assert v["codec_name"] == "h264" and v["r_frame_rate"] == "30/1"
    assert a["codec_name"] == "aac"
    audio_plan = json.loads((pipeline.workdir(job_id).root / "audio" / "audio.json").read_text())
    expected = audio_plan["duration"]
    assert float(info["format"]["duration"]) == pytest.approx(expected, abs=0.1)
    assert abs(float(v["duration"]) - float(a["duration"])) < 0.1

    # Deliverables next to the video.
    folder = Path(result.folder)
    sources = json.loads((folder / "sources.json").read_text("utf-8"))
    assert sources["source"]["revision_id"] == 123456789
    assert sources["label"] == "VERIFIED"
    assert len(sources["claims"]) == 5 and all(c["passed"] for c in sources["claims"])
    description = (folder / "description.txt").read_text("utf-8")
    assert "oldid=123456789" in description and "CC BY-SA" in description
    assert (folder / "metadata.json").is_file()
    assert "-->" in next(folder.glob("*.srt")).read_text("utf-8")
    assert "_PARTIAL" not in video.name and "_UNVERIFIED" not in video.name
    # TTS got speakable text, not digits.
    assert any("six thousand four hundred twelve meters" in c for c in synth.calls)
    assert db.get_job(job_id)["state"] == "done"


@needs_ffmpeg
def test_resume_skips_finished_steps(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    cfg.render.width, cfg.render.height = 270, 480  # keep this test fast
    pipeline, llm, synth = make_pipeline(cfg, db, source, script)
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    assert pipeline.run(job_id) is not None
    llm_calls, tts_calls = llm.work_calls, len(synth.calls)

    log: list[tuple[str, str]] = []
    assert pipeline.run(job_id, progress=lambda s, m: log.append((s, m))) is not None
    assert all(m == "already done, skipping" for _, m in log)
    assert llm.work_calls == llm_calls and len(synth.calls) == tts_calls

    # Removing the render output re-runs only render and export.
    (pipeline.workdir(job_id).root / "render" / "render.json").unlink()
    log.clear()
    pipeline.run(job_id, progress=lambda s, m: log.append((s, m)))
    rerun = {s for s, m in log if m == "started"}
    assert rerun == {"render", "export"}
    assert llm.work_calls == llm_calls


def test_failed_verification_rejects_with_reason(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    bad = script.model_copy(deep=True)
    bad.scenes[0].narration = "Its summit rises 7,777 meters above sea level."
    bad.scenes[0].claims[0].text = bad.scenes[0].narration
    pipeline, llm, _ = make_pipeline(cfg, db, source, script, bad_script=bad)
    # Every targeted rewrite of s1 comes back just as wrong.
    llm.responses["LLMSegment"] = [script_to_llm(bad).scenes[0]]
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?"))
    assert pipeline.run(job_id) is None

    job = db.get_job(job_id)
    assert job["state"] == "rejected"
    assert "7,777" in job["error"]
    rows = db.query("SELECT reason, step FROM rejections WHERE job_id = ?", (job_id,))
    assert rows and "number '7,777 meters' not found" in rows[0]["reason"]
    assert rows[0]["step"] == "verification"
    # Only s1 was rewritten (3 rounds, with the failure as feedback); no full redraft, and the
    # scene could not be dropped because 4 scenes is the minimum.
    rewrites = [m for name, m in llm.calls if name == "LLMSegment"]
    assert len(rewrites) == cfg.verification.max_rewrite_rounds
    assert "Segment s1" in rewrites[0][-1]["content"] and "7,777" in rewrites[0][-1]["content"]
    assert sum(1 for name, _ in llm.calls if name == "LLMScript") == 1
