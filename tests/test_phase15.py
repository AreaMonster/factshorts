"""Phase 1.5: engine word timestamps, longer scripts with top-ups, art direction, AI visuals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import (
    FakeDepth,
    FakeImageGen,
    FakeLLM,
    FakeMotion,
    FakeSynth,
    FakeWiki,
    needs_ffmpeg,
    new_fact_scenes,
    script_to_llm,
)
from factshorts.config import Config
from factshorts.core.art import Shot, ShotList, plan_shots, problems_in, sanitize
from factshorts.core.captions import align_word_timings
from factshorts.core.pipeline import Pipeline, Services
from factshorts.core.script import (
    LLMMoreScenes,
    ScriptWriter,
    target_scenes,
    to_segment,
    word_budget,
)
from factshorts.core.sourcing import ArticleGuess
from factshorts.core.visuals import VisualBuilder, choose_kinds
from factshorts.db import Database
from factshorts.models import Idea, Script, SourceDoc
from factshorts.render.backgrounds import cover, crossfade, kenburns_frames, motion_frames
from factshorts.render.ffmpeg import probe
from factshorts.render.themes import load_theme

# --- caption alignment with engine timestamps ---------------------------------------


def test_align_maps_spoken_tokens_to_display_words() -> None:
    spoken = [
        ("Mount", 0.0, 0.3),
        ("Everest", 0.3, 0.8),
        ("is", 0.8, 1.0),
        ("eight", 1.0, 1.3),
        ("thousand", 1.3, 1.7),
        ("eight", 1.7, 1.9),
        ("hundred", 1.9, 2.2),
        ("forty-nine", 2.2, 2.8),
        ("meters", 2.8, 3.2),
        (".", 3.2, 3.3),
    ]
    words = align_word_timings("Mount Everest is 8,849 m.", spoken, 10.0, 3.5)
    assert words is not None
    assert [w.text for w in words] == ["Mount", "Everest", "is", "8,849", "m."]
    number = words[3]
    assert number.start == pytest.approx(11.0) and number.end == pytest.approx(12.8)
    assert words[-1].end == pytest.approx(13.5)


def test_align_keeps_units_with_their_numbers() -> None:
    spoken = [
        ("it's", 0, 0.3),
        ("minus", 0.3, 0.6),
        ("forty", 0.6, 0.9),
        ("degrees", 0.9, 1.3),
        ("Celsius", 1.3, 1.8),
    ]
    words = align_word_timings("it's -40 °C.", spoken, 0.0, 2.0)
    assert words is not None
    assert [(w.text, w.start) for w in words] == [("it's", 0), ("-40", 0.3), ("°C.", 0.9)]


def test_align_returns_none_when_tokens_do_not_line_up() -> None:
    assert align_word_timings("Mount Everest", [("Mount", 0, 1)], 0.0, 1.0) is None


# --- longer scripts --------------------------------------------------------------------


def test_scene_count_and_budget_scale_with_length(cfg: Config) -> None:
    assert target_scenes(75, cfg.script) > target_scenes(30, cfg.script)
    assert word_budget(75, cfg.script, cfg.tts) > 2 * word_budget(30, cfg.script, cfg.tts)
    assert 140 <= word_budget(75, cfg.script, cfg.tts) <= 190


def test_short_draft_is_topped_up_with_new_scenes(
    cfg: Config, source: SourceDoc, script: Script
) -> None:
    cfg.script.max_topups = 2
    extra = new_fact_scenes(0, 2)
    llm = FakeLLM(
        {"LLMScript": [script_to_llm(script)], "LLMMoreScenes": [LLMMoreScenes(scenes=extra)]}
    )
    writer = ScriptWriter(
        Idea(text="How high is Mount Testa?", length=75), source, llm, "m", cfg.script, cfg.tts
    )
    result = writer.draft()
    assert len(result.scenes) > len(script.scenes)
    assert [s.id for s in result.scenes] == [f"s{i}" for i in range(1, len(result.scenes) + 1)]
    topup_calls = [m for name, m in llm.calls if name == "LLMMoreScenes"]
    assert topup_calls and "MORE scenes" in topup_calls[0][-1]["content"]


# --- art direction ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "a mountain with the number 8,849 floating above it",
        "two flags of Nepal and China side by side",
        "a timeline of key years with icons",
        'a sign that says "summit"',
        "a map with labeled countries",
    ],
)
def test_banned_image_content_is_detected(prompt: str) -> None:
    assert problems_in(prompt)


def test_sanitize_drops_offending_clauses() -> None:
    cleaned = sanitize("A snowy peak at dawn, with a label reading Everest, clouds below.")
    assert "label" not in cleaned and "snowy peak" in cleaned and "clouds below" in cleaned


def test_plan_shots_retries_then_fills_every_segment(script: Script) -> None:
    ids = [s.id for s in script.segments]
    bad = ShotList(
        shots=[
            Shot(
                segment_id=i,
                image_prompt="a chart with numbers 1 2 3",
                motion="bars grow",
                visual_interest=2,
            )
            for i in ids
        ]
    )
    good = ShotList(
        shots=[
            Shot(
                segment_id=i,
                image_prompt=f"a calm snowy valley, view {n}",
                motion="snow falls",
                visual_interest=n % 5 + 1,
            )
            for n, i in enumerate(ids)
        ]
    )
    llm = FakeLLM({"ShotList": [bad, good]})
    plan_shots(script, llm, "m", "flat vector art")
    assert len(llm.calls) == 2
    assert all(s.image_prompt.startswith("a calm snowy valley") for s in script.segments)
    assert all(not problems_in(s.image_prompt) for s in script.segments)


def test_choose_kinds_hybrid_animates_hook_and_most_interesting(script: Script) -> None:
    for i, seg in enumerate(script.scenes):
        seg.visual_interest = [2, 5, 1, 4][i]
    kinds = choose_kinds(script, "hybrid", wan_scenes=3, include_hook=True)
    assert [k for k, v in kinds.items() if v == "wan"] == ["hook", "s2", "s4"]
    assert kinds["outro"] == "parallax"


# --- background clips ---------------------------------------------------------------------


def test_cover_fills_frame_without_borders() -> None:
    img = cover(Image.new("RGB", (768, 1344), "red"), 1080, 1920)
    assert img.size == (1080, 1920)


def test_motion_frames_fill_exact_length_and_respect_min_speed() -> None:
    clip = [np.full((84, 48, 3), i * 10, np.uint8) for i in range(17)]  # 17 frames @ 24 fps
    frames = list(
        motion_frames(clip, n=90, fps=30, clip_fps=24, width=54, height=96, min_speed=0.55)
    )
    assert len(frames) == 90 and frames[0].shape == (96, 54, 3)
    # 0.7 s of clip stretched to at most 1/0.55 → the clip ends and holds before frame 90
    assert frames[-1].mean() == pytest.approx(frames[-5].mean(), abs=2)


def test_crossfade_blends_only_first_frames() -> None:
    prev = np.zeros((4, 4, 3), np.uint8)
    frames = [np.full((4, 4, 3), 200, np.uint8) for _ in range(5)]
    out = list(crossfade(iter(frames), prev, k=2))
    assert out[0].mean() < out[1].mean() < out[2].mean() == 200


def test_kenburns_produces_moving_frames() -> None:
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (96, 54, 3), np.uint8))
    frames = list(kenburns_frames(img, 10, "push_in"))
    assert len(frames) == 10 and not np.array_equal(frames[0], frames[-1])


# --- end to end with fake AI models --------------------------------------------------------


@needs_ffmpeg
def test_pipeline_with_hybrid_ai_visuals(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    cfg.render.width, cfg.render.height = 270, 480
    cfg.visuals.mode = "hybrid"
    cfg.visuals.wan_scenes = 2
    image_gen, motion = FakeImageGen(), FakeMotion()
    shots = ShotList(
        shots=[
            Shot(
                segment_id=s.id,
                image_prompt=f"a quiet snowy scene {s.id}",
                motion="snow falls",
                visual_interest=3,
            )
            for s in script.segments
        ]
    )
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script)],
            "ShotList": [shots],
        }
    )
    llm.unloaded = 0

    def unload() -> None:
        llm.unloaded += 1

    llm.unload_all = unload
    services = Services(
        llm=llm,
        wiki=FakeWiki(source),
        synth_factory=FakeSynth,
        visuals_factory=lambda theme: VisualBuilder(
            cfg, theme, image_gen=image_gen, depth=FakeDepth(), motion=motion
        ),
    )
    pipeline = Pipeline(cfg, db, services)
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    result = pipeline.run(job_id)
    assert result is not None

    wd = pipeline.workdir(job_id).root
    plan = json.loads((wd / "visuals" / "visuals.json").read_text("utf-8"))
    kinds = [s["kind"] for s in plan["scenes"]]
    assert kinds.count("wan") == 2 and kinds.count("parallax") == len(kinds) - 2
    assert motion.calls == 2 and llm.unloaded == 1
    assert all("flat 2D vector illustration" in p for p in image_gen.prompts)
    # every background clip has exactly the frames of its segment
    for sc in plan["scenes"]:
        info = probe(wd / "visuals" / sc["clip"])
        assert int(info["streams"][0]["nb_frames"]) == sc["frames"]

    folder = Path(result.folder)
    sources = json.loads((folder / "sources.json").read_text("utf-8"))
    assert sources["visuals"]["ai_generated"] is True
    assert len(sources["visuals"]["scenes"]) == len(script.segments)
    assert "AI-generated" in (folder / "description.txt").read_text("utf-8")

    # A second build reuses cached illustrations and motion clips.
    builder = VisualBuilder(
        cfg, load_theme("default"), image_gen=FakeImageGen(), depth=FakeDepth(), motion=FakeMotion()
    )
    from factshorts.models import AudioPlan

    audio = AudioPlan.model_validate_json((wd / "audio" / "audio.json").read_text("utf-8"))
    script_now = Script.model_validate_json((wd / "script.json").read_text("utf-8"))
    out = wd / "visuals_again"
    out.mkdir()
    builder.build(job_id, script_now, audio, out, llm, lambda: None, lambda m: None)
    assert builder.image_gen.prompts == [] and builder.motion.calls == 0


def _bad_s1(script: Script) -> Script:
    bad = script.model_copy(deep=True)
    bad.scenes[0].narration = "Its summit rises 7,777 meters above sea level."
    bad.scenes[0].claims[0].text = bad.scenes[0].narration
    return bad


def test_failing_scene_is_fixed_by_targeted_rewrite(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    bad = _bad_s1(script)
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(bad)],
            "LLMSegment": [script_to_llm(script).scenes[0]],  # the corrected scene
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    log: list[str] = []
    assert pipeline.run(job_id, progress=lambda s, m: log.append(m)) is not None
    assert any("rewriting s1" in m for m in log)
    final = Script.model_validate_json(
        (pipeline.workdir(job_id).root / "script.json").read_text("utf-8")
    )
    assert "7,777" not in " ".join(s.narration for s in final.segments)
    assert [s.id for s in final.scenes] == [s.id for s in script.scenes]  # nothing dropped
    assert sum(1 for name, _ in llm.calls if name == "LLMSegment") == 1
    assert sum(1 for name, _ in llm.calls if name == "LLMScript") == 1  # no full redraft


def test_scene_that_keeps_failing_is_dropped_and_replaced(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    cfg.script.min_scenes = 3
    cfg.script.max_topups = 1
    bad = _bad_s1(script)
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(bad)],
            "LLMSegment": [script_to_llm(bad).scenes[0]],  # rewrites never fix it
            "LLMMoreScenes": [LLMMoreScenes(scenes=[script_to_llm(script).scenes[0]])],
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    log: list[str] = []
    assert pipeline.run(job_id, progress=lambda s, m: log.append(m)) is not None
    assert any("dropping 1 scene(s) that still fail: s1" in m for m in log)
    final = Script.model_validate_json(
        (pipeline.workdir(job_id).root / "script.json").read_text("utf-8")
    )
    assert "7,777" not in " ".join(s.narration for s in final.segments)
    assert (
        sum(1 for name, _ in llm.calls if name == "LLMSegment")
        == cfg.verification.max_rewrite_rounds
    )


def test_short_narration_is_extended_to_min_duration(
    cfg: Config, db: Database, source: SourceDoc, script: Script, tmp_path: Path
) -> None:
    from factshorts.core.tts import build_audio

    # A budget close to the fixture's length: the draft tops up once, verification does not,
    # so the scenes added after that come from the voice step's duration check.
    cfg.script.words_per_minute = 78  # budget ≈ the fixture's ~67 words
    drafted = script.model_copy(deep=True)  # the draft step tops up once (+2 scenes)
    drafted.scenes += [to_segment(x) for x in new_fact_scenes(0, 2)]
    drafted.assign_ids()
    baseline = build_audio(drafted, FakeSynth(), cfg.tts, tmp_path, tmp_path).duration
    cfg.script.min_duration_s = baseline + 2.0
    cfg.script.max_topups = 1
    extra = new_fact_scenes(0, 2)  # first top-up (draft step)
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script)],
            "LLMMoreScenes": [
                LLMMoreScenes(scenes=extra),
                LLMMoreScenes(scenes=new_fact_scenes(2, 2)),
            ],  # voice step
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=60))
    log: list[str] = []
    assert pipeline.run(job_id, progress=lambda s, m: log.append(m)) is not None
    assert any("adding scenes" in m for m in log), log
    wd = pipeline.workdir(job_id).root
    audio = json.loads((wd / "audio" / "audio.json").read_text("utf-8"))
    assert audio["duration"] >= cfg.script.min_duration_s
    final = Script.model_validate_json((wd / "script.json").read_text("utf-8"))
    assert len(final.scenes) == len(script.scenes) + 4


@needs_ffmpeg
def test_resume_after_steps_that_rewrite_their_inputs(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    """Verification rewrites script.json (here: a targeted rewrite of s1); resuming later
    must not treat it as stale and redo it."""
    cfg.render.width, cfg.render.height = 270, 480
    bad = script.model_copy(deep=True)
    bad.scenes[0].narration = "Its summit rises 7,777 meters above sea level."
    bad.scenes[0].claims[0].text = bad.scenes[0].narration
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(bad)],
            "LLMSegment": [script_to_llm(script).scenes[0]],
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    assert pipeline.run(job_id) is not None
    (pipeline.workdir(job_id).root / "render" / "render.json").unlink()
    log: list[tuple[str, str]] = []
    pipeline.run(job_id, progress=lambda s, m: log.append((s, m)))
    assert {s for s, m in log if m == "started"} == {"render", "export"}
