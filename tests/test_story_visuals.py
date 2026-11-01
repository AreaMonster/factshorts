"""Story-style scripts (top-up placement, flow-aware rewrites, editing pass) and the
picture review/redraw loop."""

from __future__ import annotations

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
from factshorts.core.art import ImageVerdict, Shot, ShotList, plan_shots, problems_in
from factshorts.core.pipeline import Pipeline, Services
from factshorts.core.script import LLMMoreScenes, ScriptWriter
from factshorts.core.sourcing import ArticleGuess
from factshorts.core.visuals import VisualBuilder
from factshorts.db import Database
from factshorts.models import AudioPlan, Idea, Script, SourceDoc
from factshorts.render.themes import load_theme

# --- script --------------------------------------------------------------------------------


def test_topup_scenes_are_inserted_where_the_story_needs_them(
    cfg: Config, source: SourceDoc, script: Script
) -> None:
    cfg.script.max_topups = 1
    llm = FakeLLM({"LLMMoreScenes": [LLMMoreScenes(scenes=new_fact_scenes(0, 1, after="s1"))]})
    writer = ScriptWriter(Idea(text="Testa?", length=75), source, llm, "m", cfg.script, cfg.tts)
    result = writer.top_up(script)
    assert "35,000 people" in result.scenes[1].narration  # right after the old s1
    assert [s.id for s in result.scenes] == ["s1", "s2", "s3", "s4", "s5"]


def test_rewrite_prompt_shows_the_neighbors(cfg: Config, source: SourceDoc, script: Script) -> None:
    llm = FakeLLM({"LLMSegment": [script_to_llm(script).scenes[1]]})
    writer = ScriptWriter(Idea(text="Testa?"), source, llm, "m", cfg.script, cfg.tts)
    writer.rewrite_segment(script, "s2", ["s2 claim: something failed"])
    prompt = llm.calls[0][1][-1]["content"]
    assert script.scenes[0].narration in prompt and script.scenes[2].narration in prompt


def test_editing_pass_keeps_only_segments_that_still_pass(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    cfg.script.polish = True
    polished = script.model_copy(deep=True)
    # s1 gets a nicer transition and stays correct; s2 "polishing" breaks a fact.
    polished.scenes[0].narration = "So how high does it go? " + polished.scenes[0].narration
    polished.scenes[1].narration = "Ada Brenner and Tomas Kell first climbed it in 1913."
    polished.scenes[1].claims[0].text = polished.scenes[1].narration
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script), script_to_llm(polished)],  # draft, then polish
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    log: list[str] = []
    assert pipeline.run(job_id, progress=lambda s, m: log.append(m)) is not None
    final = Script.model_validate_json(
        (pipeline.workdir(job_id).root / "script.json").read_text("utf-8")
    )
    assert final.scenes[0].narration.startswith("So how high does it go?")
    assert "1931" in final.scenes[1].narration and "1913" not in final.scenes[1].narration
    assert any("editing pass" in m for m in log)


# --- art direction ---------------------------------------------------------------------------


def test_shots_use_appearance_not_names(script: Script) -> None:
    shots = ShotList(
        subject_look="a tall snowy peak with a sharp, icy summit",
        shots=[
            Shot(
                segment_id=s.id,
                image_prompt=f"Mount Testa at dawn, wide shot {s.id}",
                shows_subject=True,
                motion="snow falls",
                visual_interest=3,
            )
            for s in script.segments
        ],
    )
    llm = FakeLLM({"ShotList": [shots]})
    out, look = plan_shots(
        script, llm, "m", "flat art", context="Mount Testa is a peak.", names={"Mount", "Testa"}
    )
    assert look.startswith("a tall snowy peak")
    for seg in out.segments:
        assert not problems_in(seg.image_prompt, {"Testa"})
        assert "The main subject: a tall snowy peak" in seg.image_prompt
    first_prompt = llm.calls[0][1][0]["content"]
    assert "Mount Testa is a peak." in first_prompt  # grounded in the source


# --- picture review loop ----------------------------------------------------------------------


class FakeReviewer:
    """Rejects pictures whose prompt contains a word, until it is revised away."""

    def __init__(self, reject_word: str, fixable: bool = True) -> None:
        self.reject_word = reject_word
        self.fixable = fixable
        self.reviews = 0

    def review(self, image, narration, prompt, title, subject_look) -> ImageVerdict:
        self.reviews += 1
        bad = self.reject_word in prompt
        return ImageVerdict(
            shows="a picture", problems=["shows rings"] if bad else [], matches=not bad
        )

    def revise(self, narration, prompt, verdict, subject_look, names) -> str:
        return prompt.replace(self.reject_word, "plain") if self.fixable else prompt


def _build(cfg: Config, script: Script, tmp_path, reviewer) -> tuple:
    from factshorts.core.tts import build_audio

    cfg.render.width, cfg.render.height = 90, 160
    cfg.visuals.mode = "kenburns"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio: AudioPlan = build_audio(script, FakeSynth(), cfg.tts, audio_dir, tmp_path)
    for seg in script.segments:
        seg.image_prompt = "a ringed planet" if seg.id == "s1" else "a snowy valley"
    gen = FakeImageGen()
    builder = VisualBuilder(
        cfg,
        load_theme("default"),
        image_gen=gen,
        depth=FakeDepth(),
        motion=FakeMotion(),
        reviewer=reviewer,
    )
    out = tmp_path / "visuals"
    out.mkdir()
    plan = builder.build("job", script, audio, out, None, lambda: None, lambda m: None)
    return plan, gen


@needs_ffmpeg
def test_wrong_picture_is_redrawn(cfg: Config, script: Script, tmp_path) -> None:
    plan, gen = _build(cfg, script, tmp_path, FakeReviewer("ringed"))
    s1 = next(s for s in plan.scenes if s.segment_id == "s1")
    assert s1.review == "ok after 1 redraw(s)"
    assert "ringed" not in s1.prompt
    assert sum("plain planet" in p for p in gen.prompts) == 1
    assert all(s.review == "ok" for s in plan.scenes if s.segment_id != "s1")


@needs_ffmpeg
def test_unfixable_picture_falls_back_to_plain_subject(
    cfg: Config, script: Script, tmp_path
) -> None:
    plan, _gen = _build(cfg, script, tmp_path, FakeReviewer("ringed", fixable=False))
    s1 = next(s for s in plan.scenes if s.segment_id == "s1")
    assert s1.review.startswith("fallback: shows rings")
    assert s1.prompt.startswith("a calm, simple wide landscape")  # no subject look known


def test_hook_that_keeps_failing_is_replaced_by_first_scene(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    cfg.script.min_scenes = 3
    bad = script.model_copy(deep=True)
    bad.hook.narration = "Mount Testa rises 9,999 meters above sea level."
    bad.hook.claims[0].text = bad.hook.narration
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(bad)],
            "LLMSegment": [script_to_llm(bad).hook],  # rewrites never fix it
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    log: list[str] = []
    assert pipeline.run(job_id, progress=lambda s, m: log.append(m)) is not None
    final = Script.model_validate_json(
        (pipeline.workdir(job_id).root / "script.json").read_text("utf-8")
    )
    assert final.hook.narration == script.scenes[0].narration
    assert any("now opens the video" in m for m in log)


def test_failing_scene_inserted_mid_story_by_extension_is_dropped(
    cfg: Config, db: Database, source: SourceDoc, script: Script, tmp_path
) -> None:
    """Regression: inserted scenes renumber ids; the new failing one must still be found."""
    from factshorts.core.script import LLMNewScene, LLMSentence, to_segment
    from factshorts.core.tts import build_audio

    cfg.script.words_per_minute = 78  # budget ≈ the fixture, so only the voice step extends
    cfg.script.max_topups = 1
    drafted = script.model_copy(deep=True)
    drafted.scenes += [to_segment(x) for x in new_fact_scenes(0, 2)]
    drafted.assign_ids()
    cfg.script.min_duration_s = (
        build_audio(drafted, FakeSynth(), cfg.tts, tmp_path, tmp_path).duration + 2.0
    )
    wrong = LLMNewScene(
        insert_after="s1",
        on_screen_text="",
        visual="big_number",
        sentences=[
            LLMSentence(
                text="Its cable car takes 9,000 riders up each day.",
                source_quotes=[
                    "A cable car built in 1987 carries visitors to a viewing platform at 3,100 m."
                ],
                claim_type="number",
            )
        ],
    )
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script)],
            "LLMMoreScenes": [
                LLMMoreScenes(scenes=new_fact_scenes(0, 2)),
                LLMMoreScenes(scenes=[wrong, *new_fact_scenes(2, 1, after="s3")]),
            ],
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=60))
    log: list[str] = []
    assert pipeline.run(job_id, progress=lambda s, m: log.append(m)) is not None, log
    final = Script.model_validate_json(
        (pipeline.workdir(job_id).root / "script.json").read_text("utf-8")
    )
    text = " ".join(s.narration for s in final.segments)
    assert "9,000 riders" not in text and "150 km/h" in text
    assert any("dropped 1 new scene" in m for m in log)


def test_dedupe_removes_repeats_and_fact_free_scenes(script: Script) -> None:
    from factshorts.core.script import dedupe_script
    from factshorts.models import Segment

    s = script.model_copy(deep=True)
    repeat = s.scenes[0].model_copy(deep=True)  # same sentence again later
    stray = Segment(narration="Follow for more facts.", on_screen_text="", visual="title")
    s.scenes += [repeat, stray]
    s.assign_ids()
    out, removed = dedupe_script(s, min_scenes=3)
    texts = [x.narration for x in out.scenes]
    assert removed == 2  # the repeated sentence and the stray sign-off (same as the outro)
    assert texts.count(script.scenes[0].narration) == 1
    assert len(out.scenes) == len(script.scenes)
    assert out.outro.narration == script.outro.narration
