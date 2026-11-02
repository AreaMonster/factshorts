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
    assert "calm, simple landscape" in s1.prompt  # no subject look known


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


def test_similar_shots_are_detected() -> None:
    from factshorts.core.art import similar_shots

    def shot(sid: str, prompt: str) -> Shot:
        return Shot(
            segment_id=sid, image_prompt=prompt, shows_subject=True, motion="m", visual_interest=2
        )

    shots = {
        "s1": shot("s1", "a pale yellow planet wrapped in thick clouds against a dark sky"),
        "s2": shot("s2", "a pale yellow planet wrapped in thick clouds against a starry dark sky"),
        "s3": shot("s3", "a rocky volcanic plain under an orange hazy sky, low angle"),
    }
    assert similar_shots(shots) == [("s1", "s2")]


def test_each_segment_gets_a_different_framing(script: Script) -> None:
    shots = ShotList(
        subject_look="a snowy peak",
        shots=[
            Shot(
                segment_id=s.id,
                image_prompt=f"scene {i}: {'x ' * i}",
                shows_subject=False,
                motion="m",
                visual_interest=2,
            )
            for i, s in enumerate(script.segments)
        ],
    )
    llm = FakeLLM({"ShotList": [shots]})
    plan_shots(script, llm, "m", "flat art")
    prompt = llm.calls[0][1][0]["content"]
    framings = [
        line.split("(framing: ")[1].split(")")[0]
        for line in prompt.splitlines()
        if "(framing: " in line
    ]
    assert len(framings) == len(script.segments)
    assert len(set(framings)) == len(framings)


@needs_ffmpeg
def test_fallback_pictures_differ(cfg: Config, script: Script, tmp_path) -> None:
    from factshorts.core.tts import build_audio

    cfg.render.width, cfg.render.height = 90, 160
    cfg.visuals.mode = "kenburns"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = build_audio(script, FakeSynth(), cfg.tts, audio_dir, tmp_path)
    for seg in script.segments:
        seg.image_prompt = "a ringed planet"  # every picture keeps failing
    builder = VisualBuilder(
        cfg,
        load_theme("default"),
        image_gen=FakeImageGen(),
        depth=FakeDepth(),
        motion=FakeMotion(),
        reviewer=FakeReviewer("ringed", fixable=False),
    )
    out = tmp_path / "visuals"
    out.mkdir()
    plan = builder.build("job", script, audio, out, None, lambda: None, lambda m: None)
    prompts = [s.prompt for s in plan.scenes]
    assert all(s.review.startswith("fallback") for s in plan.scenes)
    assert len(set(prompts[:4])) == 4  # four different framings before they repeat


def test_editing_pass_cannot_move_facts_between_segments(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    cfg.script.polish = True
    polished = script.model_copy(deep=True)
    # The "polish" swaps the facts of s1 and s2 (each still checks out on its own).
    polished.scenes[0], polished.scenes[1] = (
        script.scenes[1].model_copy(deep=True),
        script.scenes[0].model_copy(deep=True),
    )
    polished.assign_ids()
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script), script_to_llm(polished)],
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    assert pipeline.run(job_id) is not None
    final = Script.model_validate_json(
        (pipeline.workdir(job_id).root / "script.json").read_text("utf-8")
    )
    assert [s.narration for s in final.scenes] == [s.narration for s in script.scenes]


def _story(*sentences_per_scene: str) -> Script:
    from factshorts.models import Claim, Segment

    def seg(text: str) -> Segment:
        from factshorts.verify.normalize import split_sentences

        return Segment(
            narration=text,
            on_screen_text="",
            claims=[Claim(text=t, source_quote="q") for t in split_sentences(text)],
        )

    return Script(
        title="Venus",
        hook=seg("Venus is hotter than Mercury."),
        scenes=[seg(t) for t in sentences_per_scene],
        outro=Segment(narration="Thanks for watching!", on_screen_text=""),
    ).assign_ids()


def test_reworded_repeat_is_removed_with_entailment() -> None:
    from factshorts.core.script import dedupe_script

    pairs = {
        (
            "The atmosphere has thick clouds that span the whole planet.",
            "These clouds cover the entire planet.",
        ): 0.98
    }
    script = _story(
        "The atmosphere has thick clouds that span the whole planet.",
        "These clouds cover the entire planet.",
        "Its pressure is 92 times Earth's.",
        "Its day lasts 243 Earth days.",
    )
    out, removed = dedupe_script(script, 3, lambda p, h: pairs.get((p, h), 0.1), 0.75)
    assert removed == 1
    assert "cover the entire planet" not in " ".join(s.narration for s in out.segments)
    assert len(out.scenes) == 3


def test_contained_sentence_is_a_repeat_but_numbers_keep_sentences_apart() -> None:
    from factshorts.core.script import dedupe_script

    script = _story(
        "At the surface it reaches 464 °C, making it the hottest planet in the solar system.",
        "This makes Venus the hottest planet in the solar system.",
        "Venus orbits 108 million km from the Sun.",
        "Mercury orbits 58 million km from the Sun.",
    )
    out, removed = dedupe_script(script, 3, None, subject_words={"venus", "mercury"})
    text = " ".join(s.narration for s in out.segments)
    assert removed == 1 and "This makes Venus the hottest" not in text
    assert "108 million" in text and "58 million" in text
