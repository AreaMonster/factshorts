"""Phase 4: music + loudness, thumbnails, scene-type graphics, themes, rerender, Whisper timing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import FakeLLM, FakeSynth, FakeWiki, needs_ffmpeg, script_to_llm
from factshorts.config import AudioConfig, Config
from factshorts.core.captions import align_by_text
from factshorts.core.pipeline import Pipeline, Services
from factshorts.core.sourcing import ArticleGuess
from factshorts.db import Database
from factshorts.models import Idea, Script, SourceDoc
from factshorts.render import audio as audio_mod
from factshorts.render.infographics import draw_infographic, plan_infographic
from factshorts.render.themes import list_themes, load_theme
from factshorts.render.thumbnail import make_thumbnail

# --- music + loudness ------------------------------------------------------------------------


def test_ambient_pad_is_stereo_audible_and_seeded() -> None:
    a = audio_mod.ambient_pad(8.0, seed=3, sr=16000)
    b = audio_mod.ambient_pad(8.0, seed=3, sr=16000)
    c = audio_mod.ambient_pad(8.0, seed=4, sr=16000)
    assert a.shape == (8 * 16000 + 1, 2)
    assert 0.3 < float(np.abs(a).max()) <= 0.51
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert abs(a[0]).max() < 1e-3 and abs(a[-1]).max() < 1e-3  # fades in and out


def test_music_files_win_over_generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_mod, "music_files", lambda: [])
    assert "generated" in audio_mod.make_music(tmp_path / "m.wav", 3.0, 1)


@needs_ffmpeg
def test_master_hits_target_loudness_with_music(tmp_path: Path) -> None:
    sr = 24000
    t = np.arange(int(6 * sr)) / sr
    speech = 0.05 * np.sin(2 * math.pi * 220 * t) * (np.sin(2 * math.pi * 1.5 * t) > 0)
    narration = tmp_path / "n.wav"
    audio_mod.write_wav(narration, speech[:, None].astype(np.float32), sr)
    music = tmp_path / "m.wav"
    audio_mod.make_music(music, 6.0, 7)
    out = tmp_path / "out.wav"
    audio_mod.master(narration, out, 6.0, -14.0, AudioConfig(), music)
    assert audio_mod.loudness(out) == pytest.approx(-14.0, abs=1.0)


# --- thumbnail -------------------------------------------------------------------------------


def test_thumbnail_is_vertical_jpeg(tmp_path: Path) -> None:
    base = Image.new("RGB", (768, 1344), (40, 120, 200))
    out = make_thumbnail(
        base, "Why is Venus hotter than Mercury?", load_theme("default"), tmp_path / "thumbnail.jpg"
    )
    img = Image.open(out)
    assert img.format == "JPEG" and img.size == (1080, 1920)
    # the title area is not just the background color any more
    assert np.asarray(img)[200:700].std() > 10


# --- scene-type graphics -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("visual", "on_screen", "narration", "kind"),
    [
        ("comparison", "Venus 464 °C vs Mercury 167 °C", "", "comparison"),
        ("comparison", "Hot planet", "It is very hot.", None),  # no numbers: no chart
        ("timeline", "First ascent", "It was climbed in 1953 after attempts in 1921.", "timeline"),
        ("icon_list", "granite, gneiss and limestone", "", "list"),
        ("quote", "The strongest greenhouse effect", "", "quote"),
        ("map_pin", "Himalayas", "", "pin"),
        ("image", "Anything", "", None),
    ],
)
def test_plan_infographic(visual: str, on_screen: str, narration: str, kind: str | None) -> None:
    info = plan_infographic(visual, on_screen, narration)
    assert (info.kind if info else None) == kind
    if info:
        layer = draw_infographic(info, load_theme("default"), 3.0, 900, 1.0, "Source: Wikipedia")
        assert layer.mode == "RGBA" and layer.getbbox() is not None


def test_comparison_labels_and_timeline_order() -> None:
    comp = plan_infographic("comparison", "Venus 464 °C vs Mercury 167 °C", "")
    assert comp.items == ["Venus 464 °C", "Mercury 167 °C"]
    line = plan_infographic("timeline", "", "In 1953, after 1924 and 1921.")
    assert line.items == ["1921", "1924", "1953"]


# --- themes ----------------------------------------------------------------------------------


def test_bundled_themes_load_and_differ() -> None:
    names = list_themes()
    assert {"default", "neon", "papercut", "watercolor"} <= set(names)
    styles = {load_theme(n).art.style for n in names}
    assert len(styles) == len(names)
    with pytest.raises(FileNotFoundError, match="available"):
        load_theme("nope")


# --- rerender --------------------------------------------------------------------------------


@needs_ffmpeg
def test_rerender_redoes_only_voice_onward(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    cfg.render.width, cfg.render.height = 270, 480
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script)],
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    assert pipeline.run(job_id) is not None
    calls = llm.work_calls

    pipeline.rerender(job_id, theme="neon", music=False)
    log: list[tuple[str, str]] = []
    result = pipeline.run(job_id, progress=lambda s, m: log.append((s, m)))
    assert result is not None
    assert {s for s, m in log if m == "started"} == {"voice", "visuals", "render", "export"}
    assert llm.work_calls == calls  # no new script, no new fact-checks
    render = json.loads((pipeline.workdir(job_id).root / "render" / "render.json").read_text())
    assert render["music_default"] is False  # the music track exists, the main video has none
    assert render["video"] == render["video_voice"] and render["video_music"]
    with pytest.raises(FileNotFoundError):
        pipeline.rerender(job_id, theme="does-not-exist")


@needs_ffmpeg
def test_rerender_keeps_script_of_jobs_with_old_fingerprints(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    cfg.render.width, cfg.render.height = 270, 480
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script)],
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    assert pipeline.run(job_id) is not None
    wd = pipeline.workdir(job_id)
    for step in ("sourcing", "premise", "script", "verification"):  # as stored before Phase 4
        db.set_step_hash(job_id, step, pipeline._legacy_hash(wd, step))
    calls = llm.work_calls

    pipeline.rerender(job_id, theme="watercolor")
    log: list[tuple[str, str]] = []
    assert pipeline.run(job_id, progress=lambda s, m: log.append((s, m))) is not None
    assert {s for s, m in log if m == "started"} == {"voice", "visuals", "render", "export"}
    assert llm.work_calls == calls


@needs_ffmpeg
def test_final_video_reports_music_and_loudness(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    cfg.render.width, cfg.render.height = 270, 480
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script)],
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    result = pipeline.run(job_id)
    assert result is not None and result.thumbnail and Path(result.thumbnail).is_file()
    sources = json.loads((Path(result.folder) / "sources.json").read_text("utf-8"))
    assert "generated ambient" in sources["video"]["music"]
    assert sources["video"]["loudness_lufs"] == pytest.approx(-14.0, abs=1.5)


# --- Whisper-style alignment ------------------------------------------------------


def test_align_by_text_handles_digits_split_by_the_transcriber() -> None:
    heard = [
        ("Mount", 0.0, 0.25),
        ("Everest", 0.26, 0.6),
        ("rises", 0.6, 1.0),
        ("8", 1.06, 1.4),
        (",849", 1.44, 2.8),
        ("meters.", 2.88, 3.3),
    ]
    words = align_by_text("Mount Everest rises 8,849 meters.", heard, 10.0, 3.5)
    assert words is not None
    by = {w.text: w for w in words}
    assert by["8,849"].start == pytest.approx(11.06, abs=0.15)
    assert by["meters."].start == pytest.approx(12.9, abs=0.1)
    assert words[-1].end == pytest.approx(13.5)


def test_align_by_text_gives_up_on_unrelated_audio() -> None:
    assert (
        align_by_text("Mount Everest rises high.", [("hello", 0, 1), ("world", 1, 2)], 0, 2) is None
    )


@needs_ffmpeg
def test_music_is_a_separate_track_with_two_finished_versions(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    cfg.render.width, cfg.render.height = 270, 480
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script)],
        }
    )
    pipeline = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30, music=False))
    result = pipeline.run(job_id)
    assert result is not None and result.video_no_music and not result.music_default
    assert (pipeline.workdir(job_id).root / "render" / "music.wav").is_file()
    with_music, plain = Path(result.video), Path(result.video_no_music)
    assert plain.name.endswith("_no-music.mp4") and with_music.is_file()
    # Same picture and length, but only one of them carries the music.
    assert ffmpeg_durations(with_music) == ffmpeg_durations(plain)
    assert audio_mod.loudness(with_music) == pytest.approx(-14.0, abs=1.5)
    assert audio_mod.loudness(plain) == pytest.approx(-14.0, abs=1.5)
    caption = Path(result.caption_txt).read_text("utf-8")
    assert caption.startswith(script.hook.narration)  # no caption answer queued: the hook
    meta = json.loads(Path(result.metadata_json).read_text("utf-8"))
    assert meta["music_default"] is False


def ffmpeg_durations(path: Path) -> tuple[float, float]:
    from factshorts.render import ffmpeg

    d = ffmpeg.stream_durations(path)
    return round(d["video"], 2), round(d["audio"], 1)
