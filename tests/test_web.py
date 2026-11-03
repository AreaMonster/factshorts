"""Phase 5: the Web UI (FastAPI TestClient), its worker, edits, settings and the doctor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from conftest import FakeLLM, FakeNLI, FakeSynth, FakeWiki, needs_ffmpeg, script_to_llm
from factshorts.config import Config, apply_overrides
from factshorts.core.doctor import config_problems, is_local_host
from factshorts.core.pipeline import Services
from factshorts.core.sourcing import ArticleGuess
from factshorts.core.tts import build_audio
from factshorts.db import Database
from factshorts.models import Script, SourceDoc, Verification
from factshorts.web.app import create_app

QUOTE = (
    "Mount Testa is the highest mountain in Fictland, with a summit elevation of 6,412 m "
    "(21,037 ft) above sea level."
)


@pytest.fixture
def llm(source: SourceDoc, script: Script) -> FakeLLM:
    return FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script)],
        }
    )


@pytest.fixture
def synth() -> FakeSynth:
    return FakeSynth()


@pytest.fixture
def client(cfg: Config, db: Database, source: SourceDoc, llm: FakeLLM, synth: FakeSynth):
    cfg.render.width, cfg.render.height = 270, 480
    services = Services(
        llm=llm, wiki=FakeWiki(source), synth_factory=lambda: synth, nli_factory=FakeNLI
    )
    app = create_app(cfg, db, services, start_worker=False)
    with TestClient(app, follow_redirects=False) as c:
        c.state = app.state.fs  # type: ignore[attr-defined]
        yield c


def drain(client: TestClient) -> None:
    while client.state.worker.process_next():
        pass


def make_video(client: TestClient) -> str:
    r = client.post(
        "/create", data={"ideas": "How high is Mount Testa?", "length": "30", "factcheck": "on"}
    )
    assert r.status_code == 303 and r.headers["location"] == "/jobs"
    job_id = client.get("/api/jobs").json()[0]["id"]
    drain(client)
    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "done"
    return job_id


# --- pages --------------------------------------------------------------------------------


def test_every_page_renders(client: TestClient) -> None:
    for path in ("/create", "/jobs", "/review", "/library", "/settings", "/login"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "factshorts" in r.text
    assert client.get("/").headers["location"] == "/create"
    assert client.get("/static/htmx.min.js").status_code == 200  # vendored, no CDN


def test_settings_shows_ollama_unreachable(client: TestClient) -> None:
    assert "Ollama is not reachable" in client.get("/settings").text


# --- create → review → approve -------------------------------------------------------------


@needs_ffmpeg
def test_create_review_and_approve(client: TestClient, cfg: Config) -> None:
    job_id = make_video(client)
    page = client.get(f"/review/{job_id}")
    assert page.status_code == 200
    assert "Its summit rises 6,412 meters above sea level." in page.text
    assert '<span class="check pass"' in page.text and ">quote</span>" in page.text
    # Both versions exist; the player switches between them and Download follows.
    assert 'data-plain="/media/' in page.text and 'id="download"' in page.text
    assert "_no-music.mp4" in page.text
    assert "TikTok caption" in page.text and "#facts" in page.text
    assert "oldid=" in page.text or "Mount Testa" in page.text  # article (revision) link
    # The thumbnail is shown and can be downloaded on its own.
    assert "Download thumbnail" in page.text
    assert f"/media/{job_id}/thumbnail.jpg?download=1" in page.text
    video = client.get(f"/api/jobs/{job_id}").json()["video"]
    mp4 = next(p.name for p in Path(video["folder"]).glob("*.mp4"))
    assert client.get(f"/media/{job_id}/{mp4}").headers["content-type"] == "video/mp4"
    thumb = client.get(f"/media/{job_id}/thumbnail.jpg?download=1")
    assert thumb.headers["content-type"] == "image/jpeg"
    assert "attachment" in thumb.headers["content-disposition"]
    queue = client.get("/review").text
    assert job_id in queue and f"/media/{job_id}/thumbnail.jpg" in queue
    assert 'class="count"' in queue  # the navigation shows how many videos wait

    assert client.post(f"/review/{job_id}/approve").status_code == 303
    video = client.get(f"/api/jobs/{job_id}").json()["video"]
    assert video["status"] == "approved"
    assert Path(video["folder"]).parent == cfg.output_dir / "approved"
    assert (Path(video["folder"]) / mp4).is_file()
    assert "How high is Mount Testa?" in client.get("/library").text
    assert "Mount Testa" not in client.get("/library?q=Everest").text.split("<main>")[1]


def test_inbox_then_make(client: TestClient) -> None:
    r = client.post(
        "/create", data={"ideas": "How high is Mount Testa?", "action": "queue", "factcheck": "on"}
    )
    assert "1 idea(s) queued" in r.text
    jobs_page = client.get("/jobs").text
    assert "How high is Mount Testa?" in jobs_page
    idea_id = client.state.db.query("SELECT id FROM ideas WHERE status = 'queued'")[0]["id"]
    r = client.post(f"/inbox/{idea_id}/make")
    assert r.status_code == 303
    assert len(client.get("/api/jobs").json()) == 1
    # The same idea again is a duplicate.
    r = client.post(
        "/create", data={"ideas": "How high is Mount Testa?", "action": "queue", "factcheck": "on"}
    )
    assert "duplicate" in r.text


def test_unchecked_factcheck_is_labeled(client: TestClient) -> None:
    client.post("/create", data={"ideas": "Why mountains are the best", "action": "queue"})
    idea = client.state.db.query("SELECT settings FROM ideas")[0]["settings"]
    assert json.loads(idea)["factcheck"] is False
    assert "no fact-check" in client.get("/jobs").text


# --- edits --------------------------------------------------------------------------------


@needs_ffmpeg
def test_edit_narration_rechecks_and_rerenders_only_that_scene(
    client: TestClient, cfg: Config, llm: FakeLLM, synth: FakeSynth
) -> None:
    job_id = make_video(client)
    first_folder = client.get(f"/api/jobs/{job_id}").json()["video"]["folder"]
    new = "Its peak stands 6,412 meters above the sea."
    llm.responses["LLMQuotedSentences"] = [
        {"sentences": [{"text": new, "source_quotes": [QUOTE], "claim_type": "number"}]}
    ]
    synth.calls.clear()
    r = client.post(
        f"/review/{job_id}/edit",
        data={"segment": "s1", "narration": new, "on_screen_text": "6,412 m"},
    )
    assert r.status_code == 303 and r.headers["location"] == f"/jobs/{job_id}"
    drain(client)

    info = client.get(f"/api/jobs/{job_id}").json()
    assert info["tasks"][-1]["state"] == "done", info["tasks"][-1]
    wd = cfg.work_dir / job_id
    script = Script.model_validate_json((wd / "script.json").read_text("utf-8"))
    assert script.scenes[0].narration == new
    assert script.scenes[0].on_screen_text == "6,412 m"
    verification = Verification.model_validate_json((wd / "verification.json").read_text("utf-8"))
    assert verification.passed
    assert any(c.segment_id == "s1" and c.text == new for c in verification.claims)
    # Only the new sentence was spoken; every other one came from the voice cache.
    assert synth.calls == ["Its peak stands six thousand four hundred twelve meters above the sea."]
    folders = list((cfg.output_dir / "pending").glob(f"*_{job_id[-6:]}"))
    assert folders == [Path(info["video"]["folder"])]  # the new version replaced the old one
    assert Path(first_folder) == folders[0] or not Path(first_folder).exists()
    assert "Edit" in client.get(f"/review/{job_id}").text


@needs_ffmpeg
def test_edit_with_an_invented_number_is_refused(
    client: TestClient, cfg: Config, llm: FakeLLM
) -> None:
    job_id = make_video(client)
    wd = cfg.work_dir / job_id
    before = (wd / "script.json").read_text("utf-8")
    bad = "Its summit rises 7,000 meters above sea level."
    llm.responses["LLMQuotedSentences"] = [
        {"sentences": [{"text": bad, "source_quotes": [QUOTE], "claim_type": "number"}]}
    ]
    client.post(f"/review/{job_id}/edit", data={"segment": "s1", "narration": bad})
    drain(client)
    task = client.get(f"/api/jobs/{job_id}").json()["tasks"][-1]
    assert task["state"] == "failed"
    assert "Not changed" in task["message"] and "7,000" in task["message"]
    assert (wd / "script.json").read_text("utf-8") == before
    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "done"
    assert "Not changed" in client.get(f"/jobs/{job_id}/stream?once=1").text


@needs_ffmpeg
def test_regenerate_scene(client: TestClient, cfg: Config, llm: FakeLLM) -> None:
    job_id = make_video(client)
    llm.responses["LLMSegment"] = [
        {
            "sentences": [
                {
                    "text": "Winds above 150 km/h are common on the north face.",
                    "source_quotes": ["Winds above 150 km/h are common on the north face."],
                    "claim_type": "number",
                }
            ],
            "on_screen_text": "150 km/h winds",
            "visual": "big_number",
        }
    ]
    client.post(f"/review/{job_id}/regenerate", data={"segment": "s3"})
    drain(client)
    assert client.get(f"/api/jobs/{job_id}").json()["tasks"][-1]["state"] == "done"
    script = Script.model_validate_json((cfg.work_dir / job_id / "script.json").read_text("utf-8"))
    assert script.scenes[2].narration == "Winds above 150 km/h are common on the north face."
    asks = [m[-1]["content"] for name, m in llm.calls if name == "LLMSegment"]
    assert len(asks) == 1 and "new version of segment s3" in asks[0]


def test_edits_wait_until_the_job_is_finished(client: TestClient) -> None:
    client.post("/create", data={"ideas": "How high is Mount Testa?", "factcheck": "on"})
    job_id = client.get("/api/jobs").json()[0]["id"]
    r = client.post(f"/review/{job_id}/edit", data={"segment": "s1", "narration": "x"})
    assert r.status_code == 409


# --- cancel, needs input ---------------------------------------------------------------


def test_cancel_queued_job(client: TestClient) -> None:
    client.post("/create", data={"ideas": "How high is Mount Testa?", "factcheck": "on"})
    job_id = client.get("/api/jobs").json()[0]["id"]
    client.post(f"/jobs/{job_id}/cancel")
    info = client.get(f"/api/jobs/{job_id}").json()
    assert info["state"] == "cancelled" and not info["busy"]
    assert client.state.worker.process_next() is False


def test_cancel_running_job_stops_at_the_next_checkpoint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post("/create", data={"ideas": "How high is Mount Testa?", "factcheck": "on"})
    job_id = client.get("/api/jobs").json()[0]["id"]
    db = client.state.db
    real_claim = db.claim_task

    def claim_then_cancel():
        task = real_claim()
        client.state.worker.cancel(job_id)  # the user presses Cancel while it runs
        return task

    monkeypatch.setattr(db, "claim_task", claim_then_cancel)
    assert client.state.worker.process_next()
    info = client.get(f"/api/jobs/{job_id}").json()
    assert info["state"] == "cancelled"
    assert info["tasks"][-1]["state"] == "cancelled"
    assert info["steps"]["sourcing"] == "cancelled"


def test_false_premise_waits_for_an_answer(client: TestClient, llm: FakeLLM) -> None:
    llm.responses["Presuppositions"] = [{"statements": ["Mount Testa is in Northland."]}]
    llm.responses["PremiseJudgement"] = [
        {
            "reason": "the source names Fictland",
            "verdict": "CONTRADICTED",
            "source_sentence": QUOTE,
        }
    ]
    client.post(
        "/create", data={"ideas": "Why is Mount Testa in Northland so high?", "factcheck": "on"}
    )
    job_id = client.get("/api/jobs").json()[0]["id"]
    drain(client)
    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "needs_input"
    live = client.get(f"/jobs/{job_id}/stream?once=1").text
    assert "Mount Testa is in Northland." in live and "Make a corrected video" in live
    client.post(f"/jobs/{job_id}/answer", data={"choice": "cancel"})
    drain(client)
    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "cancelled"


# --- live updates, media safety --------------------------------------------------------------


def test_sse_streams_the_jobs_list(client: TestClient) -> None:
    client.post("/create", data={"ideas": "How high is Mount Testa?", "factcheck": "on"})
    r = client.get("/jobs/stream?once=1")
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.startswith("event: jobs\ndata: ")
    assert "How high is Mount Testa?" in r.text


@needs_ffmpeg
def test_media_only_serves_files_of_the_video(client: TestClient, cfg: Config) -> None:
    job_id = make_video(client)
    (cfg.data_dir / "secret.txt").write_text("x")
    assert client.get(f"/media/{job_id}/..%2F..%2F..%2Fsecret.txt").status_code == 404
    assert client.get("/media/nojob/x.mp4").status_code == 404


def test_cross_site_posts_are_refused(client: TestClient) -> None:
    r = client.post("/batch", data={"per_topic": "1"}, headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    assert (
        client.post(
            "/batch", data={"per_topic": "0"}, headers={"origin": "http://testserver"}
        ).status_code
        == 303
    )


# --- settings, password --------------------------------------------------------------------


def test_settings_are_saved_over_config_and_can_be_reset(client: TestClient) -> None:
    base = client.state.base_cfg
    r = client.post(
        "/settings",
        data={
            "ollama.writer_model": "llama3.1:8b",
            "script.default_length": "45",
            "verification.nli_threshold": "0.6",
            "bool:audio.music": "1",  # checkbox shown but not ticked: music off
            "ideas.topics": "space\nvolcanoes\n",
        },
    )
    assert r.headers["location"] == "/settings?message=Saved"
    cfg = client.state.cfg
    assert cfg.ollama.writer_model == "llama3.1:8b"
    assert cfg.script.default_length == 45 and cfg.verification.nli_threshold == 0.6
    assert cfg.audio.music is False and cfg.ideas.topics == ["space", "volcanoes"]
    assert base.ollama.writer_model != "llama3.1:8b"  # config.toml itself is untouched
    assert client.state.worker.settings_version == 1  # pipelines are rebuilt

    bad = client.post("/settings", data={"verification.nli_threshold": "abc"})
    assert "error=" in bad.headers["location"]
    client.post("/settings/reset")
    assert client.state.cfg.ollama.writer_model == base.ollama.writer_model


def test_password_protects_every_page(client: TestClient) -> None:
    r = client.post("/settings/password", data={"password": "short", "confirm": "short"})
    assert "error=" in r.headers["location"]
    client.post(
        "/settings/password", data={"password": "correct horse", "confirm": "correct horse"}
    )
    stored = client.state.db.get_settings()["web.password_hash"]
    assert stored.startswith("$argon2")

    client.cookies.clear()
    assert client.get("/jobs").headers["location"] == "/login"
    assert client.get("/api/jobs").status_code == 303
    assert client.get("/static/app.css").status_code == 200
    assert (
        client.post("/login", data={"password": "wrong"})
        .headers["location"]
        .startswith("/login?error")
    )
    assert client.post("/login", data={"password": "correct horse"}).headers["location"] == "/"
    assert client.get("/jobs").status_code == 200
    client.post("/logout")
    assert client.get("/jobs").status_code == 303


def test_apply_overrides_keeps_data_dir_and_skips_unknown_keys(cfg: Config) -> None:
    new = apply_overrides(
        cfg, {"general.data_dir": "elsewhere", "nope.x": 1, "tts.voice": "af_heart"}
    )
    assert new.data_dir == cfg.data_dir and new.tts.voice == "af_heart"
    with pytest.raises(ValueError):
        apply_overrides(cfg, {"script.default_length": 5})


# --- worker restart --------------------------------------------------------------------------


def test_interrupted_tasks_run_again_after_a_restart(db: Database) -> None:
    a = db.add_task("run", "job-a")
    b = db.add_task("run", "job-b")
    db.update_task(a, state="running")
    db.update_task(b, state="cancelling")
    assert db.requeue_interrupted() == 2
    assert db.get_task(a)["state"] == "queued"
    assert db.get_task(b)["state"] == "cancelled"


def test_one_task_per_job_at_a_time(db: Database) -> None:
    first = db.add_task("run", "job-a")
    db.add_task("edit", "job-a")
    other = db.add_task("run", "job-b")
    assert db.claim_task()["id"] == first
    assert db.claim_task()["id"] == other  # job-a is busy, so its edit waits
    assert db.claim_task() is None


# --- doctor and the ui command ----------------------------------------------------------


def test_doctor_flags_keys_and_remote_llms(cfg: Config) -> None:
    assert config_problems(cfg) == []
    bad = cfg.model_copy(deep=True)
    bad.ollama.url = "https://api.example.com"
    bad.wikipedia.user_agent = "factshorts sk-abcdefghijklmnopqrstuvwxyz123456"
    problems = config_problems(bad)
    assert any("not this computer" in p for p in problems)
    assert any("looks like an API key" in p for p in problems)
    assert is_local_host("127.0.0.1") and is_local_host("localhost")
    assert not is_local_host("0.0.0.0") and not is_local_host("192.168.1.20")


def test_ui_refuses_the_network_without_a_password(tmp_path: Path, monkeypatch) -> None:
    from factshorts.cli import app as cli_app

    config = tmp_path / "config.toml"
    data = (tmp_path / "data").as_posix()
    config.write_text(f'[general]\ndata_dir = "{data}"\n', encoding="utf-8")
    result = CliRunner().invoke(cli_app, ["ui", "--host", "0.0.0.0", "--config", str(config)])
    assert result.exit_code == 1
    assert "password" in result.output


# --- pictures are kept when only one scene changes ----------------------------------------


@needs_ffmpeg
def test_visuals_keep_checked_pictures_of_unchanged_scenes(
    cfg: Config, script: Script, tmp_path: Path
) -> None:
    from conftest import FakeDepth, FakeImageGen, FakeMotion
    from factshorts.core.visuals import VisualBuilder
    from factshorts.render.themes import load_theme
    from test_story_visuals import FakeReviewer

    cfg.render.width, cfg.render.height = 90, 160
    cfg.visuals.mode = "kenburns"
    for seg in script.segments:
        seg.image_prompt = f"a snowy valley, view {seg.id}"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    out = tmp_path / "visuals"
    out.mkdir()
    reviewer = FakeReviewer("never")

    def build(s: Script, previous=None, llm=None):
        audio = build_audio(s, FakeSynth(), cfg.tts, audio_dir, tmp_path)
        builder = VisualBuilder(
            cfg, load_theme("default"), FakeImageGen(), FakeDepth(), FakeMotion(), reviewer
        )
        return builder.build(
            "job", s, audio, out, llm, lambda: None, lambda m: None, previous=previous
        )

    first = build(script)
    assert reviewer.reviews == len(script.segments)
    edited = script.model_copy(deep=True)
    edited.scenes[0].narration = "Its peak stands 6,412 meters above the sea."
    for sc in first.scenes:  # what Pipeline.edit_segment does to the saved plan
        if sc.segment_id == "s1":
            sc.narration, sc.review = edited.scenes[0].narration, ""
    # An LLM without a ShotList answer: planning new shots would fail the test.
    second = build(edited, previous=first, llm=FakeLLM({}))
    assert reviewer.reviews == len(script.segments) + 1  # only the edited scene was re-checked
    assert [s.prompt for s in second.scenes] == [s.prompt for s in first.scenes]
    assert second.scenes[1].narration == edited.scenes[0].narration
