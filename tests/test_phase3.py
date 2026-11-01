"""Phase 3: custom ideas, queue + dedupe, auto ideas, user sources, comparisons, false premises,
fact-check off, batch runs and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import FakeLLM, FakeNLI, FakeSynth, FakeWiki, FakeWikidata, script_to_llm
from factshorts.config import Config
from factshorts.core.ideas import (
    DuplicateIdea,
    IdeaStore,
    ProposedIdea,
    ProposedIdeas,
    generate_ideas,
    parse_idea_line,
    parse_ideas_text,
)
from factshorts.core.pipeline import Pipeline, Services
from factshorts.core.sourcing import ArticleGuess, merge_sources
from factshorts.db import Database
from factshorts.models import Idea, Script, SourceDoc
from factshorts.sources import user_source
from factshorts.sources.wikidata import WDValue
from factshorts.sources.wikipedia import ArticleNotFound, SearchHit, select_sections
from factshorts.verify.aggregate import Verifier
from factshorts.verify.llm_judge import Judge
from factshorts.verify.normalize import normalize_title

DEFAULTS = {"length": 75, "tone": "neutral"}

# --- parsing ----------------------------------------------------------------------------------


def test_parse_idea_line_with_settings() -> None:
    idea = parse_idea_line("Why is Venus hotter than Mercury? | length=45 | tone=funny", DEFAULTS)
    assert (idea.text, idea.length, idea.tone) == ("Why is Venus hotter than Mercury?", 45, "funny")
    assert parse_idea_line("Cats? | factcheck=no", DEFAULTS).factcheck is False


@pytest.mark.parametrize("line", ["Cats? | length", "Cats? | colour=red", "Cats? | length=5"])
def test_parse_idea_line_rejects_bad_settings(line: str) -> None:
    with pytest.raises(ValueError):
        parse_idea_line(line, DEFAULTS)


def test_parse_ideas_text_skips_comments_and_reports_line() -> None:
    text = "# my ideas\nWhy is the sky blue?\n\nHow deep is the ocean? | length=60\n"
    assert [i.text for i in parse_ideas_text(text, DEFAULTS)] == [
        "Why is the sky blue?",
        "How deep is the ocean?",
    ]
    with pytest.raises(ValueError, match="line 2"):
        parse_ideas_text("ok?\nbad? | tone=angry", DEFAULTS)


def test_normalize_title() -> None:
    assert normalize_title("Why is Venus SO hot?!") == "venus hot"


# --- queue + dedupe ---------------------------------------------------------------------------


def test_duplicates_are_detected(db: Database) -> None:
    store = IdeaStore(db)
    store.add(Idea(text="Why is Venus so hot?"))
    with pytest.raises(DuplicateIdea, match="same idea"):
        store.add(Idea(text="why is venus so HOT"))
    store.add(
        Idea(text="How old are the oldest trees?", origin="auto"),
        article="Bristlecone pine",
        angle="age of the oldest living trees",
    )
    with pytest.raises(DuplicateIdea, match="same article"):
        store.add(
            Idea(text="Which tree is the oldest?"),
            article="Bristlecone pine",
            angle="oldest living trees age",
        )
    # same article, different angle is fine
    store.add(
        Idea(text="Where do bristlecone pines grow?"),
        article="Bristlecone pine",
        angle="habitat in the mountains",
    )
    # force overrides
    store.add(Idea(text="Why is Venus so hot?", force=True))


def test_inbox_lists_custom_before_auto_and_clear(db: Database) -> None:
    store = IdeaStore(db)
    store.add(Idea(text="Auto idea about whales", origin="auto"))
    store.add(Idea(text="My own idea about volcanoes"))
    assert [s.idea.text for s in store.queued()] == [
        "My own idea about volcanoes",
        "Auto idea about whales",
    ]
    assert store.clear() == 2 and store.queued() == []


def test_old_jobs_are_backfilled_for_dedupe(db: Database) -> None:
    db.create_job("old-job", Idea(text="How tall is Mount Everest?").model_dump())
    db.update_job("old-job", state="done")
    store = IdeaStore(db)
    assert store.find_duplicate(Idea(text="How tall is Mount Everest really?")) is not None


# --- auto ideas -------------------------------------------------------------------------------


class SearchWiki(FakeWiki):
    """Search hits only for known article titles."""

    def __init__(self, doc: SourceDoc, known: list[str]) -> None:
        super().__init__(doc)
        self.known = known

    def search(self, query: str, limit: int | None = None) -> list[SearchHit]:
        return [
            SearchHit(title=t, snippet="", page_id=i)
            for i, t in enumerate(self.known)
            if t.lower() in query.lower() or query.lower() in t.lower()
        ]


def test_generate_ideas_resolves_articles_and_drops_duplicates(
    db: Database, source: SourceDoc
) -> None:
    store = IdeaStore(db)
    store.add(Idea(text="How fast can a cheetah run?"))
    llm = FakeLLM(
        {
            "ProposedIdeas": [
                ProposedIdeas(
                    ideas=[
                        ProposedIdea(
                            question="How fast can a cheetah run?",
                            article_title="Cheetah",
                            angle="speed",
                        ),
                        ProposedIdea(
                            question="Why do octopuses have three hearts?",
                            article_title="Octopus",
                            angle="circulation",
                        ),
                        ProposedIdea(
                            question="Do unicorns sleep standing up?",
                            article_title="Unicorn sleep",
                            angle="sleep",
                        ),
                        ProposedIdea(
                            question="How long do octopuses live?",
                            article_title="Octopus",
                            angle="lifespan",
                        ),
                        ProposedIdea(
                            question="How do bats see in the dark?",
                            article_title="Bat",
                            angle="echolocation",
                        ),
                    ]
                )
            ]
        }
    )
    wiki = SearchWiki(source, ["Cheetah", "Octopus", "Bat"])
    found = generate_ideas("animals", 2, llm, "m", wiki, store, DEFAULTS)
    assert [(f.idea.text, f.article) for f in found] == [
        ("Why do octopuses have three hearts?", "Octopus"),
        ("How do bats see in the dark?", "Bat"),
    ]
    assert all(f.idea.origin == "auto" and f.idea.topic == "animals" for f in found)


# --- user sources -----------------------------------------------------------------------------

LONG = (
    " ".join(["Lake Testa is a deep alpine lake in Fictland."] * 4)
    + "\n\n"
    + " ".join(["Its water stays at 4 °C all year and it is 310 m deep."] * 4)
)


def test_user_text_source(tmp_path: Path) -> None:
    doc = user_source.from_text(LONG, title="Notes")
    assert doc.kind == "user" and len(doc.text.split("\n")) == 2
    with pytest.raises(user_source.UserSourceError, match="at least"):
        user_source.from_text("Too short.")


def test_user_markdown_file_is_cleaned(tmp_path: Path) -> None:
    md = tmp_path / "lake.md"
    md.write_text(
        "# Lake Testa\n\n" + LONG.replace("deep alpine", "**deep** [alpine](x)"), encoding="utf-8"
    )
    doc = user_source.from_file(md)
    assert "**" not in doc.text and "](" not in doc.text and doc.title == "lake"


def test_user_url_source_extracts_main_text() -> None:
    paras = "".join(f"<p>{p}</p>" for p in LONG.split("\n\n"))
    html = (
        f"<html><head><title>Lake Testa facts</title></head><body><nav>Home | About</nav>"
        f"<article><h1>Lake Testa</h1>{paras}</article><footer>Cookie banner</footer>"
        "</body></html>"
    )

    class Http:
        def get_text(self, url: str, params: dict | None = None, cache: bool = True) -> str:
            return html

    doc = user_source.from_url("https://example.org/lake", Http())
    assert doc.url == "https://example.org/lake" and "310 m deep" in doc.text
    assert "Cookie banner" not in doc.text


# --- comparisons ------------------------------------------------------------------------------


def _doc(title: str, qid: str, texts: list[str]) -> SourceDoc:
    from factshorts.models import Section

    return SourceDoc(
        kind="wikipedia",
        title=title,
        url=f"https://x/{title}",
        revision_id=1,
        wikidata_id=qid,
        sha256=title,
        sections=[Section(index=i, title=f"S{i}", text=t) for i, t in enumerate(texts)],
    )


def test_merge_sources_interleaves_and_keeps_refs() -> None:
    merged = merge_sources(_doc("Venus", "Q313", ["v0", "v1"]), _doc("Mercury", "Q308", ["m0"]))
    assert [s.text for s in merged.sections] == ["v0", "m0", "v1"]
    assert [r.title for r in merged.refs] == ["Venus", "Mercury"]
    assert merged.title == "Venus and Mercury"


def test_wikidata_check_uses_the_document_the_claim_is_about() -> None:
    from factshorts.models import Claim, Segment

    venus = _doc("Venus", "Q313", ["Venus has a mean radius of 6,052 km and it is very hot."])
    mercury = _doc("Mercury", "Q308", ["Mercury has a mean radius of 2,440 km and no air."])
    merged = merge_sources(venus, mercury)
    values = {
        "Venus": {"P2120": [WDValue("P2120", "quantity", 6051.8, "km", "normal", "6051.8 km")]},
        "Mercury": {"P2120": [WDValue("P2120", "quantity", 2439.7, "km", "normal", "2439.7 km")]},
    }
    text = "Mercury has a radius of 2,440 km."
    script = Script(
        title="Venus and Mercury",
        hook=Segment(
            narration=text,
            on_screen_text="",
            claims=[
                Claim(text=text, source_quote="Mercury has a mean radius of 2,440 km and no air.")
            ],
        ),
        scenes=[],
        outro=Segment(narration="Bye now.", on_screen_text=""),
    ).assign_ids()
    v = Verifier(
        Config().verification,
        merged,
        nli=FakeNLI(),
        judge=Judge(FakeLLM({}), "j"),
        wikidata_values=values,
    ).verify(script)
    wd = next(c for c in v.claims[0].checks if c.check == "wikidata")
    assert wd.status == "pass" and "2439.7" in wd.reason


def test_focus_sections_add_relevant_later_sections() -> None:
    parts = [
        (1, "Lead", "lead"),
        (2, "Names", "n"),
        (2, "History", "h"),
        (3, "Early", "e"),
        (2, "Course", "c"),
        (2, "Condition", "d"),
        (2, "Visibility from space", "can it be seen from space"),
        (2, "References", "r"),
    ]
    titles = [s.title for s in select_sections(parts, 2, ["References"], "visible from space")]
    assert titles == ["Lead", "Names", "History", "Early", "Visibility from space"]


# --- pipeline: false premise and fact-check off -------------------------------------


def _pipeline(
    cfg: Config,
    db: Database,
    source: SourceDoc,
    script: Script,
    extra: dict | None = None,
    wiki: FakeWiki | None = None,
) -> tuple[Pipeline, FakeLLM]:
    responses = {
        "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
        "LLMScript": [script_to_llm(script)],
        **(extra or {}),
    }
    llm = FakeLLM(responses)
    services = Services(
        llm=llm,
        wiki=wiki or FakeWiki(source),
        synth_factory=FakeSynth,
        nli_factory=FakeNLI,
        wikidata=FakeWikidata(),
    )
    return Pipeline(cfg, db, services), llm


FALSE_PREMISE = {
    "Presuppositions": [{"statements": ["Mount Testa is the tallest mountain on Earth."]}],
    "PremiseJudgement": [
        {
            "reason": "it is the highest in Fictland only",
            "verdict": "CONTRADICTED",
            "source_sentence": "Mount Testa is the highest mountain in Fictland, "
            "with a summit elevation of 6,412 m (21,037 ft) "
            "above sea level.",
        }
    ],
}


def test_false_premise_waits_then_makes_a_corrected_video(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    pipeline, llm = _pipeline(cfg, db, source, script, FALSE_PREMISE)
    job_id = pipeline.create_job(
        Idea(text="Why is Mount Testa the tallest mountain on Earth?", length=30)
    )
    log: list[tuple[str, str]] = []
    assert pipeline.run(job_id, progress=lambda s, m: log.append((s, m))) is None
    job = db.get_job(job_id)
    assert job["state"] == "needs_input"
    assert "Idea assumes" in job["error"] and "highest mountain in Fictland" in job["error"]
    assert IdeaStore(db).list("needs_input")

    pipeline.answer(job_id, "corrected")
    assert pipeline.run(job_id) is not None
    script_calls = [m for name, m in llm.calls if name == "LLMScript"]
    assert "wrongly assumes" in script_calls[-1][1]["content"]
    assert IdeaStore(db).list("done")


def test_false_premise_cancel(cfg: Config, db: Database, source: SourceDoc, script: Script) -> None:
    pipeline, _ = _pipeline(cfg, db, source, script, FALSE_PREMISE)
    job_id = pipeline.create_job(Idea(text="Why is Mount Testa the tallest mountain on Earth?"))
    pipeline.run(job_id)
    pipeline.answer(job_id, "cancel")
    assert db.get_job(job_id)["state"] == "cancelled"
    rows = db.query("SELECT reason FROM rejections WHERE job_id = ?", (job_id,))
    assert rows and rows[0]["reason"].startswith("false premise")
    with pytest.raises(ValueError):
        pipeline.answer(job_id, "corrected")


class NoArticleWiki(FakeWiki):
    def search(self, query: str, limit: int | None = None) -> list:
        return []

    def fetch(self, title: str, focus: str | None = None) -> SourceDoc:
        raise ArticleNotFound(title)


def test_no_factcheck_runs_without_source_and_is_labeled(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    pipeline, llm = _pipeline(cfg, db, source, script, wiki=NoArticleWiki(source))
    job_id = pipeline.create_job(
        Idea(text="Why cats are the best pets", length=30, factcheck=False)
    )
    result = pipeline.run(job_id)
    assert result is not None and not result.verified
    assert "_UNVERIFIED" in Path(result.video).name
    folder = Path(result.folder)
    assert "UNVERIFIED" in (folder / "description.txt").read_text("utf-8")
    assert json.loads((folder / "sources.json").read_text("utf-8"))["label"] == "UNVERIFIED"
    system = next(m for name, m in llm.calls if name == "LLMScript")[0]["content"]
    assert "NOT\nfact-checked" in system or "NOT fact-checked" in system
    assert not any(name == "ClaimJudgement" for name, _ in llm.calls)


def test_fact_checked_idea_without_source_is_rejected(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    pipeline, _ = _pipeline(cfg, db, source, script, wiki=NoArticleWiki(source))
    job_id = pipeline.create_job(Idea(text="Why cats are the best pets"))
    assert pipeline.run(job_id) is None
    assert "provide a source" in db.get_job(job_id)["error"]


def test_user_text_source_in_pipeline(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    pipeline, _ = _pipeline(cfg, db, source, script)
    job_id = pipeline.create_job(
        Idea(text="How high is Mount Testa?", length=30, source_text=source.text)
    )
    result = pipeline.run(job_id)
    assert result is not None
    sources = json.loads((Path(result.folder) / "sources.json").read_text("utf-8"))
    assert sources["source"]["user_source"] is True


# --- batch + CLI ------------------------------------------------------------------------------


def test_batch_runs_inbox_first_then_auto_ideas(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    from factshorts.core.batch import run_batch

    wiki = SearchWiki(source, ["Mount Testa"])
    pipeline, _ = _pipeline(
        cfg,
        db,
        source,
        script,
        {
            "ProposedIdeas": [
                ProposedIdeas(
                    ideas=[
                        ProposedIdea(
                            question="When was Mount Testa first climbed?",
                            article_title="Mount Testa",
                            angle="first ascent",
                        )
                    ]
                )
            ]
        },
        wiki=wiki,
    )
    IdeaStore(db).add(Idea(text="How high is Mount Testa?", length=30))
    order: list[str] = []
    report = run_batch(
        pipeline, per_topic=1, topics=["mountains"], progress=lambda m: order.append(m)
    )
    assert len(report.done) == 2 and not report.rejected
    jobs = [m for m in order if m.startswith("job ")]
    assert "How high is Mount Testa?" in jobs[0] and "first climbed" in jobs[1]


def test_cli_queue_add_list_clear(tmp_path: Path) -> None:
    from factshorts.cli import app

    conf = tmp_path / "config.toml"
    conf.write_text(f'[general]\ndata_dir = "{(tmp_path / "data").as_posix()}"\n', encoding="utf-8")
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "queue",
            "add",
            "Why is the sea salty? | length=45",
            "Why is the sea salty",
            "--config",
            str(conf),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "queued #1" in r.output and "skipped" in r.output
    r = runner.invoke(app, ["queue", "list", "--config", str(conf)])
    assert "Why is the sea salty?" in r.output
    r = runner.invoke(app, ["queue", "clear", "--config", str(conf)])
    assert "Cleared 1" in r.output
