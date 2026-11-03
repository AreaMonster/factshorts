"""Phase 6: real PDF sources, the terminal review, and resuming interrupted jobs."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import FakeLLM, FakeSynth, FakeWiki, needs_ffmpeg, script_to_llm
from factshorts.config import Config
from factshorts.core.pipeline import STEPS, Pipeline, Services
from factshorts.core.sourcing import ArticleGuess
from factshorts.db import Database
from factshorts.models import Idea, Script, SourceDoc
from factshorts.sources.user_source import from_file

LINES = [
    "Mount Testa is the highest mountain in Fictland, with a summit",
    "elevation of 6,412 m above sea level. It lies on the border",
    "between Northia and Southia.",
    "",
    "The mountain was first climbed on 14 July 1931 by Ada Brenner and",
    "Tomas Kell. Its summit glacier covers about 38 km2 and is shrinking",
    "by roughly 12 m per year. Temperatures at the summit can fall to",
    "minus 45 degrees Celsius in winter, and strong winds are common on",
    "the north face of the mountain during most of the year.",
]


def make_pdf(path: Path, lines: list[str]) -> None:
    """A small but real PDF with a text layer, the way a word processor exports one: every
    line is placed separately, so the extracted text is hard-wrapped."""
    ops = ["BT", "/F1 11 Tf", "14 TL", "50 780 Td"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({escaped}) Tj T*" if line else "T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(out))


def test_pdf_source_reads_wrapped_lines_as_sentences(tmp_path: Path) -> None:
    from factshorts.verify.programmatic import quote_check

    pdf = tmp_path / "testa-notes.pdf"
    make_pdf(pdf, LINES)
    doc = from_file(pdf)
    assert doc.kind == "user" and doc.title == "testa-notes"
    assert "summit elevation of 6,412 m above sea level." in doc.text  # line break joined
    # A sentence that spans two PDF lines can be quoted exactly.
    quote = "The mountain was first climbed on 14 July 1931 by Ada Brenner and Tomas Kell."
    assert quote_check(quote, doc.text).paragraph is not None


@needs_ffmpeg
def test_terminal_review_approves(
    cfg: Config, db: Database, source: SourceDoc, script: Script, tmp_path: Path
) -> None:
    from factshorts.cli import app as cli_app

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

    config = tmp_path / "config.toml"
    config.write_text(f'[general]\ndata_dir = "{cfg.data_dir.as_posix()}"\n', encoding="utf-8")
    result = CliRunner().invoke(cli_app, ["review", "--config", str(config)], input="a\n")
    assert result.exit_code == 0, result.output
    assert "How high is Mount Testa?" in result.output
    assert "quote:pass" in result.output
    assert "approved:" in result.output
    video = db.get_video(job_id)
    assert video["status"] == "approved"
    assert Path(video["folder"]).parent == cfg.output_dir / "approved"
    again = CliRunner().invoke(cli_app, ["review", "--config", str(config)])
    assert "Nothing to review." in again.output


@needs_ffmpeg
def test_job_interrupted_mid_step_resumes_there(
    cfg: Config, db: Database, source: SourceDoc, script: Script
) -> None:
    """The program stops (crash, power cut) during the voice step: on resume, sourcing,
    script and fact-checks are kept and the voice step runs again."""
    cfg.render.width, cfg.render.height = 270, 480
    llm = FakeLLM(
        {
            "ArticleGuess": [ArticleGuess(article_title=source.title, search_query=source.title)],
            "LLMScript": [script_to_llm(script)],
        }
    )

    class Crash(BaseException):  # like KeyboardInterrupt: nothing gets to record a failure
        pass

    class CrashingSynth(FakeSynth):
        def synthesize(self, text):
            raise Crash()

    pipeline = Pipeline(
        cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=CrashingSynth)
    )
    job_id = pipeline.create_job(Idea(text="How high is Mount Testa?", length=30))
    with pytest.raises(Crash):
        pipeline.run(job_id)
    assert db.get_job(job_id)["state"] == "running"  # left behind as if the process died
    assert db.step_status(job_id, "voice")["status"] == "running"
    calls = llm.work_calls

    resumed = Pipeline(cfg, db, Services(llm=llm, wiki=FakeWiki(source), synth_factory=FakeSynth))
    log: list[tuple[str, str]] = []
    assert resumed.run(job_id, progress=lambda s, m: log.append((s, m))) is not None
    started = [s for s, m in log if m == "started"]
    assert started == STEPS[STEPS.index("voice") :]
    assert llm.work_calls == calls


# --- TikTok caption ----------------------------------------------------------------------


def test_caption_may_not_add_facts() -> None:
    from factshorts.core.social import adds_facts

    script = "A cheetah can run 93 to 104 km/h. It lives in Africa and Iran."
    assert adds_facts("Faster than a car: up to 104 km/h!", script) == []
    assert adds_facts("It can hit 120 km/h in Africa.", script) == ["120"]
    assert adds_facts("Even in Namibia it runs fast.", script) == ["Namibia"]
    assert adds_facts("It runs twice as fast as a horse.", script) == ["twice"]
    assert adds_facts("Iran has cheetahs too.", script) == []  # first word is not a name


def test_caption_uses_the_llm_line_or_falls_back_to_the_hook(script: Script) -> None:
    from factshorts.core.social import tiktok_caption

    source = SourceDoc(kind="wikipedia", title="Mount Testa", sections=[], sha256="")
    good = FakeLLM({"CaptionLine": [{"caption": "Climbed first in 1931: how high is it?"}]})
    text = tiktok_caption(script, source, "geography", "VERIFIED", True, good, "m")
    assert text.startswith("Climbed first in 1931: how high is it?")
    assert "Source: Wikipedia (Mount Testa), CC BY-SA" in text
    assert "Visuals are AI-generated." in text
    assert "#MountTesta" in text and "#Geography" in text

    invented = FakeLLM({"CaptionLine": [{"caption": "It is 9,000 m tall!"}]})
    text = tiktok_caption(script, source, None, "VERIFIED", False, invented, "m")
    assert text.startswith(script.hook.narration)  # the checked hook instead
    assert "AI-generated" not in text
    assert tiktok_caption(script, source, None, "UNVERIFIED", False).count("Not fact-checked") == 1


# --- found in the Phase 6 cheetah sample ----------------------------------------------------


def test_units_end_sentences_but_initials_do_not() -> None:
    from factshorts.verify.normalize import split_sentences

    assert split_sentences("Cheetahs can run up to 104 km/h. That's faster than any animal.") == [
        "Cheetahs can run up to 104 km/h.",
        "That's faster than any animal.",
    ]
    assert split_sentences("It rises 6,412 m. Then it drops.") == [
        "It rises 6,412 m.",
        "Then it drops.",
    ]
    assert split_sentences("J. R. R. Tolkien wrote it. He was British.") == [
        "J. R. R. Tolkien wrote it.",
        "He was British.",
    ]


def test_repeats_from_the_cheetah_sample_are_removed() -> None:
    from factshorts.core.script import dedupe_script
    from factshorts.models import Claim, Segment

    def seg(text: str, *claims: str) -> Segment:
        return Segment(
            narration=text,
            on_screen_text="",
            claims=[Claim(text=c, source_quote="q") for c in claims],
        )

    script = Script(
        title="How fast can a cheetah run?",
        hook=seg(
            "Cheetahs can run up to 104 km/h. That's faster than any other land animal.",
            "Cheetahs can run up to 104 km/h.",
        ),
        scenes=[
            seg(
                "They can run between 93 and 104 km/h. That's faster than any other land animal.",
                "They can run between 93 and 104 km/h.",
            ),
            seg(
                "Cubs are vulnerable to lions and hyenas, who also routinely steal kills from "
                "adult cheetahs. They also routinely steal kills from adult cheetahs.",
                "Cubs are vulnerable to lions and hyenas, who also routinely steal kills from "
                "adult cheetahs.",
            ),
        ],
        outro=seg("Thanks for watching!"),
    ).assign_ids()
    out, removed = dedupe_script(script, 1, subject_words={"cheetah"})
    assert removed == 2
    assert out.hook.claims  # the hook keeps its claim (its sentences split correctly)
    assert out.scenes[0].narration == "They can run between 93 and 104 km/h."
    assert out.scenes[1].narration.endswith("from adult cheetahs.")
    assert "They also" not in out.scenes[1].narration
    # A negation is not a repeat.
    assert not dedupe_script(
        script.model_copy(
            update={"scenes": [seg("They never steal kills from adult cheetahs or cubs.", "x")]}
        ),
        1,
        subject_words={"cheetah"},
    )[1]


def test_copied_source_labels_are_removed_from_quotes() -> None:
    from factshorts.core.script import LLMSegment, LLMSentence, to_segment

    s = to_segment(
        LLMSegment(
            sentences=[
                LLMSentence(
                    text="Cheetahs are fast.",
                    source_quotes=["[2] The cheetah is the fastest land animal."],
                    claim_type="general",
                )
            ],
            on_screen_text="",
            visual="image",
        )
    )
    assert s.claims[0].source_quote == "The cheetah is the fastest land animal."


@needs_ffmpeg
def test_redo_from_a_step_keeps_the_earlier_ones(
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
    pipeline.redo_from(job_id, "verification")
    log: list[tuple[str, str]] = []
    assert pipeline.run(job_id, progress=lambda s, m: log.append((s, m))) is not None
    assert [s for s, m in log if m == "started"] == STEPS[STEPS.index("verification") :]
    assert sum(1 for name, _ in llm.calls if name == "LLMScript") == 1  # no new draft


# --- found in the Phase 6 bones sample ---------------------------------------------------


def test_removing_a_repeat_does_not_turn_a_word_into_a_name(source: SourceDoc) -> None:
    """'Tourists' opening a sentence is an ordinary word because 'tourists' is used elsewhere;
    deleting that other sentence (a repeat) must not make it look like an invented name."""
    from factshorts.config import VerificationConfig
    from factshorts.models import Claim, Segment
    from factshorts.verify.aggregate import Verifier

    quote = "Around 35,000 people visit the Testa National Park each year."

    def seg(text: str) -> Segment:
        return Segment(
            narration=text, on_screen_text="", claims=[Claim(text=text, source_quote=quote)]
        )

    hook = seg("Tourists visit the park every year.")
    extra = seg("About 35,000 tourists visit the park every year.")
    outro = Segment(narration="", on_screen_text="")
    full = Script(title="Mount Testa", hook=hook, scenes=[extra], outro=outro).assign_ids()
    trimmed = Script(title="Mount Testa", hook=hook, scenes=[], outro=outro).assign_ids()
    cfg = VerificationConfig(nli_check=False, llm_judge=False, wikidata_check=False)

    fresh = Verifier(cfg, source).verify(trimmed.model_copy(deep=True))
    assert any("Tourists" in f for f in fresh.failures)  # alone, it looks like a name

    verifier = Verifier(cfg, source)
    assert verifier.verify(full.model_copy(deep=True)).passed
    assert verifier.verify(trimmed.model_copy(deep=True)).passed  # after removing a sentence


def test_last_scene_repeating_the_outro_is_removed() -> None:
    from factshorts.core.script import dedupe_script
    from factshorts.models import Claim, Segment

    def seg(text: str, fact: bool = True) -> Segment:
        claims = [Claim(text=text, source_quote="q")] if fact else []
        return Segment(narration=text, on_screen_text="", claims=claims)

    script = Script(
        title="Bones",
        hook=seg("Babies have about 300 bones."),
        scenes=[
            seg("Adults have 206 bones."),
            seg("Many bones fuse as we grow."),
            seg("Thanks for watching!", fact=False),
        ],
        outro=seg("Thanks for watching!", fact=False),
    ).assign_ids()
    out, removed = dedupe_script(script, 2)
    assert removed == 1
    assert [s.narration for s in out.segments].count("Thanks for watching!") == 1
