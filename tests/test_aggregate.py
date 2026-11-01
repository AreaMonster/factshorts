from factshorts.config import VerificationConfig
from factshorts.models import Claim, Script, SourceDoc
from factshorts.verify.aggregate import verify_script


def test_fixture_script_passes(script: Script, source: SourceDoc) -> None:
    v = verify_script(script, source, VerificationConfig())
    assert v.passed, v.failures
    assert len(v.claims) == 5
    assert all(c.paragraph for c in v.claims)


def test_later_checks_are_reported_as_skipped_not_passed(script: Script, source: SourceDoc) -> None:
    v = verify_script(script, source, VerificationConfig())
    statuses = {c.check: c.status for c in v.claims[0].checks}
    assert statuses == {
        "quote": "pass",
        "entities": "pass",
        "wikidata": "skipped",
        "nli": "skipped",
        "llm_judge": "skipped",
    }
    assert v.label == "PARTIALLY_VERIFIED"
    assert set(v.skipped_checks) == {"wikidata", "nli", "llm_judge"}


def test_label_is_verified_when_every_enabled_check_ran(script: Script, source: SourceDoc) -> None:
    cfg = VerificationConfig(wikidata_check=False, nli_check=False, llm_judge=False)
    v = verify_script(script, source, cfg)
    assert v.passed and v.label == "VERIFIED"


def test_one_failing_check_fails_the_claim(script: Script, source: SourceDoc) -> None:
    # Quote is real, but the claim's number is not in it: quote passes, entities fail.
    script.scenes[0].claims[0].text = "Its summit rises 6,999 meters above sea level."
    script.scenes[0].narration = "Its summit rises 6,999 meters above sea level."
    v = verify_script(script, source, VerificationConfig())
    verdict = next(c for c in v.claims if c.claim_id == "s1.c1")
    checks = {c.check: c.status for c in verdict.checks}
    assert checks["quote"] == "pass" and checks["entities"] == "fail"
    assert not verdict.passed
    assert not v.passed


def test_failed_quote_fails_claim_and_its_sentence(script: Script, source: SourceDoc) -> None:
    script.scenes[1].claims[0].source_quote = "The mountain was first climbed in 1931 by a team."
    v = verify_script(script, source, VerificationConfig())
    assert not v.passed
    assert any("quote not found" in f for f in v.failures)
    sentence = next(s for s in v.sentences if s.segment_id == "s2" and s.kind == "claim")
    assert not sentence.passed


def test_filler_sentence_without_entities_passes(script: Script, source: SourceDoc) -> None:
    v = verify_script(script, source, VerificationConfig())
    filler = [s for s in v.sentences if s.kind == "filler" and s.sentence == "Here's the twist."]
    assert filler and filler[0].passed


def test_on_screen_text_numbers_are_checked(script: Script, source: SourceDoc) -> None:
    script.scenes[0].on_screen_text = "7,412 meters"
    v = verify_script(script, source, VerificationConfig())
    assert not v.passed
    assert any("on-screen text" in f and "7,412" in f for f in v.failures)


def test_claim_text_linked_to_sentence_with_punctuation_differences(
    script: Script, source: SourceDoc
) -> None:
    script.scenes[1].claims = [
        Claim(
            id="s2.c1",
            text="Ada Brenner and Tomas Kell first climbed it in 1931",
            source_quote=script.scenes[1].claims[0].source_quote,
            type="date",
        )
    ]
    v = verify_script(script, source, VerificationConfig())
    assert v.passed, v.failures
