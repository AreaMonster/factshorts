"""Phase 2: Wikidata cross-check, NLI, LLM judge, check order, caching and labels."""

from __future__ import annotations

import os

import pytest

from conftest import FakeLLM, FakeNLI
from factshorts.config import VerificationConfig
from factshorts.models import Script, SourceDoc
from factshorts.sources.wikidata import WDValue, Wikidata
from factshorts.verify.aggregate import Verifier
from factshorts.verify.llm_judge import Judge
from factshorts.verify.nli import nli_check, premises
from factshorts.verify.wikidata_check import is_about_subject, wikidata_check

CFG = VerificationConfig()
PROPS = CFG.wikidata_properties
ELEVATION = {
    "P2044": [
        WDValue("P2044", "quantity", 6412.0, "m", "preferred", "6412 m"),
        WDValue("P2044", "quantity", 6400.0, "m", "normal", "6400 m"),
    ]
}


# --- Wikidata ---------------------------------------------------------------------------


class _Http:
    def __init__(self, data: dict) -> None:
        self.data = data

    def get_json(self, url: str, params: dict | None = None, cache: bool = True) -> dict:
        return self.data


def test_wikidata_statements_parse_units_ranks_and_dates() -> None:
    claims = {
        "P2044": [
            {
                "rank": "preferred",
                "mainsnak": {
                    "datavalue": {
                        "type": "quantity",
                        "value": {
                            "amount": "+8848.86",
                            "unit": "http://www.wikidata.org/entity/Q11573",
                        },
                    }
                },
            },
            {
                "rank": "deprecated",
                "mainsnak": {
                    "datavalue": {
                        "type": "quantity",
                        "value": {
                            "amount": "+8800",
                            "unit": "http://www.wikidata.org/entity/Q11573",
                        },
                    }
                },
            },
        ],
        "P571": [
            {
                "rank": "normal",
                "mainsnak": {
                    "datavalue": {
                        "type": "time",
                        "value": {"time": "+1931-07-14T00:00:00Z", "precision": 11},
                    }
                },
            }
        ],
        "P1082": [
            {
                "rank": "normal",
                "mainsnak": {
                    "datavalue": {"type": "quantity", "value": {"amount": "+35000", "unit": "1"}}
                },
            }
        ],
    }
    wd = Wikidata(_Http({"entities": {"Q1": {"claims": claims}}}))
    values = wd.statements("Q1", ["P2044", "P571", "P1082", "P2046"])
    assert [(v.value, v.unit) for v in values["P2044"]] == [(8848.86, "m")]  # no deprecated
    assert values["P571"][0].value == 1931 and values["P571"][0].kind == "time"
    assert values["P1082"][0].unit is None
    assert "P2046" not in values


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("Mount Testa is 6,412 meters high.", "pass"),
        ("Its summit is about 6.4 km above sea level.", "pass"),
        ("Testa's summit is roughly 6,400 meters tall.", "pass"),
        ("Mount Testa is 6,912 meters high.", "fail"),
        ("Its elevation is 7 km.", "fail"),
        ("Climbers start from base camps 3,100 meters high.", "na"),  # not about the subject
        ("Mount Testa gets 1,200 mm of rain.", "na"),  # no matching property keyword
    ],
)
def test_wikidata_check(text: str, status: str) -> None:
    res = wikidata_check(text, "Mount Testa", ELEVATION, PROPS, CFG)
    assert res.status == status, res.reason


def test_wikidata_check_years() -> None:
    values = {"P571": [WDValue("P571", "time", 1987.0, None, "normal", "1987-01-01")]}
    assert (
        wikidata_check("It was built in 1987.", "Testa Cable Car", values, PROPS, CFG).status
        == "pass"
    )
    assert (
        wikidata_check("It was built in 1978.", "Testa Cable Car", values, PROPS, CFG).status
        == "fail"
    )


def test_subject_detection_ignores_generic_title_words() -> None:
    assert is_about_subject("Its summit is high.", "Mount Testa")
    assert is_about_subject("Testa is high.", "Mount Testa")
    assert not is_about_subject("Every mount is high.", "Mount Testa")


# --- NLI ----------------------------------------------------------------------------------


def test_premises_resolve_pronouns_and_add_context() -> None:
    paragraph = (
        "Mount Testa is the highest mountain in Fictland. It lies on a border. "
        "Its glacier is shrinking. The town is far away."
    )
    out = premises("Its glacier is shrinking.", paragraph, "Mount Testa")
    assert out[0] == "Mount Testa's glacier is shrinking."
    assert any("It lies on a border" in p for p in out[1:])


def test_nli_check_threshold_and_early_stop() -> None:
    nli = FakeNLI()
    res = nli_check(
        nli, "It is high.", "Mount Testa is very high indeed.", "Para text.", "Mount Testa", 0.5
    )
    assert res.status == "pass" and nli.calls == 1  # the quote alone was enough
    bad = FakeNLI(fail_on=("tallest",))
    res = nli_check(
        bad, "It is the tallest from base.", "Mount Testa is the highest.", None, "Mount Testa", 0.5
    )
    assert res.status == "fail" and "contradicted" in res.reason


# --- judge ---------------------------------------------------------------------------------


def test_judge_maps_verdicts() -> None:
    for verdict, status in [("SUPPORTED", "pass"), ("NOT_SUPPORTED", "fail"), ("UNCLEAR", "fail")]:
        llm = FakeLLM({"ClaimJudgement": [{"reason": "r", "verdict": verdict}]})
        assert Judge(llm, "j").judge_claim("c", "q", None, "T").status == status
    llm = FakeLLM({"FillerJudgement": [{"reason": "a fact", "states_a_fact": True}]})
    assert Judge(llm, "j").classify_filler("Everest keeps growing.", "T").status == "fail"


def test_judge_uses_its_own_model() -> None:
    llm = FakeLLM({})
    calls: list[str | None] = []
    original = llm.structured

    def spy(messages, schema, model=None, retries=3, temperature=None):
        calls.append(model)
        return original(messages, schema, model, retries, temperature)

    llm.structured = spy
    Judge(llm, "gemma4:26b").judge_claim("c", "q", None, "T")
    assert calls == ["gemma4:26b"]


# --- aggregation -----------------------------------------------------------------------


def _verifier(
    source: SourceDoc,
    llm: FakeLLM | None = None,
    nli: FakeNLI | None = None,
    values: dict | None = None,
) -> Verifier:
    llm = llm or FakeLLM({})
    return Verifier(
        CFG,
        source,
        nli=nli or FakeNLI(),
        judge=Judge(llm, "judge"),
        wikidata_values=values if values is not None else {},
    )


def test_all_checks_run_and_label_is_verified(script: Script, source: SourceDoc) -> None:
    v = _verifier(source).verify(script)
    assert v.passed, v.failures
    assert v.label == "VERIFIED"
    checks = [c.check for c in v.claims[1].checks]
    assert checks == ["quote", "entities", "wikidata", "nli", "llm_judge"]


def test_missing_model_makes_label_partial(script: Script, source: SourceDoc) -> None:
    v = Verifier(CFG, source, nli=None, judge=Judge(FakeLLM({}), "j"), wikidata_values={}).verify(
        script
    )
    assert v.passed and v.label == "PARTIALLY_VERIFIED" and v.skipped_checks == ["nli"]


def test_expensive_checks_do_not_run_after_a_cheap_failure(
    script: Script, source: SourceDoc
) -> None:
    script.scenes[0].narration = "Its summit rises 6,999 meters above sea level."
    script.scenes[0].claims[0].text = script.scenes[0].narration
    nli = FakeNLI()
    v = _verifier(source, nli=nli).verify(script)
    verdict = next(c for c in v.claims if c.claim_id == "s1.c1")
    status = {c.check: c.status for c in verdict.checks}
    assert status["entities"] == "fail" and status["nli"] == "na" and status["llm_judge"] == "na"
    assert not v.passed


def test_results_are_cached_across_reverification(script: Script, source: SourceDoc) -> None:
    nli = FakeNLI()
    verifier = _verifier(source, nli=nli)
    verifier.verify(script)
    calls = nli.calls
    verifier.verify(script)
    assert nli.calls == calls


def test_judge_catches_what_entities_and_nli_miss(script: Script, source: SourceDoc) -> None:
    """'Highest in Fictland' → 'tallest from base to summit': same names, no numbers."""
    script.hook.narration = "Mount Testa is the tallest mountain in Fictland from base to summit."
    script.hook.claims[0].text = script.hook.narration
    llm = FakeLLM(
        {
            "ClaimJudgement": [
                {"reason": "the excerpt only says highest", "verdict": "NOT_SUPPORTED"}
            ]
        }
    )
    v = _verifier(source, llm=llm).verify(script)  # FakeNLI passes it, like the real one did
    assert not v.passed
    assert any("llm_judge" in f and "highest" in f for f in v.failures)


def test_filler_sentence_with_a_fact_is_caught(script: Script, source: SourceDoc) -> None:
    script.scenes[2].narration = "Winter nights can hit minus 45 °C. It is a dangerous place."
    llm = FakeLLM({"FillerJudgement": [{"reason": "claims danger", "states_a_fact": True}]})
    v = _verifier(source, llm=llm).verify(script)
    assert not v.passed
    assert any("states a fact without a source" in f for f in v.failures)


def test_wikidata_disagreement_fails_claim(script: Script, source: SourceDoc) -> None:
    """The article text and Wikidata disagree: the claim is dropped, as the spec requires."""
    values = {"P2044": [WDValue("P2044", "quantity", 6300.0, "m", "preferred", "6300 m")]}
    script.scenes[0].narration = "Mount Testa rises 6,412 meters high."
    script.scenes[0].claims[0].text = script.scenes[0].narration
    v = _verifier(source, values=values).verify(script)
    assert not v.passed
    assert any("disagrees with Wikidata P2044" in f for f in v.failures)


# --- real models (opt-in: needs Ollama + gemma4:26b and the NLI model cached) ----------------

REAL = pytest.mark.skipif(
    os.environ.get("FACTSHORTS_MODEL_TESTS") != "1",
    reason="set FACTSHORTS_MODEL_TESTS=1 to run against real models",
)

# Quotes from Wikipedia "Mount Everest" (CC BY-SA 4.0).
REAL_CASES = [
    (
        True,
        "Over 200 bodies remain on the mountain.",
        "Over 200 bodies remain on the mountain and have not been removed due to the dangerous "
        "conditions.",
    ),
    (
        True,
        "It lies on the China-Nepal border.",
        "It lies in the Mahalangur Himal sub-range of the Himalayas and marks part of the "
        "China–Nepal border at its summit.",
    ),
    (
        False,
        "Mount Everest is the tallest mountain from base to summit.",
        "Mount Everest is the highest mountain on Earth above sea level.",
    ),
    (
        False,
        "Over 300 bodies remain on the mountain.",
        "Over 200 bodies remain on the mountain and have not been removed due to the dangerous "
        "conditions.",
    ),
    (
        False,
        "Most climbers reach the summit without bottled oxygen.",
        "Climbers usually begin their ascent from base camps above 5,000 m (16,404 ft).",
    ),
]


@REAL
@pytest.mark.parametrize(("ok", "claim", "quote"), REAL_CASES)
def test_real_judge_and_nli(ok: bool, claim: str, quote: str) -> None:
    from factshorts.config import load_config
    from factshorts.core.llm import OllamaLLM
    from factshorts.verify.nli import TransformersNLI

    cfg = load_config()
    cfg.ensure_dirs()
    judge = Judge(OllamaLLM(cfg.ollama), cfg.ollama.judge_model)
    assert (judge.judge_claim(claim, quote, None, "Mount Everest").status == "pass") == ok
    if not ok and "tallest" not in claim:  # NLI misses superlative swaps; the judge catches them
        nli = TransformersNLI(cfg.verification.nli_model, cfg.verification.nli_device)
        assert nli_check(nli, claim, quote, None, "Mount Everest", 0.5).status == "fail"


# --- fixes found in the Phase 2 sample run ----------------------------------------------


def test_formula_on_screen_matches_spelled_out_source() -> None:
    from factshorts.verify.entities import name_in_evidence

    assert name_in_evidence("CO2", "an atmosphere of 96.5% carbon dioxide")
    assert not name_in_evidence("CO2", "an atmosphere of nitrogen")


def test_packed_sentences_keep_quotes_on_the_factual_part() -> None:
    from factshorts.core.script import LLMSegment, LLMSentence, to_segment

    seg = to_segment(
        LLMSegment(
            sentences=[
                LLMSentence(
                    text="Venus is hotter than Mercury. Here's why.",
                    source_quotes=["Q"],
                    claim_type="general",
                )
            ],
            on_screen_text="x",
            visual="title",
        )
    )
    assert seg.narration == "Venus is hotter than Mercury. Here's why."
    assert [c.text for c in seg.claims] == ["Venus is hotter than Mercury."]


def test_topup_skips_scenes_that_repeat_existing_facts(script: Script, source: SourceDoc) -> None:
    from conftest import script_to_llm
    from factshorts.config import Config
    from factshorts.core.script import LLMMoreScenes, ScriptWriter
    from factshorts.models import Idea

    cfg = Config()
    cfg.script.max_topups = 1
    repeat = script_to_llm(script).scenes[0]  # same sentences as s1
    llm = FakeLLM({"LLMMoreScenes": [LLMMoreScenes(scenes=[repeat])]})
    writer = ScriptWriter(Idea(text="Testa?", length=75), source, llm, "m", cfg.script, cfg.tts)
    before = len(script.scenes)
    assert len(writer.top_up(script).scenes) == before


def test_orbit_distance_is_not_compared_with_distance_from_earth() -> None:
    values = {"P2583": [WDValue("P2583", "quantity", 4.0e7, "km", "normal", "4e7 km")]}
    res = wikidata_check(
        "Venus orbits the Sun at a distance of 108 million km.",
        "Venus",
        values,
        CFG.wikidata_properties,
        CFG,
    )
    assert res.status == "na"
