"""Adversarial scripts: each one sneaks in a falsehood. All must be caught."""

from __future__ import annotations

import pytest

from factshorts.config import VerificationConfig
from factshorts.models import Script, SourceDoc
from factshorts.verify.aggregate import verify_script


def _replace_sentence(script: Script, seg_id: str, old: str, new: str) -> None:
    seg = next(s for s in script.segments if s.id == seg_id)
    seg.narration = seg.narration.replace(old, new)
    for c in seg.claims:
        c.text = c.text.replace(old, new)


ADVERSARIAL = {
    "wrong_number": ("s1", "6,412 meters", "6,812 meters"),
    "wrong_unit_scale": ("s1", "6,412 meters", "6,412 kilometers"),
    "wrong_year": ("s2", "in 1931", "in 1913"),
    "swapped_name": ("s2", "Tomas Kell", "Tomas Brenner-Hill"),
    "invented_person": ("s2", "Ada Brenner and Tomas Kell", "Ada Brenner and George Mallory"),
    "wrong_sign": ("s3", "minus 45 °C", "45 °C"),
    "imperial_unit": ("s1", "6,412 meters", "21,037 feet"),
    "exaggerated_rounding": ("s4", "about 12 meters", "about 20 meters"),
}


@pytest.mark.parametrize("case", sorted(ADVERSARIAL))
def test_adversarial_edit_is_caught(case: str, script: Script, source: SourceDoc) -> None:
    seg_id, old, new = ADVERSARIAL[case]
    _replace_sentence(script, seg_id, old, new)
    v = verify_script(script, source, VerificationConfig())
    assert not v.passed, f"{case} slipped through"


def test_unsupported_extra_sentence_is_caught(script: Script, source: SourceDoc) -> None:
    script.scenes[0].narration += " It grows 3 cm every decade."
    v = verify_script(script, source, VerificationConfig())
    assert not v.passed
    assert any("not linked to any claim" in f for f in v.failures)


def test_extra_sentence_with_name_but_no_claim_is_caught(script: Script, source: SourceDoc) -> None:
    script.outro.narration = "Thanks to Velden for hosting us."
    v = verify_script(script, source, VerificationConfig())
    assert not v.passed


def test_hook_with_unsupported_number_is_caught(script: Script, source: SourceDoc) -> None:
    script.hook.narration = "Mount Testa is the highest mountain in Fictland, at 9,000 meters."
    script.hook.claims[0].text = script.hook.narration
    v = verify_script(script, source, VerificationConfig())
    assert not v.passed


def test_fabricated_quote_is_caught(script: Script, source: SourceDoc) -> None:
    script.scenes[3].claims[0].source_quote = "Its glacier shrinks by about 12 meters every year."
    v = verify_script(script, source, VerificationConfig())
    assert not v.passed
