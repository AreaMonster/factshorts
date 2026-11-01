import pytest

from factshorts.config import VerificationConfig
from factshorts.verify.entities import build_vocab, extract_names
from factshorts.verify.programmatic import entity_check

EVIDENCE = (
    "Mount Testa is the highest mountain in Fictland, with a summit elevation of 6,412 m "
    "(21,037 ft) above sea level. The mountain was first climbed on 14 July 1931 by Ada "
    "Brenner and Tomas Kell. Around 35,000 people visit each year. Temperatures fall to "
    "−45 °C. The town of Velden receives about 1,200 mm of rain per year."
)
CFG = VerificationConfig()
VOCAB = build_vocab(EVIDENCE)


def check(text: str, evidence: str = EVIDENCE) -> str:
    return entity_check(text, evidence, VOCAB, CFG).status


@pytest.mark.parametrize(
    "text",
    [
        "Its summit is 6,412 meters high.",
        "Its summit is 6412 m high.",
        "Its summit is about 6.4 kilometers high.",
        "Its summit is roughly 6,400 meters high.",
        "Its summit is about 6.41 km high.",
        "It is six thousand four hundred twelve meters high.",
        "Ada Brenner and Tomas Kell climbed it in 1931.",
        "They reached the top on 14 July 1931.",
        "About 35,000 people visit every year.",
        "It can be -45 °C at the top.",
        "Velden gets about 1.2 meters of rain a year.",
    ],
)
def test_supported_entities_pass(text: str) -> None:
    assert check(text) == "pass"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Its summit is 6,500 meters high.", "number"),
        ("Its summit is 7 km high.", "number"),
        ("It was first climbed in 1932.", "number"),
        ("They reached the top on 15 July 1931.", "number"),
        ("About 3.5 million people visit every year.", "number"),
        ("Ada Brenner and George Mallory climbed it.", "name"),
        ("Edmund Hillary climbed it in 1931.", "name"),
        ("It was first climbed in August 1931.", "name"),
        ("It is 21,037 feet high.", "imperial"),
        ("It can be 45 °C at the top.", "number"),
    ],
)
def test_invented_or_changed_entities_fail(text: str, reason: str) -> None:
    res = entity_check(text, EVIDENCE, VOCAB, CFG)
    assert res.status == "fail"
    assert reason in res.reason


def test_rounding_is_bounded() -> None:
    # "1 million" for 1.4 million rounds correctly to 1 significant digit but is 29 % off.
    ev = "The city has 1,400,000 inhabitants."
    assert check("The city has 1 million inhabitants.", ev) == "fail"
    assert check("The city has 1.4 million inhabitants.", ev) == "pass"


def test_sentence_start_words_are_not_names() -> None:
    vocab = build_vocab("climbers usually start from base camps", EVIDENCE)
    assert extract_names("Climbers usually start here.", vocab) == []
    assert extract_names("Starts above 5,000 m", vocab, titlecase=True) == []


def test_unknown_capitalized_word_counts_as_name() -> None:
    assert extract_names("Zorblax climbed it.", VOCAB) == ["Zorblax"]
    assert extract_names("It was climbed by Zorblax.", VOCAB) == ["Zorblax"]


def test_known_proper_noun_at_sentence_start_is_a_name() -> None:
    vocab = build_vocab("The first climbers were Ada Brenner and Kell.")
    assert "Brenner" in extract_names("Brenner was first.", vocab)


def test_months_are_checked() -> None:
    assert extract_names("It happened in July.", VOCAB) == ["July"]


def test_accents_are_folded_for_names() -> None:
    assert check("Sagarmatha is a name.", "Its name is Sagarmāthā in Nepali.") == "pass"
