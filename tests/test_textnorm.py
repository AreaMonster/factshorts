import pytest

from factshorts.core.textnorm import number_words, to_speakable


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("8,849 m", "eight thousand eight hundred forty-nine meters"),
        ("It is 8,849 meters tall.", "It is eight thousand eight hundred forty-nine meters tall."),
        ("8,848.86 m", "eight thousand eight hundred forty-eight point eight six meters"),
        ("1 km", "one kilometer"),
        ("20 °C", "twenty degrees Celsius"),
        ("-45 °C", "minus forty-five degrees Celsius"),
        ("120 km/h", "one hundred twenty kilometers per hour"),
        ("38 km²", "thirty-eight square kilometers"),
        ("45%", "forty-five percent"),
        ("in 1953", "in nineteen fifty-three"),
        ("in 2020", "in twenty twenty"),
        ("in 2008", "in two thousand eight"),
        ("the 1950s", "the nineteen fifties"),
        ("the 21st century", "the twenty-first century"),
        ("1.5 million people", "one point five million people"),
        ("4.5 billion years", "four point five billion years"),
        ("200+ bodies", "two hundred plus bodies"),
        ("Mt. Everest", "Mount Everest"),
        ("e.g. ice", "for example ice"),
        ("rock & ice", "rock and ice"),
        ("3 3/4 km", "three and three quarters kilometers"),
    ],
)
def test_to_speakable(text: str, spoken: str) -> None:
    assert to_speakable(text) == spoken


def test_pronunciation_overrides() -> None:
    assert to_speakable("Sagarmatha is tall.", {"Sagarmatha": "Sah-gar-MAH-tah"}) == (
        "Sah-gar-MAH-tah is tall."
    )


def test_number_words_american_style() -> None:
    assert number_words(105) == "one hundred five"
    assert "," not in number_words(1_234_567)
