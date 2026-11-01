import pytest

from factshorts.verify.normalize import (
    normalize_chars,
    parse_quantities,
    round_to,
    split_sentences,
    words_to_number,
)


def values(text: str) -> list[tuple[float, str | None]]:
    return [(q.value, q.unit) for q in parse_quantities(text)]


def assert_quantities(text: str, expected: list[tuple[float, str | None]]) -> None:
    got = values(text)
    assert [u for _, u in got] == [u for _, u in expected]
    assert [v for v, _ in got] == pytest.approx([v for v, _ in expected])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8,849 m", [(8849, "m")]),
        ("8849 m", [(8849, "m")]),
        ("8 849 m", [(8849, "m")]),
        ("8 849 m", [(8849, "m")]),
        ("8 849 metres", [(8849, "m")]),
        ("8,848.86 m", [(8848.86, "m")]),
        ("8849m", [(8849, "m")]),
        ("8.8 km", [(8.8, "km")]),
        ("29,031 ft", [(29031, "ft")]),
        ("700 km (430 mi)", [(700, "km"), (430, "mi")]),
        ("120 km/h", [(120, "km/h")]),
        ("38 km²", [(38, "km2")]),
        ("20°C", [(20, "degC")]),
        ("−45 °C", [(-45, "degC")]),
        ("minus 45 °C", [(-45, "degC")]),
        ("1.5 million people", [(1_500_000, None)]),
        ("4.5 billion years", [(4.5e9, "year")]),
        ("45%", [(45, "%")]),
        ("45 percent", [(45, "%")]),
        ("3,100 m", [(3100, "m")]),
        ("1,200 mm", [(1200, "mm")]),
        ("150 km/h winds", [(150, "km/h")]),
        ("6 tonnes", [(6, "t")]),
        ("6 tons", [(6, "ton")]),
        ("5 kg", [(5, "kg")]),
        ("8+1⁄2 in", [(8.5, None)]),
    ],
)
def test_digit_quantities(text: str, expected: list[tuple[float, str | None]]) -> None:
    assert_quantities(text, expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("eight thousand eight hundred forty-nine meters", [(8849, "m")]),
        ("eight thousand eight hundred and forty-nine", [(8849, None)]),
        ("twenty-one people", [(21, None)]),
        ("two million", [(2_000_000, None)]),
        ("two thousand km", [(2000, "km")]),
        ("one hundred and five", [(105, None)]),
        ("ninety", [(90, None)]),
    ],
)
def test_word_quantities(text: str, expected: list[tuple[float, str | None]]) -> None:
    assert_quantities(text, expected)


def test_one_is_not_a_number_by_default() -> None:
    assert values("It is one of the highest peaks.") == []
    assert [q.value for q in parse_quantities("one", include_one=True)] == [1]


def test_years_and_decades() -> None:
    qs = parse_quantities("In 1953 and in the 1950s, not 2,000 people")
    assert [(q.value, q.is_year) for q in qs] == [(1953, True), (1950, True), (2000, False)]


def test_ordinals_are_not_years() -> None:
    qs = parse_quantities("the 21st century and the 3rd attempt")
    assert [(q.value, q.is_year, q.unit) for q in qs] == [(21, False, None), (3, False, None)]


def test_precision() -> None:
    p = {q.raw: q.precision for q in parse_quantities("8,800 m, 8,849 m, 8.8 km, 9,000 m")}
    assert p == {"8,800 m": 100, "8,849 m": 1, "8.8 km": pytest.approx(0.1), "9,000 m": 1000}


def test_dates_and_ranges_do_not_become_negative() -> None:
    assert [q.value for q in parse_quantities("1953-1960")] == [1953, 1960]
    assert [q.value for q in parse_quantities("COVID-19")] == [19]


def test_words_to_number() -> None:
    assert words_to_number(["three", "hundred", "forty-two"]) == 342
    assert words_to_number(["seven", "billion"]) == 7e9
    assert words_to_number(["seven", "apples"]) is None


def test_normalize_chars() -> None:
    assert normalize_chars("“Hello”  — it’s fine…") == '"Hello" - it\'s fine...'


def test_split_sentences_handles_abbreviations_and_decimals() -> None:
    text = "Mt. Everest is 8.8 km tall. The U.S. team came in 1963. Why? Because! Done."
    assert split_sentences(text) == [
        "Mt. Everest is 8.8 km tall.",
        "The U.S. team came in 1963.",
        "Why?",
        "Because!",
        "Done.",
    ]


def test_round_to() -> None:
    assert round_to(8848.86, 100) == 8800
    assert round_to(8850, 100) == 8900
    assert round_to(8.84886, 0.1) == pytest.approx(8.8)


def test_split_after_possessive() -> None:
    assert split_sentences("It is 92 times Earth's. This turns it into a fluid.") == [
        "It is 92 times Earth's.",
        "This turns it into a fluid.",
    ]
    assert split_sentences("Ask J. Smith. He knows.") == ["Ask J. Smith.", "He knows."]
