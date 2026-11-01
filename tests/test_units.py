import pytest

from factshorts.verify.units import convert, is_imperial, match_unit, same_dimension


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (8849, "m", "km", 8.849),
        (29031.7, "ft", "m", 8848.86),
        (430, "mi", "km", 692.0),
        (1, "t", "kg", 1000),
        (5, "lb", "kg", 2.268),
        (212, "degF", "degC", 100),
        (-40, "degF", "degC", -40),
        (0, "degC", "K", 273.15),
        (100, "km/h", "m/s", 27.778),
        (60, "mph", "km/h", 96.56),
        (1, "sqmi", "km2", 2.59),
        (1, "ha", "m2", 10_000),
    ],
)
def test_convert(value: float, src: str, dst: str, expected: float) -> None:
    assert convert(value, src, dst) == pytest.approx(expected, rel=1e-3)


def test_incompatible_dimensions() -> None:
    assert not same_dimension("m", "kg")
    with pytest.raises(ValueError):
        convert(1, "m", "kg")


def test_imperial_flags() -> None:
    assert is_imperial("ft") and is_imperial("degF") and is_imperial("mph")
    assert not is_imperial("km") and not is_imperial("degC")


@pytest.mark.parametrize(
    ("text", "unit"),
    [
        (" m tall", "m"),
        ("m", "m"),
        (" metres", "m"),
        (" km/h", "km/h"),
        ("°C", "degC"),
        (" degrees Celsius", "degC"),
        (" square kilometers", "km2"),
        (" million", None),
        (" M", None),
        (" months", "month"),
    ],
)
def test_match_unit(text: str, unit: str | None) -> None:
    m = match_unit(text, 0)
    assert (m[0] if m else None) == unit
