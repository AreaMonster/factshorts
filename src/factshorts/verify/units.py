"""Unit table and conversions. Every unit maps to an SI-ish base per dimension."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    key: str  # canonical short name
    dimension: str
    factor: float  # value_in_base = value * factor + offset
    offset: float = 0.0
    imperial: bool = False
    spoken_singular: str = ""
    spoken_plural: str = ""


def _u(key: str, dim: str, factor: float, singular: str, plural: str, **kw: object) -> Unit:
    return Unit(key, dim, factor, spoken_singular=singular, spoken_plural=plural, **kw)  # type: ignore[arg-type]


UNITS: dict[str, Unit] = {
    u.key: u
    for u in [
        # length (base: meter)
        _u("mm", "length", 1e-3, "millimeter", "millimeters"),
        _u("cm", "length", 1e-2, "centimeter", "centimeters"),
        _u("m", "length", 1.0, "meter", "meters"),
        _u("km", "length", 1e3, "kilometer", "kilometers"),
        _u("au", "length", 1.495978707e11, "astronomical unit", "astronomical units"),
        _u("ly", "length", 9.4607304725808e15, "light-year", "light-years"),
        _u("in", "length", 0.0254, "inch", "inches", imperial=True),
        _u("ft", "length", 0.3048, "foot", "feet", imperial=True),
        _u("yd", "length", 0.9144, "yard", "yards", imperial=True),
        _u("mi", "length", 1609.344, "mile", "miles", imperial=True),
        _u("nmi", "length", 1852.0, "nautical mile", "nautical miles", imperial=True),
        # area (base: m²)
        _u("m2", "area", 1.0, "square meter", "square meters"),
        _u("ha", "area", 1e4, "hectare", "hectares"),
        _u("km2", "area", 1e6, "square kilometer", "square kilometers"),
        _u("acre", "area", 4046.8564224, "acre", "acres", imperial=True),
        _u("sqmi", "area", 2589988.110336, "square mile", "square miles", imperial=True),
        # volume (base: m³)
        _u("l", "volume", 1e-3, "liter", "liters"),
        _u("m3", "volume", 1.0, "cubic meter", "cubic meters"),
        _u("km3", "volume", 1e9, "cubic kilometer", "cubic kilometers"),
        _u("gal", "volume", 3.785411784e-3, "gallon", "gallons", imperial=True),
        # mass (base: kg)
        _u("mg", "mass", 1e-6, "milligram", "milligrams"),
        _u("g", "mass", 1e-3, "gram", "grams"),
        _u("kg", "mass", 1.0, "kilogram", "kilograms"),
        _u("t", "mass", 1e3, "tonne", "tonnes"),
        _u("oz", "mass", 0.028349523125, "ounce", "ounces", imperial=True),
        _u("lb", "mass", 0.45359237, "pound", "pounds", imperial=True),
        _u("ton", "mass", 907.18474, "ton", "tons", imperial=True),
        # temperature (base: °C)
        _u("degC", "temperature", 1.0, "degree Celsius", "degrees Celsius"),
        _u("K", "temperature", 1.0, "kelvin", "kelvins", offset=-273.15),
        _u(
            "degF",
            "temperature",
            5 / 9,
            "degree Fahrenheit",
            "degrees Fahrenheit",
            offset=-32 * 5 / 9,
            imperial=True,
        ),
        # speed (base: m/s)
        _u("m/s", "speed", 1.0, "meter per second", "meters per second"),
        _u("km/h", "speed", 1 / 3.6, "kilometer per hour", "kilometers per hour"),
        _u("km/s", "speed", 1e3, "kilometer per second", "kilometers per second"),
        _u("mph", "speed", 0.44704, "mile per hour", "miles per hour", imperial=True),
        _u("kn", "speed", 1852 / 3600, "knot", "knots", imperial=True),
        # time (base: second)
        _u("second", "time", 1.0, "second", "seconds"),
        _u("minute", "time", 60.0, "minute", "minutes"),
        _u("hour", "time", 3600.0, "hour", "hours"),
        _u("day", "time", 86400.0, "day", "days"),
        _u("week", "time", 604800.0, "week", "weeks"),
        _u("month", "time", 2629746.0, "month", "months"),
        _u("year", "time", 31556952.0, "year", "years"),
        # ratio
        _u("%", "ratio", 1.0, "percent", "percent"),
    ]
}

# Surface forms → unit key. Case-sensitive forms are listed in CASE_SENSITIVE.
ALIASES: dict[str, str] = {
    "millimeters": "mm",
    "millimetres": "mm",
    "millimeter": "mm",
    "millimetre": "mm",
    "mm": "mm",
    "centimeters": "cm",
    "centimetres": "cm",
    "centimeter": "cm",
    "centimetre": "cm",
    "cm": "cm",
    "meters": "m",
    "metres": "m",
    "meter": "m",
    "metre": "m",
    "kilometers": "km",
    "kilometres": "km",
    "kilometer": "km",
    "kilometre": "km",
    "km": "km",
    "astronomical units": "au",
    "astronomical unit": "au",
    "light-years": "ly",
    "light years": "ly",
    "light-year": "ly",
    "light year": "ly",
    "inches": "in",
    "inch": "in",
    "feet": "ft",
    "foot": "ft",
    "ft": "ft",
    "yards": "yd",
    "yard": "yd",
    "yd": "yd",
    "nautical miles": "nmi",
    "nautical mile": "nmi",
    "nmi": "nmi",
    "miles": "mi",
    "mile": "mi",
    "mi": "mi",
    "square meters": "m2",
    "square metres": "m2",
    "square meter": "m2",
    "square metre": "m2",
    "m²": "m2",
    "m2": "m2",
    "sq m": "m2",
    "hectares": "ha",
    "hectare": "ha",
    "ha": "ha",
    "square kilometers": "km2",
    "square kilometres": "km2",
    "square kilometer": "km2",
    "square kilometre": "km2",
    "km²": "km2",
    "km2": "km2",
    "sq km": "km2",
    "acres": "acre",
    "acre": "acre",
    "square miles": "sqmi",
    "square mile": "sqmi",
    "sq mi": "sqmi",
    "mi²": "sqmi",
    "mi2": "sqmi",
    "liters": "l",
    "litres": "l",
    "liter": "l",
    "litre": "l",
    "cubic meters": "m3",
    "cubic metres": "m3",
    "m³": "m3",
    "m3": "m3",
    "cubic kilometers": "km3",
    "cubic kilometres": "km3",
    "km³": "km3",
    "km3": "km3",
    "gallons": "gal",
    "gallon": "gal",
    "milligrams": "mg",
    "milligram": "mg",
    "mg": "mg",
    "kilograms": "kg",
    "kilogram": "kg",
    "kg": "kg",
    "grams": "g",
    "gram": "g",
    "tonnes": "t",
    "tonne": "t",
    "metric tons": "t",
    "metric ton": "t",
    "ounces": "oz",
    "ounce": "oz",
    "oz": "oz",
    "pounds": "lb",
    "pound": "lb",
    "lbs": "lb",
    "lb": "lb",
    "tons": "ton",
    "ton": "ton",
    "degrees celsius": "degC",
    "degree celsius": "degC",
    "°c": "degC",
    "° c": "degC",
    "℃": "degC",
    "celsius": "degC",
    "degrees fahrenheit": "degF",
    "degree fahrenheit": "degF",
    "°f": "degF",
    "° f": "degF",
    "℉": "degF",
    "fahrenheit": "degF",
    "kelvins": "K",
    "kelvin": "K",
    "kilometers per hour": "km/h",
    "kilometres per hour": "km/h",
    "km/h": "km/h",
    "kph": "km/h",
    "km per hour": "km/h",
    "kilometers per second": "km/s",
    "kilometres per second": "km/s",
    "km/s": "km/s",
    "meters per second": "m/s",
    "metres per second": "m/s",
    "m/s": "m/s",
    "miles per hour": "mph",
    "mph": "mph",
    "knots": "kn",
    "knot": "kn",
    "seconds": "second",
    "second": "second",
    "minutes": "minute",
    "minute": "minute",
    "hours": "hour",
    "hour": "hour",
    "days": "day",
    "day": "day",
    "weeks": "week",
    "week": "week",
    "months": "month",
    "month": "month",
    "years": "year",
    "year": "year",
    "percent": "%",
    "per cent": "%",
    "%": "%",
}
# Short forms that must match case exactly (so "m" isn't "M", "K" isn't "k", "t" etc.).
CASE_SENSITIVE: dict[str, str] = {"m": "m", "g": "g", "t": "t", "K": "K", "L": "l", "AU": "au"}


def _alias_pattern() -> re.Pattern[str]:
    forms = sorted(ALIASES, key=len, reverse=True)
    alt = "|".join(re.escape(f).replace("\\ ", " ").replace(" ", r"\s+") for f in forms)
    # Unit must not be followed by a letter/digit (so "m" doesn't match "million").
    return re.compile(rf"(?:{alt})(?![\w²³])", re.I)


_ALIAS_RE = _alias_pattern()
_CS_ALT = "|".join(map(re.escape, CASE_SENSITIVE))
_CS_RE = re.compile(rf"(?:{_CS_ALT})(?![\w²³/])")


def match_unit(text: str, pos: int) -> tuple[str, int] | None:
    """Match a unit at text[pos:] (after optional whitespace). Returns (unit_key, end)."""
    m = re.compile(r"[ \t  ]*").match(text, pos)
    start = m.end() if m else pos
    # A unit directly glued with a degree sign like "20°C" or "8,849m".
    am = _ALIAS_RE.match(text, start)
    cm = _CS_RE.match(text, start)
    best: tuple[str, int] | None = None
    if am:
        best = (ALIASES[re.sub(r"\s+", " ", am.group(0).lower())], am.end())
    if cm and (best is None or cm.end() > best[1]):
        best = (CASE_SENSITIVE[cm.group(0)], cm.end())
    return best


def convert(value: float, from_key: str, to_key: str) -> float:
    a, b = UNITS[from_key], UNITS[to_key]
    if a.dimension != b.dimension:
        raise ValueError(f"cannot convert {from_key} to {to_key}")
    base = value * a.factor + a.offset
    return (base - b.offset) / b.factor


def same_dimension(a: str, b: str) -> bool:
    return UNITS[a].dimension == UNITS[b].dimension


def is_imperial(key: str) -> bool:
    return UNITS[key].imperial
