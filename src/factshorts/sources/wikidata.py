"""Wikidata: numeric and date statements of an entity, with units mapped to our unit keys."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from factshorts.core.http import CachedHttp

log = logging.getLogger(__name__)

API = "https://www.wikidata.org/w/api.php"

# Wikidata unit items → factshorts unit keys (see verify/units.py).
UNIT_QIDS = {
    "Q11573": "m",
    "Q828224": "km",
    "Q174728": "cm",
    "Q174789": "mm",
    "Q3710": "ft",
    "Q253276": "mi",
    "Q93318": "nmi",
    "Q1811": "au",
    "Q531": "ly",
    "Q218593": "in",
    "Q25343": "m2",
    "Q712226": "km2",
    "Q35852": "ha",
    "Q232291": "sqmi",
    "Q81292": "acre",
    "Q25517": "m3",
    "Q11582": "l",
    "Q4243638": "km3",
    "Q11570": "kg",
    "Q41803": "g",
    "Q191118": "t",
    "Q100995": "lb",
    "Q25267": "degC",
    "Q11579": "K",
    "Q42289": "degF",
    "Q180154": "km/h",
    "Q182429": "m/s",
    "Q211256": "mph",
    "Q128822": "kn",
    "Q11574": "second",
    "Q7727": "minute",
    "Q25235": "hour",
    "Q573": "day",
    "Q577": "year",
    "Q1092296": "year",
    "Q11229": "%",
}


@dataclass(frozen=True)
class WDValue:
    property: str
    kind: str  # "quantity" or "time"
    value: float  # amount, or the year for time values
    unit: str | None
    rank: str
    text: str  # human-readable, for reasons in verification.json


def _parse_quantity(prop: str, rank: str, v: dict) -> WDValue | None:
    unit_url = v.get("unit", "1")
    qid = unit_url.rsplit("/", 1)[-1]
    unit = None if unit_url == "1" else UNIT_QIDS.get(qid)
    if unit_url != "1" and unit is None:
        return None  # a unit we cannot compare
    amount = float(v["amount"])
    text = f"{amount:g}" + (f" {unit}" if unit else "")
    return WDValue(prop, "quantity", amount, unit, rank, text)


def _parse_time(prop: str, rank: str, v: dict) -> WDValue | None:
    m = re.match(r"([+-]\d+)-(\d\d)-(\d\d)", v.get("time", ""))
    if not m or v.get("precision", 0) < 9:  # coarser than a year
        return None
    year = int(m.group(1))
    return WDValue(prop, "time", float(year), None, rank, v["time"].lstrip("+")[:10])


class Wikidata:
    def __init__(self, http: CachedHttp) -> None:
        self.http = http

    def statements(self, qid: str, properties: list[str]) -> dict[str, list[WDValue]]:
        """Non-deprecated numeric/date values for the given properties."""
        data = self.http.get_json(
            API,
            {
                "action": "wbgetentities",
                "ids": qid,
                "props": "claims",
                "format": "json",
            },
        )
        entity = data.get("entities", {}).get(qid, {})
        out: dict[str, list[WDValue]] = {}
        for prop in properties:
            for st in entity.get("claims", {}).get(prop, []):
                rank = st.get("rank", "normal")
                dv = st.get("mainsnak", {}).get("datavalue")
                if rank == "deprecated" or not dv:
                    continue
                parsed = (
                    _parse_quantity(prop, rank, dv["value"])
                    if dv["type"] == "quantity"
                    else _parse_time(prop, rank, dv["value"])
                    if dv["type"] == "time"
                    else None
                )
                if parsed is not None:
                    out.setdefault(prop, []).append(parsed)
        return out
