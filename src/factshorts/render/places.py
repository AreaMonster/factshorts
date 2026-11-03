"""Place names in a sentence, so a map pin only appears where a scene really names a place."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

# spaCy entity labels for places: countries/cities/regions, other locations (mountains, lakes),
# and named structures (tunnels, massifs, bridges).
PLACE_LABELS = {"GPE", "LOC", "FAC"}
# Tagged as locations, but a pin on a map makes no sense for them.
NOT_ON_A_MAP = {
    "sun", "earth", "moon", "solar system", "milky way", "mercury", "venus", "mars", "jupiter",
    "saturn", "uranus", "neptune", "pluto", "space", "universe",
}  # fmt: skip


@lru_cache(maxsize=1)
def _nlp() -> Any:
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except (ImportError, OSError) as exc:  # no model: no pins, never a wrong one
        log.warning("place detection unavailable (%s); map pins are skipped", exc)
        return None


def find_places(text: str, limit: int = 3) -> list[str]:
    """Place names in `text`, in order, without a leading "the" ("the Glarus Alps" -> "Glarus
    Alps"). Empty when there are none or the language model is not installed."""
    nlp = _nlp()
    if nlp is None or not text.strip():
        return []
    places: list[str] = []
    for ent in nlp(text).ents:
        if ent.label_ not in PLACE_LABELS:
            continue
        name = re.sub(r"^the\s+", "", ent.text.strip(), flags=re.I)
        # "China-Nepal" (as in "the China-Nepal border") names two places.
        parts = name.split("-") if re.fullmatch(r"[A-Z]\w+(-[A-Z]\w+)+", name) else [name]
        for part in parts:
            on_map = part and part[0].isupper() and part.lower() not in NOT_ON_A_MAP
            if on_map and part not in places:
                places.append(part)
    return places[:limit]
