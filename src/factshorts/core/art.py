"""Art direction: turn verified narration into clean, text-free illustration prompts."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from factshorts.core.llm import LLM
from factshorts.models import Script

log = logging.getLogger(__name__)


class Shot(BaseModel):
    segment_id: str
    image_prompt: str = Field(description="25-45 words describing one illustration")
    motion: str = Field(description="3-8 words of gentle movement")
    visual_interest: int = Field(ge=1, le=5)


class ShotList(BaseModel):
    shots: list[Shot]


ART_PROMPT = """You are the art director of an animated explainer video drawn as {style_name}.
For every segment below, describe ONE illustration that shows what its narration is about.

RULES
- Describe a picture of a place, landscape, object, animal or action. Never an infographic:
  no charts, graphs, timelines, columns, icons, maps, pins, diagrams or labels.
- No text, letters, numbers, signs, flags or logos anywhere in the picture.
- No real or identifiable people. People only as tiny, faceless, generic figures seen from
  afar (hikers, scientists), never named.
- Only show what the narration states plus obviously generic surroundings (sky, clouds,
  snow, sea). Do not invent specific details that could be wrong (plants, buildings,
  weather, seasons, animals) unless the narration mentions them.
- Vary the camera between segments: wide shot, close-up, aerial view, low angle, side view.
- image_prompt: 25-45 words. motion: 3-8 words of gentle movement for animating it
  (e.g. "clouds drift slowly, snow blows"). visual_interest 1-5: 5 = real motion would make
  the segment much better.
Return one shot per segment, using the segment ids given.

VIDEO: {title}
SEGMENTS:
{segments}"""

BANNED = re.compile(
    r"\d|[\"“”]|\b(text|texts|label|labels|labeled|labelled|letter|letters|number|"
    r"numbers|numeral|flag|flags|logo|logos|sign|signs|map|maps|timeline|chart|charts|graph|"
    r"graphs|icon|icons|column|columns|pin|pins|caption|captions|written|writing|words?|"
    r"infographic|diagram|banner)\b",
    re.I,
)
FALLBACK_MOTION = "clouds drift slowly, light shifts gently"


def problems_in(prompt: str) -> list[str]:
    return sorted({m.group(0).lower() for m in BANNED.finditer(prompt)})


def sanitize(prompt: str) -> str:
    """Drop clauses that ask for text, numbers, flags, maps or charts."""
    parts = re.split(r"(?<=[.,;])\s+", prompt)
    kept = [p for p in parts if not BANNED.search(p)]
    return " ".join(kept).strip(" ,;") or ""


def plan_shots(script: Script, llm: LLM, model: str, style_name: str) -> Script:
    """Fill image_prompt / motion / visual_interest for every segment of `script`."""
    segments = "\n".join(f"[{s.id}] {s.narration}" for s in script.segments)
    messages = [
        {
            "role": "user",
            "content": ART_PROMPT.format(
                style_name=style_name, title=script.title, segments=segments
            ),
        }
    ]
    shots: dict[str, Shot] = {}
    for attempt in range(1, 3):
        result = llm.structured(messages, ShotList, model=model, temperature=0.7)
        shots = {s.segment_id.strip("[] "): s for s in result.shots}
        issues = {sid: problems_in(s.image_prompt) for sid, s in shots.items()}
        issues = {sid: p for sid, p in issues.items() if p}
        missing = [s.id for s in script.segments if s.id not in shots]
        if not issues and not missing:
            break
        log.info("shot list attempt %d: banned words %s, missing %s", attempt, issues, missing)
        feedback = []
        if issues:
            feedback.append(
                "These prompts mention forbidden things (text, numbers, flags, "
                "maps, charts...): " + "; ".join(f"{k}: {', '.join(v)}" for k, v in issues.items())
            )
        if missing:
            feedback.append(f"Missing segments: {', '.join(missing)}")
        messages = [
            *messages[:1],
            {"role": "assistant", "content": result.model_dump_json()},
            {
                "role": "user",
                "content": " ".join(feedback) + " Return the full list "
                "again with those prompts rewritten as pure pictures.",
            },
        ]
    for seg in script.segments:
        shot = shots.get(seg.id)
        prompt = sanitize(shot.image_prompt) if shot else ""
        if not prompt:
            prompt = f"an evocative scene about {script.title.rstrip('?')}, wide shot"
            prompt = sanitize(prompt) or "a wide, calm landscape under a dramatic sky"
        seg.image_prompt = prompt
        seg.motion = (sanitize(shot.motion) if shot else "") or FALLBACK_MOTION
        seg.visual_interest = shot.visual_interest if shot else 2
    return script
