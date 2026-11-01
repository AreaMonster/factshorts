"""Art direction: turn verified narration into clean, text-free, grounded illustration prompts,
and review the finished pictures with a vision model."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from factshorts.core.llm import LLM
from factshorts.models import Script

log = logging.getLogger(__name__)


class Shot(BaseModel):
    segment_id: str
    image_prompt: str = Field(description="25-45 words describing one illustration, no names")
    shows_subject: bool = Field(description="True if the main subject is visible in the shot")
    motion: str = Field(description="3-8 words of gentle movement")
    visual_interest: int = Field(ge=1, le=5)


class ShotList(BaseModel):
    subject_look: str = Field(
        description="15-30 words: how the video's main subject looks (colors, shape, surface), "
        "taken from SOURCE CONTEXT, without its name"
    )
    shots: list[Shot]


class ImageVerdict(BaseModel):
    shows: str = Field(description="One or two sentences: what the picture shows")
    problems: list[str] = Field(description="Everything wrong or misleading; empty if none")
    matches: bool


class RevisedShot(BaseModel):
    image_prompt: str


ART_PROMPT = """You are the art director of an educational animated video drawn as {style_name}.
For every segment below, describe ONE illustration that shows what its narration is about.

RULES
- The image model does not know names. Never use proper names (not "{title_hint}", not
  places or people); describe what things LOOK like instead.
- First write subject_look: how the main subject of the video looks, using only appearance
  details from SOURCE CONTEXT (colors, clouds, surface, shape). If SOURCE CONTEXT does not say,
  keep it plain. Also say what it does NOT have that drawings often add by mistake
  (e.g. "no rings, no moons" for a planet without them). Set shows_subject = true for shots
  where the main subject is visible; its look is then added to the prompt automatically, so do
  not describe it again.
- Describe a picture of a place, landscape, object or action. Never an infographic: no
  charts, graphs, timelines, columns, icons, maps, pins, diagrams or labels.
- No text, letters, numbers, signs, flags or logos anywhere in the picture.
- No real or identifiable people. People only as tiny, faceless, generic figures seen from
  afar, never named.
- Only show what the narration states plus obviously generic surroundings (sky, clouds, stars,
  snow, sea). Do not invent details that could be wrong (plants, buildings, rings, moons,
  weather, seasons, animals) unless the narration or SOURCE CONTEXT mentions them.
- VARIETY: viewers must never see the same picture twice. Use each segment's suggested
  framing. Show the whole main subject in at most a third of the shots; in the others show
  what the sentence is about: a detail, the surface or landscape, the sky or atmosphere seen
  from inside, light and distance, or an object the narration mentions. No two image_prompts
  may describe the same composition.
- image_prompt: 25-45 words. motion: 3-8 words of gentle movement. visual_interest 1-5.
Return one shot per segment, using the segment ids given.

VIDEO: {title}
SOURCE CONTEXT (what the subject is and looks like):
{context}

SEGMENTS:
{segments}"""

REVISE_PROMPT = """The illustration for this narration was rejected by a reviewer.
Narration: "{narration}"
Old image prompt: "{prompt}"
Reviewer saw: {shows}
Problems: {problems}

Write a new image prompt (25-45 words) for a picture that fits the narration and avoids these
problems. Same rules: no names, no text/numbers/labels/flags, no infographics, nothing the
narration or the subject description does not support. The main subject looks like this
(include it if the subject is visible): {subject_look}"""

REVIEW_PROMPT = """This illustration is for an educational cartoon-style video about {title}.
The main subject looks like: {subject_look}
It is shown while the narrator says: "{narration}"
Intended picture: {prompt}

Describe briefly what the picture shows. Then list only CLEAR problems:
- the wrong object (a different planet, landmark or animal), or features the subject definitely
  does not have (e.g. rings or moons around a planet that has none, Earth's continents on
  another planet);
- any text, letters, numbers, labels, flags or logos;
- realistic depictions of identifiable people;
- something that plainly contradicts the narration.
Do NOT count as problems: the flat cartoon style, simplification, exact color shades, missing
details, the size, position, distance or framing of the subject, or a general picture for a
sentence without facts (such as "Thanks for watching!").
matches = true when there are no clear problems."""

# Suggested framing per segment, rotated so consecutive shots look different.
FRAMINGS = [
    "a wide establishing view",
    "a close-up of one detail the sentence is about",
    "a view from the ground or surface looking up",
    "seen from far away, the main subject small in the frame",
    "a view from inside the scene (inside the clouds, on the surface, in the landscape)",
    "an aerial view looking down",
    "a dramatic low-angle view",
    "an abstract close-up of textures and light",
]

BANNED = re.compile(
    r"\d|[\"“”]|\b(text|texts|label|labels|labeled|labelled|letter|letters|number|"
    r"numbers|numeral|flag|flags|logo|logos|sign|signs|map|maps|timeline|chart|charts|graph|"
    r"graphs|icon|icons|column|columns|pin|pins|caption|captions|written|writing|words?|"
    r"infographic|diagram|banner)\b",
    re.I,
)
FALLBACK_MOTION = "clouds drift slowly, light shifts gently"


def problems_in(prompt: str, names: set[str] | None = None) -> list[str]:
    found = {m.group(0).lower() for m in BANNED.finditer(prompt)}
    for name in names or set():
        if re.search(rf"\b{re.escape(name)}\b", prompt, re.I):
            found.add(f"name '{name}'")
    return sorted(found)


def sanitize(prompt: str) -> str:
    """Drop clauses that ask for text, numbers, flags, maps or charts."""
    parts = re.split(r"(?<=[.,;])\s+", prompt)
    kept = [p for p in parts if not BANNED.search(p)]
    return " ".join(kept).strip(" ,;") or ""


def _strip_names(prompt: str, names: set[str]) -> str:
    for name in names:
        prompt = re.sub(rf"\b{re.escape(name)}('s)?\b", "the subject", prompt, flags=re.I)
    return prompt


def similar_shots(shots: dict[str, Shot], threshold: float = 80) -> list[tuple[str, str]]:
    """Pairs of shots whose prompts describe nearly the same picture."""
    items = list(shots.items())
    return [
        (a, b)
        for i, (a, sa) in enumerate(items)
        for b, sb in items[i + 1 :]
        if fuzz.token_set_ratio(sa.image_prompt, sb.image_prompt) >= threshold
    ]


def compose_prompt(image_prompt: str, shows_subject: bool, subject_look: str) -> str:
    if shows_subject and subject_look:
        return f"{image_prompt.rstrip('. ')}. The main subject: {subject_look}"
    return image_prompt


def plan_shots(
    script: Script,
    llm: LLM,
    model: str,
    style_name: str,
    context: str = "",
    names: set[str] | None = None,
) -> tuple[Script, str]:
    """Fill image_prompt / motion / visual_interest for every segment of `script`.

    Returns the script and the subject's look. `names` (e.g. the article title's words) must not
    appear in prompts: the image model would guess what they look like."""
    names = {n for n in (names or set()) if len(n) > 2}
    segments = "\n".join(
        f"[{s.id}] (framing: {FRAMINGS[i % len(FRAMINGS)]}) {s.narration}"
        for i, s in enumerate(script.segments)
    )
    messages = [
        {
            "role": "user",
            "content": ART_PROMPT.format(
                style_name=style_name,
                title=script.title,
                segments=segments,
                context=context.strip() or "(none)",
                title_hint=next(iter(sorted(names)), "the subject's name"),
            ),
        }
    ]
    shots: dict[str, Shot] = {}
    look = ""
    for attempt in range(1, 3):
        result = llm.structured(messages, ShotList, model=model, temperature=0.7)
        shots = {s.segment_id.strip("[] "): s for s in result.shots}
        look = result.subject_look
        issues = {sid: problems_in(s.image_prompt, names) for sid, s in shots.items()}
        issues = {sid: p for sid, p in issues.items() if p}
        missing = [s.id for s in script.segments if s.id not in shots]
        alike = similar_shots(shots)
        if not issues and not missing and not alike:
            break
        log.info("shot list attempt %d: problems %s, missing %s", attempt, issues, missing)
        feedback = []
        if issues:
            feedback.append(
                "These prompts mention forbidden things (names, text, numbers, "
                "flags, maps, charts...): "
                + "; ".join(f"{k}: {', '.join(v)}" for k, v in issues.items())
            )
        if missing:
            feedback.append(f"Missing segments: {', '.join(missing)}")
        if alike:
            feedback.append(
                "These shots look almost the same; make each one a different picture: "
                + "; ".join(f"{a} and {b}" for a, b in alike)
            )
        messages = [
            *messages[:1],
            {"role": "assistant", "content": result.model_dump_json()},
            {
                "role": "user",
                "content": " ".join(feedback) + " Return the full list "
                "again with those prompts rewritten as pure pictures.",
            },
        ]
    look = sanitize(_strip_names(look, names))
    for seg in script.segments:
        shot = shots.get(seg.id)
        prompt = sanitize(_strip_names(shot.image_prompt, names)) if shot else ""
        shows = shot.shows_subject if shot else True
        if not prompt:
            prompt, shows = "a simple, calm wide shot of the main subject", True
        seg.image_prompt = compose_prompt(prompt, shows, look)
        seg.motion = (sanitize(shot.motion) if shot else "") or FALLBACK_MOTION
        seg.visual_interest = shot.visual_interest if shot else 2
    return script, look


class ImageReviewer:
    """A vision model checks each finished picture against its narration."""

    def __init__(self, llm: LLM, model: str) -> None:
        self.llm = llm
        self.model = model

    def review(
        self, image: Path, narration: str, prompt: str, title: str, subject_look: str
    ) -> ImageVerdict:
        data = base64.b64encode(image.read_bytes()).decode()
        content = REVIEW_PROMPT.format(
            title=title,
            subject_look=subject_look or "(not given)",
            narration=narration,
            prompt=prompt,
        )
        verdict = self.llm.structured(
            [{"role": "user", "content": content, "images": [data]}],
            ImageVerdict,
            model=self.model,
            temperature=0.0,
        )
        if verdict.problems:
            verdict.matches = False
        return verdict

    def revise(
        self, narration: str, prompt: str, verdict: ImageVerdict, subject_look: str, names: set[str]
    ) -> str:
        result = self.llm.structured(
            [
                {
                    "role": "user",
                    "content": REVISE_PROMPT.format(
                        narration=narration,
                        prompt=prompt,
                        shows=verdict.shows,
                        problems="; ".join(verdict.problems),
                        subject_look=subject_look,
                    ),
                }
            ],
            RevisedShot,
            model=self.model,
            temperature=0.5,
        )
        return sanitize(_strip_names(result.image_prompt, names)) or prompt
