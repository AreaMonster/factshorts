"""A short caption to paste into TikTok (or Reels/Shorts) when a video is finished.

The LLM writes one catchy line from the checked script. It may not add anything: every number
and name in its line must already be in the script, otherwise the checked hook is used."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from factshorts.core.llm import LLM
from factshorts.models import Script, SourceDoc

log = logging.getLogger(__name__)

MAX_LINE = 150

CAPTION_PROMPT = """Write the caption for a short TikTok video. One or two short sentences, at
most {max_chars} characters, that make people curious to watch: a question or the most
surprising fact. Plain American English, no hashtags, no emojis, no "Did you know" opener.
Use ONLY facts from the SCRIPT. Do not add any number, date, name or claim that is not in it.

TITLE: {title}
SCRIPT:
{script}"""


class CaptionLine(BaseModel):
    caption: str = Field(description="The caption line, without hashtags")


NUMBER_WORDS = frozenset(
    {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "trillion",
        "twice",
        "thrice",
        "half",
        "double",
        "triple",
        "dozen",
    }
)


def _numbers(text: str) -> set[str]:
    return {d for n in re.findall(r"\d[\d,.]*", text) if (d := re.sub(r"[^\d]", "", n))}


def adds_facts(line: str, script_text: str) -> list[str]:
    """Numbers and names in `line` that the script does not contain."""
    have_words = set(re.findall(r"[a-z]+", script_text.lower()))
    extra = sorted(_numbers(line) - _numbers(script_text))
    for sentence in re.split(r"(?<=[.!?])\s+", line):
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
        for i, word in enumerate(words):
            low = word.lower().split("'")[0]
            is_name = i > 0 and word[0].isupper()  # the first word is capitalized anyway
            if (is_name or low in NUMBER_WORDS) and low not in have_words:
                extra.append(word)
    return extra


def tags_for(source: SourceDoc, topic: str | None) -> list[str]:
    tags = ["#facts", "#learnontiktok", "#didyouknow"]
    words = [r.title for r in source.refs] if source.kind == "wikipedia" else []
    for word in [*words, topic or ""]:
        word = re.sub(r"\s*\([^)]*\)", "", word)  # "Mercury (planet)" -> "Mercury"
        tag = "#" + re.sub(r"[^A-Za-z0-9]", "", word.title())
        if len(tag) > 2 and tag.lower() not in [t.lower() for t in tags]:
            tags.append(tag)
    return tags[:6]


def credit(source: SourceDoc) -> str:
    if source.kind == "wikipedia":
        titles = ", ".join(r.title for r in source.refs)
        return f"Source: Wikipedia ({titles}), CC BY-SA"
    if source.kind == "none":
        return "Opinion video, not fact-checked."
    return f"Source: {source.title}"


def tiktok_caption(
    script: Script,
    source: SourceDoc,
    topic: str | None,
    label: str,
    ai_visuals: bool,
    llm: LLM | None = None,
    model: str | None = None,
) -> str:
    text = "\n".join([script.title, *(s.narration for s in script.segments)])
    line = ""
    if llm is not None:
        try:
            raw = llm.structured(
                [
                    {
                        "role": "user",
                        "content": CAPTION_PROMPT.format(
                            max_chars=MAX_LINE,
                            title=script.title,
                            script="\n".join(s.narration for s in script.segments),
                        ),
                    }
                ],
                CaptionLine,
                model=model,
                temperature=0.7,
            )
            line = " ".join(raw.caption.replace("#", "").split())
            extra = adds_facts(line, text)
            if extra or not line or len(line) > MAX_LINE + 40:
                log.info("caption '%s' rejected (adds: %s); using the hook", line, extra)
                line = ""
        except Exception as exc:  # the caption is a convenience; never fail the video
            log.warning("caption: %s", exc)
            line = ""
    if not line:
        line = script.hook.narration
    parts = [line, ""]
    if label == "UNVERIFIED":
        parts.append("Not fact-checked: opinion/creative video.")
    parts.append(credit(source))
    if ai_visuals:
        parts.append("Visuals are AI-generated.")
    parts += ["", " ".join(tags_for(source, topic))]
    return "\n".join(parts).strip() + "\n"
