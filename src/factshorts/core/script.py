"""Generate the video script from the source text only (RAG), as validated JSON."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from factshorts.config import ScriptConfig, TTSConfig
from factshorts.core.llm import LLM
from factshorts.models import Claim as ClaimModel
from factshorts.models import ClaimType, Idea, SceneType, Script, Segment, SourceDoc
from factshorts.verify.normalize import split_paragraphs, split_sentences

log = logging.getLogger(__name__)


class LLMSentence(BaseModel):
    text: str = Field(description="One short narration sentence")
    source_quotes: list[str] = Field(
        description="Sentences copied character-for-character from SOURCE that prove this "
        "sentence; empty only if the sentence states no fact",
        max_length=2,
    )
    claim_type: ClaimType = Field(description="number, date, name or general")


class LLMSegment(BaseModel):
    sentences: list[LLMSentence] = Field(min_length=1, max_length=3)
    on_screen_text: str
    visual: SceneType


class LLMScript(BaseModel):
    title: str
    hook: LLMSegment
    scenes: list[LLMSegment] = Field(min_length=1, max_length=8)
    outro: LLMSegment


TONES = {
    "neutral": "clear, calm and confident",
    "funny": "light and playful with gentle jokes; jokes never change or add facts",
    "dramatic": "suspenseful and vivid, like a documentary trailer, without exaggerating facts",
    "kid-friendly": "simple words and short sentences for children aged 8 to 12",
}

SYSTEM_PROMPT = """You write scripts for short vertical fact videos (TikTok, YouTube Shorts).

STRICT RULES
1. Use only facts stated in SOURCE. If unsure, leave it out. Do not add outside knowledge.
2. Narration is a list of short sentences. For every sentence that states a fact, put in
   source_quotes the complete SOURCE sentence(s) that prove it, copied character-for-character
   without the [n] label. Do not shorten, merge or paraphrase quotes.
3. Every number, year, date and name in a sentence must appear in its source_quotes.
4. Sentences with empty source_quotes must not contain any fact, number or name
   (e.g. "But here is the twist." or "Follow for more facts.").
5. Metric units only: meters, kilometers, kilograms, degrees Celsius, km/h. Never feet, miles,
   pounds or Fahrenheit. Write numbers as digits. You may round a source number
   (8,848.86 m -> 8,849 meters) but never change it.
6. American English spelling.
7. Each scene: 1-2 short sentences; on_screen_text of at most 8 words that only repeats
   facts from that scene's sentences.
8. The hook is one short, punchy sentence that makes viewers keep watching. It must not
   misrepresent the facts (no clickbait).
9. The outro is one short closing line without new facts (empty source_quotes).
10. Total narration words (hook + all scenes + outro) must be at most {budget}.
    Use {min_scenes} to {max_scenes} scenes.
11. Tone: {tone_desc}. Tone may change wording, never facts.
12. visual per scene, one of: big_number (one striking number), comparison (two values),
    timeline (dates), map_pin (a place), icon_list (a few items), quote (a notable sentence),
    image. The hook and outro use "title".
Reply with JSON only."""

USER_PROMPT = """IDEA: {idea}

SOURCE ({title}, Wikipedia):
{source}"""


class ScriptError(RuntimeError):
    pass


def word_budget(length_s: int, cfg: ScriptConfig, tts: TTSConfig) -> int:
    """Words that fit in `length_s` at the configured speaking rate, minus pause time."""
    segments = (cfg.min_scenes + cfg.max_scenes) // 2 + 2
    pause_s = (segments * tts.scene_pause_ms + tts.lead_in_ms + tts.tail_ms) / 1000
    return int((length_s - pause_s) * cfg.words_per_minute / 60)


def format_source(source: SourceDoc, max_words: int) -> str:
    """Numbered sentences grouped by section, cut at `max_words` (lead first)."""
    lines: list[str] = []
    n = 0
    words = 0
    for section in source.sections:
        lines.append(f"\n## {section.title}")
        for para in split_paragraphs(section.text):
            parts = []
            for sentence in split_sentences(para):
                n += 1
                parts.append(f"[{n}] {sentence}")
                words += len(sentence.split())
            lines.append(" ".join(parts))
            if words >= max_words:
                return "\n".join(lines).strip()
    return "\n".join(lines).strip()


def to_script(raw: LLMScript) -> Script:
    def seg(s: LLMSegment, visual: SceneType | None = None) -> Segment:
        sentences = [x for x in s.sentences if x.text.strip()]
        return Segment(
            narration=" ".join(_end_sentence(x.text) for x in sentences),
            on_screen_text=s.on_screen_text.strip(),
            visual=visual or s.visual,
            claims=[
                ClaimModel(text=_end_sentence(x.text), source_quote=q.strip(), type=x.claim_type)
                for x in sentences
                for q in x.source_quotes
                if q.strip()
            ],
        )

    script = Script(
        title=raw.title.strip(),
        hook=seg(raw.hook, "title"),
        scenes=[seg(s) for s in raw.scenes],
        outro=seg(raw.outro, "title"),
    )
    return script.assign_ids()


def _end_sentence(text: str) -> str:
    text = " ".join(text.split())
    return text if text[-1:] in ".!?" else f"{text}."


def generate_script(
    idea: Idea,
    source: SourceDoc,
    llm: LLM,
    model: str,
    cfg: ScriptConfig,
    tts: TTSConfig,
    feedback: list[str] | None = None,
) -> Script:
    budget = word_budget(idea.length, cfg, tts)
    system = SYSTEM_PROMPT.format(
        budget=budget,
        min_scenes=cfg.min_scenes,
        max_scenes=cfg.max_scenes,
        tone_desc=TONES[idea.tone],
    )
    user = USER_PROMPT.format(
        idea=idea.text, title=source.title, source=format_source(source, cfg.max_source_words)
    )
    if feedback:
        user += (
            "\n\nA previous script failed fact-checking. Avoid these problems "
            "(use different, verifiable facts or quote the source exactly):\n- "
            + "\n- ".join(feedback[:12])
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    best: Script | None = None
    for attempt in range(1, cfg.max_json_retries + 1):
        raw = llm.structured(messages, LLMScript, model=model, retries=cfg.max_json_retries)
        script = to_script(raw)
        problems = check_shape(script, budget, cfg)
        log.info(
            "script attempt %d: %d words (budget %d), %d scenes%s",
            attempt,
            script.word_count(),
            budget,
            len(script.scenes),
            f", problems: {problems}" if problems else "",
        )
        if not problems:
            return script
        if best is None or script.word_count() < best.word_count():
            best = script
        messages = [
            *messages[:2],
            {"role": "assistant", "content": raw.model_dump_json()},
            {
                "role": "user",
                "content": "Fix these problems and return the full JSON again: "
                + "; ".join(problems),
            },
        ]
    assert best is not None
    return fit_budget(best, budget, cfg)


def check_shape(script: Script, budget: int, cfg: ScriptConfig) -> list[str]:
    problems = []
    words = script.word_count()
    if words > budget * (1 + cfg.budget_tolerance):
        problems.append(f"narration has {words} words, the maximum is {budget}")
    elif words < budget * 0.6:
        log.warning("narration has only %d words (budget %d); video will be short", words, budget)
    if not cfg.min_scenes <= len(script.scenes) <= cfg.max_scenes:
        problems.append(
            f"{len(script.scenes)} scenes; use {cfg.min_scenes} to {cfg.max_scenes} scenes"
        )
    long_texts = [s.id for s in script.segments if len(s.on_screen_text.split()) > 8]
    if long_texts:
        problems.append(f"on_screen_text longer than 8 words in {', '.join(long_texts)}")
    return problems


def fit_budget(script: Script, budget: int, cfg: ScriptConfig) -> Script:
    """Last resort: drop scenes from the end until the word budget fits."""
    limit = budget * (1 + cfg.budget_tolerance)
    while script.word_count() > limit and len(script.scenes) > cfg.min_scenes:
        dropped = script.scenes.pop()
        log.info("dropping scene %s to fit word budget", dropped.id)
    if script.word_count() > limit:
        raise ScriptError(
            f"script has {script.word_count()} words, budget is {budget}; the LLM could not "
            "shorten it"
        )
    if len(script.scenes) > cfg.max_scenes:
        script.scenes = script.scenes[: cfg.max_scenes]
    return script.assign_ids()
