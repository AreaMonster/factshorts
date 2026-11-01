"""Generate the video script from the source text only (RAG), as validated JSON."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from factshorts.config import ScriptConfig, TTSConfig
from factshorts.core.llm import LLM
from factshorts.models import Claim as ClaimModel
from factshorts.models import ClaimType, Idea, SceneType, Script, Segment, SourceDoc
from factshorts.verify.normalize import split_paragraphs, split_sentences

log = logging.getLogger(__name__)


class LLMSentence(BaseModel):
    text: str = Field(description="One narration sentence of 8 to 16 words")
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
    scenes: list[LLMSegment] = Field(min_length=1, max_length=16)
    outro: LLMSegment


class LLMMoreScenes(BaseModel):
    scenes: list[LLMSegment] = Field(min_length=1, max_length=8)


TONES = {
    "neutral": "clear, calm and confident",
    "funny": "light and playful with gentle jokes; jokes never change or add facts",
    "dramatic": "suspenseful and vivid, like a documentary trailer, without exaggerating facts",
    "kid-friendly": "simple words and short sentences for children aged 8 to 12",
}

SYSTEM_PROMPT = """You write scripts for vertical fact videos (TikTok, YouTube Shorts) that are \
about {length} seconds long.

STRICT RULES
1. Use only facts stated in SOURCE. If unsure, leave it out. Do not add outside knowledge.
2. Narration is a list of sentences. For every sentence that states a fact, put in
   source_quotes the complete SOURCE sentence(s) that prove it, copied character-for-character
   without the [n] label. Do not shorten, merge or paraphrase quotes.
3. Every number, year, date and name in a sentence must appear in its source_quotes.
4. Sentences with empty source_quotes must not contain any fact, number or name
   (e.g. "But here is the twist." or "Follow for more facts.").
5. Metric units only: meters, kilometers, kilograms, degrees Celsius, km/h. Never feet, miles,
   pounds or Fahrenheit. Write numbers as digits. You may round a source number
   (8,848.86 m -> 8,849 meters) but never change it.
6. American English spelling.
7. Write exactly {scenes} scenes. Each scene has 1-2 sentences with about {wps} words in
   total, and on_screen_text of at most 8 words that only repeats facts from that scene.
8. The hook is one punchy sentence that makes viewers keep watching. It must not
   misrepresent the facts (no clickbait).
9. The outro is one short closing line without new facts (empty source_quotes).
10. Total narration: about {budget} words and never more (hook + all scenes + outro). Use many
    different facts from SOURCE; do not repeat a fact.
11. Tone: {tone_desc}. Tone may change wording, never facts.
12. visual per scene, one of: big_number (one striking number), comparison (two values),
    timeline (dates), map_pin (a place), icon_list (a few items), quote (a notable sentence),
    image. The hook and outro use "title".
Reply with JSON only."""

USER_PROMPT = """IDEA: {idea}

SOURCE ({title}, Wikipedia):
{source}"""

TOPUP_PROMPT = """The script is too short: {words} words, but it needs about {budget}.
Write {count} MORE scenes (same JSON fields and rules) with NEW facts from SOURCE that the \
script does not use yet. Do not repeat any fact. Return only the new scenes.{avoid}"""


class ScriptError(RuntimeError):
    pass


def target_scenes(length_s: int, cfg: ScriptConfig) -> int:
    return max(cfg.min_scenes, min(cfg.max_scenes, round(length_s / cfg.seconds_per_scene)))


def word_budget(length_s: int, cfg: ScriptConfig, tts: TTSConfig) -> int:
    """Words that fit in `length_s` at the configured speaking rate, minus pause time."""
    segments = target_scenes(length_s, cfg) + 2
    pause_s = (
        segments * (tts.scene_pause_ms + tts.sentence_pause_ms) + tts.lead_in_ms + tts.tail_ms
    ) / 1000
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


def to_segment(s: LLMSegment, visual: SceneType | None = None) -> Segment:
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


def to_script(raw: LLMScript) -> Script:
    script = Script(
        title=raw.title.strip(),
        hook=to_segment(raw.hook, "title"),
        scenes=[to_segment(s) for s in raw.scenes],
        outro=to_segment(raw.outro, "title"),
    )
    return script.assign_ids()


def _end_sentence(text: str) -> str:
    text = " ".join(text.split())
    return text if text[-1:] in ".!?" else f"{text}."


class ScriptWriter:
    """Holds the conversation with the LLM so follow-up calls (top-ups) keep the context."""

    def __init__(
        self, idea: Idea, source: SourceDoc, llm: LLM, model: str, cfg: ScriptConfig, tts: TTSConfig
    ) -> None:
        self.idea, self.source, self.llm, self.model, self.cfg = idea, source, llm, model, cfg
        self.budget = word_budget(idea.length, cfg, tts)
        self.scenes = target_scenes(idea.length, cfg)
        self.wps = max(8, round((self.budget - 22) / self.scenes))  # hook ~14 + outro ~8
        system = SYSTEM_PROMPT.format(
            length=idea.length,
            budget=self.budget,
            scenes=self.scenes,
            wps=self.wps,
            tone_desc=TONES[idea.tone],
        )
        user = USER_PROMPT.format(
            idea=idea.text,
            title=source.title,
            source=format_source(source, cfg.max_source_words),
        )
        self.base = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def draft(self, feedback: list[str] | None = None) -> Script:
        messages = list(self.base)
        if feedback:
            messages[1] = {
                "role": "user",
                "content": messages[1]["content"]
                + "\n\nA previous script failed fact-checking. Avoid these problems (use "
                "different, verifiable facts or quote the source exactly):\n- "
                + "\n- ".join(feedback[:12]),
            }
        best: Script | None = None
        for attempt in range(1, self.cfg.max_json_retries + 1):
            raw = self.llm.structured(
                messages, LLMScript, model=self.model, retries=self.cfg.max_json_retries
            )
            script = to_script(raw)
            problems = check_shape(script, self.budget, self.cfg)
            log.info(
                "script attempt %d: %d words (budget %d), %d scenes%s",
                attempt,
                script.word_count(),
                self.budget,
                len(script.scenes),
                f", problems: {problems}" if problems else "",
            )
            if not problems:
                best = script
                break
            if best is None or script.word_count() < best.word_count():
                best = script
            messages = [
                *messages[:2],
                {"role": "assistant", "content": raw.model_dump_json()},
                {
                    "role": "user",
                    "content": "Fix these problems and return the full JSON "
                    "again: " + "; ".join(problems),
                },
            ]
        assert best is not None
        script = fit_budget(best, self.budget, self.cfg)
        return self.top_up(script)

    def top_up(
        self, script: Script, avoid: list[str] | None = None, target_words: int | None = None
    ) -> Script:
        """Add scenes with new facts until the narration reaches the word budget
        (or `target_words`, used when the spoken result came out too short)."""
        target = target_words or self.budget
        for _ in range(self.cfg.max_topups):
            words = script.word_count()
            missing = target - words
            room = self.cfg.max_scenes - len(script.scenes)
            if missing <= 0 or room <= 0:
                break
            count = max(1, min(room, round(missing / self.wps)))
            hint = ""
            if avoid:
                hint = (
                    "\nThese earlier sentences failed fact-checking; do not reuse them:\n- "
                    + "\n- ".join(avoid[:10])
                )
            messages = [
                *self.base,
                {"role": "assistant", "content": script_to_json(script)},
                {
                    "role": "user",
                    "content": TOPUP_PROMPT.format(
                        words=words, budget=target, count=count, avoid=hint
                    ),
                },
            ]
            more = self.llm.structured(
                messages, LLMMoreScenes, model=self.model, retries=self.cfg.max_json_retries
            )
            new = [to_segment(s) for s in more.scenes][:room]
            script.scenes.extend(new)
            script.assign_ids()
            log.info(
                "top-up: +%d scenes, now %d words (budget %d)",
                len(new),
                script.word_count(),
                self.budget,
            )
        if script.word_count() < self.budget * 0.6:
            log.warning(
                "narration has only %d words (budget %d); video will be short",
                script.word_count(),
                self.budget,
            )
        return script


def script_to_json(script: Script) -> str:
    """Script in the LLM's own JSON shape, so follow-up calls see what they produced."""

    def seg(s: Segment) -> dict:
        return {
            "sentences": [
                {
                    "text": t,
                    "source_quotes": [c.source_quote for c in s.claims if c.text == t],
                    "claim_type": next((c.type for c in s.claims if c.text == t), "general"),
                }
                for t in split_sentences(s.narration)
            ],
            "on_screen_text": s.on_screen_text,
            "visual": s.visual,
        }

    return json.dumps(
        {
            "title": script.title,
            "hook": seg(script.hook),
            "scenes": [seg(s) for s in script.scenes],
            "outro": seg(script.outro),
        }
    )


def generate_script(
    idea: Idea,
    source: SourceDoc,
    llm: LLM,
    model: str,
    cfg: ScriptConfig,
    tts: TTSConfig,
    feedback: list[str] | None = None,
) -> Script:
    return ScriptWriter(idea, source, llm, model, cfg, tts).draft(feedback)


def check_shape(script: Script, budget: int, cfg: ScriptConfig) -> list[str]:
    problems = []
    words = script.word_count()
    if words > budget * (1 + cfg.budget_tolerance):
        problems.append(f"narration has {words} words, the maximum is {budget}")
    if len(script.scenes) > cfg.max_scenes:
        problems.append(f"{len(script.scenes)} scenes; use at most {cfg.max_scenes} scenes")
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
