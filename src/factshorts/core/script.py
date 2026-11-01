"""Generate the video script from the source text only (RAG), as validated JSON."""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from factshorts.config import ScriptConfig, TTSConfig
from factshorts.core.llm import LLM
from factshorts.models import Claim as ClaimModel
from factshorts.models import ClaimType, Idea, SceneType, Script, Segment, SourceDoc
from factshorts.verify.normalize import normalize_chars, split_paragraphs, split_sentences

log = logging.getLogger(__name__)


class LLMSentence(BaseModel):
    text: str = Field(description="One narration sentence, 5 to 20 words, plain words")
    source_quotes: list[str] = Field(
        description="Sentences copied character-for-character from SOURCE that prove this "
        "sentence; empty only if the sentence states no fact",
        max_length=2,
    )
    claim_type: ClaimType = Field(description="number, date, name or general")


class LLMSegment(BaseModel):
    sentences: list[LLMSentence] = Field(min_length=1, max_length=4)
    on_screen_text: str
    visual: SceneType


class LLMScript(BaseModel):
    story_plan: list[str] = Field(
        description="The story beats in order: the question, each step that leads to the "
        "answer, and the payoff. One short line each.",
        min_length=3,
        max_length=8,
    )
    title: str
    hook: LLMSegment
    scenes: list[LLMSegment] = Field(min_length=1, max_length=16)
    outro: LLMSegment


class LLMNewScene(LLMSegment):
    insert_after: str = Field(description="Id of the segment this scene follows (hook, s1, ...)")


class LLMMoreScenes(BaseModel):
    scenes: list[LLMNewScene] = Field(min_length=1, max_length=8)


TONES = {
    "neutral": "clear, calm and confident",
    "funny": "light and playful with gentle jokes; jokes never change or add facts",
    "dramatic": "suspenseful and vivid, like a documentary trailer, without exaggerating facts",
    "kid-friendly": "simple words and short sentences for children aged 8 to 12",
}

SYSTEM_PROMPT = """You write scripts for vertical fact videos (TikTok, YouTube Shorts) that are \
about {length} seconds long. Viewers should enjoy watching it to the end and understand
everything the first time they hear it.

STORY
A. Tell ONE story that answers the IDEA. Do not list facts. Plan the beats first in
   story_plan: open with the most surprising fact or the question, then build up step by step,
   each scene following from the one before, to the answer, then a short payoff.
B. Every sentence connects to the previous one, like a good teacher talking: "But...",
   "That's because...", "So...", "And it gets stranger:", "Here's the thing:". Only say
   "because" or "that's why" when SOURCE itself states that cause.
C. Plain English for a curious 12-year-old: short sentences, everyday words, one idea per
   sentence. If a technical term is needed (e.g. "greenhouse effect"), explain it in simple
   words that SOURCE supports.
D. Choose facts that help answer the IDEA; skip trivia that does not fit the story.
E. Vary how sentences start; do not begin every sentence with the subject's name.
G. Explain only what SOURCE explains. Never describe how a process works (how heat gets
   trapped, what happens over time, what something "means") unless SOURCE says it. A simple
   true story told with SOURCE's own facts beats a clever explanation SOURCE does not support.
F. Never make the same point twice. Every scene adds something new; the payoff is said once,
   and the outro is a short sign-off that does not repeat it.

FACT RULES (strict)
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
7. Write exactly {scenes} scenes. Each scene has 2-3 sentences with about {wps} words in
   total, and on_screen_text of at most 8 words that only repeats facts from that scene.
8. The hook is 1-2 punchy sentences that make viewers keep watching (a surprising fact or a
   question the video answers). It must not misrepresent the facts (no clickbait).
9. The outro is one short closing line without new facts (empty source_quotes).
10. Total narration: about {budget} words and never more (hook + all scenes + outro).
    Do not repeat a fact.
11. Tone: {tone_desc}. Tone may change wording, never facts.
12. visual per scene, one of: big_number (one striking number), comparison (two values),
    timeline (dates), map_pin (a place), icon_list (a few items), quote (a notable sentence),
    image. The hook and outro use "title".
Reply with JSON only."""

CREATIVE_PROMPT = """You write scripts for vertical videos (TikTok, YouTube Shorts) that are \
about {length} seconds long. This is an opinion, story or creative video and it is NOT
fact-checked, so do not state specific facts, numbers, dates or names as true; present things
as opinions, ideas, questions or a story. Leave every source_quotes list empty.

Tell ONE engaging story or argument with natural transitions, plain words for a curious
12-year-old, and no repeated points. Write exactly {scenes} scenes of 2-3 sentences (about
{wps} words each) with on_screen_text of at most 8 words; a 1-2 sentence hook and a short
outro; about {budget} words in total. Tone: {tone_desc}. visual per scene, one of:
big_number, comparison, timeline, map_pin, icon_list, quote, image (hook and outro: "title").
Write story_plan first. Reply with JSON only."""

USER_PROMPT = """IDEA: {idea}

SOURCE ({title}):
{source}"""

REWRITE_PROMPT = """Segment {seg_id} of the script failed fact-checking:
- {problems}

Rewrite only segment {seg_id} so that every sentence is fully supported by SOURCE (same rules:
quotes copied exactly, every number and name inside its quote, metric units, nothing beyond
what the quote says). The fix is usually to say LESS: state the fact plainly, the way SOURCE
says it, and drop explanations or causes SOURCE does not state. Keep the same topic if SOURCE
supports it; otherwise use a different fact from SOURCE that the script does not use yet.
It must still flow in the story: it follows "{before}" and leads into "{after}".
{role}Return only the rewritten segment."""

ROLE_HINTS = {
    "hook": "This is the hook: one punchy sentence that does not misrepresent facts. ",
    "outro": "This is the outro: one short closing line without facts (empty source_quotes). ",
    "scene": "Write 2-3 sentences. ",
}

TOPUP_PROMPT = """The script is too short: {words} words, but it needs about {budget}.
Write {count} MORE scenes (same JSON fields and rules) with NEW facts from SOURCE that the \
script does not use yet and that deepen the story. For each, set insert_after to the id of the
segment it should follow so the story still flows (e.g. "s2"). Do not repeat any fact. Return
only the new scenes.{avoid}"""

POLISH_PROMPT = """This script passed fact-checking. Now make it flow like one story that a
curious 12-year-old enjoys from start to end: natural transitions between sentences and scenes,
simpler words, varied sentence openings, technical terms explained simply.
If two sentences say the same thing, keep only the better one. Keep every remaining fact,
number and name exactly as it is, keep each sentence's source_quotes unchanged, add no new
facts, and keep the same segments in the same order. Return the full
script JSON."""


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


def _is_tag_on(sentence: str) -> bool:
    """Short connective like "Here's why." that carries no fact of its own."""
    return len(sentence.split()) <= 4 and not re.search(r"\d|[A-Z][a-z]+.*[A-Z]", sentence)


def to_segment(s: LLMSegment, visual: SceneType | None = None) -> Segment:
    """Build a Segment. If the LLM packed several sentences into one entry, its quotes go to
    the factual sentences only; short tag-ons ("Here's why.") are left as claim-free filler."""
    narration: list[str] = []
    claims: list[ClaimModel] = []
    for x in s.sentences:
        parts = [_end_sentence(t) for t in split_sentences(x.text)] if x.text.strip() else []
        narration += parts
        quotes = [q.strip() for q in x.source_quotes if q.strip()]
        factual = [t for t in parts if not (len(parts) > 1 and _is_tag_on(t))] or parts
        claims += [
            ClaimModel(text=t, source_quote=q, type=x.claim_type) for t in factual for q in quotes
        ]
    return Segment(
        narration=" ".join(narration),
        on_screen_text=s.on_screen_text.strip(),
        visual=visual or s.visual,
        claims=claims,
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
        self,
        idea: Idea,
        source: SourceDoc,
        llm: LLM,
        model: str,
        cfg: ScriptConfig,
        tts: TTSConfig,
        note: str | None = None,
    ) -> None:
        self.idea, self.source, self.llm, self.model, self.cfg = idea, source, llm, model, cfg
        self.budget = word_budget(idea.length, cfg, tts)
        self.scenes = target_scenes(idea.length, cfg)
        self.wps = max(8, round((self.budget - 22) / self.scenes))  # hook ~14 + outro ~8
        template = SYSTEM_PROMPT if idea.factcheck else CREATIVE_PROMPT
        system = template.format(
            length=idea.length,
            budget=self.budget,
            scenes=self.scenes,
            wps=self.wps,
            tone_desc=TONES[idea.tone],
        )
        user = USER_PROMPT.format(
            idea=idea.text,
            title=f"{source.title}, Wikipedia"
            if source.kind == "wikipedia"
            else f"{source.title}, provided by the user",
            source=format_source(source, cfg.max_source_words) or "(none)",
        )
        if note:
            user += (
                f"\n\nIMPORTANT: {note} Make the video gently correct this misconception "
                "using SOURCE, and never present the false assumption as true."
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

    def rewrite_segment(self, script: Script, seg_id: str, problems: list[str]) -> Script:
        """Targeted rewrite of one failing segment; the rest of the script stays as it is."""
        role = "hook" if seg_id == "hook" else "outro" if seg_id == "outro" else "scene"
        segs = script.segments
        idx = next(i for i, x in enumerate(segs) if x.id == seg_id)
        before = segs[idx - 1].narration if idx > 0 else "(the start of the video)"
        after = segs[idx + 1].narration if idx + 1 < len(segs) else "(the end of the video)"
        messages = [
            *self.base,
            {"role": "assistant", "content": script_to_json(script)},
            {
                "role": "user",
                "content": REWRITE_PROMPT.format(
                    seg_id=seg_id,
                    problems="\n- ".join(problems[:8]) or "unsupported",
                    role=ROLE_HINTS[role],
                    before=before,
                    after=after,
                ),
            },
        ]
        raw = self.llm.structured(
            messages, LLMSegment, model=self.model, retries=self.cfg.max_json_retries
        )
        new = to_segment(raw, "title" if role != "scene" else None)
        log.info("rewrote %s: %s", seg_id, new.narration)
        return replace_segment(script, seg_id, new)

    def polish(self, script: Script) -> Script:
        """Editing pass for flow and plain language. The result must be re-verified."""
        messages = [
            *self.base,
            {"role": "assistant", "content": script_to_json(script)},
            {"role": "user", "content": POLISH_PROMPT},
        ]
        raw = self.llm.structured(
            messages, LLMScript, model=self.model, retries=self.cfg.max_json_retries
        )
        polished = to_script(raw)
        if len(polished.scenes) != len(script.scenes):
            log.info("polish changed the number of scenes; keeping the original")
            return script
        return polished

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
            new = [(x.insert_after.strip().lower(), to_segment(x)) for x in more.scenes]
            new = [(after, seg) for after, seg in new if not _repeats(seg, script)][:room]
            for after, seg in new:
                ids = [x.id for x in script.scenes]  # new scenes have no id yet
                pos = 0 if after == "hook" else ids.index(after) + 1 if after in ids else len(ids)
                script.scenes.insert(pos, seg)
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


def replace_segment(script: Script, seg_id: str, new: Segment) -> Script:
    if seg_id == "hook":
        script.hook = new
    elif seg_id == "outro":
        script.outro = new
    else:
        script.scenes = [new if s.id == seg_id else s for s in script.scenes]
    return script.assign_ids()


def _repeats(seg: Segment, script: Script) -> bool:
    """A new scene that only restates sentences already in the script adds nothing."""
    existing = [t.casefold() for s in script.segments for t in split_sentences(s.narration)]
    new = [t.casefold() for t in split_sentences(seg.narration)]
    return bool(new) and all(any(fuzz.ratio(n, e) >= 85 for e in existing) for n in new)


def dedupe_script(script: Script, min_scenes: int) -> tuple[Script, int]:
    """Remove sentences that repeat an earlier one (with their claims) and scenes left without
    any fact. The outro is kept as it is. Returns the script and how many sentences went."""
    script = script.model_copy(deep=True)
    seen = [normalize_chars(t).casefold() for t in split_sentences(script.outro.narration)]
    removed = 0
    for seg in [script.hook, *script.scenes]:
        kept: list[str] = []
        for t in split_sentences(seg.narration):
            key = normalize_chars(t).casefold()
            if any(fuzz.ratio(key, s) >= 85 for s in seen):
                removed += 1
                continue
            kept.append(t)
            seen.append(key)
        seg.narration = " ".join(kept)
        keep_keys = {normalize_chars(t).casefold() for t in kept}
        seg.claims = [c for c in seg.claims if normalize_chars(c.text).casefold() in keep_keys]
    factual = [s for s in script.scenes if s.narration and s.claims]
    if len(factual) >= min_scenes:
        script.scenes = factual
    else:
        script.scenes = [s for s in script.scenes if s.narration]
    if not script.hook.narration and script.scenes:
        script.hook = script.scenes.pop(0).model_copy(update={"visual": "title"})
    return script.assign_ids(), removed


def script_to_json(script: Script) -> str:
    """Script in the LLM's own JSON shape, so follow-up calls see what they produced."""

    def seg(s: Segment) -> dict:
        return {
            "id": s.id,
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
            "story_plan": [],
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
