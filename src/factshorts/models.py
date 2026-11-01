"""Typed data passed between pipeline steps."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from factshorts.config import Tone

SceneType = Literal[
    "big_number", "comparison", "timeline", "map_pin", "icon_list", "image", "quote", "title"
]
ClaimType = Literal["number", "date", "name", "general"]
CheckStatus = Literal["pass", "fail", "skipped", "na"]  # na = check does not apply
SegmentRole = Literal["hook", "scene", "outro"]


def utcnow() -> datetime:
    return datetime.now(UTC)


# --- Ideas -----------------------------------------------------------------


class Idea(BaseModel):
    text: str
    length: int = Field(default=75, ge=20, le=180)  # target seconds
    tone: Tone = "neutral"
    factcheck: bool = True
    force: bool = False
    source_url: str | None = None
    source_text: str | None = None
    source_file: str | None = None
    origin: Literal["custom", "auto"] = "custom"
    topic: str | None = None


# --- Sources ---------------------------------------------------------------


class Section(BaseModel):
    index: int
    title: str
    text: str


class SourceDoc(BaseModel):
    kind: Literal["wikipedia", "user"]
    title: str
    url: str | None = None
    language: str = "en"
    revision_id: int | None = None
    page_id: int | None = None
    wikidata_id: str | None = None
    sections: list[Section]
    sha256: str
    fetched_at: datetime = Field(default_factory=utcnow)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)

    @property
    def revision_url(self) -> str | None:
        if self.kind != "wikipedia" or self.revision_id is None:
            return self.url
        return f"https://{self.language}.wikipedia.org/w/index.php?oldid={self.revision_id}"


# --- Script ----------------------------------------------------------------


class Claim(BaseModel):
    id: str = ""
    text: str
    source_quote: str
    type: ClaimType = "general"


class Segment(BaseModel):
    """Hook, a scene, or the outro. All are spoken and all are verified."""

    id: str = ""
    role: SegmentRole = "scene"
    narration: str
    on_screen_text: str
    visual: SceneType = "title"
    claims: list[Claim] = Field(default_factory=list)
    image_prompt: str = ""  # what the AI illustration shows (generic, no real people)
    motion: str = ""  # what moves when the scene is animated
    visual_interest: int = 3  # 1-5: how much the scene gains from real AI motion


class Script(BaseModel):
    title: str
    hook: Segment
    scenes: list[Segment]
    outro: Segment

    @property
    def segments(self) -> list[Segment]:
        return [self.hook, *self.scenes, self.outro]

    def assign_ids(self) -> Script:
        self.hook.id, self.hook.role = "hook", "hook"
        self.outro.id, self.outro.role = "outro", "outro"
        for i, scene in enumerate(self.scenes, start=1):
            scene.id, scene.role = f"s{i}", "scene"
        for seg in self.segments:
            for j, claim in enumerate(seg.claims, start=1):
                claim.id = f"{seg.id}.c{j}"
        return self

    def word_count(self) -> int:
        return sum(len(seg.narration.split()) for seg in self.segments)


# --- Verification ----------------------------------------------------------


class CheckResult(BaseModel):
    check: str
    status: CheckStatus
    score: float | None = None
    reason: str = ""


class ClaimVerdict(BaseModel):
    claim_id: str
    segment_id: str
    text: str
    source_quote: str
    paragraph: str | None = None
    checks: list[CheckResult]
    passed: bool


class SentenceVerdict(BaseModel):
    segment_id: str
    sentence: str
    claim_ids: list[str]
    kind: Literal["claim", "filler", "unsupported"]
    checks: list[CheckResult]
    passed: bool


class Verification(BaseModel):
    enabled_checks: list[str]
    claims: list[ClaimVerdict]
    sentences: list[SentenceVerdict]
    passed: bool
    failures: list[str] = Field(default_factory=list)
    attempt: int = 1

    @property
    def skipped_checks(self) -> list[str]:
        """Enabled checks that did not actually run where they should have."""
        skipped = {c.check for v in self.claims for c in v.checks if c.status == "skipped"}
        if any(
            c.check == "filler" and c.status == "skipped" for s in self.sentences for c in s.checks
        ):
            skipped.add("llm_judge")  # the judge could not classify a claim-less sentence
        return sorted(skipped & set(self.enabled_checks))

    @property
    def label(self) -> Literal["VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"]:
        if not self.passed:
            return "UNVERIFIED"
        return "PARTIALLY_VERIFIED" if self.skipped_checks else "VERIFIED"


# --- Audio -----------------------------------------------------------------


class WordTiming(BaseModel):
    text: str
    start: float
    end: float


class SentenceAudio(BaseModel):
    text: str
    spoken: str
    wav: str
    start: float
    duration: float
    words: list[WordTiming]


class SegmentAudio(BaseModel):
    segment_id: str
    start: float
    end: float
    sentences: list[SentenceAudio]


class AudioPlan(BaseModel):
    voice: str
    sample_rate: int
    wav: str
    duration: float
    segments: list[SegmentAudio]


# --- Visuals ---------------------------------------------------------------


class SceneVisual(BaseModel):
    segment_id: str
    kind: Literal["wan", "parallax", "kenburns", "none"]
    frames: int
    prompt: str = ""
    motion: str = ""
    seed: int = 0
    image: str | None = None  # file name in work/<job>/visuals
    clip: str | None = None  # background clip with exactly `frames` frames
    review: str = ""  # vision check outcome, e.g. "ok", "redrawn 1x", "fallback: ..."


class VisualsPlan(BaseModel):
    mode: str
    style: str = ""
    models: dict[str, str] = Field(default_factory=dict)
    scenes: list[SceneVisual] = Field(default_factory=list)

    def clip_for(self, segment_id: str) -> str | None:
        return next((s.clip for s in self.scenes if s.segment_id == segment_id), None)


# --- Output ----------------------------------------------------------------


class RenderResult(BaseModel):
    video: str
    duration: float
    frames: int
    encoder: str


class ExportResult(BaseModel):
    folder: str
    video: str
    sources_json: str
    metadata_json: str
    description_txt: str
    srt: str | None = None
    verified: bool
