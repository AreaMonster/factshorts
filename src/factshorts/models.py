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
CheckStatus = Literal["pass", "fail", "skipped"]
SegmentRole = Literal["hook", "scene", "outro"]


def utcnow() -> datetime:
    return datetime.now(UTC)


# --- Ideas -----------------------------------------------------------------


class Idea(BaseModel):
    text: str
    length: Literal[30, 45, 60] = 45
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
        """Enabled checks that did not actually run on some claim."""
        return sorted(
            {c.check for v in self.claims for c in v.checks if c.status == "skipped"}
            & set(self.enabled_checks)
        )

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
