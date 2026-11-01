"""Offline test fixtures: fake LLM, fake TTS, fake Wikipedia, and no network."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

from factshorts.config import Config
from factshorts.core.script import LLMNewScene, LLMScript, LLMSegment, LLMSentence
from factshorts.core.tts import SpokenWord, SynthResult
from factshorts.db import Database
from factshorts.models import Script, SourceDoc

FIXTURES = Path(__file__).parent / "fixtures"

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_send = httpx.Client.send
    allow_local = os.environ.get("FACTSHORTS_MODEL_TESTS") == "1"  # opt-in: local Ollama only
    if allow_local:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # models must already be in the local cache

    def blocked(self: Any, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        if allow_local and request.url.host in ("127.0.0.1", "localhost"):
            return real_send(self, request, **kwargs)
        raise RuntimeError(f"network access in tests: {request.url}")

    monkeypatch.setattr(httpx.Client, "send", blocked)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    c = Config()
    c.general.data_dir = str(tmp_path / "data")
    c.render.encoder = "libx264"
    c.render.workers = 2
    c.visuals.mode = "off"  # AI visuals are tested separately with fake models
    c.visuals.device = "cpu"
    c.script.max_topups = 0
    c.script.polish = False  # the editing pass has its own test
    c.ensure_dirs()
    return c


@pytest.fixture
def db(cfg: Config) -> Database:
    database = Database(cfg.db_path)
    yield database
    database.close()


@pytest.fixture
def source() -> SourceDoc:
    return SourceDoc.model_validate_json((FIXTURES / "source_testa.json").read_text("utf-8"))


@pytest.fixture
def script() -> Script:
    data = json.loads((FIXTURES / "script_testa.json").read_text("utf-8"))
    return Script.model_validate(data).assign_ids()


def script_to_llm(script: Script) -> LLMScript:
    """Express a fixture Script the way the LLM returns it (sentences with quotes)."""
    from factshorts.verify.normalize import split_sentences

    def seg(s: Any) -> LLMSegment:
        sentences = []
        for sentence in split_sentences(s.narration):
            quotes = [c.source_quote for c in s.claims if c.text == sentence]
            ctype = next((c.type for c in s.claims if c.text == sentence), "general")
            sentences.append(LLMSentence(text=sentence, source_quotes=quotes, claim_type=ctype))
        return LLMSegment(sentences=sentences, on_screen_text=s.on_screen_text, visual=s.visual)

    return LLMScript(
        story_plan=["the question", "the explanation", "the payoff"],
        title=script.title,
        hook=seg(script.hook),
        scenes=[seg(s) for s in script.scenes],
        outro=seg(script.outro),
    )


# The judge agrees by default; tests that exercise the judge queue their own answers.
JUDGE_DEFAULTS = {
    "ClaimJudgement": {"reason": "stated in the excerpt", "verdict": "SUPPORTED"},
    "FillerJudgement": {"reason": "a transition", "states_a_fact": False},
}


class FakeNLI:
    """Entailment by word overlap: good enough to tell matching claims from unrelated ones."""

    def __init__(self, fail_on: tuple[str, ...] = ()) -> None:
        self.fail_on = fail_on
        self.calls = 0

    def scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        self.calls += 1
        if any(f in hypothesis for f in self.fail_on):
            return {"entailment": 0.02, "neutral": 0.08, "contradiction": 0.9}
        return {"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01}


class FakeWikidata:
    def __init__(self, values: dict | None = None) -> None:
        self.values = values or {}

    def statements(self, qid: str, properties: list[str]) -> dict:
        return {k: v for k, v in self.values.items() if k in properties}


NEW_FACTS = [  # (narration, quote) pairs from the fixture article not used by the script
    (
        "About 35,000 people visit the park every year.",
        "Around 35,000 people visit the Testa National Park each year.",
    ),
    (
        "A cable car built in 1987 climbs to 3,100 meters.",
        "A cable car built in 1987 carries visitors to a viewing platform at 3,100 m.",
    ),
    (
        "Winds above 150 km/h are common on the north face.",
        "Winds above 150 km/h are common on the north face.",
    ),
    (
        "The town of Velden gets about 1,200 mm of rain a year.",
        "The nearby town of Velden receives about 1,200 mm of rain per year.",
    ),
]


def as_new(seg: LLMSegment, after: str = "") -> LLMNewScene:
    return LLMNewScene(**seg.model_dump(), insert_after=after)


def new_fact_scenes(start: int = 0, count: int = 2, after: str = "") -> list[LLMNewScene]:
    """Top-up scenes with facts the fixture script does not use yet."""
    return [
        LLMNewScene(
            insert_after=after,
            sentences=[LLMSentence(text=t, source_quotes=[q], claim_type="number")],
            on_screen_text="",
            visual="big_number",
        )
        for t, q in NEW_FACTS[start : start + count]
    ]


class FakeLLM:
    """Returns queued responses per schema name; records every call."""

    def __init__(self, responses: dict[str, list[Any]]) -> None:
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def structured(self, messages, schema, model=None, retries=3, temperature=None):
        self.calls.append((schema.__name__, messages))
        if schema.__name__ not in self.responses and schema.__name__ in JUDGE_DEFAULTS:
            return schema.model_validate(JUDGE_DEFAULTS[schema.__name__])
        queue = self.responses[schema.__name__]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return item if isinstance(item, schema) else schema.model_validate(item)


class FakeSynth:
    """Deterministic 'speech': a quiet tone whose length depends on the text, with
    evenly spaced word timestamps like a real engine reports."""

    name = "fake-voice"
    sample_rate = 22050
    settings = "fake"

    def __init__(self, timestamps: bool = True) -> None:
        self.calls: list[str] = []
        self.timestamps = timestamps

    def synthesize(self, text: str) -> SynthResult:
        self.calls.append(text)
        seconds = 0.25 + 0.045 * len(text)
        t = np.arange(int(seconds * self.sample_rate)) / self.sample_rate
        tone = 0.3 * np.sin(2 * math.pi * 220 * t)
        words = text.split()
        step = seconds / max(1, len(words))
        spoken = [SpokenWord(w, i * step, (i + 1) * step) for i, w in enumerate(words)]
        return SynthResult((tone * 32767).astype(np.int16), spoken if self.timestamps else None)


class FakeImageGen:
    """Solid-color 'illustrations', one color per prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompts: list[str], seeds: list[int]) -> list[Image.Image]:
        self.prompts += prompts
        return [Image.new("RGB", (96, 168), ((s * 37) % 256, (s * 91) % 256, 120)) for s in seeds]


class FakeDepth:
    def estimate(self, img: Image.Image) -> np.ndarray:
        return np.tile(np.linspace(0, 1, img.height, dtype=np.float32)[:, None], (1, img.width))

    def close(self) -> None:
        pass


class FakeMotion:
    def __init__(self) -> None:
        self.calls = 0

    def animate(self, image: Image.Image, prompt: str, seed: int) -> list[np.ndarray]:
        self.calls += 1
        base = np.asarray(image.resize((48, 84)))
        return [np.roll(base, i, axis=1) for i in range(17)]

    def close(self) -> None:
        pass


class FakeWiki:
    def __init__(self, doc: SourceDoc) -> None:
        self.doc = doc

    def search(self, query: str, limit: int | None = None) -> list[Any]:
        from factshorts.sources.wikipedia import SearchHit

        return [SearchHit(title=self.doc.title, snippet="", page_id=1)]

    def fetch(self, title: str) -> SourceDoc:
        return self.doc
