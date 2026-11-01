"""Load and validate config.toml."""

from __future__ import annotations

import os
import sys
import tomllib
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tone = Literal["neutral", "funny", "dramatic", "kid-friendly"]
EncoderPref = Literal["auto", "nvenc", "amf", "qsv", "libx264"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneralConfig(_Section):
    data_dir: str = ""
    log_level: str = "INFO"


class OllamaConfig(_Section):
    url: str = "http://127.0.0.1:11434"
    writer_model: str = "qwen3:14b"
    judge_model: str = "gemma4:26b"
    temperature: float = 0.4
    num_ctx: int = 12288
    timeout_s: float = 600
    think: bool = False


class WikipediaConfig(_Section):
    language: str = "en"
    user_agent: str = "factshorts/0.1 (local fact-video tool; contact: you@example.com)"
    max_sections: int = 3
    focus_sections: int = 2  # extra sections chosen by how well they match the idea
    skip_sections: list[str] = Field(
        default_factory=lambda: [
            "See also",
            "References",
            "Notes",
            "External links",
            "Further reading",
            "Bibliography",
            "Sources",
            "Citations",
        ]
    )
    search_limit: int = 5


class CacheConfig(_Section):
    http_ttl_hours: float = 168


class ScriptConfig(_Section):
    default_length: int = Field(default=75, ge=20, le=180)
    default_tone: Tone = "neutral"
    words_per_minute: int = 160
    seconds_per_scene: float = 9.0
    min_scenes: int = 4
    max_scenes: int = 16
    max_topups: int = 2
    polish: bool = True  # editing pass for flow after fact-checking (re-verified)
    # Videos targeting >= 60 s must end up at least this long (TikTok pays for > 1 minute).
    min_duration_s: float = 62.0
    max_json_retries: int = 3
    budget_tolerance: float = 0.10
    max_source_words: int = 4000


class WikidataProperty(_Section):
    label: str
    keywords: list[str]


DEFAULT_WIKIDATA_PROPERTIES = {
    "P2044": WikidataProperty(
        label="elevation above sea level",
        keywords=["tall", "high", "height", "elevation", "altitude", "above sea level", "summit"],
    ),
    "P2048": WikidataProperty(label="height", keywords=["tall", "high", "height"]),
    "P2043": WikidataProperty(label="length", keywords=["long", "length"]),
    "P2049": WikidataProperty(label="width", keywords=["wide", "width"]),
    "P2386": WikidataProperty(label="diameter", keywords=["diameter", "across"]),
    "P2120": WikidataProperty(label="radius", keywords=["radius"]),
    "P4511": WikidataProperty(label="vertical depth", keywords=["deep", "depth"]),
    "P2046": WikidataProperty(label="area", keywords=["area", "covers", "square"]),
    "P1082": WikidataProperty(
        label="population", keywords=["population", "inhabitants", "residents", "people live"]
    ),
    "P2067": WikidataProperty(label="mass", keywords=["weigh", "weight", "mass", "heavy"]),
    "P2583": WikidataProperty(
        label="distance from Earth", keywords=["from earth", "away from earth", "away from us"]
    ),
    "P571": WikidataProperty(
        label="inception",
        keywords=[
            "founded",
            "built",
            "established",
            "created",
            "formed",
            "constructed",
            "completed",
        ],
    ),
    "P569": WikidataProperty(label="date of birth", keywords=["born", "birth"]),
    "P570": WikidataProperty(label="date of death", keywords=["died", "death"]),
    "P575": WikidataProperty(label="time of discovery", keywords=["discovered", "discovery"]),
    "P1619": WikidataProperty(label="date of official opening", keywords=["opened"]),
}


class VerificationConfig(_Section):
    quote_check: bool = True
    entity_check: bool = True
    wikidata_check: bool = True
    nli_check: bool = True
    llm_judge: bool = True
    max_rounding_error: float = 0.05
    # A near-copy of a source sentence (this similar, 0-100) is replaced by the exact sentence
    # before checking; the claim is then verified against the real text. 0 disables it.
    snap_quote_min_ratio: float = 80
    conversion_tolerance: float = 0.005
    # Rounds of targeted scene rewrites before a failing scene is dropped (or the idea rejected).
    max_rewrite_rounds: int = 3
    premise_check: bool = True  # stop with needs_input if the source contradicts the idea
    max_script_attempts: int = 3
    require_review: bool = True
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    nli_device: str = "cpu"
    nli_threshold: float = 0.5
    wikidata_properties: dict[str, WikidataProperty] = Field(
        default_factory=lambda: dict(DEFAULT_WIKIDATA_PROPERTIES)
    )


class TTSConfig(_Section):
    engine: Literal["kokoro", "piper"] = "kokoro"
    voice: str = "am_michael"
    speed: float = 1.0
    piper_voice: str = "en_US-ryan-high"
    length_scale: float = 1.0
    sentence_pause_ms: int = 250
    scene_pause_ms: int = 450
    lead_in_ms: int = 150
    tail_ms: int = 600
    use_whisper: bool = False
    pronunciations: dict[str, str] = Field(default_factory=dict)


class RenderConfig(_Section):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    theme: str = "default"
    encoder: EncoderPref = "auto"
    workers: int = 0
    max_hw_sessions: int = 4
    video_quality: int = 20
    audio_bitrate: str = "192k"
    lufs_target: float = -14.0


class VisualsConfig(_Section):
    # parallax: AI illustration + 3D depth camera move on every scene (default).
    mode: Literal["hybrid", "parallax", "wan", "kenburns", "off"] = "parallax"
    image_model: str = "Tongyi-MAI/Z-Image-Turbo"
    image_steps: int = 9
    image_width: int = 768
    image_height: int = 1344
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    video_model: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    wan_scenes: int = 4
    wan_include_hook: bool = True
    wan_steps: int = 30
    wan_width: int = 480
    wan_height: int = 832
    wan_frames: int = 81
    wan_fps: int = 24
    min_playback_speed: float = 0.55
    crossfade_frames: int = 8
    device: str = "cuda"
    # A vision model checks every illustration against its narration; mismatches are redrawn.
    review_images: bool = True
    review_model: str = ""  # empty = [ollama].judge_model
    max_redraws: int = 2


class IdeasConfig(_Section):
    topics: list[str] = Field(
        default_factory=lambda: [
            "space",
            "human body",
            "animals",
            "geography",
            "history",
            "technology",
            "Switzerland",
        ]
    )
    per_topic: int = 2  # auto ideas per topic in a batch run


class OutputConfig(_Section):
    dir: str = "output"


class WebConfig(_Section):
    host: str = "127.0.0.1"
    port: int = 8000
    password_hash: str = ""


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    wikipedia: WikipediaConfig = Field(default_factory=WikipediaConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    script: ScriptConfig = Field(default_factory=ScriptConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    visuals: VisualsConfig = Field(default_factory=VisualsConfig)
    ideas: IdeasConfig = Field(default_factory=IdeasConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    # Directory the config file was loaded from; relative paths resolve against it.
    base_dir: Path = Field(default=PROJECT_ROOT, exclude=True)

    @cached_property
    def data_dir(self) -> Path:
        if self.general.data_dir:
            p = Path(self.general.data_dir).expanduser()
            return p if p.is_absolute() else (self.base_dir / p).resolve()
        return default_data_dir()

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def output_dir(self) -> Path:
        p = Path(self.output.dir).expanduser()
        return p if p.is_absolute() else self.data_dir / p

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def models_dir(self) -> Path:
        return self.cache_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "factshorts.db"

    def ensure_dirs(self) -> None:
        # Hugging Face downloads (voices, image/video models) go to the data folder too.
        os.environ.setdefault("HF_HOME", str(self.models_dir / "hf"))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        for d in (self.work_dir, self.output_dir, self.cache_dir, self.models_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        for sub in ("pending", "approved", "rejected"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)


def default_data_dir() -> Path:
    if sys.platform == "win32":
        drive = os.environ.get("SYSTEMDRIVE", "C:")
        return Path(f"{drive}\\") / "factshorts-data"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "factshorts"


def find_config_file() -> Path | None:
    env = os.environ.get("FACTSHORTS_CONFIG")
    if env:
        return Path(env)
    for candidate in (Path.cwd() / "config.toml", PROJECT_ROOT / "config.toml"):
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path | None = None) -> Config:
    path = path or find_config_file()
    if path is None:
        return Config()
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    cfg = Config.model_validate(raw)
    cfg.base_dir = path.resolve().parent
    return cfg
