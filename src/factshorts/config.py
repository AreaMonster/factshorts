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
    judge_model: str = "llama3.1:8b"
    temperature: float = 0.4
    num_ctx: int = 12288
    timeout_s: float = 600
    think: bool = False


class WikipediaConfig(_Section):
    language: str = "en"
    user_agent: str = "factshorts/0.1 (local fact-video tool; contact: you@example.com)"
    max_sections: int = 3
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
    default_length: Literal[30, 45, 60] = 45
    default_tone: Tone = "neutral"
    words_per_minute: int = 150
    min_scenes: int = 4
    max_scenes: int = 7
    max_json_retries: int = 3
    budget_tolerance: float = 0.10
    max_source_words: int = 4000


class VerificationConfig(_Section):
    quote_check: bool = True
    entity_check: bool = True
    wikidata_check: bool = True
    nli_check: bool = True
    llm_judge: bool = True
    max_rounding_error: float = 0.05
    conversion_tolerance: float = 0.005
    max_script_attempts: int = 3
    require_review: bool = True


class TTSConfig(_Section):
    voice: str = "en_US-ryan-high"
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
