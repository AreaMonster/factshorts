"""Health checks shared by `factshorts doctor` and the Settings page's health panel."""

from __future__ import annotations

import contextlib
import ipaddress
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx

from factshorts.config import Config

Status = Literal["ok", "warn", "fail"]

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
# Values that look like credentials for a paid/cloud service. This tool must never need one.
KEY_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),  # OpenAI/Anthropic style
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),  # Google
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),  # Hugging Face token
    re.compile(r"\b(ghp|gho|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abp]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS
]
KEY_NAMES = re.compile(r"(api_?key|secret|token|access_key)", re.I)


@dataclass
class Check:
    name: str
    status: Status
    detail: str


def is_local_host(host: str) -> bool:
    if host.lower() in LOCAL_HOSTS or host.lower().endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def config_problems(cfg: Config) -> list[str]:
    """Anything that looks like an API key, or an LLM endpoint that is not on this machine."""
    problems = []
    host = urlparse(cfg.ollama.url).hostname or ""
    if not is_local_host(host):
        problems.append(
            f"[ollama].url points to '{host}', not this computer; factshorts only uses a local LLM"
        )

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(f"{prefix}[{i}]", v)
        elif isinstance(value, str) and prefix != "web.password_hash":
            if any(p.search(value) for p in KEY_PATTERNS):
                problems.append(f"{prefix} looks like an API key; no keys are needed or allowed")
            elif KEY_NAMES.search(prefix.rsplit(".", 1)[-1]) and value:
                problems.append(f"{prefix} is set; factshorts uses no API keys or tokens")

    walk("", cfg.model_dump())
    return problems


def _hf_cached(repo: str, filename: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo, filename), str)
    except Exception:
        return False


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path):
            pass
        return True
    except OSError:
        return False


def run_checks(cfg: Config, timeout: float = 3.0) -> list[Check]:
    checks: list[Check] = []
    add = lambda name, ok, detail, warn=False: checks.append(  # noqa: E731
        Check(name, "ok" if ok else "warn" if warn else "fail", detail)
    )

    v = sys.version_info
    add("Python", v >= (3, 11), f"{v.major}.{v.minor}.{v.micro} (3.11+ needed)")

    from factshorts.render import ffmpeg

    tools = [shutil.which(t) for t in ("ffmpeg", "ffprobe")]
    with contextlib.suppress(Exception):
        tools = [ffmpeg.ffmpeg_bin(), ffmpeg.ffprobe_bin()]
    add(
        "ffmpeg",
        all(tools),
        "found" if all(tools) else "ffmpeg/ffprobe not found (winget install ffmpeg)",
    )
    if all(tools):
        hw = [name for name, codec in ffmpeg.HW_ENCODERS.items() if ffmpeg.encoder_works(codec)]
        add(
            "Hardware encoder",
            bool(hw),
            ", ".join(hw) if hw else "none works; libx264 (CPU) will be used, which is slower",
            warn=True,
        )

    models: list[str] = []
    try:
        r = httpx.get(cfg.ollama.url.rstrip("/") + "/api/tags", timeout=timeout)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        add("Ollama", True, f"reachable at {cfg.ollama.url} ({len(models)} models)")
    except Exception as exc:
        add("Ollama", False, f"not reachable at {cfg.ollama.url}: {exc}. Start the Ollama app.")
    if models:
        for role, name in (
            ("Writer model", cfg.ollama.writer_model),
            ("Judge model", cfg.ollama.judge_model),
        ):
            present = name in models or f"{name}:latest" in models
            add(role, present, name if present else f"{name} is not pulled: ollama pull {name}")

    if cfg.tts.engine == "kokoro":
        try:
            import kokoro  # noqa: F401

            cached = _hf_cached("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
            add(
                "Voice (Kokoro)",
                True,
                f"{cfg.tts.voice}; model "
                + ("downloaded" if cached else "downloads on first use (about 330 MB)"),
            )
        except ImportError:
            add("Voice (Kokoro)", False, "the kokoro package is not installed")
    else:
        from factshorts.core.tts import piper_voice_files

        try:
            rel, _ = piper_voice_files(cfg.tts.piper_voice)
            present = (cfg.models_dir / "piper" / rel).is_file()
            add(
                "Voice (Piper)",
                present,
                cfg.tts.piper_voice + ("" if present else ": downloads on first use"),
                warn=True,
            )
        except ValueError as exc:
            add("Voice (Piper)", False, str(exc))

    if cfg.verification.nli_check:
        cached = _hf_cached(cfg.verification.nli_model, "config.json")
        add(
            "NLI model",
            cached,
            cfg.verification.nli_model + ("" if cached else ": downloads on first use"),
            warn=True,
        )
    if cfg.visuals.mode != "off":
        cached = _hf_cached(cfg.visuals.image_model, "model_index.json")
        add(
            "Image model",
            cached,
            cfg.visuals.image_model + ("" if cached else ": downloads on first use (large)"),
            warn=True,
        )
        try:
            import torch

            gpu = torch.cuda.is_available()
            name = torch.cuda.get_device_name(0) if gpu else "no CUDA GPU"
            add("GPU", gpu, name + ("" if gpu else "; illustrations will be very slow"), warn=True)
        except Exception as exc:
            add("GPU", False, f"torch not usable: {exc}", warn=True)

    for label, path in (("Data folder", cfg.data_dir), ("Output folder", cfg.output_dir)):
        ok = _writable(path)
        add(label, ok, str(path) + ("" if ok else " is not writable"))

    problems = config_problems(cfg)
    add("Config", not problems, "; ".join(problems) or "no API keys, local LLM only")
    if not is_local_host(cfg.web.host) and not cfg.web.password_hash:
        add(
            "Web UI",
            False,
            f"host {cfg.web.host} is reachable from the network but no password is set",
        )
    return checks
