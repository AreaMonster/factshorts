"""Ollama client: chat with JSON-schema structured output, validation and retries."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from factshorts.config import OllamaConfig

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def structured(
        self,
        messages: list[dict[str, str]],
        schema: type[M],
        model: str | None = None,
        retries: int = 3,
        temperature: float | None = None,
    ) -> M: ...


class OllamaLLM:
    def __init__(self, cfg: OllamaConfig, transport: httpx.BaseTransport | None = None) -> None:
        self.cfg = cfg
        self.client = httpx.Client(
            base_url=cfg.url.rstrip("/"), timeout=cfg.timeout_s, transport=transport
        )

    def list_models(self) -> list[str]:
        resp = self.client.get("/api/tags")
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))

    def unload_all(self) -> None:
        """Free the GPU: ask Ollama to unload every loaded model (keep_alive=0)."""
        try:
            loaded = self.client.get("/api/ps").json().get("models", [])
            for m in loaded:
                self.client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})
        except httpx.HTTPError as exc:
            log.warning("could not unload Ollama models: %s", exc)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        fmt: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.cfg.writer_model,
            "messages": messages,
            "stream": False,
            "think": self.cfg.think,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        if fmt is not None:
            payload["format"] = fmt
        delay = 2.0
        for attempt in range(1, 4):
            try:
                resp = self.client.post("/api/chat", json=payload)
            except httpx.TransportError as exc:
                if attempt == 3:
                    raise LLMError(
                        f"cannot reach Ollama at {self.cfg.url} ({exc}). Is it running?"
                    ) from exc
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code == 404:
                raise LLMError(
                    f"model '{payload['model']}' not found in Ollama;"
                    f" run: ollama pull {payload['model']}"
                )
            if resp.status_code >= 400:
                raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
            return resp.json()["message"]["content"]
        raise LLMError("unreachable")

    def structured(
        self,
        messages: list[dict[str, str]],
        schema: type[M],
        model: str | None = None,
        retries: int = 3,
        temperature: float | None = None,
    ) -> M:
        json_schema = schema.model_json_schema()
        convo = list(messages)
        last_error = ""
        for attempt in range(1, retries + 1):
            raw = self.chat(convo, model=model, fmt=json_schema, temperature=temperature)
            try:
                return schema.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)[:500]
                log.warning("invalid JSON from LLM (attempt %d): %s", attempt, last_error)
                convo = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": "Your JSON was invalid: "
                        f"{last_error}\nReturn corrected JSON only, matching the schema.",
                    },
                ]
        raise LLMError(f"LLM returned invalid JSON {retries} times: {last_error}")
