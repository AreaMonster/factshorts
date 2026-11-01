"""HTTP client with User-Agent, retries with backoff and an SQLite response cache."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import httpx

from factshorts.db import Database

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpError(RuntimeError):
    pass


class CachedHttp:
    def __init__(
        self,
        user_agent: str,
        db: Database | None = None,
        ttl_hours: float = 168,
        retries: int = 4,
        timeout_s: float = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.db = db
        self.ttl_s = ttl_hours * 3600
        self.retries = retries
        self.client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip"},
            timeout=timeout_s,
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _key(url: str, params: dict[str, Any] | None) -> str:
        blob = json.dumps([url, sorted((params or {}).items())], default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get_text(self, url: str, params: dict[str, Any] | None = None, cache: bool = True) -> str:
        key = self._key(url, params)
        if cache and self.db is not None:
            hit = self.db.cache_get(key, self.ttl_s)
            if hit is not None:
                return hit
        body = self._get_with_retries(url, params)
        if cache and self.db is not None:
            self.db.cache_put(key, url, body)
        return body

    def get_json(self, url: str, params: dict[str, Any] | None = None, cache: bool = True) -> Any:
        return json.loads(self.get_text(url, params, cache))

    def get_bytes(self, url: str) -> bytes:
        resp = self._request(url, None)
        return resp.content

    def _get_with_retries(self, url: str, params: dict[str, Any] | None) -> str:
        return self._request(url, params).text

    def _request(self, url: str, params: dict[str, Any] | None) -> httpx.Response:
        delay = 1.0
        last_error = ""
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.client.get(url, params=params)
                if resp.status_code in RETRY_STATUS:
                    last_error = f"HTTP {resp.status_code}"
                    retry_after = resp.headers.get("Retry-After", "")
                    wait = float(retry_after) if retry_after.isdigit() else delay
                    log.warning("%s for %s, retry %d in %.1fs", last_error, url, attempt, wait)
                    time.sleep(wait)
                    delay *= 2
                    continue
                if resp.status_code >= 400:
                    raise HttpError(f"HTTP {resp.status_code} for {url}")
                return resp
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("network error for %s (%s), retry %d", url, last_error, attempt)
                time.sleep(delay)
                delay *= 2
        raise HttpError(f"giving up on {url} after {self.retries} attempts ({last_error})")
