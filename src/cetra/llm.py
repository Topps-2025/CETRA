"""Minimal, dependency-light client for the DeepSeek chat completions API.

The client is deliberately small: it uses only the standard library, retries
transient failures, supports JSON mode, and keeps a thread-safe token counter.
It never logs the API key.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding the environment.

    Searches the current working directory and the repository root (two levels
    above this file). This keeps `python -I` runs working without extra
    dependencies.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.append(Path.cwd() / ".env")
        candidates.append(Path(__file__).resolve().parents[2] / ".env")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


@dataclass
class LLMResponse:
    content: str
    reasoning_content: str
    usage: dict[str, Any]
    model: str
    latency_s: float
    raw: dict[str, Any] = field(repr=False)


class TokenCounter:
    """Thread-safe aggregate token accounting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.calls = 0
        self.by_phase: dict[str, dict[str, int]] = {}

    def add(self, usage: dict[str, Any], phase: str = "") -> None:
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("completion_tokens_details") or {}
        reasoning = int(details.get("reasoning_tokens", 0) or 0)
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.reasoning_tokens += reasoning
            self.calls += 1
            phase_stats = self.by_phase.setdefault(
                phase or "unspecified",
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
            )
            phase_stats["calls"] += 1
            phase_stats["prompt_tokens"] += prompt
            phase_stats["completion_tokens"] += completion
            phase_stats["reasoning_tokens"] += reasoning

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "by_phase": {k: dict(v) for k, v in self.by_phase.items()},
            }


class LLMError(RuntimeError):
    """Raised when the API cannot return usable content."""


class DeepSeekClient:
    """Small DeepSeek client with JSON mode and retries."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 180,
        max_retries: int = 5,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise LLMError(
                "DEEPSEEK_API_KEY is not set. Put it in .env or export it before running."
            )
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.counter = TokenCounter()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                data["_latency_s"] = time.time() - started
                return data
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = LLMError(f"HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except Exception as exc:  # noqa: BLE001 - network errors are retried
                last_error = exc
            if attempt < self.max_retries:
                wait = min(30.0, (2**attempt) + random.random())
                time.sleep(wait)
        raise LLMError(f"request failed after {self.max_retries} attempts: {last_error}")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: str | None = "json",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        reasoning_effort: str = "low",
        phase: str = "",
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        raw = self._post(payload)
        if "error" in raw:
            raise LLMError(f"API error: {raw['error']}")
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = raw.get("usage") or {}
        self.counter.add(usage, phase=phase)
        return LLMResponse(
            content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content") or "",
            usage=usage,
            model=raw.get("model", self.model),
            latency_s=float(raw.get("_latency_s", 0.0)),
            raw=raw,
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        reasoning_effort: str = "low",
        phase: str = "",
    ) -> tuple[dict[str, Any], LLMResponse]:
        response = self.chat(
            messages,
            response_format="json",
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            phase=phase,
        )
        text = response.content.strip()
        finish_reason = (response.raw.get("choices") or [{}])[0].get("finish_reason")
        if finish_reason == "length":
            raise LLMError(
                "truncated JSON (finish_reason=length); retry with a larger max_tokens "
                "or a shorter answer"
            )
        if not text:
            raise LLMError(
                "empty content; reasoning_content="
                f"{response.reasoning_content[:300]!r}; finish_reason={finish_reason!r}"
            )
        try:
            return json.loads(text), response
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1]), response
                except json.JSONDecodeError:
                    pass
            raise LLMError(f"could not parse JSON from content: {text[:300]!r}")
