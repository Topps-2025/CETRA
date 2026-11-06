"""Minimal DeepSeek client for the memory-construction prototype.

The key is read from the environment or the project ``.env`` file; it is never
printed.  The client uses the OpenAI-compatible ``/chat/completions`` endpoint
and requests JSON output.  The rest of the pipeline treats the LLM as a
pluggable judge, not as an oracle.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parents[3]
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_seconds": round(self.latency_seconds, 3),
        }


@dataclass
class DeepSeekClient:
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    timeout: int = 90
    max_retries: int = 2
    seed: int = 42
    usage: LLMUsage = field(default_factory=LLMUsage)
    last_error: str = ""

    def __post_init__(self) -> None:
        _load_dotenv()
        self.model = os.environ.get("DEEPSEEK_MODEL", self.model)
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", self.base_url).rstrip("/")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    def available(self) -> bool:
        return bool(self.api_key)

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": self.seed,
            "response_format": {"type": "json_object"},
            # DeepSeek Flash enables thinking by default; disable it so the
            # completion budget is spent on JSON, not reasoning tokens.
            "thinking": {"type": "disabled"},
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed = time.time() - started
                self.usage.calls += 1
                self.usage.latency_seconds += elapsed
                if response.status_code != 200:
                    last_error = RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    continue
                body = response.json()
                usage = body.get("usage", {}) or {}
                self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0))
                self.usage.completion_tokens += int(usage.get("completion_tokens", 0))
                content = body["choices"][0]["message"].get("content", "")
                if not content:
                    last_error = RuntimeError("empty content")
                    continue
                return json.loads(content)
            except Exception as exc:  # network, JSON, schema
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        self.last_error = str(last_error)[:300] if last_error else "unknown"
        raise RuntimeError(self.last_error)
