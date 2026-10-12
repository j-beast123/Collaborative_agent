"""Provider-independent chat calls with retries, usage logging and a JSONL cache."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.common import usage
from src.common.config import IntegrationUnavailable
from src.data.schema import digest

RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ChatRequest:
    system: str
    user: str
    prompt_version: str
    params: dict = field(default_factory=dict)


@dataclass
class ChatResult:
    status: str  # "ok" or "failed"
    text: Optional[str]
    provider: str
    model: str
    revision: Optional[str]
    usage: dict
    latency_s: float
    attempts: int
    error: Optional[str] = None
    cache_hit: bool = False
    started_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChatBackend:
    provider = "abstract"
    model: str = ""
    revision: Optional[str] = None

    def identity(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "revision": self.revision}

    def complete(self, request: ChatRequest) -> ChatResult:
        raise NotImplementedError


def _retry_delay(retry_after: Optional[str], attempt: int) -> float:
    """Retry-After seconds when numeric (it may also be an HTTP date), else capped backoff."""
    try:
        return min(300.0, max(0.0, float(retry_after)))
    except (TypeError, ValueError):
        return min(60.0, 2.0 ** attempt)


class OpenAICompatibleBackend(ChatBackend):
    """POST {base_url}/chat/completions with bounded exponential backoff."""

    provider = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: Optional[str],
        model: Optional[str],
        api_key_env: Optional[str],
        revision: Optional[str] = None,
        max_retries: int = 5,
        timeout_s: float = 180.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not base_url or not model or not api_key_env:
            raise IntegrationUnavailable("Chat backend needs base_url, model and api_key_env in config")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.revision = revision
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.sleep = sleep

    def identity(self) -> dict[str, Any]:
        return dict(super().identity(), base_url=self.base_url)

    def complete(self, request: ChatRequest) -> ChatResult:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise IntegrationUnavailable(f"{self.api_key_env} is not set")
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            **request.params,
            **({"usage": {"include": True}} if usage.openrouter(self.base_url) else {}),  # reports the call's cost
        }).encode()
        started = time.time()
        error = None
        for attempt in range(1, self.max_retries + 2):
            http = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(http, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode())
                text = body["choices"][0]["message"]["content"]
                usage.record(request.prompt_version, body.get("model", self.model), body.get("usage"), attempt)
                return ChatResult(
                    status="ok",
                    text=text,
                    provider=self.provider,
                    model=body.get("model", self.model),
                    revision=self.revision,
                    usage=body.get("usage") or {},
                    latency_s=time.time() - started,
                    attempts=attempt,
                    started_at=started,
                )
            except urllib.error.HTTPError as exc:
                error = f"HTTP {exc.code}"
                if exc.code not in RETRYABLE_HTTP:
                    break
                delay = _retry_delay(exc.headers.get("Retry-After") if exc.headers else None, attempt)
            except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                # OSError covers URLError, timeouts, connection resets and RemoteDisconnected.
                error = f"{type(exc).__name__}: {exc}"
                delay = _retry_delay(None, attempt)
            if attempt <= self.max_retries:
                self.sleep(delay)
        return ChatResult(
            status="failed", text=None, provider=self.provider, model=self.model,
            revision=self.revision, usage={}, latency_s=time.time() - started,
            attempts=attempt, error=error, started_at=started,
        )


class ScriptedBackend(ChatBackend):
    """Offline test double: returns scripted texts in order (or via a function)."""

    provider = "scripted"

    def __init__(self, responses, model: str = "scripted-model") -> None:
        self.responses = responses if callable(responses) else iter(responses)
        self.model = model
        self.calls = []

    def complete(self, request: ChatRequest) -> ChatResult:
        self.calls.append(request)
        text = self.responses(request) if callable(self.responses) else next(self.responses)
        status = "failed" if text is None else "ok"
        return ChatResult(status, text, self.provider, self.model, None, {}, 0.0, 1,
                          error=None if text is not None else "scripted failure")


class CachedChat:
    """Append-only JSONL cache in front of a backend; only successful calls are cached."""

    def __init__(self, backend: ChatBackend, cache_path: Optional[str | Path]) -> None:
        self.backend = backend
        self.path = Path(cache_path) if cache_path else None
        self.entries: dict[str, dict] = {}
        self.lock = threading.Lock()  # callers may share one cache across threads
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry["result"]

    def key(self, request: ChatRequest) -> str:
        return digest({
            "backend": self.backend.identity(),
            "system_sha": digest(request.system),
            "user": request.user,
            "params": request.params,
            "prompt_version": request.prompt_version,
        })

    def complete(self, request: ChatRequest) -> ChatResult:
        key = self.key(request)
        if key in self.entries:
            return ChatResult(**dict(self.entries[key], cache_hit=True))
        result = self.backend.complete(request)
        if result.status == "ok" and self.path:
            with self.lock:
                self.entries[key] = result.to_dict()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "result": result.to_dict()}, ensure_ascii=False) + "\n")
        return result


def parse_json_object(text: Optional[str]) -> tuple[Optional[dict], str]:
    """Strict JSON object parse; tolerates only surrounding code fences/whitespace."""
    if text is None:
        return None, "no_response"
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped[stripped.find("{"):] if "{" in stripped else stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        return None, "invalid_json"
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(value, dict):
        return None, "invalid_json"
    return value, "valid"


OPENROUTER = "https://openrouter.ai/api/v1"  # the endpoint generator.doctor_call uses


def openrouter_chat(model: str, cache_path) -> "CachedChat":
    """Cached chat on an OpenRouter model (OPENROUTER_API_KEY)."""
    return CachedChat(build_chat_backend({"backend": "openai_compatible", "base_url": OPENROUTER, "model_id": model,
                                          "api_key_env": "OPENROUTER_API_KEY", "revision": "openrouter-unpinned"}),
                      cache_path)


def build_chat_backend(settings: dict[str, Any]) -> ChatBackend:
    if settings.get("backend") != "openai_compatible":
        raise IntegrationUnavailable(f"Unsupported chat backend: {settings.get('backend')}")
    return OpenAICompatibleBackend(
        base_url=settings.get("base_url"),
        model=settings.get("model_id"),
        api_key_env=settings.get("api_key_env"),
        revision=settings.get("revision"),
        max_retries=settings.get("max_retries", 5),
    )
