"""Jev (TypeSafe "System One") through OpenRouter's Decisions API: typed decisions with probabilities, no text."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from src.common import usage
from src.common.config import IntegrationUnavailable

TYPES = ("choice", "score", "noul")


def http_post(url: str, payload: dict, key: str, timeout_s: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode())


def parse_answers(body: Any, questions: dict[str, dict]) -> dict[str, dict]:
    """{question id: {"value", "probabilities", "confidence"}}; raises when an answer is missing or not of the asked type."""
    answers = body.get("answers") if isinstance(body, dict) and isinstance(body.get("answers"), dict) else None
    if answers is None:
        raise IntegrationUnavailable("Jev response has no `answers` object keyed by question id")
    parsed = {}
    for qid, question in questions.items():
        answer = answers.get(qid)
        kind = question["type"]
        if not isinstance(answer, dict) or (kind not in answer and "value" not in answer):
            raise IntegrationUnavailable(f"Jev response has no {kind} answer for question {qid}")
        value = answer.get(kind, answer.get("value"))
        if kind == "choice" and value not in question["criteria"]:
            raise IntegrationUnavailable(f"Jev answered {value!r} to {qid}, which is not one of its options")
        if kind == "noul" and not (isinstance(value, (int, float)) and 0 <= value <= 1):
            raise IntegrationUnavailable(f"Jev answered {value!r} to the yes/no question {qid}; a probability was expected")
        parsed[qid] = {"value": value, "probabilities": answer.get("probabilities"), "confidence": answer.get("confidence")}
    return parsed


class JevClient:
    def __init__(self, *, endpoint: Optional[str], model: Optional[str], api_key_env: Optional[str], max_retries: int = 5,
                 timeout_s: float = 60.0, post: Optional[Callable] = None, sleep: Callable[[float], None] = time.sleep) -> None:
        if not endpoint or not model or not api_key_env:
            raise IntegrationUnavailable("Jev is not configured: endpoint, model_id and api_key_env are required")
        self.endpoint, self.model, self.api_key_env = endpoint, model, api_key_env
        self.max_retries, self.timeout_s, self.post, self.sleep = max_retries, timeout_s, post or http_post, sleep
        self.calls = 0

    def identity(self) -> dict[str, Any]:
        return {"provider": "jev", "model": self.model, "endpoint": self.endpoint}

    def decide(self, state: str, questions: dict[str, dict], purpose: str = "jev") -> tuple[dict[str, dict], dict]:
        """(answers per question, call record). A transport error is retried; a response of an unknown shape is an error."""
        for qid, question in questions.items():
            if question.get("type") not in TYPES or (question["type"] == "choice" and not isinstance(question.get("criteria"), dict)):
                raise ValueError(f"question {qid} is not a typed Jev question")
        key = os.environ.get(self.api_key_env)
        if not key:
            raise IntegrationUnavailable(f"{self.api_key_env} is not set")
        started, error = time.time(), None
        for attempt in range(1, self.max_retries + 2):
            try:
                body = self.post(self.endpoint, {"model": self.model, "state": state, "questions": questions}, key.strip(), self.timeout_s)
            except urllib.error.HTTPError as exc:
                error = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"
                if exc.code not in (408, 409, 425, 429) and exc.code < 500:  # a request the API rejects is not retried
                    raise IntegrationUnavailable(f"Jev rejected the request: {error}") from None
                if attempt <= self.max_retries:
                    self.sleep(min(2 ** attempt, 30))
                continue
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt <= self.max_retries:
                    self.sleep(min(2 ** attempt, 30))
                continue
            self.calls += 1
            spent = (body.get("usage") or {}) if isinstance(body, dict) else {}
            usage.record(purpose, body.get("model", self.model) if isinstance(body, dict) else self.model,
                         {"prompt_tokens": spent.get("input_tokens"), "completion_tokens": spent.get("output_tokens"),
                          "cost": spent.get("cost")}, attempt)
            return parse_answers(body, questions), {"provider": "jev", "model": body.get("model", self.model), "attempts": attempt,
                                                    "latency_s": time.time() - started, "usage": body.get("usage"), "id": body.get("id")}
        raise IntegrationUnavailable(f"Jev call failed after {self.max_retries + 1} attempts: {error}")
