"""API usage ledger: one JSONL row per paid chat call, and its cost summary."""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

ENV, STAGE_ENV, ITERATION_ENV = "API_USAGE_LOG", "API_USAGE_STAGE", "API_USAGE_ITERATION"
GROUPS = ("model", "stage", "purpose")
_lock = threading.Lock()
_stage = threading.local()                  # per-thread role label (with_role); falls back to the env variable


def current_stage():
    return getattr(_stage, "value", None) or os.environ.get(STAGE_ENV)
_case: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("api_usage_case", default=None)


def openrouter(base_url: str) -> bool:
    return "openrouter.ai" in base_url


@contextmanager
def case(case_id: Optional[str]):
    """Calls made inside (in this thread) are recorded with the case id."""
    token = _case.set(str(case_id) if case_id is not None else None)
    try:
        yield
    finally:
        _case.reset(token)


def record(purpose: str, model: str, usage: Optional[dict], attempts: int = 1, *,
           latency_seconds: Optional[float] = None, request_seconds: Optional[float] = None,
           retry_wait_seconds: Optional[float] = None) -> None:
    path = os.environ.get(ENV)
    if not path:
        return
    usage = usage or {}
    iteration = os.environ.get(ITERATION_ENV)
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "purpose": purpose,
           "stage": current_stage(), "iteration": int(iteration) if iteration not in (None, "") else None,
           "case_id": _case.get(), "model": model,
           "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
           "cost_usd": usage.get("cost"), "retry_count": attempts - 1}
    if latency_seconds is not None:
        row["latency_seconds"] = max(0.0, float(latency_seconds))
    if request_seconds is not None:
        row["request_seconds"] = max(0.0, float(request_seconds))
    if retry_wait_seconds is not None:
        row["retry_wait_seconds"] = max(0.0, float(retry_wait_seconds))
    with _lock:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


def summarize(paths: list, prices: Optional[dict] = None) -> dict:
    """Totals of ledgers overall and by model, by stage and by purpose.
    prices: {model: {"input": $, "output": $}} per 1M tokens, for rows without a reported cost."""
    def empty():
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "unpriced_calls": 0, "retries": 0}

    total, groups = empty(), {name: defaultdict(empty) for name in GROUPS}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cost, price = row["cost_usd"], (prices or {}).get(row["model"])
            if cost is None and price and row["input_tokens"] is not None:
                cost = (row["input_tokens"] * price["input"] + (row["output_tokens"] or 0) * price["output"]) / 1e6
            for bucket in [total] + [groups[name][str(row.get(name))] for name in GROUPS]:
                bucket["calls"] += 1
                bucket["input_tokens"] += row["input_tokens"] or 0
                bucket["output_tokens"] += row["output_tokens"] or 0
                bucket["retries"] += row.get("retry_count") or 0
                bucket["cost_usd"] += cost or 0.0
                bucket["unpriced_calls"] += cost is None
    return {"total_usd": total["cost_usd"], **total,
            **{f"by_{name}": dict(sorted(groups[name].items())) for name in GROUPS}}
