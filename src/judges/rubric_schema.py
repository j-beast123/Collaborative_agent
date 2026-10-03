"""Rubric schema: every score level is self-contained."""

from __future__ import annotations

import re
from typing import Any

LEVELS = ("0", "1", "2")
CONTEXT_FIELDS = ("user_side_context", "agent_side_state")
LEVEL_FIELDS = CONTEXT_FIELDS + ("behavior",)
SCORE_NORMALIZATION = {0: 0.0, 1: 0.5, 2: 1.0}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def build_levels(user_side_context: str, agent_side_state: str, behaviors: dict[str, str]) -> dict[str, dict]:
    """Self-contained levels from one shared context and three behaviors."""
    return {level: {"user_side_context": user_side_context, "agent_side_state": agent_side_state,
                    "behavior": behaviors[level]} for level in LEVELS}


def validate_rubric(rubric: dict[str, Any]) -> list[str]:
    """Problems with a rubric; empty when it follows the schema."""
    problems = []
    if not isinstance(rubric.get("name"), str) or not rubric["name"].strip():
        problems.append("missing name")
    applicability = rubric.get("applicability")
    if not isinstance(applicability, dict) or any(
            not isinstance(applicability.get(f), str) or not applicability[f].strip() for f in CONTEXT_FIELDS):
        return problems + ["applicability needs user_side_context and agent_side_state"]
    scores = rubric.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(LEVELS):
        return problems + ["scores must have exactly levels 0, 1, 2"]
    for level in LEVELS:
        entry = scores[level]
        if not isinstance(entry, dict) or any(not isinstance(entry.get(f), str) or not entry[f].strip()
                                              for f in LEVEL_FIELDS):
            problems.append(f"level {level} needs {', '.join(LEVEL_FIELDS)}")
            continue
        for f in CONTEXT_FIELDS:
            if _norm(entry[f]) != _norm(applicability[f]):
                problems.append(f"level {level} {f} differs from the rubric's applicability "
                                "(a different state belongs in a separate rubric)")
    if not problems and len({_norm(scores[level]["behavior"]) for level in LEVELS}) != 3:
        problems.append("levels 0/1/2 must describe different behaviors")
    return problems


def normalize_score(label: Any) -> float:
    """0 -> 0.0, 1 -> 0.5, 2 -> 1.0 (reward scale)."""
    if label not in SCORE_NORMALIZATION:
        raise ValueError(f"invalid rubric score {label!r}")
    return SCORE_NORMALIZATION[label]
