"""Interfaces for controlled future continuation rollouts."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable
import json


@dataclass
class RolloutRecord:
    state_id: str
    case_id: str
    rollout_index: int
    seed: int
    final_answer: str
    success: bool
    model_name: str
    simulator_name: str
    prompt_version: str
    generation_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_rollouts(
    *,
    state: dict[str, Any],
    num_rollouts: int,
    continuation_fn: Callable[[dict[str, Any], int], RolloutRecord],
) -> list[RolloutRecord]:
    """Run independent continuations from exactly the supplied state.

    The caller owns the model/simulator implementation and final-answer
    evaluator. Keeping that dependency explicit prevents accidental mixing of
    continuation policies across value estimates.
    """
    if num_rollouts != 9:
        raise ValueError("Production effectiveness targets require exactly K=9 rollouts")
    records = [continuation_fn(state, index) for index in range(9)]
    if len({record.rollout_index for record in records}) != 9:
        raise ValueError("Rollout indices must be unique")
    if len({record.seed for record in records}) != 9:
        raise ValueError("Rollout seeds must be unique")
    if any(record.state_id != state["state_id"] for record in records):
        raise ValueError("Rollout state identity mismatch")
    if len({record.model_name for record in records}) != 1 or len({record.simulator_name for record in records}) != 1:
        raise ValueError("Rollout policy metadata is inconsistent")
    if len({record.prompt_version for record in records}) != 1:
        raise ValueError("Rollout prompt versions are inconsistent")
    if len({json.dumps(record.generation_config, sort_keys=True) for record in records}) != 1:
        raise ValueError("Rollout generation configs are inconsistent")
    return records
