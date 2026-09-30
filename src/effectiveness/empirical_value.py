"""Empirical value targets from future continuation outcomes."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import sqrt
from typing import Any


@dataclass
class EmpiricalValue:
    state_id: str
    case_id: str
    turn_id: int
    num_rollouts: int
    num_success: int
    empirical_value: float
    standard_error: float
    confidence_interval_95: list[float]
    continuation_policy: str
    simulator: str
    generation_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def empirical_value(
    *,
    state_id: str,
    case_id: str,
    turn_id: int,
    outcomes: list[int | bool],
    continuation_policy: str,
    simulator: str,
    generation_config: dict[str, Any],
) -> EmpiricalValue:
    """Build a Bernoulli value target and a simple normal CI.

    The raw rollout count is retained because later value fitting can use a
    binomial likelihood rather than treating every target as equally certain.
    """
    if len(outcomes) != 9:
        raise ValueError("Effectiveness targets require exactly 9 continuation outcomes")
    if any(type(outcome) not in (bool, int) or outcome not in (0, 1) for outcome in outcomes):
        raise ValueError("Continuation outcomes must be binary bool/int values")
    successes = sum(int(outcome) for outcome in outcomes)
    count = 9
    value = successes / count
    standard_error = sqrt(value * (1.0 - value) / count)
    margin = 1.96 * standard_error
    return EmpiricalValue(
        state_id=state_id,
        case_id=case_id,
        turn_id=turn_id,
        num_rollouts=count,
        num_success=successes,
        empirical_value=value,
        standard_error=standard_error,
        confidence_interval_95=[max(0.0, value - margin), min(1.0, value + margin)],
        continuation_policy=continuation_policy,
        simulator=simulator,
        generation_config=generation_config,
    )
