"""Pluggable decisions of the case tree. A stage that needs a missing one raises UnresolvedDecision; the team-decided
implementations are in src/online_rl/tree/team.py."""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Callable, Optional

class UnresolvedDecision(RuntimeError):
    pass


@dataclass(frozen=True)
class Strategies:
    """multi_target(per_node_R_O) -> float           s(tau) when the trajectory rubric matches several source responses
    r_a_aggregator(scores, rubric_weights) -> float  several raw rubric scores of one response -> R_A
    credit_assignment(tree, leaf_outcomes, node_r_a) -> dict   final outcomes + R_A -> Agent training targets
    agent_update(tree, targets) -> dict              GRPO wiring for those targets
    prompt_unit_value(unit_result) -> float          one trajectory's rubric verification -> its s(tau) in the prompt fitness"""
    multi_target: Optional[Callable] = None
    r_a_aggregator: Optional[Callable] = None
    credit_assignment: Optional[Callable] = None
    agent_update: Optional[Callable] = None
    prompt_unit_value: Optional[Callable] = None

    def need(self, name: str) -> Callable:
        fn = getattr(self, name)
        if fn is None: raise UnresolvedDecision(f'no strategy provided for {name}')
        return fn

    def missing(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is None]
