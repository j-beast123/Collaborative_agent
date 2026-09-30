"""Represent paired interaction states and their value differences."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .effectiveness import effectiveness


@dataclass
class PairedState:
    pair_id: str
    case_id: str
    parent_state_id: str
    before_state_id: str
    after_state_id: str
    transition_label: str
    value_before: float | None = None
    value_after: float | None = None
    effectiveness: float | None = None

    def finalize(self, value_before: float, value_after: float) -> None:
        self.value_before = value_before
        self.value_after = value_after
        self.effectiveness = effectiveness(value_before, value_after)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
