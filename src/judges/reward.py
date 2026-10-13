"""Composite reward (Step 3), independent of any training backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .rubric_schema import normalize_score

COMPONENTS = ("outcome", "appropriateness", "effectiveness")
RUBRIC_AGGREGATIONS = ("weighted_mean", "weighted_softmin")
MISSING_POLICIES = ("omit", "renormalize", "error")  # omit: the main pipeline; renormalize: the MedMCQA protocol


@dataclass(frozen=True)
class RewardSpec:
    weights: dict  # {"outcome": w_O, "appropriateness": w_A, "effectiveness": w_E}
    tau: float
    effectiveness_transform: Optional[Callable[[float], float]] = None
    missing_component_policy: str = "renormalize"
    tau_effectiveness: Optional[float] = None  # soft-min temperature of the effectiveness turns; tau when None
    rubric_aggregation: str = "weighted_mean"  # how the applicable rubrics of one response are combined
    tau_rubric: Optional[float] = None         # temperature of weighted_softmin

    def __post_init__(self):
        if set(self.weights) != set(COMPONENTS) or any(
                not isinstance(w, (int, float)) or w < 0 for w in self.weights.values()):
            raise ValueError(f"weights need non-negative values for {COMPONENTS}")
        if not sum(self.weights.values()) > 0:
            raise ValueError("at least one reward weight must be positive")
        if not self.tau > 0 or (self.tau_effectiveness is not None and not self.tau_effectiveness > 0):
            raise ValueError("soft-min temperature must be positive")
        if self.weights["effectiveness"] > 0 and self.effectiveness_transform is None:
            raise ValueError("w_E > 0 requires an effectiveness transform")
        if self.missing_component_policy not in MISSING_POLICIES:
            raise ValueError(f"missing_component_policy must be one of {MISSING_POLICIES}")
        if self.rubric_aggregation not in RUBRIC_AGGREGATIONS:
            raise ValueError(f"rubric_aggregation must be one of {RUBRIC_AGGREGATIONS}")
        if self.rubric_aggregation == "weighted_softmin" and not (self.tau_rubric or 0) > 0:
            raise ValueError("weighted_softmin needs a positive rubric temperature")


@dataclass
class TurnSignals:
    turn_id: int
    rubric_labels: dict = field(default_factory=dict)  # rubric_id -> NA|0|1|2
    delta_effectiveness: Optional[float] = None
    effectiveness_applicable: bool = True  # False: the provider does not cover this turn; it has no R^E_t


def weighted_soft_min(scores: list[float], weights: list[float], tau: float) -> Optional[float]:
    """-tau * log(sum_k alpha_k * exp(-s_k / tau)) with alpha_k = w_k / sum(w), computed stably; None for an
    empty list. One score gives that score exactly; equal scores give that score whatever the weights."""
    if not tau > 0:
        raise ValueError("soft-min temperature must be positive")
    total = sum(weights)
    if not scores or not total > 0:
        return None
    exponents = [-s / tau for s in scores]
    top = max(exponents)
    return -tau * (top + math.log(sum(w / total * math.exp(e - top) for w, e in zip(weights, exponents))))


def turn_appropriateness(labels: dict[str, Any], weights: Optional[dict[str, float]] = None,
                         method: str = "weighted_mean", tau: Optional[float] = None) -> Optional[float]:
    """R^A_t of one response from its rubric labels (uniform weights unless weights are given); None if no rubric
    applies.
    """
    applicable = [(normalize_score(label), 1.0 if weights is None else weights.get(rubric_id, 0.0))
                  for rubric_id, label in labels.items() if label != "NA"]
    applicable = [(score, weight) for score, weight in applicable if weight > 0]
    if not applicable:
        return None
    scores, used = [s for s, _ in applicable], [w for _, w in applicable]
    if method == "weighted_softmin":
        return weighted_soft_min(scores, used, tau)
    return sum(w * s for s, w in applicable) / sum(used)


def appropriateness_detail(labels: dict[str, Any], weights: Optional[dict[str, float]] = None,
                           method: str = "weighted_mean", tau: Optional[float] = None) -> dict[str, Any]:
    """What went into a response's appropriateness: the applicable rubrics with their normalized score, raw weight and
    normalized weight alpha, the aggregate, and no_applicable_rubric when nothing applied (the response then has no ...
    """
    applicable = {rubric_id: {"score": normalize_score(label), "weight": 1.0 if weights is None else weights.get(rubric_id, 0.0)}
                  for rubric_id, label in labels.items() if label != "NA"}
    total = sum(entry["weight"] for entry in applicable.values())
    for entry in applicable.values():
        entry["alpha"] = entry["weight"] / total if total > 0 else None
    value = turn_appropriateness(labels, weights, method, tau)
    return {"applicable_rubrics": applicable, "no_match_rubrics": sorted(r for r, label in labels.items() if label == "NA"),
            "aggregation": method, "temperature": tau if method == "weighted_softmin" else None,
            "appropriateness": value, "no_applicable_rubric": value is None}


def soft_min(values: list[float], tau: float) -> Optional[float]:
    """-tau * log(mean(exp(-v / tau))), computed stably; None for an empty list."""
    if not tau > 0:
        raise ValueError("soft-min temperature must be positive")
    if not values:
        return None
    exponents = [-v / tau for v in values]
    top = max(exponents)
    return -tau * (top + math.log(sum(math.exp(e - top) for e in exponents) / len(values)))


def rubric_weights(rubric_set: dict) -> Optional[dict[str, float]]:
    """{rubric_id: weight} of an active rubric set; None (equal weights) when it carries no weights."""
    rubrics = rubric_set.get("rubrics", [])
    return {r["rubric_id"]: r["weight"] for r in rubrics} if rubrics and all("weight" in r for r in rubrics) else None


def weighted_reward(components: dict[str, Optional[float]], spec: RewardSpec) -> dict[str, Any]:
    """{reward, weights_used, unavailable_components} of component values."""
    active = {name: w for name, w in spec.weights.items() if w > 0}
    unavailable = [name for name in active if components[name] is None]
    if unavailable and spec.missing_component_policy == "error":
        raise ValueError(f"reward components unavailable: {unavailable}")
    used = {name: w for name, w in active.items() if components[name] is not None}
    if not used:
        raise ValueError("all positively weighted reward components are unavailable")
    scale = 1.0 if spec.missing_component_policy == "omit" else sum(active.values()) / sum(used.values())
    weights_used = {name: w * scale for name, w in used.items()}
    return {"reward": sum(w * components[name] for name, w in weights_used.items()), "weights_used": weights_used,
            "unavailable_components": unavailable}


def group_mean_imputation(values: list[Optional[float]]) -> list[Optional[float]]:
    """Values of one component over the responses of a group: a missing one becomes the mean of the available ones, so
    a response gains or loses nothing against its group by having no value.
    """
    available = [v for v in values if v is not None]
    return [v if v is not None else (sum(available) / len(available) if available else None) for v in values]


def compute_reward(turns: list[TurnSignals], outcome: float, spec: RewardSpec,
                   rubric_weights: Optional[dict[str, float]] = None, combine: bool = True) -> dict[str, Any]:
    """combine False: the components only (reward None); the caller combines them, e.g. after a group-level
    imputation (weighted_reward)."""
    aggregation = (spec.rubric_aggregation, spec.tau_rubric)
    appropriateness = [turn_appropriateness(t.rubric_labels, rubric_weights, *aggregation) for t in turns]
    components = {"outcome": float(outcome),
                  "appropriateness": soft_min([a for a in appropriateness if a is not None], spec.tau)}
    effectiveness = None
    if spec.weights["effectiveness"] > 0:
        missing = [t.turn_id for t in turns if t.effectiveness_applicable and t.delta_effectiveness is None]
        if missing:
            raise ValueError(f"w_E > 0 but no effectiveness value for turns {missing}")
        for t in turns:
            if t.effectiveness_applicable and not -1.0 <= t.delta_effectiveness <= 1.0:
                raise ValueError(f"delta_effectiveness out of [-1, 1] at turn {t.turn_id}")
        effectiveness = [spec.effectiveness_transform(t.delta_effectiveness) if t.effectiveness_applicable else None
                         for t in turns]
        components["effectiveness"] = soft_min([e for e in effectiveness if e is not None],
                                               spec.tau_effectiveness or spec.tau)
    else:
        components["effectiveness"] = None

    combined = weighted_reward(components, spec) if combine else {"reward": None, "weights_used": {}, "unavailable_components": []}
    return {**combined, "components": components, "turn_appropriateness": appropriateness,
            "turn_appropriateness_details": [appropriateness_detail(t.rubric_labels, rubric_weights, *aggregation) for t in turns],
            "turn_effectiveness": effectiveness}
