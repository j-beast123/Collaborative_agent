"""Reward computation for one trajectory, for a training backend to call."""

from __future__ import annotations

from typing import Any, Optional

from src.common.config import IntegrationUnavailable
from src.effectiveness.history import history_at
from src.effectiveness.providers import EffectivenessProvider, build_transform
from src.judges.agent_state import is_collaboration_turn
from src.judges.reward import RewardSpec, TurnSignals, compute_reward


def load_reward_spec(config: dict) -> RewardSpec:
    """Reward settings from config["reward"]; weights are placeholders to tune."""
    section = config["reward"]
    weights = section["weights"]
    transform = None
    if weights.get("effectiveness", 0) > 0:
        name = (section.get("effectiveness") or {}).get("transform")
        if name is None:
            raise IntegrationUnavailable("w_E > 0 requires reward.effectiveness.transform")
        transform = build_transform(name)
    aggregation = config.get("rubric_aggregation") or {}  # how the rubrics of one response are combined
    return RewardSpec(weights=weights, tau=section["softmin_temperature"], effectiveness_transform=transform,
                      missing_component_policy=section.get("missing_component_policy", "renormalize"),
                      rubric_aggregation=aggregation.get("method", "weighted_mean"), tau_rubric=aggregation.get("temperature"),
                      tau_effectiveness=(section.get("effectiveness") or {}).get("softmin_temperature"))


def trajectory_reward(
    trajectory: dict,
    labels_by_item: dict[str, dict],
    spec: RewardSpec,
    *,
    include_initial_turn: bool,
    provider: Optional[EffectivenessProvider] = None,
    rubric_weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """labels_by_item: {"<trajectory_id>:turn<i>": {rubric_id: NA|0|1|2}} from the rubric scorer."""
    if str(trajectory.get("protocol", "")).startswith("agentclinic"):
        raise IntegrationUnavailable("AgentClinic rewards must use src.grpo.reward.SerlReward (A1-only calibrated predictor)")
    if spec.weights["effectiveness"] > 0 and provider is None:
        raise IntegrationUnavailable("w_E > 0 requires an effectiveness provider")
    turns = []
    for index in range(len(trajectory.get("turns", []))):
        if not is_collaboration_turn(trajectory, index, include_initial_turn=include_initial_turn):
            continue
        item_id = f"{trajectory['trajectory_id']}:turn{index}"
        if item_id not in labels_by_item:
            raise ValueError(f"no rubric scores for {item_id}")
        delta = None
        if spec.weights["effectiveness"] > 0:
            history = dict(history_at(trajectory, index), ground_truth=trajectory.get("ground_truth"))
            delta = provider.predict_delta(history, trajectory["turns"][index]["text"])
        turns.append(TurnSignals(turn_id=index, rubric_labels=labels_by_item[item_id], delta_effectiveness=delta))
    outcome = 1.0 if trajectory.get("is_correct") else 0.0
    result = compute_reward(turns, outcome, spec, rubric_weights)
    result.update(trajectory_id=trajectory["trajectory_id"], scored_turns=[t.turn_id for t in turns],
                  effectiveness_provider=provider.identity() if provider else None)
    return result
