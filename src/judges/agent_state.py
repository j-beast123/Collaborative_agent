"""Agent-side state computed before each agent response."""

from __future__ import annotations

from typing import Any, Optional

from src.belief.local_judgment import map_judgment_to_roles

STATE_FIELDS = (
    "focal_claim", "agent_current_position", "clinician_current_position",
    "agent_evidence_so_far", "clinician_evidence_so_far", "unresolved_points",
)


def is_collaboration_turn(trajectory: dict, turn_index: int, *, include_initial_turn: bool = False) -> bool:
    """Agent turn in the scored scope."""
    turns = trajectory["turns"]
    if not 0 <= turn_index < len(turns) or turns[turn_index].get("speaker") != "agent":
        return False
    if turns[turn_index].get("final"):
        return False
    ordinal = sum(1 for turn in turns[: turn_index + 1] if turn.get("speaker") == "agent")
    return include_initial_turn or int(turns[turn_index].get("round", ordinal)) >= 2


def resolve_include_initial_turn(config: dict) -> bool:
    """scoring.include_initial_turn from config (METHOD_TBD)."""
    from src.common.config import IntegrationUnavailable

    value = (config.get("scoring") or {}).get("include_initial_turn")
    if value is None:
        if config.get("mode") == "production":
            raise IntegrationUnavailable("METHOD_TBD: set scoring.include_initial_turn (true/false) for production runs")
        return False
    if type(value) is not bool:
        raise ValueError("scoring.include_initial_turn must be true or false")
    return value


def pre_response_state(trajectory: dict, turn_index: int) -> Optional[dict[str, Any]]:
    """Agent-side state before the agent turn at turn_index."""
    turn = trajectory["turns"][turn_index]
    if turn.get("speaker") != "agent":
        raise ValueError("pre_response_state is defined for agent turns")
    if "pre_response_state" in turn:  # stored with the turn (AgentClinic trajectories)
        return turn["pre_response_state"]
    status = turn.get("local_judgment_status")
    if status == "not_applicable_turn_1":
        return {
            "state_status": "initial_turn_fixed",
            "note": "First round: the agent gives its initial answer; belief/judgment is fixed and not computed.",
        }
    judgment = turn.get("local_judgment")
    if status != "valid" or not isinstance(judgment, dict):
        return None
    claim_order = trajectory.get("claim_order")
    if claim_order not in ("doctor_first", "agent_first"):
        # A1/A2 cannot be mapped to agent/clinician; say so rather than guess.
        return {"state_status": "unavailable_missing_claim_order"}
    mapped = map_judgment_to_roles(judgment, claim_order)
    reconstruction = turn.get("reconstruction") or {}
    return {
        "state_status": "computed",
        **{name: reconstruction.get(name) for name in STATE_FIELDS},
        "better_supported_position": mapped["q1"]["better_supported_position"],
        "better_supported_reason": mapped["q1"]["reason"],
        "evidence_sufficient": mapped["q2"]["answer"] == "Y",
        "evidence_sufficiency_reason": mapped["q2"]["reason"],
    }
