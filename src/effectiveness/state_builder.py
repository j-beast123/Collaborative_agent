"""State and paired-state schema helpers."""

from __future__ import annotations

from typing import Any
import hashlib
import json


def make_state(
    *,
    state_id: str,
    case_id: str,
    turn_id: int,
    history: list[dict[str, str]],
    state_type: str,
    parent_state_id: str | None = None,
    response_label: str | None = None,
    question: str = "",
    options: dict[str, str] | None = None,
    trajectory_id: str = "",
) -> dict[str, Any]:
    """Create a leakage-safe state record.

    Outcomes and empirical values are intentionally absent from the model input
    portion of the record and are attached later as supervision metadata.
    """
    if not trajectory_id:
        raise ValueError("trajectory_id is required")
    if state_type not in {"before_agent", "after_agent"}:
        raise ValueError("state_type must be before_agent or after_agent")
    if state_type == "after_agent":
        if not history or history[-1].get("speaker") != "agent":
            raise ValueError("after_agent state must end with one agent response")
        if parent_state_id is None:
            raise ValueError("after_agent state requires parent_state_id")
    pair_id = f"{case_id}:{trajectory_id}:{turn_id}"
    state = {
        "state_id": state_id,
        "case_id": case_id,
        "trajectory_id": trajectory_id,
        "turn_id": turn_id,
        "agent_turn_id": turn_id,
        "pair_id": pair_id,
        "parent_state_id": parent_state_id,
        "state_type": state_type,
        "response_label": response_label,
        "agent_response_text": response_label,
        "question": question,
        "options": options or {},
        "history": history,
    }
    state["state_hash"] = hashlib.sha256(
        json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return state


def validate_pair(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before["pair_id"] != after["pair_id"]:
        raise ValueError("before/after pair_id mismatch")
    if before["case_id"] != after["case_id"] or before.get("trajectory_id") != after.get("trajectory_id"):
        raise ValueError("before/after trajectory identity mismatch")
    for key in ("turn_id", "agent_turn_id", "question", "options", "pair_id"):
        if before.get(key) != after.get(key):
            raise ValueError(f"before/after {key} mismatch")
    if after.get("parent_state_id") != before.get("state_id"):
        raise ValueError("after parent_state_id must point to before state")
    if before["state_type"] != "before_agent" or after["state_type"] != "after_agent":
        raise ValueError("invalid before/after state types")
    if after["history"][:-1] != before["history"]:
        raise ValueError("after state must contain exactly one additional turn")
    if after["history"][-1].get("speaker") != "agent":
        raise ValueError("the additional turn must be an agent response")
    if after.get("agent_response_text") != after["history"][-1].get("text"):
        raise ValueError("agent_response_text must match final history turn")


def build_agent_state_pair(trajectory: dict[str, Any], agent_turn_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an auditable pair around one original agent turn."""
    turns = trajectory.get("turns", [])
    if not 0 <= agent_turn_index < len(turns) or turns[agent_turn_index].get("speaker") != "agent":
        raise ValueError("agent_turn_index must point to an agent turn")
    common = {
        "case_id": trajectory["case_id"],
        "trajectory_id": trajectory.get("trajectory_id", ""),
        "question": trajectory["question"],
        "options": trajectory.get("options", {}),
    }
    turn_id = int(turns[agent_turn_index].get("turn_id", agent_turn_index))
    before = make_state(
        state_id=f"{common['trajectory_id']}:{turn_id}:before",
        turn_id=turn_id,
        history=[dict(turn) for turn in turns[:agent_turn_index]],
        state_type="before_agent",
        **common,
    )
    after = make_state(
        state_id=f"{common['trajectory_id']}:{turn_id}:after",
        turn_id=turn_id,
        history=[dict(turn) for turn in turns[: agent_turn_index + 1]],
        state_type="after_agent",
        parent_state_id=before["state_id"],
        response_label=turns[agent_turn_index].get("text", ""),
        **common,
    )
    validate_pair(before, after)
    return before, after
