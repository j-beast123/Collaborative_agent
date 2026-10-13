"""History / response representation shared by labels, the model and providers."""

from __future__ import annotations

from typing import Any, Optional

HISTORY_KEYS = ("history_id", "case_id", "original_case_id", "source_trajectory_id", "question", "options",
                "scenario", "claim_order", "belief_enabled", "round", "turns")


def history_at(trajectory: dict, turn_index: int) -> dict[str, Any]:
    """History before the agent turn at turn_index of a trajectory."""
    turns = trajectory["turns"]
    turn = turns[turn_index]
    if turn.get("speaker") != "agent" or turn.get("final"):
        raise ValueError(f"turn {turn_index} is not an agent response turn")
    ordinal = sum(1 for t in turns[: turn_index + 1] if t.get("speaker") == "agent")
    return {
        "history_id": f"{trajectory['trajectory_id']}:turn{turn_index}",
        "case_id": str(trajectory["case_id"]),
        "original_case_id": str(trajectory.get("original_case_id", trajectory["case_id"])),
        "source_trajectory_id": str(trajectory["trajectory_id"]),
        "question": trajectory.get("question", ""),
        "options": trajectory.get("options", {}),
        "scenario": trajectory.get("scenario", ""),
        "claim_order": trajectory.get("claim_order"),
        "belief_enabled": trajectory.get("belief_enabled", True),
        "round": int(turn.get("round", ordinal)),
        "turns": [{k: t.get(k) for k in ("speaker", "text", "round")} for t in turns[:turn_index]],
    }


INPUT_FORMATS = ("v1", "turn_v1")
TURNS = tuple(f"A{turn}" for turn in range(1, 10))  # the Agent turns a turn-conditioned predictor can be asked about


def format_input(history: dict, response: Optional[str] = None, input_format: str = "v1",
                 turn: Optional[str] = None) -> str:
    """Model input. BEFORE: [CASE] + [HISTORY]. AFTER: additionally [AGENT RESPONSE]."""
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"input_format must be one of {INPUT_FORMATS}")
    options = "\n".join(f"{key}. {value}" for key, value in (history.get("options") or {}).items())
    dialogue = "\n".join(f"{t['speaker'].title()}: {t['text']}" for t in history["turns"]) or "(no dialogue yet)"
    state = "before_response" if response is None else "after_response"
    parts = [f"[STATE] {state}"]
    if input_format == "turn_v1":
        turn = turn or history.get("intervention_turn") or "A1"
        if turn not in TURNS:
            raise ValueError(f"intervention turn must be one of {TURNS}")
        parts.append(f"Intervention turn: {turn}")
    parts += [
        "[CASE]",
        f"Scenario: {history.get('scenario') or 'No separate patient vignette is provided.'}",
        f"Question: {history.get('question', '')}",
        f"Answer choices:\n{options}",
        "[HISTORY]",
        dialogue,
    ]
    if response is not None:
        parts += ["[AGENT RESPONSE]", response]
    return "\n".join(parts)


def continuation_trajectory(history: dict, ground_truth: dict, *, max_rounds: int,
                            final_decision_role: str, response: Optional[str] = None) -> tuple[dict, int]:
    """(trajectory, agent_turn_index) that src/collaboration/rerun.py can continue from."""
    turns = [dict(t) for t in history["turns"]]
    turns.append({"speaker": "agent", "text": response or "", "round": history["round"]})
    trajectory = {
        "case_id": history["case_id"], "trajectory_id": history["source_trajectory_id"],
        "question": history["question"], "options": history["options"], "scenario": history["scenario"],
        "claim_order": history["claim_order"], "belief_enabled": history["belief_enabled"],
        "ground_truth": ground_truth, "max_discussion_rounds": max_rounds,
        "final_decision_role": final_decision_role, "turns": turns,
    }
    return trajectory, len(turns) - 1
