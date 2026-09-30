"""Dataset utilities for rollout-derived value targets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import hashlib
from typing import Any


@dataclass
class ValueExample:
    state_id: str
    case_id: str
    text: str
    num_rollouts: int
    num_success: int
    empirical_value: float
    trajectory_id: str = ""
    agent_turn_id: int = -1
    pair_id: str = ""
    state_type: str = ""
    state_hash: str = ""
    split: str = ""


def compute_state_hash(state: dict[str, Any]) -> str:
    payload = {key: value for key, value in state.items() if key != "state_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def validate_state(state: dict[str, Any]) -> None:
    required = {"state_id", "case_id", "trajectory_id", "turn_id", "agent_turn_id", "pair_id", "state_type", "scenario", "question", "options", "history", "state_hash"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"State missing required fields: {sorted(missing)}")
    if not state["trajectory_id"] or state["state_type"] not in {"before_agent", "after_agent"}:
        raise ValueError("Invalid trajectory_id or state_type")
    expected_pair = f"{state['case_id']}:{state['trajectory_id']}:{state['turn_id']}"
    if state["pair_id"] != expected_pair:
        raise ValueError("Invalid pair_id")
    if state["state_hash"] != compute_state_hash(state):
        raise ValueError("State content/hash mismatch")


def build_state_text(state: dict[str, Any]) -> str:
    """Serialize only information available at the state being evaluated."""
    forbidden = ("ground_truth", "final_answer", "is_correct", "correct_option", "rollout", "rubric")
    serialized = json.dumps(state, ensure_ascii=False).lower()
    if any(marker in serialized for marker in forbidden):
        raise ValueError("State contains a forbidden outcome/rubric marker")
    history = state.get("history", [])
    dialogue = "\n".join(
        f"{turn.get('speaker', 'unknown')}: {turn.get('text', '')}"
        for turn in history
    ) or "None"
    options = state.get("options", {})
    option_text = "\n".join(f"{key}: {value}" for key, value in options.items())
    return (
        f"Scenario:\n{state.get('scenario', '') or 'No separate patient vignette is provided.'}\n\n"
        f"Question:\n{state.get('question', '')}\n\n"
        f"Answer choices:\n{option_text}\n\n"
        f"Interaction history:\n{dialogue}"
    )


def make_value_example(state: dict[str, Any], target: dict[str, Any]) -> ValueExample:
    """Join a leakage-safe state with an empirical supervision target."""
    validate_state(state)
    for key in ("state_id", "case_id", "state_hash"):
        if key not in target:
            raise ValueError(f"Target missing required field: {key}")
    count = target["num_rollouts"]
    successes = target["num_success"]
    if type(count) is not int or count != 9:
        raise ValueError("num_rollouts must be the integer K=9")
    if type(successes) is not int or not 0 <= successes <= count:
        raise ValueError("num_success must be an integer in [0, 9]")
    if target["state_id"] != state["state_id"]:
        raise ValueError("Target/state_id mismatch")
    if target["case_id"] != state["case_id"]:
        raise ValueError("Target/case_id mismatch")
    if target["state_hash"] != state["state_hash"]:
        raise ValueError("Target/state_hash mismatch")
    empirical = successes / 9
    if "target_value" in target and not math.isclose(float(target["target_value"]), empirical):
        raise ValueError("target_value does not match num_success / 9")
    return ValueExample(
        state_id=str(state["state_id"]),
        case_id=str(state["case_id"]),
        text=build_state_text(state),
        num_rollouts=count,
        num_success=successes,
        empirical_value=empirical,
        trajectory_id=str(state.get("trajectory_id", target.get("trajectory_id", ""))),
        agent_turn_id=int(state.get("agent_turn_id", target.get("agent_turn_id", -1))),
        pair_id=str(state.get("pair_id", target.get("pair_id", ""))),
        state_type=str(state.get("state_type", "")),
        state_hash=str(state.get("state_hash", target.get("state_hash", ""))),
    )


def split_by_case(
    examples: list[ValueExample],
    *,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[ValueExample]]:
    """Split by case_id so states from one case never cross a split."""
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("Ratios must leave a positive test split")
    import random

    case_ids = sorted({example.case_id for example in examples})
    random.Random(seed).shuffle(case_ids)
    train_end = int(len(case_ids) * train_ratio)
    dev_end = train_end + int(len(case_ids) * dev_ratio)
    split_cases = {
        "train": set(case_ids[:train_end]),
        "dev": set(case_ids[train_end:dev_end]),
        "test": set(case_ids[dev_end:]),
    }
    output: dict[str, list[ValueExample]] = {key: [] for key in split_cases}
    for example in examples:
        split = next(name for name, ids in split_cases.items() if example.case_id in ids)
        example.split = split
        output[split].append(example)
    return output
