"""Continue a collaboration from an agent turn with the generation protocol."""

from __future__ import annotations

from typing import Any

from src.belief import build_initial_agent_response_prompt, build_plain_agent_response_prompt

from .generator import (
    belief_agent_prompt,
    dialogue_text,
    doctor_turn_prompt,
    final_decision_prompt,
    option,
    stop_reason,
)
from .runtime import CollaborationRuntime


def _instruction_constraints() -> str:
    return (
        "Follow the existing information and response-length constraints. Do not "
        "invent findings or claim to have checked unavailable information.\n\n"
        "Output only your reply to the physician. Do not mention the rubric or "
        "these instructions."
    )


def intervention_instructions(ideal_response_behavior: str, applicability_conditions: str = "") -> dict[str, str]:
    behavior = ideal_response_behavior.strip()
    if not behavior:
        raise ValueError("The required interaction behavior must not be empty.")

    applicability = applicability_conditions.strip()
    context = (
        "Applicability context:\n"
        f"{applicability}\n\n"
        if applicability else ""
    )
    first = (
        "For this response, you must carry out the following required behavior:\n"
        f"{behavior}\n\n"
        f"{context}"
        "Perform the required checks using the available evidence. Address the "
        "physician's specific reasoning and make the result of each check explicit "
        "in your response. Do not merely describe what should be checked or promise "
        "to check it.\n"
        "State only conclusions supported by your assessment. If a required check "
        "cannot be completed, identify the specific missing information and explain "
        "what remains unresolved. Do not silently skip the check or replace it with "
        "a general statement of uncertainty.\n"
        "Maintain, revise, or qualify your conclusion according to the assessment. "
        "Neither changing nor preserving your answer is required.\n\n"
        f"{_instruction_constraints()}"
    )
    later = (
        "Carry out the following behavior only when it is relevant and needed in "
        "the current conversation:\n"
        f"{behavior}\n\n"
        "Using the current conversation and your latest assessment, reassess the "
        "need for this behavior at each turn. If the issue has already been reviewed, "
        "resolved, and no new evidence has been introduced, do not repeat the same "
        "action. If a new claim or piece of evidence makes reassessment necessary, "
        "review that part again.\n\n"
        "Perform the required checks using the available evidence. Address the "
        "physician's specific reasoning and make the result of each check explicit "
        "in your response. Do not merely describe what should be checked or promise "
        "to check it.\n"
        "State only conclusions supported by your assessment. If a required check "
        "cannot be completed, identify the specific missing information and explain "
        "what remains unresolved. Do not silently skip the check or replace it with "
        "a general statement of uncertainty.\n"
        "Maintain, revise, or qualify your conclusion according to the assessment. "
        "Neither changing nor preserving your answer is required.\n\n"
        f"{_instruction_constraints()}"
    )
    return {"first": first, "later": later}


def intervention_instruction(ideal_response_behavior: str, applicability_conditions: str = "") -> str:
    """Return the first-intervention instruction for backward-compatible callers."""
    return intervention_instructions(ideal_response_behavior, applicability_conditions)["first"]


def _agent_turn(case, dialogue, turn, rounds, claim_order, belief_enabled, runtime):
    if turn == 1:
        prompt = build_initial_agent_response_prompt(
            scenario=case.get("scenario", ""), question=case["question"],
            options=case["options"], clinician_message=dialogue[-1]["text"],
        )
        return prompt, None, None, "not_applicable_turn_1"
    if not belief_enabled:
        prompt = build_plain_agent_response_prompt(
            scenario=case.get("scenario", ""), question=case["question"], options=case["options"],
            dialogue_history=dialogue_text(dialogue), current_round=turn, max_rounds=rounds,
            is_final_round=turn == rounds,
        )
        return prompt, None, None, "not_applicable_belief_disabled"
    return belief_agent_prompt(
        case, dialogue, turn, rounds, claim_order,
        reconstruct=runtime.reconstruct,
        judge=lambda p: runtime.agent(runtime.judgment_system, p, 384),
    )


def _setup(trajectory: dict, turn_index: int, allow_initial_turn: bool = False):
    turns = trajectory["turns"]
    if not 0 <= turn_index < len(turns) or turns[turn_index].get("speaker") != "agent":
        raise ValueError(f"turn {turn_index} is not an agent turn")
    if turns[turn_index].get("final"):
        raise ValueError("the final-decision turn is the outcome, not a collaboration turn")
    if int(turns[turn_index].get("round", 1)) < 2 and not allow_initial_turn:
        raise ValueError("round 1 (initial response) is outside the scored scope unless include_initial_turn is set")
    case = {key: trajectory.get(key, "") for key in ("question", "options", "scenario")}
    dialogue = [
        {k: v for k, v in turn.items() if k in ("speaker", "text", "round")}
        for turn in turns[:turn_index]
    ]
    return case, dialogue, int(turns[turn_index]["round"])


def agent_response_at(trajectory: dict, turn_index: int, runtime: CollaborationRuntime,
                      instruction: str | dict[str, str] = "", allow_initial_turn: bool = False) -> dict[str, Any]:
    """Generate the agent response at an existing agent turn from the same history,
    with the generation protocol (e.g. a Base or SERL agent answering one history)."""
    case, dialogue, start_round = _setup(trajectory, turn_index, allow_initial_turn)
    prompt, reconstruction, judgment, status = _agent_turn(
        case, dialogue, start_round, int(trajectory["max_discussion_rounds"]), trajectory["claim_order"],
        trajectory.get("belief_enabled", True), runtime)
    if prompt is None:
        return {"status": status, "response": None}
    if instruction:
        prompt = prompt + "\n\n" + (instruction.get("first", "") if isinstance(instruction, dict) else instruction)
    return {"status": "completed", "response": runtime.agent(runtime.agent_system, prompt, runtime.max_tokens),
            "reconstruction": reconstruction, "local_judgment": judgment, "local_judgment_status": status}


def continue_collaboration(trajectory: dict, turn_index: int, runtime: CollaborationRuntime, *,
                           instruction: str | dict[str, str] = "", fixed_response: str = None,
                           allow_initial_turn: bool = False,
                           max_agent_turns: int | None = None) -> dict[str, Any]:
    # rounds and final decision role come from the trajectory record (the
    # effectiveness labels build trajectories that carry the ContinuationSpec values).
    """Replay the history before turn_index, then continue to the final decision."""
    case, dialogue, start_round = _setup(trajectory, turn_index, allow_initial_turn)
    rounds = int(trajectory["max_discussion_rounds"])
    claim_order = trajectory["claim_order"]
    belief_enabled = trajectory.get("belief_enabled", True)
    if max_agent_turns is not None and max_agent_turns < 1:
        raise ValueError("max_agent_turns must be at least 1")
    end_round = rounds if max_agent_turns is None else min(rounds, start_round + max_agent_turns - 1)

    termination = None
    intervention_response = None
    for turn in range(start_round, end_round + 1):
        if turn > start_round:
            doctor = runtime.doctor(runtime.doctor_system, doctor_turn_prompt(case, dialogue, turn, rounds))
            dialogue.append({"speaker": "doctor", "text": doctor, "round": turn})
        if turn == start_round and fixed_response is not None:
            response, reconstruction, judgment, status = fixed_response, None, None, "fixed_response"
        else:
            prompt, reconstruction, judgment, status = _agent_turn(
                case, dialogue, turn, rounds, claim_order, belief_enabled, runtime)
            if prompt is None:
                return {"status": status, "dialogue": dialogue, "intervention_response": intervention_response}
            if instruction:
                if isinstance(instruction, dict):
                    turn_instruction = instruction.get("first", "") if turn == start_round else instruction.get("later", "")
                else:
                    turn_instruction = instruction
                prompt = prompt + "\n\n" + turn_instruction
            elif turn > start_round and runtime.continuation_guideline:
                prompt = prompt + "\n\n" + runtime.continuation_guideline
            response = runtime.agent(runtime.agent_system, prompt, runtime.max_tokens)
        if turn == start_round:
            intervention_response = response
        dialogue.append({
            "speaker": "agent", "text": response, "round": turn, "intervention": turn == start_round,
            "reconstruction": reconstruction, "local_judgment": judgment, "local_judgment_status": status,
        })
        termination = stop_reason(turn, dialogue[-2]["text"], response)
        if termination and (max_agent_turns is None or turn == end_round):
            break

    # Same final decision maker as the original run (trajectories written before
    # final_decision_role was recorded always used the doctor).
    final_role = trajectory.get("final_decision_role", "doctor")
    if final_role == "doctor":
        final = runtime.doctor(runtime.doctor_system, final_decision_prompt(case, dialogue))
    elif final_role == "agent":
        final = runtime.agent(runtime.agent_system, final_decision_prompt(case, dialogue), runtime.max_tokens)
    else:
        raise ValueError(f"unknown final_decision_role {final_role!r}")
    dialogue.append({"speaker": final_role, "text": final, "round": rounds + 1, "final": True})
    final_option = option(final)
    return {
        "status": "completed" if final_option else "final_option_parse_failure",
        "final_option": final_option or None,
        "correct": bool(final_option) and final_option == trajectory["ground_truth"]["correct_option"],
        "termination_reason": termination or "max_agent_turns",
        "intervention_response": intervention_response,
        "dialogue": dialogue,
    }


def rerun_from_turn(trajectory: dict, turn_index: int, instruction: str | dict[str, str], runtime: CollaborationRuntime,
                    allow_initial_turn: bool = False,
                    max_agent_turns: int | None = None) -> dict[str, Any]:
    """Step 2-1: rerun from the failed turn with the rubric's ideal-behavior instruction."""
    return continue_collaboration(trajectory, turn_index, runtime, instruction=instruction,
                                  allow_initial_turn=allow_initial_turn,
                                  max_agent_turns=max_agent_turns)
