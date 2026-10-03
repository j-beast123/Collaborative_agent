"""Collaboration trajectories of the AgentClinic protocols (protocol.ProtocolSpec), in the SERL trajectory schema."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from src.common import usage
from src.evaluation.collaboration_states import STATES

from .adapter import agent_view, doctor_view, render
from .outcome import DiagnosisJudge
from .pipeline import QUESTION, Appender
from .protocol import (ASSESSMENT, TWO_TURN, ClinicRuntime, ProtocolSpec, agent_message, continue_dialogue, elicit_state,
                       standalone_prompt, stated)

PROTOCOL = TWO_TURN.name  # "agentclinic_two_turn"; trajectories of another spec carry that spec's name


def rounds_of(dialogue: list[dict]) -> list[int]:
    """The collaboration turn of every message: D_t and A_t are turn t, FINAL belongs to the last turn."""
    rounds = []
    for message in dialogue:
        stage = message["stage"]
        rounds.append(int(stage[1:]) if stage != "FINAL" else (rounds[-1] if rounds else 1))
    return rounds


def standalone_assessment(case: dict, runtime: ClinicRuntime) -> str:
    """The Agent's diagnosis from its own information, before any Doctor message (greedy when available)."""
    return stated((runtime.agent_greedy or runtime.agent)(runtime.agent_system, standalone_prompt(case)), ASSESSMENT)


def agent_state(case: dict, elicited: dict) -> dict:
    """pre_response_state of an Agent turn: the elicited state and the information the Agent has."""
    return {**elicited, "clinical_information_available_to_agent": render(agent_view(case))}


def elicit_states(case: dict, dialogue: list[dict], runtime: ClinicRuntime) -> dict[str, dict]:
    """{stage: elicited state} for the Agent turns of a dialogue, each from the dialogue before that turn.
    The calls are independent of the responses, so eliciting after the dialogue was written is the same."""
    return {m["stage"]: elicit_state(case, dialogue[:i], runtime) for i, m in enumerate(dialogue) if m["speaker"] == "agent"}


def to_trajectory(case: dict, dialogue: list[dict], *, trajectory_id: str, standalone: Optional[str],
                  states: dict[str, dict], judge: DiagnosisJudge, status: str = "completed",
                  spec: ProtocolSpec = TWO_TURN, **extra) -> dict:
    """states: {stage: elicited state} of the Agent turns. extra: fields such as history_id, iteration, checkpoint_id."""
    def correct(diagnosis):
        return judge.equivalent(case["gold"], diagnosis)["equivalent"] if diagnosis else None

    final = dialogue[-1].get("final_diagnosis") if dialogue and dialogue[-1]["stage"] == "FINAL" else None
    verdict = judge.equivalent(case["gold"], final or "")
    # the Doctor's first stated diagnosis: its D1 opening, or under agent_first its first reply (after A1)
    first_doctor = next((m for m in dialogue if m["speaker"] == "doctor"), {})
    doctor_initial = first_doctor.get("provisional_diagnosis") or first_doctor.get("final_diagnosis") or None
    agent_ok, doctor_ok = correct(standalone), correct(doctor_initial)
    turns, rounds = [], rounds_of(dialogue)
    for index, message in enumerate(dialogue):
        turn = {"speaker": message["speaker"], "stage": message["stage"], "round": rounds[index],
                "text": message["text"]}
        if message["speaker"] == "agent":
            turn.update(assessment=message.get("assessment") or None,
                        pre_response_state=agent_state(case, states[message["stage"]]))
        elif message["stage"] == "FINAL":
            turn["final"] = True
        else:
            turn["provisional_diagnosis"] = message.get("provisional_diagnosis") or None
        turns.append(turn)
    return {
        "trajectory_id": trajectory_id, "case_id": case["case_id"], "original_case_id": case["case_id"],
        "split": case.get("split"), "protocol": spec.name, "status": status,
        "trajectory_status": "valid" if status == "completed" and verdict["equivalent"] is not None else "invalid",
        "scenario": render(agent_view(case)),
        "doctor_information": "" if spec.doctor_view == "none" else render(agent_view(case) if spec.doctor_view == "full" else doctor_view(case)),
        "question": QUESTION, "options": {}, "ground_truth": {"correct_diagnosis": case["gold"]},
        "max_discussion_rounds": spec.max_turns, "final_decision_role": "doctor", "first_collaboration_round": 1,
        "early_stopping": spec.early_stopping, "agent_first": spec.agent_first, "doctor_view": spec.doctor_view, "num_turns": sum(m["speaker"] == "agent" for m in dialogue),
        "stop": extra.pop("stop", None),
        "agent_initial_position": standalone or None, "doctor_initial_claim": doctor_initial,
        "agent_initial_correct": agent_ok, "doctor_initial_correct": doctor_ok,
        "collaboration_state": STATES.get((agent_ok, doctor_ok)),
        "final_diagnosis": final, "is_correct": bool(verdict["equivalent"]), "outcome_method": verdict["method"],
        "turns": turns, **extra,
    }


def dialogue_of(trajectory: dict, upto: Optional[int] = None) -> list[dict]:
    """The protocol dialogue of a trajectory (its first upto turns), as protocol.py reads it."""
    return [{"speaker": t["speaker"], "stage": t["stage"], "text": t["text"],
             **{k: t[k] for k in ("assessment", "provisional_diagnosis") if t.get(k)}}
            for t in trajectory["turns"][:upto]]


def collaborate(case: dict, history: list[dict], runtime: ClinicRuntime, judge: DiagnosisJudge, *,
                trajectory_id: str, max_attempts: int = 3, spec: ProtocolSpec = TWO_TURN,
                standalone: Optional[str] = None, **extra) -> dict:
    """The Agent's standalone assessment (asked here unless given), one natural collaboration from the Doctor's
    D1 under the protocol spec, and the Agent-side state before every Agent turn."""
    with usage.case(case["case_id"]):
        standalone = standalone if standalone is not None else standalone_assessment(case, runtime)
        for attempt in range(1, max_attempts + 1):  # a Doctor output that is not valid JSON is a format failure, rerun
            states = {}
            result = continue_dialogue(case, history, runtime, states=states, spec=spec)
            if result["status"] == "completed":
                break
        return to_trajectory(case, result["dialogue"], trajectory_id=trajectory_id, standalone=standalone,
                             states=states, judge=judge, status=result["status"], spec=spec,
                             stop=result.get("stop"), attempts=attempt, **extra)


def from_prepared(case: dict, row: dict, runtime: ClinicRuntime, judge: DiagnosisJudge, **extra) -> dict:
    """The base Agent's trajectory from a prepared history row of the label pipeline: the frozen dialogue
    is kept as it is, and only the Agent-side states are elicited (no Doctor call, no new response)."""
    natural = row["natural"]
    with usage.case(case["case_id"]):
        return to_trajectory(case, natural["dialogue"], trajectory_id=f"{row['history_id']}:base",
                             standalone=row["agent_standalone_diagnosis"],
                             states=elicit_states(case, natural["dialogue"], runtime), judge=judge,
                             status=natural["status"], history_id=row["history_id"], **extra)


def intervention_message(trajectory: dict, turn_index: int, instruction: str, case: dict, runtime: ClinicRuntime,
                         spec: ProtocolSpec = TWO_TURN) -> dict:
    """Rubric validation: the Agent's message at turn_index under the instruction, generated once and not continued
    (the candidate's intervention, before its compliance is checked)."""
    if trajectory["turns"][turn_index]["speaker"] != "agent":
        raise ValueError("the intervention turn must be an Agent turn")
    with usage.case(case["case_id"]):
        return agent_message(case, dialogue_of(trajectory, turn_index), trajectory["turns"][turn_index]["stage"], runtime,
                             instruction, spec)


def rerun_from_turn(trajectory: dict, turn_index: int, instruction: str, case: dict, runtime: ClinicRuntime,
                    judge: DiagnosisJudge, spec: ProtocolSpec = TWO_TURN, fixed: Optional[dict] = None) -> dict:
    """Rubric validation: regenerate the Agent turn at turn_index under the instruction, continue naturally.
    fixed ({"text", "assessment"}): that message is the turn instead, as it is; only what follows is generated."""
    if trajectory["turns"][turn_index]["speaker"] != "agent":
        raise ValueError("the intervention turn must be an Agent turn")
    with usage.case(case["case_id"]):
        result = continue_dialogue(case, dialogue_of(trajectory, turn_index), runtime,
                                   guideline="" if fixed else instruction, fixed=fixed, spec=spec)
        verdict = judge.equivalent(case["gold"], result["final_diagnosis"] or "")
    status = result["status"] if result["status"] != "completed" or verdict["equivalent"] is not None else "outcome_unjudged"
    return {"status": status, "final_diagnosis": result["final_diagnosis"], "correct": bool(verdict["equivalent"]),
            "intervention_response": result["dialogue"][turn_index]["text"], "dialogue": result["dialogue"]}



def generate_trajectories(cases: dict[str, dict], histories: list[dict], runtime: ClinicRuntime, judge: DiagnosisJudge,
                          path: Path, *, suffix: str, workers: int, max_attempts: int = 3,
                          spec: ProtocolSpec = TWO_TURN, reuse_standalone: bool = False, **extra) -> list[dict]:
    """One natural collaboration per history, appended to path; finished histories are not run again."""
    log = Appender(path, key=lambda r: r["trajectory_id"])

    def run(row):
        log.add(collaborate(cases[row["case_id"]], row["history"], runtime, judge, max_attempts=max_attempts, spec=spec,
                            standalone=row["agent_standalone_diagnosis"] if reuse_standalone else None,
                            trajectory_id=f"{row['history_id']}:{suffix}", history_id=row["history_id"], **extra))

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, [row for row in histories if f"{row['history_id']}:{suffix}" not in log.keys]))
    return sorted(log.rows, key=lambda r: r["trajectory_id"])


def summary(trajectories: list[dict]) -> dict:
    valid = [t for t in trajectories if t["trajectory_status"] == "valid"]
    return {"trajectories": len(trajectories), "valid": len(valid), "failures": sum(not t["is_correct"] for t in valid),
            "accuracy": sum(t["is_correct"] for t in valid) / len(valid) if valid else None}
