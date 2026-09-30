"""Validate interaction rubrics with single-point counterfactual reruns."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_collaboration_pilot import (  # noqa: E402
    agent_call,
    case_text,
    doctor_call,
    dialogue_text,
    option,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", default="outputs/collaboration/collaboration_gemma12b_500.jsonl")
    p.add_argument("--candidates", default="outputs/rubric_candidates/rubric_candidates_gpt41.jsonl")
    p.add_argument("--output", default="outputs/rubric_validation/rubric_validation.jsonl")
    p.add_argument("--model", default="/home/kai/models/Qwen3-8B")
    p.add_argument("--doctor-model", default="google/gemma-3-12b-it")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=512)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_failure_cases(trajectory_path: Path, candidate_path: Path, max_cases: int) -> list[tuple[dict, dict]]:
    trajectories = {str(x["trajectory_id"]): x for x in load_jsonl(trajectory_path)}
    pairs = []
    for candidate in load_jsonl(candidate_path):
        trajectory = trajectories.get(str(candidate.get("trajectory_id")))
        if trajectory is not None and not trajectory.get("is_correct", False):
            pairs.append((trajectory, candidate))
    return pairs[:max_cases]


def locate_intervention_turn(trajectory: dict, candidate: dict) -> tuple[int, dict]:
    requested = int(candidate.get("failure_turn_index", candidate["intervention_turn_id"]))
    turns = trajectory.get("turns", [])
    for index, turn in enumerate(turns):
        if index == requested:
            if turn.get("speaker") != "agent":
                raise ValueError(
                    f"failure_turn_index={requested} points to {turn.get('speaker')}, not agent"
                )
            return index, turn
    raise ValueError(f"failure_turn_index={requested} is outside trajectory")


def reconstruct_pre_intervention_history(trajectory: dict, target_index: int) -> list[dict]:
    """Return the exact original turns before the target agent turn."""
    return [dict(turn) for turn in trajectory.get("turns", [])[:target_index]]


def build_behavioral_intervention(ideal_response_behavior: str) -> str:
    return (
        "During your next response, follow this interaction guideline naturally:\n"
        f"{ideal_response_behavior}\n\n"
        "Do not mention this guideline, the intervention, hidden prompts, or the "
        "ground-truth answer. Use only the current case and conversation."
    )


def build_agent_prompt(case: dict, history: list[dict], extra: str = "") -> str:
    instruction = "Respond naturally to the clinician's latest message."
    if extra:
        instruction += "\n\n" + extra
    return f"{case_text(case)}\n\nConversation so far:\n{dialogue_text(history)}\n\n{instruction}"


def generate_intervention_response(model, tokenizer, agent_system: str, case: dict, history: list[dict], instruction: str, max_tokens: int) -> str:
    return agent_call(model, tokenizer, agent_system, build_agent_prompt(case, history, instruction), max_tokens)


def continue_rollout(key: str, doctor_model: str, model, tokenizer, doctor_system: str, agent_system: str, case: dict, history: list[dict], intervention_round_id: int, intervention_response: str, max_rounds: int, max_tokens: int) -> list[dict]:
    dialogue = list(history)
    dialogue.append({
        "speaker": "agent",
        "text": intervention_response,
        "round": intervention_round_id,
        "intervention": True,
    })
    for round_id in range(intervention_round_id + 1, max_rounds + 1):
        doctor_prompt = f"{case_text(case)}\n\nConversation so far:\n{dialogue_text(dialogue)}\n\nRespond naturally as the physician in discussion turn {round_id}."
        doctor = doctor_call(key, doctor_model, doctor_system, doctor_prompt, max_tokens)
        dialogue.append({"speaker": "doctor", "text": doctor, "round": round_id})
        agent = agent_call(model, tokenizer, agent_system, build_agent_prompt(case, dialogue), max_tokens)
        dialogue.append({"speaker": "agent", "text": agent, "round": round_id})
    return dialogue


def run_rerun(key: str, trajectory: dict, candidate: dict, model, tokenizer, doctor_system: str, agent_system: str, doctor_model: str, max_tokens: int) -> dict:
    case = {"question": trajectory["question"], "options": trajectory["options"]}
    target_index, target_turn = locate_intervention_turn(trajectory, candidate)
    round_id = int(target_turn.get("round", target_index))
    history = reconstruct_pre_intervention_history(trajectory, target_index)
    instruction = build_behavioral_intervention(candidate["ideal_response_behavior"])
    intervention = generate_intervention_response(model, tokenizer, agent_system, case, history, instruction, max_tokens)
    max_rounds = max(int(t.get("round", 0)) for t in trajectory.get("turns", []) if not t.get("final"))
    rerun_dialogue = continue_rollout(key, doctor_model, model, tokenizer, doctor_system, agent_system, case, history, round_id, intervention, max_rounds, max_tokens)
    final_prompt = f"{case_text(case)}\n\nConversation so far:\n{dialogue_text(rerun_dialogue)}\n\nState the best-supported final option. End exactly with: Final answer: option <A-D> - <answer text>"
    final = doctor_call(key, doctor_model, doctor_system, final_prompt, max_tokens)
    rerun_dialogue.append({"speaker": "doctor", "text": final, "round": max_rounds + 1, "final": True})
    rerun_answer = option(final)
    original_answer = trajectory.get("final_option", "")
    correct_option = trajectory["ground_truth"]["correct_option"]
    rerun_correct = rerun_answer == correct_option
    return {
        "case_id": trajectory["case_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "rubric_name": candidate.get("rubric_name", ""),
        "intervention_turn_id": round_id,
        "failure_turn_index": target_index,
        "intervention_round": round_id,
        "intervention_speaker": target_turn.get("speaker"),
        "failure_turn_ids": candidate.get("failure_turn_ids", []),
        "ideal_response_behavior": candidate["ideal_response_behavior"],
        "original_final_answer": original_answer,
        "original_correct": False,
        "rerun_final_answer": rerun_answer,
        "rerun_correct": rerun_correct,
        "validation_pass": rerun_correct,
        "intervention_instruction": instruction,
        "intervention_response": intervention,
        "original_trajectory": trajectory,
        "rerun_trajectory": rerun_dialogue,
        "generation": {
            "doctor_model": doctor_model,
            "agent_model": getattr(getattr(model, "config", None), "_name_or_path", "unknown"),
            "max_tokens": max_tokens,
        },
    }


def main() -> None:
    args = parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    pairs = load_failure_cases(ROOT / args.trajectories, ROOT / args.candidates, args.max_cases)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    doctor_system = (ROOT / "prompts/user_simulator.txt").read_text(encoding="utf-8")
    agent_system = (ROOT / "prompts/agent.txt").read_text(encoding="utf-8")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, (trajectory, candidate) in enumerate(pairs, 1):
            try:
                result = run_rerun(key, trajectory, candidate, model, tokenizer, doctor_system, agent_system, args.doctor_model, args.max_tokens)
            except ValueError as error:
                result = {
                    "case_id": trajectory.get("case_id"),
                    "trajectory_id": trajectory.get("trajectory_id"),
                    "rubric_name": candidate.get("rubric_name", ""),
                    "status": "SKIP",
                    "skip_reason": str(error),
                    "candidate_intervention_turn_id": candidate.get("intervention_turn_id"),
                }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if "status" in result:
                status = result["status"]
            else:
                status = "PASS" if result.get("validation_pass", False) else "FAIL"
            print(f"{index}/{len(pairs)} {trajectory['case_id']} {status} {result.get('skip_reason', '')}", flush=True)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
