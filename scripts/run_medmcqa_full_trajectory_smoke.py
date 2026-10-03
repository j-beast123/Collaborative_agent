"""Generate full MedMCQA collaborations with per-Agent token traces.

This is the small pre-training smoke test for the multi-turn path. It samples
four independent conversations for each selected case, continues through all
configured collaboration rounds, asks the Doctor for a final decision, and
scores every Agent turn with Jev using only the history available at that
turn. It does not update model weights yet.

The saved token traces are the exact prompt and completion tokens used for
each Agent response, together with generation transition log-probabilities.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.belief import build_initial_agent_response_prompt, build_plain_agent_response_prompt
from src.collaboration.generator import (
    case_text,
    dialogue_text,
    doctor_call,
    final_decision_prompt,
)
from src.judges.jev import JevClient
from src.judges.reward import soft_min, weighted_soft_min
from src.judges.rubric_schema import normalize_score
from src.judges.rubric_scoring import TWO_STAGE_JEV_PROTOCOL, two_stage_jev_score
try:
    from run_medmcqa_grpo_pilot import load_rubrics, rubric_set_metadata, visible_case
except ModuleNotFoundError:
    from scripts.run_medmcqa_grpo_pilot import load_rubrics, rubric_set_metadata, visible_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/medmcqa/medicine_train_random300_seed20261004.jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=2)
    parser.add_argument("--rollouts-per-case", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", default="outputs/medmcqa_full_trajectory_smoke")
    parser.add_argument("--model", default="/home/kai/models/Qwen3-8B")
    parser.add_argument("--rubric-set", default="configs/rubrics/rubric_guidance_set_11_observable_v2.json")
    parser.add_argument("--doctor-model", default="google/gemma-3-12b-it")
    parser.add_argument("--jev-endpoint", default="https://openrouter.ai/api/alpha/decisions")
    parser.add_argument("--jev-model", default="typesafe/jev-1.13")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-completion-length", type=int, default=400)
    parser.add_argument("--max-tokens-doctor", type=int, default=512)
    parser.add_argument("--generation-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="generation top-k; when omitted, use the model generation_config value",
    )
    parser.add_argument("--rubric-temperature", type=float, default=0.5)
    parser.add_argument("--rubric-weight", type=float, default=0.3)
    parser.add_argument("--jev-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-unvalidated-rubrics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cases(path: Path, start_index: int, num_cases: int) -> list[dict]:
    rows = read_jsonl(path)
    selected = rows[start_index : start_index + num_cases]
    if len(selected) != num_cases:
        raise ValueError(f"requested {num_cases} cases from index {start_index}, found {len(selected)}")
    required = {"case_id", "question", "options"}
    for case in selected:
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"case {case.get('case_id')} is missing fields: {missing}")
    return selected


def chat_prompt(tokenizer: Any, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resolve_generation_top_k(model: Any, requested: int | None) -> int:
    value = requested
    if value is None:
        value = getattr(getattr(model, "generation_config", None), "top_k", 0)
    value = 0 if value is None else int(value)
    if value < 0:
        raise ValueError("top-k must be non-negative; use 0 to disable top-k filtering")
    return value


def generate_with_trace(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int,
                        temperature: float, top_p: float, top_k: int) -> dict:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    sequences = generated.sequences
    completion_ids = sequences[:, prompt_length:]
    transition_scores = model.compute_transition_scores(
        sequences,
        generated.scores,
        normalize_logits=True,
    )
    completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()
    return {
        "prompt_input_ids": encoded["input_ids"][0].detach().cpu().tolist(),
        "prompt_attention_mask": encoded["attention_mask"][0].detach().cpu().tolist(),
        "completion_ids": completion_ids[0].detach().cpu().tolist(),
        "completion_logprobs": transition_scores[0].detach().cpu().tolist(),
        "sampling_temperature": temperature,
        "sampling_top_p": top_p,
        "sampling_top_k": top_k,
        "completion_text": completion_text,
    }


def discussion_prompt(case: dict, dialogue: list[dict], turn: int, rounds: int) -> str:
    return (
        f"{case_text(case)}\n\n"
        "Grounding constraint: Use only the question, answer choices, dialogue, and established domain knowledge. "
        "If no patient vignette or event description is provided, do not invent one.\n\n"
        f"Conversation so far:\n{dialogue_text(dialogue)}\n\n"
        f"Respond naturally as the physician. This is discussion turn {turn} of {rounds}. "
        "Do not give the final decision yet; state your current reasoning and what you need from the collaborator."
    )


def final_option(text: str) -> str:
    matches = re.findall(r"final\s+answer\s*[:\-]?\s*(?:option\s+)?([A-D])\b", text, re.I)
    return matches[-1].upper() if matches else ""


def final_outcome(case: dict, option: str) -> float | None:
    correct = str((case.get("ground_truth") or {}).get("correct_option") or "").strip().upper()
    selected = str(option or "").strip().upper()
    if not correct or not selected:
        return None
    return float(selected == correct)


def score_agent_turn(client: JevClient, case: dict, dialogue: list[dict], target_index: int,
                     rubrics: list[dict], rubric_temperature: float,
                     rubric_metadata: dict | None = None) -> dict:
    target = dialogue[target_index]
    history_state = {
        "trajectory_id": f"full_smoke:{case['case_id']}",
        "turn_id": target_index,
        "dialogue_before_target_turn": dialogue[:target_index],
        "case_context": visible_case(case),
    }
    target_state = {
        "trajectory_id": f"full_smoke:{case['case_id']}",
        "turn_id": target_index,
        "dialogue_through_target_turn": dialogue[: target_index + 1],
        "target_turn": target,
        "case_context": visible_case(case),
    }
    scored = two_stage_jev_score(
        client,
        history_state=history_state,
        target_state=target_state,
        rubrics=rubrics,
        purpose="medmcqa_full_trajectory_rubric",
    )
    labels = scored["labels"]
    numeric_labels = {
        rid: int(value) if isinstance(value, str) and value in {"0", "1", "2"} else value
        for rid, value in labels.items()
    }
    applicable = [normalize_score(value) for value in numeric_labels.values() if value != "NA"]
    raw = weighted_soft_min(applicable, [1.0] * len(applicable), rubric_temperature) if applicable else None
    return {
        "turn_id": target_index,
        "stage": target.get("stage"),
        "labels": labels,
        "normalized_labels": numeric_labels,
        "raw_rubric_adherence": raw,
        "applicability_labels": scored["applicability_labels"],
        "applicability_probabilities": scored["applicability_probabilities"],
        "behavior_scores": scored["behavior_scores"],
        "behavior_probabilities": scored["behavior_probabilities"],
        "jev_calls": scored["calls"],
        "jev_call": scored["calls"],
        "jev_call_count": scored["call_count"],
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        **(rubric_metadata or {}),
    }


def generate_trajectory(model: Any, tokenizer: Any, case: dict, doctor_system: str, agent_system: str,
                        key: str, args: argparse.Namespace) -> dict:
    dialogue: list[dict] = []
    agent_traces: list[dict] = []
    for round_index in range(1, args.rounds + 1):
        doctor = doctor_call(
            key,
            args.doctor_model,
            doctor_system,
            discussion_prompt(case, dialogue, round_index, args.rounds),
            args.max_tokens_doctor,
        )
        dialogue.append({"speaker": "doctor", "stage": f"D{round_index}", "text": doctor, "round": round_index})
        if round_index == 1:
            user_prompt = build_initial_agent_response_prompt(
                scenario=case.get("scenario", ""), question=case["question"],
                options=case["options"], clinician_message=doctor,
            )
        else:
            user_prompt = build_plain_agent_response_prompt(
                scenario=case.get("scenario", ""), question=case["question"],
                options=case["options"], dialogue_history=dialogue_text(dialogue),
                current_round=round_index, max_rounds=args.rounds,
                is_final_round=round_index == args.rounds,
            )
        prompt = chat_prompt(tokenizer, agent_system, user_prompt)
        trace = generate_with_trace(
            model, tokenizer, prompt, args.max_completion_length,
            args.generation_temperature, args.top_p, args.top_k,
        )
        agent_index = len(dialogue)
        dialogue.append({
            "speaker": "agent", "stage": f"A{round_index}",
            "text": trace["completion_text"], "round": round_index,
            "trace_index": len(agent_traces),
        })
        agent_traces.append({
            "stage": f"A{round_index}",
            "dialogue_index": agent_index,
            "prompt_text": prompt,
            **trace,
        })

    final_text = doctor_call(
        key,
        args.doctor_model,
        doctor_system,
        final_decision_prompt(case, dialogue),
        args.max_tokens_doctor,
    )
    dialogue.append({"speaker": "doctor", "stage": "FINAL", "text": final_text, "final": True})
    return {
        "dialogue": dialogue,
        "agent_traces": agent_traces,
        "final_decision_text": final_text,
        "final_option": final_option(final_text),
    }


def main() -> None:
    args = parse_args()
    if args.num_cases < 1 or args.rollouts_per_case < 1 or args.rounds < 1:
        raise ValueError("num-cases, rollouts-per-case, and rounds must be positive")
    if args.jev_workers < 1:
        raise ValueError("jev-workers must be positive")
    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"{args.api_key_env} is not set")

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectories.jsonl"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --overwrite")
    if args.overwrite:
        trajectory_path.unlink(missing_ok=True)

    cases = load_cases(ROOT / args.input, args.start_index, args.num_cases)
    rubric_path = ROOT / args.rubric_set
    rubrics = load_rubrics(rubric_path)
    rubric_metadata = rubric_set_metadata(rubric_path)
    validation_status = rubric_metadata.get("rubric_set_validation_status")
    if validation_status and validation_status not in {"validated", "passed"} and not args.allow_unvalidated_rubrics:
        raise ValueError(
            f"rubric set is not validated for rollout collection: {validation_status}; "
            "revalidate it first or use --allow-unvalidated-rubrics only for an audit run"
        )
    doctor_system = (ROOT / "prompts/user_simulator.txt").read_text(encoding="utf-8")
    agent_system = (ROOT / "prompts/agent.txt").read_text(encoding="utf-8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; launch this smoke test on the GPU server")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model.to(torch.device("cuda:0"))
    model.eval()
    args.top_k = resolve_generation_top_k(model, args.top_k)
    (output / "config.json").write_text(json.dumps({
        "input": args.input,
        "start_index": args.start_index,
        "num_cases": args.num_cases,
        "rollouts_per_case": args.rollouts_per_case,
        "rounds": args.rounds,
        "total_trajectories": args.num_cases * args.rollouts_per_case,
        "rubric_set": args.rubric_set,
        **rubric_metadata,
        "rubrics": len(rubrics),
        "rubric_ids": [rubric["rubric_id"] for rubric in rubrics],
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        "rubric_aggregation": "weighted_softmin_with_equal_rubric_weights",
        "turn_aggregation": "soft_min_over_available_agent_turns",
        "rubric_temperature": args.rubric_temperature,
        "rubric_weight": args.rubric_weight,
        "outcome_weight": 0.0,
        "effectiveness_weight": 0.0,
        "sampling": {
            "temperature": args.generation_temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "seed": args.seed,
        "training": False,
        "allow_unvalidated_rubrics": args.allow_unvalidated_rubrics,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    client = JevClient(endpoint=args.jev_endpoint, model=args.jev_model, api_key_env=args.api_key_env)

    records = []
    label_counts: Counter[str] = Counter()
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for case_number, case in enumerate(cases, 1):
            for rollout_index in range(args.rollouts_per_case):
                trajectory = generate_trajectory(
                    model, tokenizer, case, doctor_system, agent_system, key, args,
                )
                agent_indices = [
                    index for index, turn in enumerate(trajectory["dialogue"])
                    if turn["speaker"] == "agent"
                ]
                with ThreadPoolExecutor(max_workers=min(args.jev_workers, len(agent_indices))) as pool:
                    futures = [pool.submit(
                        score_agent_turn,
                        client, case, trajectory["dialogue"], index, rubrics, args.rubric_temperature,
                        rubric_metadata,
                    ) for index in agent_indices]
                    turn_scores = [future.result() for future in futures]
                available = [row["raw_rubric_adherence"] for row in turn_scores
                             if row["raw_rubric_adherence"] is not None]
                conversation_value = soft_min(available, args.rubric_temperature) if available else None
                for row in turn_scores:
                    label_counts.update(str(value) for value in row["labels"].values())
                record = {
                    "trajectory_id": f"{case['case_id']}:rollout_{rollout_index}",
                    "case_id": case["case_id"],
                    "case_index": args.start_index + case_number - 1,
                    "rollout_index": rollout_index,
                    "case_context": visible_case(case),
                    "ground_truth": case.get("ground_truth"),
                    "dialogue": trajectory["dialogue"],
                    "agent_traces": trajectory["agent_traces"],
                    "turn_scores": turn_scores,
                    **rubric_metadata,
                    "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
                    "conversation_rubric_adherence": conversation_value,
                    "scaled_rubric_reward": None if conversation_value is None else args.rubric_weight * conversation_value,
                    "final_decision_text": trajectory["final_decision_text"],
                    "final_option": trajectory["final_option"],
                    "outcome": final_outcome(case, trajectory["final_option"]),
                    "jev_calls": sum(row["jev_call_count"] for row in turn_scores),
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"TRAJECTORIES: {len(records)}/{args.num_cases * args.rollouts_per_case} "
                    f"| case={case_number}/{len(cases)} | jev_calls={client.calls}",
                    flush=True,
                )

    rewards = [row["conversation_rubric_adherence"] for row in records
               if row["conversation_rubric_adherence"] is not None]
    summary = {
        "status": "completed",
        "trajectories": len(records),
        "cases": len(cases),
        "agent_turns": sum(len(row["turn_scores"]) for row in records),
        "final_decisions": sum(bool(row["final_decision_text"]) for row in records),
        "jev_calls": client.calls,
        "rubric_score_counts": dict(label_counts),
        "conversation_rubric_adherence_mean": sum(rewards) / len(rewards) if rewards else None,
        "conversation_rubric_adherence_min": min(rewards) if rewards else None,
        "conversation_rubric_adherence_max": max(rewards) if rewards else None,
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        **rubric_metadata,
        "output": str(output.relative_to(ROOT)),
        "training": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
