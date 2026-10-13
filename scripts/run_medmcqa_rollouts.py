"""Generate and Jev-score 100 MedMCQA prompts x 4 Agent rollouts.

This is a rollout-only preflight for the rubric-only GRPO experiment. It does
not update model weights. For each case, the Doctor's opening message is
frozen, Qwen generates four Agent responses, and Jev scores each response
against the visible rubric set. Ground truth is retained only in the local
rollout record for offline analysis; it is never sent to Jev or the model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.belief import build_initial_agent_response_prompt
from src.collaboration.generator import doctor_call, doctor_turn_prompt
from src.judges.jev import JevClient
from src.judges.rubric_scoring import TWO_STAGE_JEV_PROTOCOL
from run_medmcqa_grpo_pilot import (
    load_rubrics,
    rubric_set_metadata,
    score_one,
    visible_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/interim/medmcqa/medicine_train_random300_seed20261004.jsonl",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=100, help="number of cases handled by this process")
    parser.add_argument("--output", default="outputs/medmcqa_rollouts_100x4")
    parser.add_argument("--model", default="/opt/anonymous/models/Qwen3-8B")
    parser.add_argument(
        "--rubric-set",
        default="configs/rubrics/rubric_guidance_set_11_observable_v2.json",
    )
    parser.add_argument("--doctor-model", default="google/gemma-3-12b-it")
    parser.add_argument("--jev-endpoint", default="https://openrouter.ai/api/alpha/decisions")
    parser.add_argument("--jev-model", default="typesafe/jev-1.13")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=400)
    parser.add_argument("--max-tokens-doctor", type=int, default=512)
    parser.add_argument("--rubric-temperature", type=float, default=0.5)
    parser.add_argument("--rubric-weight", type=float, default=0.3)
    parser.add_argument("--generation-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--jev-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cases(path: Path, start_index: int, num_cases: int) -> list[dict]:
    rows = read_jsonl(path)
    selected = rows[start_index : start_index + num_cases]
    if len(selected) != num_cases:
        raise ValueError(
            f"requested {num_cases} cases from index {start_index}, found {len(selected)}"
        )
    required = {"case_id", "question", "options"}
    for case in selected:
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"case {case.get('case_id')} is missing fields: {missing}")
    return selected


def chat_prompt(tokenizer: Any, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_responses(
    model: Any,
    tokenizer: Any,
    prompt: str,
    num_generations: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_generations,
            max_new_tokens=max_completion_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completions = generated[:, input_length:]
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in completions]


def score_rollout(
    client: JevClient,
    case: dict,
    doctor_text: str,
    response: str,
    rubrics: list[dict],
    rubric_temperature: float,
    rubric_weight: float,
) -> dict:
    try:
        result = score_one(client, case, doctor_text, response, rubrics, rubric_temperature)
        raw = result["raw_rubric_adherence"]
        return {
            "status": "scored",
            "labels": result["labels"],
            "normalized_labels": result["normalized_labels"],
            "raw_rubric_reward": raw,
            "raw_rubric_adherence": raw,
            "scaled_rubric_reward": None if raw is None else rubric_weight * raw,
            "applicability_labels": result["applicability_labels"],
            "applicability_probabilities": result["applicability_probabilities"],
            "behavior_scores": result["behavior_scores"],
            "behavior_probabilities": result["behavior_probabilities"],
            "jev_calls": result["jev_calls"],
            "jev_call": result["jev_calls"],
            "jev_call_count": result["jev_call_count"],
            "scoring_protocol": result["scoring_protocol"],
        }
    except Exception as exc:  # keep one failed API call from losing all rollouts
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "labels": {},
            "normalized_labels": {},
            "raw_rubric_reward": None,
            "raw_rubric_adherence": None,
            "scaled_rubric_reward": None,
            "jev_calls": None,
            "jev_call": None,
            "jev_call_count": 0,
        }


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["rollout_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    args = parse_args()
    if args.num_generations != 4:
        raise ValueError("this run is fixed to four rollouts per case")
    if args.num_cases < 1:
        raise ValueError("num-cases must be positive")
    if args.jev_workers < 1:
        raise ValueError("jev-workers must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rollout_path = output / "rollouts.jsonl"
    doctor_path = output / "doctor_contexts.jsonl"
    summary_path = output / "summary.json"
    if output.exists() and any(output.iterdir()) and not args.resume and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --resume or --overwrite")
    if args.overwrite:
        rollout_path.unlink(missing_ok=True)
        doctor_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    cases = load_cases(ROOT / args.input, args.start_index, args.num_cases)
    rubric_path = ROOT / args.rubric_set
    rubrics = load_rubrics(rubric_path)
    rubric_metadata = rubric_set_metadata(rubric_path)
    doctor_system = (ROOT / "prompts/user_simulator.txt").read_text(encoding="utf-8")
    agent_system = (ROOT / "prompts/agent.txt").read_text(encoding="utf-8")

    config = {
        "input": args.input,
        "start_index": args.start_index,
        "num_cases": args.num_cases,
        "num_generations": args.num_generations,
        "total_requested_rollouts": args.num_cases * args.num_generations,
        "model": args.model,
        "doctor_model": args.doctor_model,
        "rubric_set": args.rubric_set,
        **rubric_metadata,
        "rubrics": len(rubrics),
        "jev_model": args.jev_model,
        "max_completion_length": args.max_completion_length,
        "rubric_weight": args.rubric_weight,
        "outcome_weight": 0.0,
        "effectiveness_weight": 0.0,
        "seed": args.seed,
        "training": False,
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    doctor_by_case: dict[str, str] = {}
    if args.resume and doctor_path.exists():
        for line in doctor_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                doctor_by_case[item["case_id"]] = item["doctor_text"]
    with doctor_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            if case["case_id"] not in doctor_by_case:
                doctor_text = doctor_call(
                    key,
                    args.doctor_model,
                    doctor_system,
                    doctor_turn_prompt(case, [], 1, 1),
                    args.max_tokens_doctor,
                )
                doctor_by_case[case["case_id"]] = doctor_text
                handle.write(json.dumps({
                    "case_id": case["case_id"],
                    "doctor_text": doctor_text,
                }, ensure_ascii=False) + "\n")
            print(f"DOCTOR CONTEXT: {index}/{len(cases)}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; launch this rollout job on the GPU server")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)

    client = JevClient(
        endpoint=args.jev_endpoint,
        model=args.jev_model,
        api_key_env=args.api_key_env,
    )
    existing_ids = load_existing_ids(rollout_path) if args.resume else set()
    lock = Lock()
    completed = 0
    errors = 0
    raw_rewards: list[float] = []
    label_counts: Counter[str] = Counter()
    mode = "a" if args.resume else "w"
    with rollout_path.open(mode, encoding="utf-8") as handle:
        for case_index, case in enumerate(cases, 1):
            case_id = case["case_id"]
            doctor_text = doctor_by_case[case_id]
            user_prompt = build_initial_agent_response_prompt(
                scenario=case.get("scenario", ""),
                question=case["question"],
                options=case["options"],
                clinician_message=doctor_text,
            )
            prompt = chat_prompt(tokenizer, agent_system, user_prompt)
            responses = generate_responses(
                model,
                tokenizer,
                prompt,
                args.num_generations,
                args.max_completion_length,
                args.generation_temperature,
                args.top_p,
            )
            jobs = []
            for generation_index, response in enumerate(responses):
                rollout_id = f"{case_id}:rollout_{generation_index}"
                if rollout_id not in existing_ids:
                    jobs.append((rollout_id, generation_index, response))
            with ThreadPoolExecutor(max_workers=min(args.jev_workers, max(1, len(jobs)))) as pool:
                futures = [
                    pool.submit(
                        score_rollout,
                        client,
                        case,
                        doctor_text,
                        response,
                        rubrics,
                        args.rubric_temperature,
                        args.rubric_weight,
                    )
                    for _, _, response in jobs
                ]
                scored = [future.result() for future in futures]
            with lock:
                for (rollout_id, generation_index, response), score in zip(jobs, scored):
                    record = {
                        "rollout_id": rollout_id,
                        "case_index": args.start_index + case_index - 1,
                        "case_id": case_id,
                        "generation_index": generation_index,
                        "case_context": visible_case(case),
                        "doctor_text": doctor_text,
                        "response": response,
                        "ground_truth": case.get("ground_truth"),
                        **rubric_metadata,
                        **score,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed += 1
                    if score["status"] == "error":
                        errors += 1
                    elif score["raw_rubric_adherence"] is not None:
                        raw_rewards.append(score["raw_rubric_adherence"])
                        label_counts.update(str(value) for value in score["labels"].values())
                handle.flush()
            print(
                f"ROLLOUTS: {case_index * args.num_generations}/{args.num_cases * args.num_generations} "
                f"| scored={completed - errors} error={errors}",
                flush=True,
            )

    summary = {
        "status": "completed",
        "cases": len(cases),
        "rollouts_requested": len(cases) * args.num_generations,
        "rollouts_written_this_run": completed,
        "score_errors": errors,
        "rubric_score_counts": dict(label_counts),
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        **rubric_metadata,
        "raw_rubric_reward_mean": sum(raw_rewards) / len(raw_rewards) if raw_rewards else None,
        "raw_rubric_reward_min": min(raw_rewards) if raw_rewards else None,
        "raw_rubric_reward_max": max(raw_rewards) if raw_rewards else None,
        "raw_rubric_adherence_mean": sum(raw_rewards) / len(raw_rewards) if raw_rewards else None,
        "raw_rubric_adherence_min": min(raw_rewards) if raw_rewards else None,
        "raw_rubric_adherence_max": max(raw_rewards) if raw_rewards else None,
        "jev_calls": client.calls,
        "output": str(output.relative_to(ROOT)),
        "training": False,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
