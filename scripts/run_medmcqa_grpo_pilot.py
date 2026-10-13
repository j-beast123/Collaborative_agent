"""One-case MedMCQA GRPO pilot with Jev-only rubric reward.

The Doctor's opening message is frozen as the prompt context. The policy samples
the Agent response online; Jev scores that response against visible rubrics using
only the case context, the Doctor message, and the target response. Ground truth,
future turns, and final outcomes are never sent to Jev.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.collaboration.generator import doctor_call, doctor_turn_prompt
from src.belief import build_initial_agent_response_prompt
from src.judges.jev import JevClient
from src.judges.reward import weighted_soft_min
from src.judges.rubric_schema import normalize_score
from src.judges.rubric_scoring import TWO_STAGE_JEV_PROTOCOL, two_stage_jev_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/interim/medmcqa/medicine_train_random300_seed20261004.jsonl",
    )
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--output", default="outputs/grpo_medmcqa_rubric_only_pilot")
    parser.add_argument("--model", default="/opt/anonymous/models/Qwen3-8B")
    parser.add_argument(
        "--rubric-set",
        default="configs/rubrics/rubric_guidance_set_11_observable_v2.json",
    )
    parser.add_argument("--doctor-model", default="google/gemma-3-12b-it")
    parser.add_argument("--jev-endpoint", default="https://openrouter.ai/api/alpha/decisions")
    parser.add_argument("--jev-model", default="typesafe/jev-1.13")
    parser.add_argument("--jev-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=400)
    parser.add_argument("--max-tokens-doctor", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--rubric-weight", type=float, default=0.3)
    parser.add_argument("--rubric-temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-unvalidated-rubrics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_case(path: Path, index: int) -> dict:
    rows = read_jsonl(path)
    if not 0 <= index < len(rows):
        raise ValueError(f"case-index {index} is outside 0..{len(rows) - 1}")
    case = rows[index]
    required = {"case_id", "question", "options"}
    missing = sorted(required - set(case))
    if missing:
        raise ValueError(f"case is missing fields: {missing}")
    return case


def load_rubrics(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for rubric in payload.get("rubrics", []):
        condition = str(rubric.get("applicability_condition") or "").strip()
        required = str(rubric.get("required_behavior") or "").strip()
        scores = rubric.get("scores") or {}
        if not rubric.get("rubric_id") or not condition or not required:
            raise ValueError(f"invalid visible rubric: {rubric.get('rubric_id')}")
        if any(not isinstance(scores.get(level), str) or not scores[level].strip() for level in ("0", "1", "2")):
            raise ValueError(f"rubric {rubric['rubric_id']} must contain string scores 0/1/2")
        result.append({
            "rubric_id": str(rubric["rubric_id"]),
            "name": str(rubric.get("name") or rubric["rubric_id"]),
            "applicability_condition": condition,
            "required_behavior": required,
            "scores": {level: scores[level] for level in ("0", "1", "2")},
        })
    if not result:
        raise ValueError(f"no rubrics found in {path}")
    if len({rubric["rubric_id"] for rubric in result}) != len(result):
        raise ValueError(f"duplicate rubric_id values in {path}")
    return result


def rubric_set_metadata(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rubrics = payload.get("rubrics") or []
    canonical = json.dumps(
        rubrics, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    content_sha256 = hashlib.sha256(canonical).hexdigest()
    declared_version = payload.get("rubric_set_version")
    if declared_version and declared_version != content_sha256[: len(str(declared_version))]:
        raise ValueError(
            f"rubric_set_version does not match rubric content in {path}: "
            f"declared={declared_version} computed={content_sha256[:len(str(declared_version))]}"
        )
    return {
        "rubric_set_version": declared_version,
        "rubric_set_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rubric_set_content_sha256": content_sha256,
        "rubric_set_schema_version": payload.get("schema_version"),
        "rubric_set_validation_status": payload.get("validation_status"),
        "rubric_set_parent_version": payload.get("parent_rubric_set_version"),
        "rubric_set_parent_content_sha256": payload.get("parent_rubric_set_content_sha256"),
    }


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    if isinstance(completion, list):
        return "".join(str(message.get("content") or "") for message in completion if isinstance(message, dict))
    return str(completion or "")


def visible_case(case: dict) -> dict:
    return {key: case.get(key) for key in ("question", "scenario", "options")}


def score_one(
    client: JevClient,
    case: dict,
    doctor_text: str,
    response: str,
    rubrics: list[dict],
    rubric_temperature: float,
) -> dict:
    history_state = {
        "trajectory_id": f"grpo_pilot:{case['case_id']}",
        "turn_id": 1,
        "dialogue_before_target_turn": [
            {"turn_id": 0, "speaker": "doctor", "text": doctor_text},
        ],
        "case_context": visible_case(case),
    }
    target_state = {
        "trajectory_id": f"grpo_pilot:{case['case_id']}",
        "turn_id": 1,
        "dialogue_through_target_turn": [
            {"turn_id": 0, "speaker": "doctor", "text": doctor_text},
            {"turn_id": 1, "speaker": "agent", "text": response},
        ],
        "target_turn": {"turn_id": 1, "speaker": "agent", "text": response},
        "case_context": visible_case(case),
    }
    scored = two_stage_jev_score(
        client,
        history_state=history_state,
        target_state=target_state,
        rubrics=rubrics,
        purpose="grpo_rubric_adherence",
    )
    labels = scored["labels"]
    normalized = dict(labels)
    applicable = [normalize_score(value) for value in normalized.values() if value != "NA"]
    raw_value = (
        weighted_soft_min(applicable, [1.0] * len(applicable), rubric_temperature)
        if applicable else None
    )
    return {
        "labels": labels,
        "normalized_labels": normalized,
        "raw_rubric_adherence": raw_value,
        "raw_rubric_reward": raw_value,
        "applicability_labels": scored["applicability_labels"],
        "applicability_probabilities": scored["applicability_probabilities"],
        "behavior_scores": scored["behavior_scores"],
        "behavior_probabilities": scored["behavior_probabilities"],
        "jev_calls": scored["calls"],
        "jev_call": scored["calls"],
        "jev_call_count": scored["call_count"],
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        "response": response,
    }


def build_reward(case: dict, doctor_text: str, rubrics: list[dict], client: JevClient,
                 output: Path, args: argparse.Namespace, rubric_metadata: dict | None = None):
    lock = Lock()
    reward_path = output / "reward_log.jsonl"
    reward_path.parent.mkdir(parents=True, exist_ok=True)

    def reward_func(prompts, completions, trainer_state=None, **_):
        responses = [completion_text(completion) for completion in completions]
        with ThreadPoolExecutor(max_workers=min(4, len(responses))) as pool:
            futures = [pool.submit(score_one, client, case, doctor_text, response, rubrics, args.rubric_temperature)
                       for response in responses]
            scored = [future.result() for future in futures]
        available = [row["raw_rubric_adherence"] for row in scored if row["raw_rubric_adherence"] is not None]
        group_mean = sum(available) / len(available) if available else None
        rewards = []
        with lock, reward_path.open("a", encoding="utf-8") as handle:
            for row in scored:
                raw = row["raw_rubric_adherence"]
                # TRL 1.14 treats None as an unscorable completion, excludes it
                # from the group baseline, and gives it zero advantage. Do not
                # invent a rubric-adherence value for an all-NA response.
                reward = None if raw is None else args.rubric_weight * raw
                rewards.append(reward)
                handle.write(json.dumps({
                    "global_step": getattr(trainer_state, "global_step", None),
                    "case_id": case["case_id"],
                    "reward": reward,
                    "rubric_weight": args.rubric_weight,
                    "raw_rubric_reward": raw,
                    "raw_rubric_adherence": raw,
                    "group_mean_rubric_reward": group_mean,
                    "group_mean_rubric_adherence": group_mean,
                    "no_applicable_rubric": raw is None,
                    "rubric_scores": row["labels"],
                    "applicability_labels": row["applicability_labels"],
                    "applicability_probabilities": row["applicability_probabilities"],
                    "behavior_scores": row["behavior_scores"],
                    "behavior_probabilities": row["behavior_probabilities"],
                    "response": row["response"],
                    "jev_calls": row["jev_calls"],
                    "jev_call_count": row["jev_call_count"],
                    "scoring_protocol": row["scoring_protocol"],
                    **(rubric_metadata or {}),
                }, ensure_ascii=False) + "\n")
        return rewards

    return reward_func


def main() -> None:
    args = parse_args()
    if not args.num_generations >= 2:
        raise ValueError("GRPO needs at least two generations per prompt")
    if not 0 < args.rubric_weight:
        raise ValueError("rubric-weight must be positive")
    if not 0 < args.beta:
        raise ValueError("beta must be positive for this KL-enabled pilot")

    output = ROOT / args.output
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --overwrite or a new output")
    output.mkdir(parents=True, exist_ok=True)
    trainer_output = output / "trainer"
    case = load_case(ROOT / args.input, args.case_index)
    rubric_path = ROOT / args.rubric_set
    rubrics = load_rubrics(rubric_path)
    rubric_metadata = rubric_set_metadata(rubric_path)
    validation_status = rubric_metadata.get("rubric_set_validation_status")
    if validation_status and validation_status not in {"validated", "passed"} and not args.allow_unvalidated_rubrics:
        raise ValueError(
            f"rubric set is not validated for training: {validation_status}; "
            "revalidate it first or use --allow-unvalidated-rubrics only for a deliberate smoke test"
        )

    visible = visible_case(case)
    (output / "case_context.json").write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "config.json").write_text(json.dumps({
        "input": args.input,
        "case_index": args.case_index,
        "case_id": case["case_id"],
        "model": args.model,
        "rubric_set": args.rubric_set,
        **rubric_metadata,
        "rubrics": len(rubrics),
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        "allow_unvalidated_rubrics": args.allow_unvalidated_rubrics,
        "jev_model": args.jev_model,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "rubric_weight": args.rubric_weight,
        "outcome_weight": 0.0,
        "effectiveness_weight": 0.0,
        "beta": args.beta,
        "loss_type": "grpo",
        "seed": args.seed,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    key = os.environ.get(args.jev_api_key_env)
    if not key:
        raise RuntimeError(f"{args.jev_api_key_env} is not set")
    doctor_system = (ROOT / "prompts/user_simulator.txt").read_text(encoding="utf-8")
    doctor_text = doctor_call(
        key,
        args.doctor_model,
        doctor_system,
        doctor_turn_prompt(case, [], 1, 1),
        args.max_tokens_doctor,
    )
    (output / "frozen_doctor_context.json").write_text(json.dumps({
        "case_id": case["case_id"],
        "dialogue": [{"speaker": "doctor", "text": doctor_text, "round": 1}],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; launch this pilot on the GPU server")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], use_rslora=True,
    ))
    model.to(device)
    model.print_trainable_parameters()

    agent_system = (ROOT / "prompts/agent.txt").read_text(encoding="utf-8")
    agent_prompt = build_initial_agent_response_prompt(
        scenario=case.get("scenario", ""), question=case["question"], options=case["options"],
        clinician_message=doctor_text,
    )
    dataset = Dataset.from_list([{
        "prompt": [
            {"role": "system", "content": agent_system},
            {"role": "user", "content": agent_prompt},
        ],
    }])
    client = JevClient(endpoint=args.jev_endpoint, model=args.jev_model, api_key_env=args.jev_api_key_env)
    reward = build_reward(case, doctor_text, rubrics, client, output, args, rubric_metadata)
    from src.grpo.callbacks import JsonlLog

    trainer_args = GRPOConfig(
        output_dir=str(trainer_output), report_to="none",
        num_generations=args.num_generations,
        per_device_train_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=1, max_steps=1,
        learning_rate=1e-5, temperature=0.7, top_p=0.9,
        bf16=True, gradient_checkpointing=True, logging_steps=1, logging_first_step=True,
        save_strategy="steps", save_steps=1, save_total_limit=2,
        beta=args.beta, loss_type="grpo", seed=args.seed,
        chat_template_kwargs={"enable_thinking": False},
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=reward, args=trainer_args,
        train_dataset=dataset, processing_class=tokenizer,
        callbacks=[JsonlLog(output / "train_log.jsonl", iteration=0)],
    )
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    summary = {
        "status": "completed",
        "case_id": case["case_id"],
        "global_step": trainer.state.global_step,
        "training_loss": result.training_loss,
        "reward_calls": client.calls,
        "output": str(output.relative_to(ROOT)),
        "grpo": {"num_generations": args.num_generations, "beta": args.beta, "loss_type": "grpo"},
        "reward": {"rubric": args.rubric_weight, "outcome": 0.0, "effectiveness": 0.0},
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        **rubric_metadata,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
