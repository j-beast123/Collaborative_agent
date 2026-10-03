"""One-case MedMCQA GRPO pilot with Jev-only rubric reward.

The Doctor's opening message is frozen as the prompt context. The policy samples
the Agent response online; Jev scores that response against visible rubrics using
only the case context, the Doctor message, and the target response. Ground truth,
future turns, and final outcomes are never sent to Jev.
"""

from __future__ import annotations

import argparse
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
from src.judges.reward import turn_appropriateness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/interim/medmcqa/medicine_train_random300_seed20261004.jsonl",
    )
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--output", default="outputs/grpo_medmcqa_rubric_only_pilot")
    parser.add_argument("--model", default="/home/kai/models/Qwen3-8B")
    parser.add_argument(
        "--rubric-set",
        default="outputs/rubric_candidates/rubric_guidance_set_11.json",
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
    return result


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


def jev_questions(rubrics: list[dict]) -> dict[str, dict]:
    questions = {}
    for rubric in rubrics:
        questions[rubric["rubric_id"]] = {
            "type": "choice",
            "instructions": (
                "Judge only the target Agent turn. Use the dialogue before and through the target turn. "
                "First determine whether the applicability condition is observable before the target turn. "
                "If it is absent or unsupported, choose NA. If it applies, choose the level whose behavior "
                "the target response actually demonstrates. Mentioning or promising an action is not performing it. "
                f"Applicability condition: {rubric['applicability_condition']} "
                f"Required behavior when applicable: {rubric['required_behavior']}"
            ),
            "criteria": {
                "NA": "The applicability condition is absent or unsupported before the target turn.",
                "0": rubric["scores"]["0"],
                "1": rubric["scores"]["1"],
                "2": rubric["scores"]["2"],
            },
        }
    return questions


def score_one(
    client: JevClient,
    case: dict,
    doctor_text: str,
    response: str,
    rubrics: list[dict],
    rubric_temperature: float,
) -> dict:
    state = json.dumps(
        {
            "trajectory_id": f"grpo_pilot:{case['case_id']}",
            "turn_id": 1,
            "dialogue_through_target_turn": [
                {"turn_id": 0, "speaker": "doctor", "text": doctor_text},
                {"turn_id": 1, "speaker": "agent", "text": response},
            ],
            "target_turn": {"turn_id": 1, "speaker": "agent", "text": response},
            "agent_pre_response_state": None,
            "case_context": visible_case(case),
        },
        ensure_ascii=False,
    )
    answers, call = client.decide(state, jev_questions(rubrics), purpose="grpo_rubric_reward")
    labels = {rid: answer["value"] for rid, answer in answers.items()}
    normalized = {
        rid: (int(label) if isinstance(label, str) and label in {"0", "1", "2"} else label)
        for rid, label in labels.items()
    }
    raw_value = turn_appropriateness(normalized, method="weighted_softmin", tau=rubric_temperature)
    return {
        "labels": labels,
        "normalized_labels": normalized,
        "raw_rubric_reward": raw_value,
        "jev_call": call,
        "response": response,
    }


def build_reward(case: dict, doctor_text: str, rubrics: list[dict], client: JevClient, output: Path, args: argparse.Namespace):
    lock = Lock()
    reward_path = output / "reward_log.jsonl"
    reward_path.parent.mkdir(parents=True, exist_ok=True)

    def reward_func(prompts, completions, trainer_state=None, **_):
        responses = [completion_text(completion) for completion in completions]
        with ThreadPoolExecutor(max_workers=min(4, len(responses))) as pool:
            futures = [pool.submit(score_one, client, case, doctor_text, response, rubrics, args.rubric_temperature)
                       for response in responses]
            scored = [future.result() for future in futures]
        available = [row["raw_rubric_reward"] for row in scored if row["raw_rubric_reward"] is not None]
        group_mean = sum(available) / len(available) if available else 0.0
        rewards = []
        with lock, reward_path.open("a", encoding="utf-8") as handle:
            for row in scored:
                raw = row["raw_rubric_reward"]
                used = group_mean if raw is None else raw
                reward = args.rubric_weight * used
                rewards.append(reward)
                handle.write(json.dumps({
                    "global_step": getattr(trainer_state, "global_step", None),
                    "case_id": case["case_id"],
                    "reward": reward,
                    "rubric_weight": args.rubric_weight,
                    "raw_rubric_reward": raw,
                    "group_mean_rubric_reward": group_mean,
                    "no_applicable_rubric": raw is None,
                    "rubric_scores": row["labels"],
                    "response": row["response"],
                    "jev_call": row["jev_call"],
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
    rubrics = load_rubrics(ROOT / args.rubric_set)

    visible = visible_case(case)
    (output / "case_context.json").write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "config.json").write_text(json.dumps({
        "input": args.input,
        "case_index": args.case_index,
        "case_id": case["case_id"],
        "model": args.model,
        "rubric_set": args.rubric_set,
        "rubrics": len(rubrics),
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
    reward = build_reward(case, doctor_text, rubrics, client, output, args)
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
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
