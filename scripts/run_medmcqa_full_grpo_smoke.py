"""Two-case multi-turn GRPO smoke test for the MedMCQA collaboration.

The script samples four full conversations per case, scores every Agent turn
with Jev, aggregates rubric adherence with rubric-level and dialogue-level
soft-mins, and performs one LoRA policy-gradient/GRPO-style update. Doctor
messages never enter the policy loss. Exact prompt/completion token IDs and
generation log-probabilities are stored in the trajectory JSONL.

This is deliberately a small smoke path, not the final high-throughput
trainer. It exists to verify that full multi-turn rollout, reward aggregation,
token traces, KL regularization, and one parameter update are connected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from run_medmcqa_full_trajectory_smoke import (
        generate_trajectory,
        load_cases,
        score_agent_turn,
        visible_case,
    )
    from run_medmcqa_grpo_pilot import load_rubrics
except ModuleNotFoundError:
    from scripts.run_medmcqa_full_trajectory_smoke import (
        generate_trajectory,
        load_cases,
        score_agent_turn,
        visible_case,
    )
    from scripts.run_medmcqa_grpo_pilot import load_rubrics
from src.judges.jev import JevClient
from src.judges.reward import soft_min


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/medmcqa/medicine_train_random300_seed20261004.jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=2)
    parser.add_argument("--rollouts-per-case", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", default="outputs/medmcqa_full_grpo_smoke")
    parser.add_argument("--model", default="/home/kai/models/Qwen3-8B")
    parser.add_argument("--rubric-set", default="outputs/rubric_candidates/rubric_guidance_set_11.json")
    parser.add_argument("--doctor-model", default="google/gemma-3-12b-it")
    parser.add_argument("--jev-endpoint", default="https://openrouter.ai/api/alpha/decisions")
    parser.add_argument("--jev-model", default="typesafe/jev-1.13")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-completion-length", type=int, default=400)
    parser.add_argument("--max-tokens-doctor", type=int, default=512)
    parser.add_argument("--generation-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rubric-temperature", type=float, default=0.5)
    parser.add_argument("--rubric-weight", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--jev-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def record_trajectory(case: dict, case_index: int, rollout_index: int, trajectory: dict,
                      turn_scores: list[dict], rubric_weight: float,
                      rubric_temperature: float) -> dict:
    available = [row["raw_rubric_adherence"] for row in turn_scores
                 if row["raw_rubric_adherence"] is not None]
    conversation_value = soft_min(available, rubric_temperature) if available else None
    return {
        "trajectory_id": f"{case['case_id']}:rollout_{rollout_index}",
        "case_id": case["case_id"],
        "case_index": case_index,
        "rollout_index": rollout_index,
        "case_context": visible_case(case),
        "ground_truth": case.get("ground_truth"),
        "dialogue": trajectory["dialogue"],
        "agent_traces": trajectory["agent_traces"],
        "turn_scores": turn_scores,
        "conversation_rubric_adherence": conversation_value,
        "scaled_rubric_reward": None if conversation_value is None else rubric_weight * conversation_value,
        "final_decision_text": trajectory["final_decision_text"],
        "final_option": trajectory["final_option"],
    }


def sequence_logprobs(model, prompt_ids: list[int], completion_ids: list[int], *, grad: bool):
    import torch
    import torch.nn.functional as F

    if not completion_ids:
        return torch.empty(0, device=next(model.parameters()).device)
    full_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=next(model.parameters()).device)
    attention = torch.ones_like(full_ids)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        logits = model(input_ids=full_ids, attention_mask=attention, use_cache=False).logits[:, :-1]
        targets = full_ids[:, 1:]
        token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = len(prompt_ids) - 1
    end = start + len(completion_ids)
    return token_logprobs[0, start:end]


def reference_logprobs(model, prompt_ids: list[int], completion_ids: list[int]):
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("the LoRA model does not expose disable_adapter for KL reference scoring")
    with model.disable_adapter():
        return sequence_logprobs(model, prompt_ids, completion_ids, grad=False)


def compute_grpo_loss(model, records: list[dict], beta: float, clip_epsilon: float, *, backward: bool = False):
    import torch

    grouped = defaultdict(list)
    for record in records:
        grouped[record["case_id"]].append(record)
    advantages = {}
    for case_id, members in grouped.items():
        available_rewards = [
            float(member["scaled_rubric_reward"])
            for member in members
            if member["scaled_rubric_reward"] is not None
        ]
        if not available_rewards:
            continue
        group_mean = sum(available_rewards) / len(available_rewards)
        rewards = torch.tensor(
            [float(member["scaled_rubric_reward"])
             if member["scaled_rubric_reward"] is not None else group_mean
             for member in members],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        normalized = (rewards - mean) / (std + 1e-8) if float(std) > 0 else torch.zeros_like(rewards)
        for member, advantage in zip(members, normalized):
            advantages[member["trajectory_id"]] = advantage

    jobs = []
    for record in records:
        if record["trajectory_id"] not in advantages:
            continue
        advantage = advantages[record["trajectory_id"]]
        for trace in record["agent_traces"]:
            jobs.append((trace, advantage))
    if not jobs:
        raise RuntimeError("no Agent completion tokens were available for the GRPO loss")

    policy_values = []
    kl_values = []
    used = 0
    for trace, advantage in jobs:
        current = sequence_logprobs(model, trace["prompt_input_ids"], trace["completion_ids"], grad=True)
        old = torch.tensor(trace["completion_logprobs"], dtype=current.dtype, device=current.device)
        if current.numel() == 0 or old.numel() != current.numel():
            del current, old
            continue
        ratio = torch.exp(current - old)
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = torch.minimum(ratio * advantage, clipped * advantage)
        policy_term = -surrogate.mean()

        ref = reference_logprobs(model, trace["prompt_input_ids"], trace["completion_ids"])
        log_ratio = ref - current
        kl_term = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
        if backward:
            ((policy_term + beta * kl_term) / len(jobs)).backward()
        policy_values.append(policy_term.detach())
        kl_values.append(kl_term.detach())
        used += 1
        del current, old, ref, ratio, clipped, surrogate, policy_term, kl_term
    if not policy_values:
        raise RuntimeError("no Agent completion tokens had matching old log-probabilities")
    policy_loss = torch.stack(policy_values).mean()
    kl_loss = torch.stack(kl_values).mean() if kl_values else torch.zeros_like(policy_loss)
    total_loss = policy_loss + beta * kl_loss
    return total_loss, policy_loss, kl_loss, used


def main() -> None:
    args = parse_args()
    if args.rollouts_per_case != 4:
        raise ValueError("the smoke test uses exactly four rollouts per case")
    if args.num_cases < 1 or args.rounds < 1:
        raise ValueError("num-cases and rounds must be positive")
    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"{args.api_key_env} is not set")

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --overwrite")
    if args.overwrite:
        for name in ("trajectories.jsonl", "summary.json", "train_log.jsonl", "config.json"):
            (output / name).unlink(missing_ok=True)

    cases = load_cases(ROOT / args.input, args.start_index, args.num_cases)
    rubrics = load_rubrics(ROOT / args.rubric_set)
    doctor_system = (ROOT / "prompts/user_simulator.txt").read_text(encoding="utf-8")
    agent_system = (ROOT / "prompts/agent.txt").read_text(encoding="utf-8")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; launch this smoke test on the GPU server")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], use_rslora=True,
    ))
    model.to(torch.device("cuda:0"))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.eval()
    client = JevClient(endpoint=args.jev_endpoint, model=args.jev_model, api_key_env=args.api_key_env)

    records = []
    trajectory_path = output / "trajectories.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for case_number, case in enumerate(cases, 1):
            for rollout_index in range(args.rollouts_per_case):
                trajectory = generate_trajectory(
                    model, tokenizer, case, doctor_system, agent_system, key, args,
                )
                agent_indices = [index for index, turn in enumerate(trajectory["dialogue"])
                                 if turn["speaker"] == "agent"]
                with ThreadPoolExecutor(max_workers=min(args.jev_workers, len(agent_indices))) as pool:
                    futures = [pool.submit(
                        score_agent_turn, client, case, trajectory["dialogue"], index,
                        rubrics, args.rubric_temperature,
                    ) for index in agent_indices]
                    turn_scores = [future.result() for future in futures]
                record = record_trajectory(
                    case, args.start_index + case_number - 1, rollout_index,
                    trajectory, turn_scores, args.rubric_weight, args.rubric_temperature,
                )
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"TRAJECTORIES: {len(records)}/{len(cases) * 4} | Jev calls={client.calls}", flush=True)

    model.train()
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    total_loss, policy_loss, kl_loss, used_completions = compute_grpo_loss(
        model, records, args.beta, args.clip_epsilon, backward=True,
    )
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")

    train_row = {
        "global_step": 1,
        "loss": float(total_loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "used_agent_completions": used_completions,
        "cases": len(cases),
        "trajectories": len(records),
    }
    (output / "train_log.jsonl").write_text(json.dumps(train_row, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "completed",
        "implementation": "custom_multi_turn_grpo_smoke",
        "cases": len(cases),
        "trajectories": len(records),
        "agent_turns": sum(len(row["agent_traces"]) for row in records),
        "jev_calls": client.calls,
        "loss": train_row,
        "reward": {"rubric": args.rubric_weight, "outcome": 0.0, "effectiveness": 0.0},
        "aggregation": {"rubric": "weighted_softmin", "turns": "soft_min"},
        "output": str(output.relative_to(ROOT)),
    }
    (output / "config.json").write_text(json.dumps({
        "input": args.input, "start_index": args.start_index, "num_cases": args.num_cases,
        "rollouts_per_case": args.rollouts_per_case, "rounds": args.rounds,
        "rubric_set": args.rubric_set, "rubrics": len(rubrics),
        "rubric_temperature": args.rubric_temperature, "rubric_weight": args.rubric_weight,
        "beta": args.beta, "learning_rate": args.learning_rate,
        "clip_epsilon": args.clip_epsilon, "training": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
