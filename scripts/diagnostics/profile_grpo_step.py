"""Where a GRPO step of a run spends its time, measured without training: for a few of the run's training prompts the
policy samples num_generations responses and the reward is computed as in training (continuations under the Doctor,
outcome judgment, and whatever the condition's reward uses). No weight is updated and nothing of the run is written to.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from src.agentclinic.runtime import build_clinic, frozen_histories, load_policy
from src.common.io import read_jsonl
from src.common.main_config import load_main_config, open_run
from src.grpo.dataset import grpo_rows
from src.grpo.environment import ClinicEnvironment, turn_samples
from src.grpo.reward import SerlReward
from src.grpo.trainer import NO_RUBRICS, a1_states, spec_of, training_stages, trajectories_path
from src.training.reward_pipeline import load_reward_spec

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch

    config = open_run(load_main_config(args.config, ROOT), ROOT, run_id=args.run_id, resume=True)
    if config["reward"]["weights"]["appropriateness"] > 0 or config["reward"]["weights"]["effectiveness"] > 0:
        raise SystemExit("this profile covers a reward of the outcome alone; a rubric or predictor reward needs its models")
    out = ROOT / args.output
    out.mkdir(parents=True)
    model, tokenizer = load_policy(config["policy"], None, trainable=False, device=config["devices"]["policy"])
    clinic = build_clinic(config, ROOT, model, tokenizer)
    protocol = spec_of(clinic.source)
    stages, split = training_stages(config, protocol), config["environment"]["train_split"]
    histories = frozen_histories(config, ROOT, clinic.cases)
    own = read_jsonl(trajectories_path(config, ROOT, args.iteration))
    later = turn_samples(own, clinic.cases, split, stages)
    environment = ClinicEnvironment(clinic.cases, histories, clinic.runtime, clinic.judge, a1_states=a1_states(config, ROOT, args.iteration),
                                    a2=later, collaboration_states={t["history_id"]: t.get("collaboration_state") for t in own},
                                    max_attempts=config["rollout"]["max_attempts"], spec=protocol)
    reward = SerlReward(environment, load_reward_spec(config), rubric_set=NO_RUBRICS, scorer=None, provider=None,
                        log_path=out / "reward_log.jsonl", iteration=args.iteration, scored_stages=tuple(config["reward"]["scored_stages"]),
                        effectiveness_stages=(), workers=config["rollout"]["workers"], generation=clinic.generation,
                        max_failed_share=1.0, credit_assignment=config["reward"]["credit_assignment"],
                        rubric_weight_strategy=config["rubric_importance"]["strategy"])
    rows = grpo_rows(clinic.cases, histories, clinic.runtime.agent_system, split=split, a2=later, stages=stages, spec=protocol)
    chosen = [rows[round(i * (len(rows) - 1) / max(args.groups - 1, 1))] for i in range(args.groups)]  # spread over the turns
    grpo, groups = config["grpo"], []
    for row in chosen:
        text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, do_sample=True, temperature=grpo["temperature"], top_p=grpo["top_p"],
                                       max_new_tokens=grpo["max_completion_length"], num_return_sequences=grpo["num_generations"])
        generation_seconds = time.perf_counter() - started
        completions = [tokenizer.decode(g[inputs["input_ids"].shape[1]:], skip_special_tokens=True) for g in generated]
        started = time.perf_counter()
        reward(prompts=[None] * len(completions), completions=[[{"role": "assistant", "content": c}] for c in completions],
               completion_ids=[[]] * len(completions), sample_id=[row["sample_id"]] * len(completions))
        groups.append({"sample_id": row["sample_id"], "stage": row["stage"], "prompt_tokens": int(inputs["input_ids"].shape[1]),
                       "policy_generation_seconds": generation_seconds, "reward_call_seconds": time.perf_counter() - started})
    timing = read_jsonl(out / "reward_timing.jsonl")
    for group, row in zip(groups, timing):
        group.update(reward=row)
    kinds = sorted({k for row in timing for k in row["sum_seconds"]})
    summary = {
        "groups": len(groups), "responses_per_group": grpo["num_generations"], "workers": config["rollout"]["workers"],
        "mean_policy_generation_seconds": mean(g["policy_generation_seconds"] for g in groups),
        "mean_reward_wall_seconds": mean(r["reward_wall_seconds"] for r in timing),
        "mean_sum_seconds_by_call": {k: mean(r["sum_seconds"].get(k, 0.0) for r in timing) for k in kinds},
        "mean_calls_by_kind": {k: mean(r["calls"].get(k, 0) for r in timing) for k in kinds},
        "mean_seconds_per_call": {k: sum(r["sum_seconds"].get(k, 0.0) for r in timing) / max(sum(r["calls"].get(k, 0) for r in timing), 1) for k in kinds},
        "mean_continuation_agent_turns_per_response": mean(t for r in timing for t in r["continuation_agent_turns"]) if any(r["continuation_agent_turns"] for r in timing) else 0.0,
        "note": "sum_seconds add up the concurrent continuations of a group; the wall time is what the step waits for"}
    (out / "profile.json").write_text(json.dumps({"config": args.config, "run_id": args.run_id, "summary": summary, "groups": groups}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
