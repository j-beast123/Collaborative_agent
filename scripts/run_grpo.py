"""SERL policy side on the AgentClinic protocol: trajectories and GRPO training (TRL GRPOTrainer)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agentclinic.runtime import frozen_histories
from src.agentclinic.selection import split_eligible_cases
from src.common.config import IntegrationUnavailable, read_config
from src.common.main_config import load_main_config, open_run, training_device_problems
from src.common.io import read_jsonl
from src.grpo.dataset import grpo_rows
from src.grpo.environment import turn_samples
from src.grpo.trainer import (base_trajectories, checkpoint_trajectories, effectiveness_settings, iteration_dir,
                              spec_of, trajectories_path, training_stages,
                              load_rubric_set, previous_adapter, train_iteration)
from src.training.reward_pipeline import load_reward_spec

ROOT = Path(__file__).resolve().parents[1]


def check(config: dict, iteration: int) -> dict:
    reward = load_reward_spec(config)
    source = read_config(ROOT / config["environment"]["config"])
    cases = split_eligible_cases(source, ROOT)
    spec = spec_of(source)
    stages, split = training_stages(config, spec), config["environment"]["train_split"]
    second = turn_samples(read_jsonl(trajectories_path(config, ROOT, iteration), missing_ok=True), cases, split, stages)
    rows = grpo_rows(cases, frozen_histories(config, ROOT, cases),
                     (ROOT / source["prompts"]["consultant"]).read_text(encoding="utf-8"),
                     split=split, a2=second, stages=stages, spec=spec)
    problems, rubric_set = [], None
    if set(stages) - {"A1"} and not second:
        problems.append(f"later-turn training samples: {trajectories_path(config, ROOT, iteration)} holds no collaboration that reached D2")
    try:
        rubric_set = load_rubric_set(config, ROOT, iteration)
    except (OSError, IntegrationUnavailable) as error:
        problems.append(f"rubric set: {error}")
    effectiveness = effectiveness_settings(config, iteration)
    if reward.weights["effectiveness"] > 0:
        checkpoint = ROOT / (effectiveness.get("checkpoint") or "")
        for name in ("predictor_config.json", "trainable_state.pt"):
            if not (checkpoint / name).is_file():
                problems.append(f"effectiveness checkpoint: no {name} in {checkpoint}")
        if not (ROOT / (effectiveness.get("calibration") or "missing")).is_file():
            problems.append(f"effectiveness calibration file is missing: {effectiveness.get('calibration')}")
    problems += training_device_problems(config)
    try:
        previous_adapter(config, ROOT, iteration)
    except IntegrationUnavailable as error:
        problems.append(str(error))
    grpo = config["grpo"]
    responses = grpo["max_steps"] * grpo["per_device_train_batch_size"] if grpo.get("max_steps") \
        else len(rows) * grpo["num_generations"] * grpo.get("num_train_epochs", 1)
    return {"iteration": iteration, "output_dir": str(iteration_dir(config, ROOT, iteration).relative_to(ROOT)),
            "train_prompts": len(rows), "train_prompts_by_stage": {s: sum(r["stage"] == s for r in rows) for s in stages},
            "responses_per_prompt": grpo["num_generations"],
            "max_active_categories": (config.get("rubric_registry") or {}).get("max_active_categories"),
            "rubric_schedule": (config.get("rubric_registry") or {}).get("schedule"),
            "sampled_responses": responses,
            "reward_weights": reward.weights, "softmin_temperature": reward.tau,
            "active_rubrics": [r["rubric_id"] for r in rubric_set["rubrics"]] if rubric_set else None,
            "effectiveness_checkpoint": effectiveness.get("checkpoint"),
            "ready": not problems, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("init", "base", "check", "train", "generate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", help="the run this stage belongs to (printed by init)")
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--resume", action="store_true", help="train: continue from the last trainer checkpoint")
    parser.add_argument("--split", help="generate: evaluation split (with --output)")
    parser.add_argument("--output", help="generate: evaluation trajectories file")
    parser.add_argument("--allow-test", action="store_true", help="generate: permit the held-out TEST split")
    parser.add_argument("--set", action="append", default=[], metavar="a.b=JSON")
    parser.add_argument("--allow-stand-ins", action="store_true",
                        help="train although the decided rubric scorer (Jev) is not connected and its stand-in would score")
    args = parser.parse_args()
    config = load_main_config(args.config, ROOT, args.set)
    if args.stage == "init":
        print(json.dumps({"run_id": open_run(config, ROOT, purpose="SERL iterations")["run_id"]}))
        return
    if not args.run_id:
        raise SystemExit(f"{args.stage} needs --run-id (create a run with the init stage)")
    config = open_run(config, ROOT, run_id=args.run_id, resume=True)
    if args.stage in ("check", "train") and args.iteration is None:
        raise SystemExit(f"{args.stage} needs --iteration")
    if bool(args.split) != bool(args.output):
        raise SystemExit("--split and --output go together")
    if args.split == "test" and not args.allow_test:
        raise SystemExit("the TEST split is held out: pass --allow-test for the final evaluation")
    if args.stage == "base":
        result = base_trajectories(config, ROOT)
    elif args.stage == "check":
        result = check(config, args.iteration)
    elif args.stage == "train":
        if config["reward"]["weights"]["appropriateness"] > 0:
            from src.common.main_config import refuse_stand_ins

            refuse_stand_ins(config, ["rubric_scorer"], "GRPO training with an Appropriateness reward", args.allow_stand_ins)
        result = train_iteration(config, ROOT, args.iteration, resume=args.resume)
    else:
        if args.iteration is None and not args.split:
            raise SystemExit("generate needs --iteration, or --split and --output for the base Agent")
        result = checkpoint_trajectories(config, ROOT, args.iteration, split=args.split,
                                         output=Path(args.output) if args.output else None)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if args.stage == "check" and not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
