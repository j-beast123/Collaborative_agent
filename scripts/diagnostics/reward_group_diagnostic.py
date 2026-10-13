"""Per iteration of a GRPO run: how many groups (one prompt, num_generations sampled responses) had no reward difference."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def diagnose(run: Path) -> list[dict]:
    rows = []
    for directory in sorted(run.glob("iter*"), key=lambda p: int(p.name[4:])):
        log = directory / "reward_log.jsonl"
        if not log.is_file():
            continue
        groups = defaultdict(list)
        for r in read_jsonl(log):
            groups[(r.get("global_step"), r.get("sample_id"))].append(r)
        complete = {k: v for k, v in groups.items() if all(x.get("status") == "completed" for x in v)}
        rewards = {k: [x["reward"] for x in v] for k, v in complete.items()}
        outcomes = {k: [x.get("outcome") for x in v] for k, v in complete.items()}
        mixed = sum(len(set(v)) > 1 for v in rewards.values())
        after = directory / "trajectories.jsonl"
        valid = [t for t in read_jsonl(after) if t["trajectory_status"] == "valid"] if after.is_file() else []
        rows.append({"iteration": int(directory.name[4:]), "finished": (directory / "checkpoint.json").is_file(),
                     "total_groups": len(groups), "groups_with_every_response_completed": len(complete),
                     "all_zero_outcome_groups": sum(set(v) == {0.0} for v in outcomes.values()),
                     "all_one_outcome_groups": sum(set(v) == {1.0} for v in outcomes.values()),
                     "mixed_reward_groups": mixed, "nonzero_gradient_group_fraction": mixed / len(complete) if complete else None,
                     "mean_reward": sum(sum(v) for v in rewards.values()) / sum(len(v) for v in rewards.values()) if rewards else None,
                     "train_success_after_iteration": sum(t["is_correct"] for t in valid) / len(valid) if valid else None})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = {"run": args.run, "iterations": diagnose(ROOT / args.run)}
    text = json.dumps(result, indent=2)
    if args.output:
        Path(ROOT / args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
