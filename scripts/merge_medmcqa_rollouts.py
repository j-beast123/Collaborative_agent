"""Merge GPU-sharded MedMCQA rollout records into one ordered JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards-root",
        default="outputs/medmcqa_rollouts_100x4_shards",
        help="directory containing gpu_0/, gpu_1/, ... shard directories",
    )
    parser.add_argument(
        "--output",
        default="outputs/medmcqa_rollouts_100x4",
    )
    parser.add_argument("--expected-rollouts", type=int, default=400)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    shards_root = ROOT / args.shards_root
    shard_paths = sorted(shards_root.glob("gpu_*/rollouts.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard rollouts found under {shards_root}")

    records: list[dict] = []
    for path in shard_paths:
        records.extend(read_jsonl(path))
    records.sort(key=lambda row: (int(row["case_index"]), int(row["generation_index"])))

    ids = [row["rollout_id"] for row in records]
    duplicates = [rollout_id for rollout_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate rollout IDs: {duplicates[:5]}")
    if len(records) != args.expected_rollouts:
        raise ValueError(
            f"expected {args.expected_rollouts} rollouts, found {len(records)}; "
            "wait for all GPU shards before merging"
        )

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rollout_path = output / "rollouts.jsonl"
    rollout_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )

    doctors = {}
    for path in sorted(shards_root.glob("gpu_*/doctor_contexts.jsonl")):
        for row in read_jsonl(path):
            doctors[row["case_id"]] = row
    (output / "doctor_contexts.jsonl").write_text(
        "".join(json.dumps(doctors[key], ensure_ascii=False) + "\n" for key in sorted(doctors)),
        encoding="utf-8",
    )

    scored = [row for row in records if row.get("status") == "scored"]
    errors = [row for row in records if row.get("status") == "error"]
    rewards = [row["raw_rubric_reward"] for row in scored if row.get("raw_rubric_reward") is not None]
    label_counts = Counter(
        str(value)
        for row in scored
        for value in (row.get("labels") or {}).values()
    )
    summary = {
        "status": "completed",
        "shards": [str(path.parent.relative_to(ROOT)) for path in shard_paths],
        "cases": len(doctors),
        "rollouts": len(records),
        "scored": len(scored),
        "score_errors": len(errors),
        "rubric_score_counts": dict(label_counts),
        "raw_rubric_reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "raw_rubric_reward_min": min(rewards) if rewards else None,
        "raw_rubric_reward_max": max(rewards) if rewards else None,
        "output": str(output.relative_to(ROOT)),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
