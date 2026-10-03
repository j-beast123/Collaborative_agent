"""Merge and validate GPU-sharded full-trajectory GRPO rollout records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards-root",
        default="outputs/medmcqa_full_trajectories_100x4_shards",
    )
    parser.add_argument(
        "--output",
        default="outputs/medmcqa_full_trajectories_100x4",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--expected-cases", type=int, default=100)
    parser.add_argument("--rollouts-per-case", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def single_identity(records: list[dict], field: str):
    values = {record.get(field) for record in records}
    if len(values) != 1 or None in values:
        raise ValueError(f"trajectory shards mix or omit {field}: {sorted(map(str, values))}")
    return next(iter(values))


def main() -> None:
    args = parse_args()
    shards_root = ROOT / args.shards_root
    shard_paths = sorted(shards_root.glob("gpu_*/trajectories.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard trajectories found under {shards_root}")

    records = [row for path in shard_paths for row in read_jsonl(path)]
    records.sort(key=lambda row: (int(row["case_index"]), int(row["rollout_index"])))
    expected_trajectories = args.expected_cases * args.rollouts_per_case
    if len(records) != expected_trajectories:
        raise ValueError(f"expected {expected_trajectories} trajectories, found {len(records)}")

    trajectory_ids = [str(record["trajectory_id"]) for record in records]
    duplicate_ids = [key for key, count in Counter(trajectory_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"duplicate trajectory IDs: {duplicate_ids[:5]}")

    by_case = defaultdict(list)
    case_indices = {}
    for record in records:
        case_id = str(record["case_id"])
        case_index = int(record["case_index"])
        by_case[case_id].append(record)
        if case_id in case_indices and case_indices[case_id] != case_index:
            raise ValueError(f"case {case_id} has inconsistent case_index values")
        case_indices[case_id] = case_index
        if not record.get("agent_traces") or not record.get("turn_scores"):
            raise ValueError(f"incomplete full trajectory: {record['trajectory_id']}")
        if len(record["agent_traces"]) != len(record["turn_scores"]):
            raise ValueError(f"Agent trace/score count mismatch: {record['trajectory_id']}")
    wrong = {case_id: len(rows) for case_id, rows in by_case.items()
             if len(rows) != args.rollouts_per_case}
    if wrong:
        raise ValueError(f"each case must have {args.rollouts_per_case} trajectories: {wrong}")
    expected_rollout_indices = set(range(args.rollouts_per_case))
    malformed_rollouts = {
        case_id: sorted(int(row["rollout_index"]) for row in rows)
        for case_id, rows in by_case.items()
        if {int(row["rollout_index"]) for row in rows} != expected_rollout_indices
    }
    if malformed_rollouts:
        raise ValueError(f"case rollout indices are incomplete or duplicated: {malformed_rollouts}")
    expected_indices = set(range(args.start_index, args.start_index + args.expected_cases))
    if set(case_indices.values()) != expected_indices or len(by_case) != args.expected_cases:
        raise ValueError(
            f"case coverage mismatch: cases={len(by_case)} "
            f"indices={sorted(set(case_indices.values()))[:5]}..."
        )

    identities = {
        field: single_identity(records, field)
        for field in (
            "rubric_set_version", "rubric_set_content_sha256",
            "rubric_set_validation_status", "scoring_protocol",
        )
    }
    if identities["rubric_set_validation_status"] not in {"validated", "passed"}:
        raise ValueError("full training rollouts must use a validated rubric set")
    sampling = {
        (
            trace.get("sampling_temperature"),
            trace.get("sampling_top_p"),
            trace.get("sampling_top_k"),
        )
        for record in records
        for trace in record["agent_traces"]
    }
    if len(sampling) != 1 or None in next(iter(sampling)):
        raise ValueError(f"trajectory shards mix or omit sampling settings: {sampling}")
    temperature, top_p, top_k = next(iter(sampling))

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --overwrite")
    output_path = output / "trajectories.jsonl"
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    known_outcomes = [record.get("outcome") for record in records if record.get("outcome") is not None]
    summary = {
        "status": "completed",
        "shards": [str(path.parent.relative_to(ROOT)) for path in shard_paths],
        "cases": len(by_case),
        "trajectories": len(records),
        "agent_turns": sum(len(record["agent_traces"]) for record in records),
        "known_outcomes": len(known_outcomes),
        "accuracy": sum(known_outcomes) / len(known_outcomes) if known_outcomes else None,
        **identities,
        "sampling": {"temperature": temperature, "top_p": top_p, "top_k": top_k},
        "output": str(output_path.relative_to(ROOT)),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
