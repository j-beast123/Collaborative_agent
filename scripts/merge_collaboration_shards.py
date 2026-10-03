"""Merge collaboration trajectory shards and report final-answer accuracy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    root = ROOT / args.shards_root
    trajectory_paths = sorted(root.glob("gpu_*/trajectories.jsonl"))
    if not trajectory_paths:
        raise FileNotFoundError(f"no trajectory shards under {root}")
    rows = [row for path in trajectory_paths for row in read_jsonl(path)]
    failures = [
        row
        for path in sorted(root.glob("gpu_*/trajectories_failures.jsonl"))
        for row in read_jsonl(path)
    ]
    ids = [row["trajectory_id"] for row in rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate trajectory IDs: {duplicates[:5]}")
    if len(rows) + len(failures) != args.expected:
        raise ValueError(
            f"expected {args.expected} completed cases, found {len(rows)} valid + {len(failures)} invalid"
        )
    rows.sort(key=lambda row: row["case_id"])
    destination = ROOT / args.output
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "trajectories.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (destination / "trajectories_failures.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    correct = sum(bool(row.get("is_correct")) for row in rows)
    summary = {
        "status": "completed",
        "expected_cases": args.expected,
        "valid_trajectories": len(rows),
        "invalid_trajectories": len(failures),
        "correct": correct,
        "incorrect": len(rows) - correct,
        "accuracy_over_valid": correct / len(rows) if rows else None,
        "accuracy_over_all": correct / args.expected if args.expected else None,
        "termination_reasons": dict(Counter(row.get("termination_reason") for row in rows)),
        "output": str(destination.relative_to(ROOT)),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
