"""Print per-GPU collaboration progress and correctness counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="GPU shard output directory")
    parser.add_argument("--total", type=int, required=True, help="Total assigned cases")
    parser.add_argument("--gpus", type=int, default=7)
    args = parser.parse_args()

    base = args.total // args.gpus
    remainder = args.total % args.gpus
    grand = {"assigned": args.total, "done": 0, "correct": 0, "incorrect": 0, "errors": 0}
    print(f"Output: {args.root}")
    print("GPU | done/assigned | correct | incorrect | error | valid accuracy")
    print("----+---------------+---------+-----------+-------+---------------")

    for gpu in range(args.gpus):
        assigned = base + (1 if gpu < remainder else 0)
        directory = Path(args.root) / f"gpu_{gpu}"
        valid = rows(directory / "trajectories.jsonl")
        errors = rows(directory / "trajectories_failures.jsonl")
        correct = sum(row.get("is_correct") is True for row in valid)
        incorrect = sum(row.get("is_correct") is False for row in valid)
        error_count = len(errors)
        done = correct + incorrect + error_count
        accuracy = 100 * correct / (correct + incorrect) if correct + incorrect else 0.0
        print(
            f"GPU{gpu} | {done:3d}/{assigned:<3d}       | {correct:7d} "
            f"| {incorrect:9d} | {error_count:5d} | {accuracy:6.1f}%"
        )
        grand["done"] += done
        grand["correct"] += correct
        grand["incorrect"] += incorrect
        grand["errors"] += error_count

    valid_total = grand["correct"] + grand["incorrect"]
    accuracy = 100 * grand["correct"] / valid_total if valid_total else 0.0
    print("----+---------------+---------+-----------+-------+---------------")
    print(
        f"ALL | {grand['done']:3d}/{grand['assigned']:<3d}       | "
        f"{grand['correct']:7d} | {grand['incorrect']:9d} | {grand['errors']:5d} "
        f"| {accuracy:6.1f}%"
    )
    print(f"Remaining: {grand['assigned'] - grand['done']}")


if __name__ == "__main__":
    main()
