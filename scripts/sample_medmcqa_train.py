"""Create a reproducible, leakage-free Medicine subset from the MedMCQA train Arrow stream."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


OPTIONS = ("A", "B", "C", "D")
ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_arrow_rows(path: Path) -> list[dict]:
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_stream(source).read_all().to_pylist()


def normalize_row(row: dict, source_index: int) -> dict:
    correct_index = int(row["cop"])
    if correct_index not in range(4):
        raise ValueError(f"unexpected MedMCQA cop={row['cop']} for source row {source_index}")
    options = {letter: row[key] for letter, key in zip(OPTIONS, ("opa", "opb", "opc", "opd"))}
    correct_option = OPTIONS[correct_index]
    return {
        "case_id": f"medmcqa_medicine_train_{source_index:06d}",
        "source_dataset": "openlifescienceai/medmcqa",
        "source_question_id": row["id"],
        "source_split": "train",
        "language": "en",
        "question": row["question"],
        "options": options,
        "ground_truth": {
            "correct_option": correct_option,
            "correct_text": options[correct_option],
        },
        "original_ground_truth": None,
        "reference_explanation": row.get("exp"),
        "subject": row.get("subject_name"),
        "topic": row.get("topic_name"),
        "metadata": {
            "source_record_index": source_index,
            "choice_type": row.get("choice_type"),
            "sampling_source": "MedMCQA train Arrow stream",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-arrow", required=True, type=Path)
    parser.add_argument("--exclude-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    excluded_ids = {row["source_question_id"] for row in read_jsonl(args.exclude_jsonl)}
    source_rows = load_arrow_rows(args.source_arrow)
    pool = [
        (index, row)
        for index, row in enumerate(source_rows)
        if row.get("subject_name") == "Medicine" and row.get("id") not in excluded_ids
    ]
    if len(pool) < args.size:
        raise SystemExit(f"Medicine pool after exclusions has {len(pool)} rows; need {args.size}")

    selected = random.Random(args.seed).sample(pool, args.size)
    selected.sort(key=lambda pair: pair[0])
    rows = [normalize_row(row, index) for index, row in selected]
    selected_ids = [row["source_question_id"] for row in rows]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("duplicate source_question_id in sampled rows")
    if set(selected_ids) & excluded_ids:
        raise ValueError("sample overlaps excluded source IDs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "source": str(args.source_arrow),
        "excluded_source": str(args.exclude_jsonl),
        "subject": "Medicine",
        "source_split": "train",
        "seed": args.seed,
        "requested_size": args.size,
        "selected_size": len(rows),
        "pool_size_after_exclusion": len(pool),
        "excluded_count": len(excluded_ids),
        "selected_source_question_ids": selected_ids,
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_size": manifest["selected_size"],
        "pool_size_after_exclusion": manifest["pool_size_after_exclusion"],
        "excluded_count": manifest["excluded_count"],
        "seed": manifest["seed"],
        "output": str(args.output),
        "manifest": str(args.manifest),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
