"""Leakage-safe value examples and case-level splitting."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .value_dataset import ValueExample, build_state_text, make_value_example, split_by_case


def load_value_examples(path: str | Path) -> list[ValueExample]:
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "state" not in row or "target" not in row:
            raise ValueError(f"{path}: each row must contain raw state and target objects")
        example = make_value_example(row["state"], row["target"])
        safe_text = build_state_text(row["state"])
        if "text" in row and row["text"] != safe_text:
            raise ValueError(f"{path}: serialized text does not match leakage-safe state text")
        example.text = safe_text
        examples.append(example)
    return examples


def write_split_rows(rows: list[dict], output_dir: str | Path, seed: int = 42) -> None:
    """Write raw {state,target} rows so the loader can revalidate them."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples = [make_value_example(row["state"], row["target"]) for row in rows]
    case_splits = split_by_case(examples, seed=seed)
    allowed = {split: {example.case_id for example in values} for split, values in case_splits.items()}
    for split, case_ids in allowed.items():
        selected = [row for row in rows if str(row["state"]["case_id"]) in case_ids]
        selected_cases = {str(row["state"]["case_id"]) for row in selected}
        if selected_cases != case_ids:
            raise ValueError(f"Split {split} does not preserve its case set")
        (output / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n",
            encoding="utf-8",
        )


def write_split_examples(examples: list[ValueExample], output_dir: str | Path, seed: int = 42) -> None:
    raise ValueError("Use write_split_rows with raw state/target rows; flat examples are not reloadable")
