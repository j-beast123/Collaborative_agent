"""Load effectiveness pair labels and turn them into BEFORE / AFTER training examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.common.io import read_jsonl
from src.data.schema import identity

from .history import format_input

REQUIRED = ("case_id", "original_case_id", "history_id", "response_id", "split", "history", "response",
            "before", "after", "metadata")


@dataclass
class Example:
    kind: str  # "before" | "after"
    history_id: str
    response_id: Optional[str]
    case_id: str
    original_case_id: str
    text: str
    num_success: int
    num_rollouts: int
    intervention_turn: str = "A1"

    @property
    def empirical_value(self) -> float:
        return self.num_success / self.num_rollouts

    @property
    def parent_state_id(self) -> str:
        """Groups candidates of one history for within-history ranking (AFTER only)."""
        return self.history_id if self.kind == "after" else ""


def _check_counts(counts: Any, where: str) -> None:
    if not isinstance(counts, dict) or type(counts.get("trials")) is not int or counts["trials"] < 1:
        raise ValueError(f"{where}: trials must be a positive integer")
    if type(counts.get("successes")) is not int or not 0 <= counts["successes"] <= counts["trials"]:
        raise ValueError(f"{where}: successes must be an integer in [0, trials]")


def validate_pair(row: dict, *, split: Optional[str] = None, protocol_hash: Optional[str] = None,
                  expected_k: Optional[int] = None) -> None:
    missing = [k for k in REQUIRED if k not in row]
    if missing:
        raise ValueError(f"label row missing {missing}")
    where = f"{row['history_id']}/{row['response_id']}"
    _check_counts(row["before"], where + " before")
    _check_counts(row["after"], where + " after")
    if row["before"]["trials"] != row["after"]["trials"]:
        raise ValueError(f"{where}: BEFORE and AFTER must use the same K")
    if expected_k is not None and row["after"]["trials"] != expected_k:
        raise ValueError(f"{where}: K={row['after']['trials']} does not match continuation k={expected_k}")
    if split is not None and row["split"] != split:
        raise ValueError(f"{where}: row split {row['split']!r} used as {split!r}")
    if row["split"] == "eval":
        raise ValueError(f"{where}: EVAL rows never train the effectiveness model")
    if protocol_hash is not None and row["metadata"].get("continuation_protocol_hash") != protocol_hash:
        raise ValueError(f"{where}: labels were produced under a different continuation protocol")
    if not isinstance(row["response"], str) or not row["response"].strip():
        raise ValueError(f"{where}: empty response")


def load_pairs(path: str | Path, **checks) -> list[dict]:
    rows = read_jsonl(path)
    keys = set()
    for row in rows:
        validate_pair(row, **checks)
        key = (row["history_id"], row["response_id"])
        if key in keys:
            raise ValueError(f"duplicate pair {key}")
        keys.add(key)
    return rows


def to_examples(pairs: list[dict], input_format: str = "v1") -> list[Example]:
    """input_format: how the text is built (history.format_input); "turn_v1" names the intervention turn of
    the row (intervention_turn, A1 when the row has none: the A1 labels were written before A2 labels existed)."""
    examples, before = [], {}
    for row in pairs:
        turn = row.get("intervention_turn") or "A1"
        counts = (row["before"]["successes"], row["before"]["trials"])
        if row["history_id"] in before and before[row["history_id"]] != counts:
            raise ValueError(f"history {row['history_id']} has inconsistent BEFORE counts across candidates")
        if row["history_id"] not in before:
            before[row["history_id"]] = counts
            examples.append(Example("before", row["history_id"], None, row["case_id"], row["original_case_id"],
                                    format_input(row["history"], None, input_format, turn), *counts, turn))
        examples.append(Example("after", row["history_id"], row["response_id"], row["case_id"],
                                row["original_case_id"], format_input(row["history"], row["response"], input_format, turn),
                                row["after"]["successes"], row["after"]["trials"], turn))
    return examples


def assert_case_disjoint(splits: dict[str, list[dict]]) -> None:
    """No original clinical case may appear in two splits."""
    seen = {}
    for split, rows in splits.items():
        for row in rows:
            other = seen.setdefault(row["original_case_id"], split)
            if other != split:
                raise ValueError(f"original case {row['original_case_id']} appears in {other} and {split}")


def assert_manifest_splits(splits: dict[str, list[dict]], manifest: dict, source_dataset: str) -> None:
    """Each row's case must be in the case-level split manifest under the split it is used for."""
    for split, rows in splits.items():
        for row in rows:
            member = manifest["members"].get(identity(source_dataset, str(row["case_id"])))
            if member is None:
                raise ValueError(f"case {row['case_id']} is not in the split manifest")
            if member["split"] == "eval":
                raise ValueError(f"held-out EVAL case {row['case_id']} in effectiveness labels")
            if member["split"] != split:
                raise ValueError(f"case {row['case_id']} belongs to {member['split']}, used as {split}")
