"""Case-level split and pilot sampling. Both depend only on case ids and case content."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from src.data.schema import digest, identity
from src.data.split import EXTERNAL_ASSIGNMENT, validate_manifest

from .adapter import diagnosis_named_in_private_evidence, eligibility, leaf_values, load_cases

SPLITS = ("train", "dev", "test")


def split_cases(case_ids: list[str], seed: int, fractions: dict[str, float]) -> dict[str, str]:
    """{case_id: split}; deterministic in (seed, case_id), so a case never changes split."""
    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError(f"fractions must cover {SPLITS} and sum to 1")
    ordered = sorted(case_ids, key=lambda cid: hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest())
    n_train = round(len(ordered) * fractions["train"])
    n_dev = round(len(ordered) * fractions["dev"])
    return {cid: "train" if i < n_train else "dev" if i < n_train + n_dev else "test" for i, cid in enumerate(ordered)}


def split_manifest(splits: dict[str, str], *, source_dataset: str, dataset_version: str, seed: int,
                   fractions: dict[str, float]) -> dict:
    """The case-level split in the manifest format of src/data/split.py (TEST is that format's "eval"),
    so effectiveness training can check that no case crosses splits."""
    name = {"train": "train", "dev": "dev", "test": "eval"}
    members = {}
    for case_id in sorted(splits):
        key = identity(source_dataset, case_id)
        members[key] = {"original_key": key, "group": key, "split": name[splits[case_id]],
                        "dataset_version": dataset_version, "synthetic": False}
    payload = {"schema_version": 1, "algorithm": EXTERNAL_ASSIGNMENT, "seed": seed,
               "proportions": {name[k]: v for k, v in fractions.items()}, "duplicate_groups": [],
               "assignment": "src/agentclinic/selection.py:split_cases (sha256 of seed and case id)",
               "group_counts": {s: sum(m["split"] == s for m in members.values()) for s in ("train", "dev", "eval")},
               "members": members}
    payload["manifest_hash"] = digest(payload)
    return validate_manifest(payload)


def split_fractions(config: dict, splits: dict[str, str]) -> tuple[int, dict[str, float]]:
    """(seed, fractions) for split_manifest: the configured ones, or for a frozen dataset version the shares of its
    split (the assignment itself is the version's; src/data/split.py wants an integer seed and proportions)."""
    if (config.get("dataset") or {}).get("manifest"):
        return int(str(config["split"]["seed"]).split(":")[0]), {s: sum(v == s for v in splits.values()) / len(splits) for s in SPLITS}
    return config["split"]["seed"], config["split"]["fractions"]


def eligible_cases(cases: list[dict], min_test_findings: int, exclude_named_diagnosis: bool = False
                   ) -> tuple[list[dict], dict[str, int]]:
    """(eligible cases, {reason: count}) from the case content only."""
    kept, reasons = [], {}
    for case in cases:
        why = eligibility(case)
        if not why and len(leaf_values(case["private_evidence"]["Test_Results"])) < min_test_findings:
            why = [f"fewer than {min_test_findings} test-result values"]
        if not why and exclude_named_diagnosis and diagnosis_named_in_private_evidence(case):
            why = ["diagnosis named in consultant-only evidence"]
        for reason in why:
            reasons[reason] = reasons.get(reason, 0) + 1
        if not why:
            kept.append(case)
    return kept, reasons


def load_manifest(path: Path, dataset_path: Path) -> dict:
    """A frozen dataset version, checked against its own hash and the dataset file it was made from."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != manifest["manifest_sha256"]:
        raise ValueError(f"{path} was edited after it was frozen")
    if hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError(f"{dataset_path} is not the dataset file {path} was frozen from")
    return manifest


def select_eligible(config: dict, root) -> tuple[list[dict], dict[str, str], dict[str, int]]:
    """(eligible cases, {case_id: split}, {reason a case is left out: count}) of a run config."""
    dataset = config["dataset"]
    source = load_cases(root / dataset["path"], (config.get("doctor_view") or {}).get("patient_fields"))
    if dataset.get("manifest"):
        manifest = load_manifest(root / dataset["manifest"], root / dataset["path"])
        splits = {case_id: row["split"] for case_id, row in manifest["cases"].items()}
        reasons: dict[str, int] = {}
        for row in manifest["not_in_main"].values():
            reasons[row["status"]] = reasons.get(row["status"], 0) + 1
        return [c for c in source if c["case_id"] in splits], splits, reasons
    eligible, reasons = eligible_cases(source, dataset["min_test_findings"],
                                       bool(dataset.get("exclude_diagnosis_named_in_private_evidence")))
    return eligible, split_cases([c["case_id"] for c in eligible], config["split"]["seed"], config["split"]["fractions"]), reasons


def sample_pilot(cases: list[dict], splits: dict[str, str], *, split: str, n: int, seed: int) -> list[dict]:
    pool = sorted((c for c in cases if splits[c["case_id"]] == split), key=lambda c: c["case_id"])
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))), key=lambda c: c["case_id"])


def split_eligible_cases(config: dict, root, *, allowed_splits: tuple | None = None) -> dict[str, dict]:
    """{case_id: case with its split} for every eligible case of a run config."""
    root = Path(root)
    if allowed_splits is not None and config["dataset"].get("manifest"):
        dataset = config["dataset"]
        manifest = load_manifest(root / dataset["manifest"], root / dataset["path"])
        splits = {cid: row["split"] for cid, row in manifest["cases"].items() if row["split"] in allowed_splits}
        eligible = load_cases(root / dataset["path"], (config.get("doctor_view") or {}).get("patient_fields"), case_ids=set(splits))
    else:
        eligible, splits, _ = select_eligible(config, root)
        if allowed_splits is not None:
            eligible = [c for c in eligible if splits[c["case_id"]] in allowed_splits]
    return {c["case_id"]: dict(c, split=splits[c["case_id"]]) for c in eligible}
