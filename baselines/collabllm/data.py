"""Verify original bytes and ordering without downloading or resampling data."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.online_rl.guards import assert_not_protected, question_sha256

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "configs/online_rl/tree_v21_five_gpu_shared5_random10_promptv7.json"


def audit_data(*, train=None, train_order=None, test=None):
    spec = json.loads(Path(__file__).with_name("adaptation_spec.json").read_text())
    if hashlib.sha256((ROOT / REFERENCE).read_bytes()).hexdigest() != spec["reference_config_sha256"]:
        raise RuntimeError("Reference protocol changed from the recorded comparison setting")
    from baselines.common.protocol import apply_common, settings
    from src.online_rl.tree.experiment import load_config
    cfg = apply_common(load_config(spec['reference_outcome_only_config']))
    count = settings()['train_cases']
    paths = {
        "train": (train or ROOT / cfg["data"]["train"], cfg["data"]["train_sha256"]),
        "train_order": (train_order or ROOT / cfg["five_gpu"]["train_order_file"],
                        cfg["five_gpu"]["train_order_sha256"]),
        "test": (test or ROOT / cfg["five_gpu"]["eval_file"], cfg["five_gpu"]["eval_sha256"]),
    }
    files = {}
    for role, (path, expected) in paths.items():
        path = Path(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        files[role] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual,
                       "status": "missing" if actual is None else "match" if actual == expected else "mismatch"}
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
              "reference_config": REFERENCE,
              "reference_config_sha256": hashlib.sha256((ROOT / REFERENCE).read_bytes()).hexdigest(),
              "train_verified": False, "test_verified": False, "all_verified": False,
              "errors": [], "train_case_order": None, "test_case_order": None}
    cases = tests = None
    if all(files[r]["status"] == "match" for r in ("train", "train_order")):
        try:
            cases = _cases(paths["train"][0])
            order = json.loads(Path(paths["train_order"][0]).read_text())
            ids = [c["case_id"] for c in cases]
            if (len(ids) != count or len(set(ids)) != count or len(order) != count
                    or len(set(order)) != count or order != ids):
                raise ValueError(f"Train/order must preserve the same {count} unique IDs in original file row order")
            if set(ids) & set(cfg["data"]["excluded_calibration_case_ids"]):
                raise ValueError("Calibration case in training data")
            if any(c.get("source_split") != "train" for c in cases):
                raise ValueError("Non-train source in training data")
            assert_not_protected(cases, "CollabLLM train pilot")
            report.update(train_verified=True, train_case_order=order)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            report["errors"].append(str(exc))
    if files["test"]["status"] == "match":
        try:
            tests = _cases(paths["test"][0])
            ids = [c["case_id"] for c in tests]
            if len(ids) != 622 or len(set(ids)) != 622:
                raise ValueError("Test must contain 622 unique cases")
            report.update(test_verified=True, test_case_order=ids)
        except (ValueError, KeyError, TypeError) as exc:
            report["errors"].append(str(exc))
    if report["train_verified"] and report["test_verified"]:
        if {question_sha256(c["question"]) for c in cases} & {question_sha256(c["question"]) for c in tests}:
            report["errors"].append("Train/test question overlap")
        else:
            report["all_verified"] = True
    return report


def _cases(path):
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for case in cases:
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError("Missing question")
        if case["ground_truth"]["correct_option"] not in case["options"]:
            raise ValueError("Gold option absent from options")
    return cases


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("train", "train-order", "test", "out"):
        p.add_argument("--" + flag)
    args = p.parse_args()
    result = audit_data(train=args.train, train_order=args.train_order, test=args.test)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["all_verified"] else 1)
