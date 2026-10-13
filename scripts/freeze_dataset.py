"""Freeze a dataset version for the main experiment: the eligible cases and their case-level split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from scripts.audit_eligibility import audit_case, diagnosis_key
from src.agentclinic.adapter import load_cases
from src.agentclinic.selection import eligible_cases, split_cases
from src.common.config import read_config

ROOT = Path(__file__).resolve().parents[1]
MAIN_STATUSES = ("clean_eligible", "eligible_low_information")
ELIGIBILITY = ("A case is eligible if (1) the Doctor and the Agent have asymmetric information, (2) the Agent has at least one "
               "clinically meaningful private evidence item (a physical-examination finding or a test result), and (3) neither "
               "the Doctor's nor the Agent's input directly reveals the target diagnosis through its name, its abbreviation or "
               "a trivial lexical equivalent.")


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Whole numbers that sum to total, in proportion to the weights (largest remainder; ties by name)."""
    scale = sum(weights.values())
    exact = {k: total * w / scale if scale else 0.0 for k, w in weights.items()}
    result = {k: int(v) for k, v in exact.items()}
    for key in sorted(exact, key=lambda k: (-(exact[k] - result[k]), k))[: total - sum(result.values())]:
        result[key] += 1
    return result


def make_split(rows: list[dict], seed: str, test: int, dev: int) -> dict[str, str]:
    by_stratum: dict[str, list[dict]] = {}
    for row in rows:
        by_stratum.setdefault(row["stratum"], []).append(row)
    kept = {name: sum(r["earlier_split"] == "test" for r in members) for name, members in by_stratum.items()}
    if sum(kept.values()) > test:
        raise SystemExit("the earlier TEST split is larger than the requested TEST")
    # TEST: every stratum's proportional share of the whole TEST, less what the kept cases already cover
    deficit = {name: max(0.0, test * len(members) / len(rows) - kept[name]) for name, members in by_stratum.items()}
    new_test = allocate(test - sum(kept.values()), deficit)
    free = {name: len(members) - kept[name] - new_test[name] for name, members in by_stratum.items()}
    new_dev = allocate(dev, free)
    splits = {}
    for name, members in by_stratum.items():
        ordered = sorted((r for r in members if r["earlier_split"] != "test"),
                         key=lambda r: hashlib.sha256(f"{seed}:{r['case_id']}".encode()).hexdigest())
        for index, row in enumerate(ordered):
            splits[row["case_id"]] = "test" if index < new_test[name] else "dev" if index < new_test[name] + new_dev[name] else "train"
        splits.update({r["case_id"]: "test" for r in members if r["earlier_split"] == "test"})
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the first label run's config: dataset path, earlier rule and split")
    parser.add_argument("--version", required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--dev", type=int, required=True)
    parser.add_argument("--audit", help="the audit directory this decision was taken from (recorded)")
    args = parser.parse_args()
    path = ROOT / "data/splits" / f"{args.version}.json"
    if path.exists():
        raise SystemExit(f"{path} exists: a frozen dataset version is not written again")
    config = read_config(args.config)
    dataset_path = ROOT / config["dataset"]["path"]
    source = load_cases(dataset_path, (config.get("doctor_view") or {}).get("patient_fields"))
    earlier_eligible, _ = eligible_cases(source, config["dataset"]["min_test_findings"],
                                         bool(config["dataset"].get("exclude_diagnosis_named_in_private_evidence")))
    earlier = split_cases([c["case_id"] for c in earlier_eligible], config["split"]["seed"], config["split"]["fractions"])
    audit = [dict(audit_case(case, config["dataset"]["min_test_findings"]), earlier_split=earlier.get(case["case_id"])) for case in source]
    pool = [r for r in audit if r["recommended_status"] in MAIN_STATUSES]
    repeated = {key for key, n in Counter(diagnosis_key(r["correct_diagnosis"]) for r in pool).items() if n > 1}
    for row in pool:
        row["diagnosis_key"] = diagnosis_key(row["correct_diagnosis"])
        row["stratum"] = (("low_information" if row["recommended_status"] == "eligible_low_information" else "standard") + "/"
                          + ("repeated_diagnosis" if row["diagnosis_key"] in repeated else "single_diagnosis"))
    seed = f'{config["split"]["seed"]}:{args.version}'
    splits = make_split(pool, seed, args.test, args.dev)
    seen = {r["diagnosis_key"] for r in pool if splits[r["case_id"]] != "test"}
    cases = {r["case_id"]: {
        "split": splits[r["case_id"]], "status": r["recommended_status"], "low_information_kind": r["low_information_kind"],
        "private_real_test_results": r["private_real_test_results"], "private_examination_findings": r["private_examination_findings"],
        "stratum": r["stratum"], "diagnosis_key": r["diagnosis_key"],
        "diagnosis_also_in_train_or_dev": r["diagnosis_key"] in seen if splits[r["case_id"]] == "test" else None,
        "split_in_the_first_label_run": r["earlier_split"]} for r in sorted(pool, key=lambda r: r["case_id"])}
    sizes = {s: sum(c["split"] == s for c in cases.values()) for s in ("train", "dev", "test")}
    test_rows = [c for c in cases.values() if c["split"] == "test"]
    manifest = {
        "dataset_version": args.version, "frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dataset": "agentclinic_medqa_extended", "dataset_path": config["dataset"]["path"],
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "raw_cases": len(source),
        "eligibility": {"definition": ELIGIBILITY, "statuses_in_main": list(MAIN_STATUSES), "audit": "scripts/audit_eligibility.py",
                        "audit_output": args.audit, "status_counts": dict(Counter(r["recommended_status"] for r in audit)),
                        "eligible": len(pool), "uses": "raw case content only; no model output, outcome, trajectory or score"},
        "split": {
            "level": "case", "seed": seed, "sizes": sizes, "procedure": "scripts/freeze_dataset.py:make_split",
            "order": 'sha256("<seed>:<case_id>") inside a stratum',
            "strata": {name: {s: sum(c["stratum"] == name and c["split"] == s for c in cases.values()) for s in ("train", "dev", "test")}
                       for name in sorted({c["stratum"] for c in cases.values()})},
            "low_information_cases": {s: sum(c["status"] == "eligible_low_information" and c["split"] == s for c in cases.values())
                                      for s in ("train", "dev", "test")},
            "test_composition": {
                "kept_from_the_first_label_run_TEST": sum(c["split_in_the_first_label_run"] == "test" for c in test_rows),
                "TRAIN_of_the_first_label_run": sum(c["split_in_the_first_label_run"] == "train" for c in test_rows),
                "DEV_of_the_first_label_run": sum(c["split_in_the_first_label_run"] == "dev" for c in test_rows),
                "never_in_a_run": sum(c["split_in_the_first_label_run"] is None for c in test_rows),
                "note": "TEST cases that were TRAIN or DEV of the fixed two-turn run were seen by models of that run. Nothing of that "
                        "run (labels, predictor, calibration, rubrics, weights, trajectories) is used in the main experiment."},
            "diagnosis_overlap": {
                "test_cases_whose_diagnosis_also_occurs_in_train_or_dev": sum(bool(c["diagnosis_also_in_train_or_dev"]) for c in test_rows),
                "test_cases_with_a_diagnosis_not_in_train_or_dev": sum(c["diagnosis_also_in_train_or_dev"] is False for c in test_rows),
                "note": "not a leak: different clinical cases. The per-case flag allows a diagnosis-non-overlap subset as a sensitivity analysis."}},
        "cases": cases,
        "not_in_main": {r["case_id"]: {"status": r["recommended_status"], "split_in_the_first_label_run": r["earlier_split"],
                                       "manual_review": r["manual_review"]}
                        for r in sorted(audit, key=lambda r: r["case_id"]) if r["recommended_status"] not in MAIN_STATUSES},
    }
    if sizes != {"train": len(pool) - args.test - args.dev, "dev": args.dev, "test": args.test}:
        raise SystemExit(f"split sizes {sizes}")
    if any(c["split"] != "test" for c in cases.values() if c["split_in_the_first_label_run"] == "test"):
        raise SystemExit("an earlier TEST case left TEST")
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    path.chmod(0o444)
    print(json.dumps({k: manifest[k] for k in ("dataset_version", "raw_cases", "eligibility", "split", "manifest_sha256")}
                     | {"path": str(path.relative_to(ROOT))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
