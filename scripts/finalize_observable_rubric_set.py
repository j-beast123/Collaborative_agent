"""Retain revalidated observable rubrics and write a trainable versioned set."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rubric-set",
        default="configs/rubrics/rubric_guidance_set_11_observable_v2.json",
    )
    parser.add_argument("--validation-glob", required=True)
    parser.add_argument(
        "--output",
        default="outputs/rubric_candidates/rubric_guidance_set_observable_v2_validated.json",
    )
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def content_sha256(rubrics: list[dict]) -> str:
    canonical = json.dumps(
        rubrics, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    source_path = ROOT / args.rubric_set
    output_path = ROOT / args.output
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_path}; use --overwrite or a new path")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_rubrics = source.get("rubrics") or []
    source_hash = content_sha256(source_rubrics)
    source_version = source.get("rubric_set_version")
    if not source_version or source_version != source_hash[:len(str(source_version))]:
        raise ValueError("source rubric_set_version does not match its rubric content")

    paths = sorted(glob.glob(str(ROOT / args.validation_glob)))
    if not paths:
        raise FileNotFoundError(f"no validation files matched: {args.validation_glob}")
    latest = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[str(row["rubric_id"])] = row

    expected_ids = [str(rubric["rubric_id"]) for rubric in source_rubrics]
    missing = sorted(set(expected_ids) - set(latest))
    if missing:
        raise ValueError(f"validation results are missing rubric IDs: {missing}")
    for rubric_id in expected_ids:
        row = latest[rubric_id]
        if row.get("rubric_set_version") != source_version:
            raise ValueError(f"validation version mismatch for {rubric_id}")
        if row.get("rubric_set_content_sha256") != source_hash:
            raise ValueError(f"validation content hash mismatch for {rubric_id}")
        if row.get("validation_protocol") != "ideal_behavior_task_recovery_v2":
            raise ValueError(f"validation protocol mismatch for {rubric_id}")

    statuses = Counter(latest[rubric_id].get("validation_status") for rubric_id in expected_ids)
    if statuses.get("error") or statuses.get("intervention_generation_failure"):
        raise ValueError(f"validation contains unresolved errors: {dict(statuses)}")
    passed_ids = {rubric_id for rubric_id in expected_ids if latest[rubric_id].get("validation_status") == "pass"}
    if args.require_all and len(passed_ids) != len(expected_ids):
        raise ValueError(f"not every revised rubric passed: {dict(statuses)}")
    retained = [rubric for rubric in source_rubrics if rubric["rubric_id"] in passed_ids]
    if not retained:
        raise ValueError("no revised rubric passed validation")

    retained_hash = content_sha256(retained)
    result = {
        "schema_version": source.get("schema_version"),
        "rubric_set_version": retained_hash[:16],
        "parent_rubric_set": args.rubric_set,
        "parent_rubric_set_version": source_version,
        "parent_rubric_set_content_sha256": source_hash,
        "revision": "observable-dialogue rubrics retained after task-recovery intervention validation",
        "validation_status": "passed",
        "validation": {
            "protocol": "ideal_behavior_task_recovery_v2",
            "source_files": [display_path(Path(path)) for path in paths],
            "status_counts": dict(statuses),
            "source_rubrics": len(source_rubrics),
            "retained_rubrics": len(retained),
        },
        "rubrics": retained,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": display_path(output_path),
        "rubric_set_version": result["rubric_set_version"],
        "retained": len(retained),
        "statuses": dict(statuses),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
