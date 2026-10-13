"""Build a conditional guidance set from validated rubric candidates."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--validations-glob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    passed: dict[str, dict] = {}
    validation_paths = sorted(Path(p) for p in glob.glob(args.validations_glob))
    if not validation_paths:
        raise SystemExit(f"no validation files matched: {args.validations_glob}")
    for path in validation_paths:
        for row in read_jsonl(path):
            if row.get("validation_status") == "pass":
                rubric_id = row.get("rubric_id")
                if rubric_id:
                    passed[str(rubric_id)] = row

    candidates = {}
    for row in read_jsonl(ROOT / args.candidates):
        rubric_id = row.get("rubric_id")
        if rubric_id in passed and row.get("analysis_status") == "valid":
            candidates[str(rubric_id)] = row

    missing = sorted(set(passed) - set(candidates))
    if missing:
        raise SystemExit(f"validated rubric candidates missing: {missing}")

    rubrics = []
    for rubric_id in sorted(candidates, key=lambda value: candidates[value].get("source_case_id", "")):
        row = candidates[rubric_id]
        candidate = row["candidate_rubric"]
        applicability = candidate.get("applicability") or {}
        if isinstance(applicability, dict):
            # The agent must decide applicability from the visible dialogue. The
            # hidden pre-response agent state is intentionally not exposed.
            condition = applicability.get("user_side_context", "")
        else:
            condition = applicability
        scores = candidate.get("scores") or {}
        ideal = (scores.get("2") or {}).get("behavior", "")
        score_behaviors = {
            level: (scores.get(level) or {}).get("behavior", "")
            for level in ("0", "1", "2")
        }
        if not condition or not all(score_behaviors.values()):
            raise SystemExit(f"rubric {rubric_id} has no usable condition or score-2 behavior")
        rubrics.append(
            {
                "rubric_id": rubric_id,
                "source_case_id": row.get("source_case_id"),
                "source_trajectory_id": row.get("source_trajectory_id"),
                "name": candidate.get("name", rubric_id),
                "applicability_condition": condition,
                "required_behavior": ideal,
                "scores": score_behaviors,
            }
        )

    output = {
        "schema_version": "conditional-rubric-guidance-v1",
        "rubric_set_version": digest(rubrics),
        "source_candidates": str(Path(args.candidates)),
        "source_validations": args.validations_glob,
        "selection": "validated pass candidates; score-2 behavior; visible-dialogue applicability only",
        "rubrics": rubrics,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "rubric_count": len(rubrics), "rubric_set_version": output["rubric_set_version"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
