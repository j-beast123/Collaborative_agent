"""Build rerun-validation inputs for a revised observable rubric guidance set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judges.failure_analysis import load_latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rubric-set",
        default="configs/rubrics/rubric_guidance_set_11_observable_v2.json",
    )
    parser.add_argument(
        "--source-candidates",
        default="outputs/rubric_candidates/rubric_candidates_300_gpt56_sol.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/rubric_candidates/rubric_guidance_set_11_observable_v2_revalidation.jsonl",
    )
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
    rubric_path = ROOT / args.rubric_set
    source_path = ROOT / args.source_candidates
    output_path = ROOT / args.output
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_path}; use --overwrite or a new path")

    rubric_set = json.loads(rubric_path.read_text(encoding="utf-8"))
    rubrics = rubric_set.get("rubrics") or []
    if not rubrics:
        raise ValueError(f"rubric set is empty: {rubric_path}")
    semantic_hash = content_sha256(rubrics)
    version = rubric_set.get("rubric_set_version")
    if not version or version != semantic_hash[:len(str(version))]:
        raise ValueError("rubric_set_version does not match the revised rubric content")

    source_by_trajectory = load_latest(source_path)
    records = []
    seen_sources = set()
    for rubric in rubrics:
        source_trajectory_id = str(rubric.get("source_trajectory_id") or "")
        if not source_trajectory_id or source_trajectory_id in seen_sources:
            raise ValueError(f"missing or duplicate source trajectory: {source_trajectory_id!r}")
        seen_sources.add(source_trajectory_id)
        source = source_by_trajectory.get(source_trajectory_id)
        if source is None:
            raise ValueError(f"source candidate is missing for {source_trajectory_id}")
        if source.get("analysis_status") != "valid" or not isinstance(source.get("analysis"), dict):
            raise ValueError(f"source candidate is not valid for {source_trajectory_id}")

        record = copy.deepcopy(source)
        analysis = record["analysis"]
        analysis.update({
            "rubric_name": rubric["name"],
            "applicability": rubric["applicability_condition"],
            "ideal_response_behavior": rubric["required_behavior"],
            "score_0": rubric["scores"]["0"],
            "score_1": rubric["scores"]["1"],
            "score_2": rubric["scores"]["2"],
        })
        # The revised intervention is dialogue-observable only. Original
        # evidence spans and intervention turn remain solely for replay.
        analysis.pop("agent_side_state", None)
        analysis.pop("user_side_context", None)
        record.update({
            "rubric_id": rubric["rubric_id"],
            "candidate_rubric": {
                "name": rubric["name"],
                "applicability_condition": rubric["applicability_condition"],
                "required_behavior": rubric["required_behavior"],
                "scores": rubric["scores"],
            },
            "rubric_set_version": version,
            "rubric_set_content_sha256": semantic_hash,
            "rubric_revision": rubric.get("revision"),
            "revalidation_reason": "observable applicability and behavior scope revision",
        })
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": display_path(output_path),
        "rubrics": len(records),
        "rubric_set_version": version,
        "rubric_set_content_sha256": semantic_hash,
        "source_candidates": args.source_candidates,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
