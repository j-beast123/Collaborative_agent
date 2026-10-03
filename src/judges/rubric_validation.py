"""Rubric Validation: rerun each candidate's failed turn under its ideal behavior num_reruns times; pass when at least
min_successes reruns end with the correct final diagnosis (main: 3 reruns, keep at 2).

`intervene` (frozen score-2 response) and `behavior` (behavioral gate) are diagnostic-only options, off in the main config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.collaboration.rerun import intervention_instructions


GENERATION_FAILURE = "intervention_generation_failure"
DECIDED = ("pass", "fail", GENERATION_FAILURE)


def load_validations(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["rubric_id"]] = record
    return latest


def validate_candidates(
    records: list[dict],
    trajectories: dict[str, dict],
    rerun: Callable[[dict, int, str, int], dict],
    *,
    num_reruns: int,
    min_successes: int,
    output_path: str | Path,
    intervene: Callable[[dict, dict, str], dict] | None = None,
    behavior: Callable[[dict, dict, str], dict] | None = None,
    validation_metadata: dict | None = None,
) -> dict[str, int]:
    """rerun(trajectory, turn_index, instruction, rerun_index) -> rerun result."""
    if type(num_reruns) is not int or num_reruns < 1 or not 1 <= min_successes <= num_reruns:
        raise ValueError("Need num_reruns >= 1 and 1 <= min_successes <= num_reruns")
    output = Path(output_path)
    existing = load_validations(output)
    if validation_metadata:
        identity_fields = (
            "rubric_set_version", "rubric_set_content_sha256", "validation_protocol",
        )
        for rubric_id, row in existing.items():
            mismatches = {
                key: (row.get(key), validation_metadata.get(key))
                for key in identity_fields
                if validation_metadata.get(key) is not None and row.get(key) != validation_metadata.get(key)
            }
            if mismatches:
                raise ValueError(
                    f"existing validation output has incompatible metadata for {rubric_id}: {mismatches}"
                )
    done = {rid for rid, r in existing.items() if r["validation_status"] in DECIDED}
    counts = {"pass": 0, "fail": 0, "error": 0, "skipped_done": 0, **({GENERATION_FAILURE: 0} if intervene else {})}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            if record.get("analysis_status") != "valid":
                continue
            if record["rubric_id"] in done:
                counts["skipped_done"] += 1
                continue
            trajectory = trajectories[record["source_trajectory_id"]]
            analysis = record["analysis"]
            instruction = intervention_instructions(
                analysis["ideal_response_behavior"],
                analysis.get("applicability", ""),
            )
            first_instruction = instruction["first"]
            reruns, error, intervention = [], None, None
            if intervene is not None:
                try:
                    intervention = intervene(record, trajectory, first_instruction)
                except Exception as exc:  # the generation or its check could not be run: retried on the next run
                    error = f"intervention {type(exc).__name__}: {exc}"
            frozen = intervention is not None and intervention["status"] == "frozen"
            for index in range(num_reruns if (intervene is None or frozen) else 0):
                try:
                    result = rerun(trajectory, analysis["intervention_turn_id"], instruction, index,
                                   *([{"text": intervention["text"], "assessment": intervention.get("assessment") or ""}]
                                     if frozen else []))
                except Exception as exc:  # one failed rerun must not stop the whole validation
                    error = f"{type(exc).__name__}: {exc}"
                    break
                reruns.append(result)
                if result.get("status") != "completed":
                    error = f"rerun status {result.get('status')}"
                    break
            successes = sum(bool(r.get("correct")) for r in reruns)
            status = "error" if error else GENERATION_FAILURE if intervene is not None and not frozen else \
                ("pass" if successes >= min_successes else "fail")
            behavioral = None  # diagnostic-only behavioral gate
            if behavior is not None and not error:
                try:
                    behavioral = behavior(record, trajectory, first_instruction)
                except Exception as exc:  # the behavior samples could not be scored: retried on the next run
                    error, status = f"behavior {type(exc).__name__}: {exc}", "error"
            if behavioral is not None:
                shown = sum(x.get("label") == 2 for x in behavioral["samples"])
                behavioral.update(shown=shown, absent_before=behavioral["before"] != 2,
                                  passed=behavioral["before"] != 2 and shown >= min_successes)
                status = "pass" if behavioral["passed"] else "fail"
            counts[status] += 1
            handle.write(json.dumps({
                **(validation_metadata or {}),
                "rubric_id": record["rubric_id"],
                "source_trajectory_id": record["source_trajectory_id"],
                "intervention_turn_id": analysis["intervention_turn_id"],
                "validation_status": status,
                "successes": successes,
                "num_reruns": num_reruns,
                "min_successes": min_successes,
                "error": error,
                **({"intervention": intervention} if intervene is not None else {}),
                **({"criterion": "behavior", "behavior": behavioral,
                    "task_recovery": {"successes": successes, "reruns": len(reruns), "min_successes": min_successes,
                                      "intervention_status": (intervention or {}).get("status")}} if behavior is not None else {}),
                "reruns": reruns,
            }, ensure_ascii=False) + "\n")
            handle.flush()
    return counts


def passed_records(records: list[dict], validations: dict[str, dict], required: bool) -> list[dict]:
    """Candidates allowed into consolidation."""
    if not required:
        return records
    return [r for r in records if validations.get(r.get("rubric_id"), {}).get("validation_status") == "pass"]
