"""The unfiltered rubric set (ablation "No-filtering Rubric")."""

from __future__ import annotations

from typing import Optional

from src.data.schema import digest

SCHEDULE = "no_filtering"
DISABLED = ("validation rerun (no candidate is rejected)", "consolidation into categories (no candidate is merged)",
            "active-category cap (no candidate is pruned)", "selection by failure frequency",
            "any filter on effectiveness or task success")


def raw_rubric_set(analyses: list[dict], iteration: int, previous: Optional[dict] = None, importance: str = "failure_frequency") -> dict:
    """previous: the unfiltered set of iteration - 1 (None at iteration 0)."""
    if previous is not None and (previous.get("rubric_schedule") != SCHEDULE or previous["iteration"] != iteration - 1):
        raise ValueError("the previous set must be the unfiltered set of the iteration before")
    rubrics = [dict(r) for r in (previous or {}).get("rubrics", [])]
    known, dropped = {r["rubric_id"] for r in rubrics}, []
    for record in analyses:
        if record.get("analysis_status") != "valid":
            continue
        if record["rubric_id"] in known:
            dropped.append({"rubric_id": record["rubric_id"], "source_trajectory_id": record["source_trajectory_id"],
                            "reason": "the same analysis record is already in the set"})
            continue
        known.add(record["rubric_id"])
        candidate = record["candidate_rubric"]
        rubrics.append({"rubric_id": record["rubric_id"], "category_id": record["rubric_id"], "category_name": candidate["name"],
                        "name": candidate["name"], "applicability": candidate["applicability"], "scores": candidate["scores"],
                        "failure_count": 1, "weight": 1.0, "discovered_iteration": iteration,
                        "source_rubric_candidates": [record["rubric_id"]], "source_trajectories": [record["source_trajectory_id"]],
                        "source_case_id": record["source_case_id"], "origin": "raw_candidate", "status": "active"})
    payload = {"iteration": iteration, "rubric_schedule": SCHEDULE, "rubrics": rubrics, "excluded_categories": [],
               "max_active_categories": None, "rubric_importance": importance, "canonical_rubric_per_category": False,
               "cumulative": True, "filtering_disabled": list(DISABLED),
               "discovered_by_iteration": {str(i): sum(r["discovered_iteration"] == i for r in rubrics) for i in range(iteration + 1)},
               "technical_dedup": dropped}
    payload["rubric_set_version"] = digest(payload)[:16]
    return payload
