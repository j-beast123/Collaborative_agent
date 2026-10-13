"""Rubric filtering and weighting by the effectiveness gap (the main method's "weighting and remove" step)."""

from __future__ import annotations

from statistics import mean
from typing import Optional

HIGH, LOW = 2, 0
RETAINED, INSUFFICIENT, NONPOSITIVE, VALIDATION_FAILED = "retained", "insufficient_support", "nonpositive_effectiveness_gap", "validation_failed"
BEYOND_BUDGET = "beyond_rubric_budget"


def effectiveness_gap(observations: list[dict], rubric_id: str, *, high: int = HIGH, low: int = LOW, min_group_n: int = 1) -> dict:
    """observations: one row per scored response, {"labels": {rubric_id: "NA" | 0 | 1 | 2}, "effectiveness": float | None}."""
    groups = {high: [], low: []}
    applicable = partial = without = 0
    for row in observations:
        label = (row.get("labels") or {}).get(rubric_id)
        if label in (None, "NA"):
            continue
        applicable += 1
        if row.get("effectiveness") is None:
            without += 1
        elif label in groups:
            groups[label].append(float(row["effectiveness"]))
        else:
            partial += 1
    high_n, low_n = len(groups[high]), len(groups[low])
    supported = high_n >= min_group_n and low_n >= min_group_n
    gap = mean(groups[high]) - mean(groups[low]) if supported else None
    return {"applicable_n": applicable, "high_n": high_n, "low_n": low_n, "partial_n": partial,
            "applicable_without_effectiveness_n": without,
            "high_mean_effectiveness": mean(groups[high]) if high_n else None,
            "low_mean_effectiveness": mean(groups[low]) if low_n else None,
            "effectiveness_gap": gap,
            "status": INSUFFICIENT if gap is None else RETAINED if gap > 0 else NONPOSITIVE}


def filter_and_weight(rubrics: list[dict], observations: list[dict], *, high: int = HIGH, low: int = LOW,
                      min_group_n: int = 1, max_rubrics: Optional[int] = None) -> tuple[list[dict], list[dict]]:
    """(the rubrics kept, each with weight = its effectiveness gap; one row per rubric with the numbers and the decision)."""
    kept, rows = [], []
    for rubric in rubrics:
        measured = effectiveness_gap(observations, rubric["rubric_id"], high=high, low=low, min_group_n=min_group_n)
        retained = measured["status"] == RETAINED
        row = {**measured, "retained": retained, "final_weight": measured["effectiveness_gap"] if retained else None,
               "removal_reason": None if retained else measured["status"]}
        rows.append({"rubric_id": rubric["rubric_id"], **row})
        if retained:
            kept.append({**rubric, "weight": measured["effectiveness_gap"],
                         **{name: measured[name] for name in ("high_n", "low_n", "high_mean_effectiveness",
                                                              "low_mean_effectiveness", "effectiveness_gap")}})
    if max_rubrics is not None and len(kept) > max_rubrics:  # the budget: the largest gaps, in the rubrics' own order
        ranked = sorted(kept, key=lambda r: (-r["effectiveness_gap"], str(r["rubric_id"])))
        dropped = {r["rubric_id"] for r in ranked[max_rubrics:]}
        kept = [r for r in kept if r["rubric_id"] not in dropped]
        for row in rows:
            if row["rubric_id"] in dropped:
                row.update(retained=False, final_weight=None, removal_reason=BEYOND_BUDGET, status=BEYOND_BUDGET)
    return kept, rows


def assert_positive_weights(rubric_set: dict) -> Optional[str]:
    """Why a rubric set is not a valid effectiveness-gap set; None when every rubric has a weight > 0."""
    bad = [r["rubric_id"] for r in rubric_set.get("rubrics", []) if not (isinstance(r.get("weight"), (int, float)) and r["weight"] > 0)]
    return f"rubrics without a positive weight: {bad}" if bad else None
