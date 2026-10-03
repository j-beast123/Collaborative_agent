"""RQ3: fixed vs evolving rubrics, rubric category evolution, failure resolution."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

ARMS = ("fixed_initial", "evolving", "fixed_final_oracle")


def rubric_schedule(arm: str, iterations: int, evolving_sets: list[str]) -> list[str]:
    """Rubric-set version used for training at each iteration of an arm."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if len(evolving_sets) < 1 or (arm != "fixed_initial" and len(evolving_sets) < iterations):
        raise ValueError("evolving_sets must contain R_0 .. R_(iterations-1)")
    if arm == "fixed_initial":
        return [evolving_sets[0]] * iterations
    if arm == "evolving":
        return list(evolving_sets[:iterations])
    return [evolving_sets[iterations - 1]] * iterations


def bubble_rows(registry: list[dict], top_k: int | None = None) -> list[dict]:
    """Tidy rows for the category x iteration bubble matrix from a category registry."""
    categories = [c for c in registry if c.get("status") != "merged"]
    totals = defaultdict(int)
    for category in registry:
        for iteration, count in category.get("frequency_history", {}).items():
            totals[iteration] += count
    keep = {c["category_id"] for c in categories}
    if top_k is not None:
        ranked = sorted(categories, key=lambda c: (-sum(c.get("frequency_history", {}).values()), c["category_id"]))
        keep = {c["category_id"] for c in ranked[:top_k]}
    counts = defaultdict(int)
    names = {}
    for category in categories:
        key = category["category_id"] if category["category_id"] in keep else "OTHER"
        names[key] = category["name"] if key != "OTHER" else "OTHER"
        for iteration, count in category.get("frequency_history", {}).items():
            counts[(key, int(iteration))] += count
    return [{"category_id": key, "category_name": names[key], "iteration": iteration, "failure_count": count,
             "proportion": count / totals[str(iteration)] if totals[str(iteration)] else 0.0}
            for (key, iteration), count in sorted(counts.items(), key=lambda x: (x[0][1], x[0][0]))]


def probe_transitions(results: list[dict]) -> list[dict]:
    """Failure resolution on a FIXED probe set, per category and consecutive iterations."""
    table = defaultdict(dict)
    for row in results:
        table[(row["category_id"], row["probe_id"])][int(row["iteration"])] = bool(row["success"])
    counts = defaultdict(lambda: {"fail_to_success": 0, "fail_to_fail": 0, "success_to_fail": 0,
                                  "success_to_success": 0})
    for (category, _), by_iteration in table.items():
        iterations = sorted(by_iteration)
        for t, nxt in zip(iterations, iterations[1:]):
            before, after = by_iteration[t], by_iteration[nxt]
            key = ("success" if before else "fail") + "_to_" + ("success" if after else "fail")
            counts[(category, t, nxt)][key] += 1
    return [{"category_id": c, "from_iteration": t, "to_iteration": n, **v}
            for (c, t, n), v in sorted(counts.items())]


def arm_config(arm: str, base: dict[str, Any]) -> dict[str, Any]:
    """Per-arm run config: identical matched settings, only the rubric schedule differs."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    matched = ("theta_0", "train_split_manifest", "eval_split_manifest", "iterations", "trajectories_per_iteration",
               "optimizer", "outcome_reward", "seed")
    missing = [k for k in matched if base.get(k) is None]
    return {"arm": arm, **{k: base.get(k) for k in matched}, "unresolved": missing,
            "rubric_update": arm == "evolving", "diagnostic_only": arm == "fixed_final_oracle"}
