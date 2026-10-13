"""Iteration-to-iteration rubric retention (roles.rubric_manager.retention_policy; the method uses reactivation)."""

from __future__ import annotations

from typing import Optional

POLICIES = ("current_only", "cumulative", "reactivation")
SELECTION_ORDER = ("current_failure_frequency (descending)", "validation_success_rate (descending)",
                   "historical_failure_frequency (descending)", "category_id")


def selection_key(category: dict, stats: dict[str, dict]) -> tuple:
    """Order in which categories compete for the active set: this iteration's failure frequency, then the
    rerun success rate of their candidates, then the frequency over all iterations, then the category id."""
    own = stats[category["category_id"]]
    return (-own["failure_count"], -(own.get("validation_success_rate") or 0.0),
            -own.get("historical_failure_count", own["failure_count"]), category["category_id"])


class RetentionStrategy:
    name = "abstract"

    def plan(self, categories: list[dict], stats: dict[str, dict], max_active: Optional[int],
             min_failure_count: Optional[int]) -> tuple[list[dict], list[dict]]:
        """Returns (plan entries {category, action, reason}, excluded entries {category, reason})."""
        raise NotImplementedError

    @staticmethod
    def _observed(categories, stats, max_active, min_failure_count):
        ranked = sorted(categories, key=lambda c: selection_key(c, stats))
        observed, excluded = [], []
        for category in ranked:
            count = stats[category["category_id"]]["failure_count"]
            if count == 0:
                continue
            if min_failure_count is not None and count < min_failure_count:
                excluded.append({"category": category, "reason": f"failure_count < {min_failure_count}"})
            elif max_active is not None and len(observed) >= max_active:
                excluded.append({"category": category, "reason": f"outside the top {max_active} by failure_count"})
            else:
                observed.append(category)
        return observed, excluded


class CurrentOnly(RetentionStrategy):
    name = "current_only"

    def plan(self, categories, stats, max_active, min_failure_count):
        observed, excluded = self._observed(categories, stats, max_active, min_failure_count)
        plan = [{"category": c, "action": "construct", "reason": "observed this iteration"} for c in observed]
        excluded += [{"category": c, "reason": "not observed this iteration"}
                     for c in categories if stats[c["category_id"]]["failure_count"] == 0]
        return plan, excluded


class Cumulative(RetentionStrategy):
    name = "cumulative"

    def plan(self, categories, stats, max_active, min_failure_count):
        observed, excluded = self._observed(categories, stats, max_active, min_failure_count)
        plan = [{"category": c, "action": "construct", "reason": "observed this iteration"} for c in observed]
        chosen = {c["category_id"] for c in observed}
        for category in categories:
            if category["category_id"] in chosen:
                continue
            if category.get("rubrics"):
                plan.append({"category": category, "action": "reuse", "reason": "carried from an earlier iteration"})
            elif stats[category["category_id"]]["failure_count"] == 0:
                excluded.append({"category": category, "reason": "not observed and never active"})
        excluded = [e for e in excluded if e["category"]["category_id"] not in
                    {p["category"]["category_id"] for p in plan}]
        return plan, excluded


class Reactivation(RetentionStrategy):
    name = "reactivation"

    def plan(self, categories, stats, max_active, min_failure_count):
        observed, excluded = self._observed(categories, stats, max_active, min_failure_count)
        plan = [{"category": c, "action": "reuse" if c.get("rubrics") else "construct",
                 "reason": "observed this iteration" if not c.get("rubrics")
                 else "still observed" if c.get("status") == "active" else "reactivated"} for c in observed]
        excluded += [{"category": c, "reason": "dormant (not observed this iteration)"}
                     for c in categories if stats[c["category_id"]]["failure_count"] == 0]
        return plan, excluded


def build_retention(name: Optional[str]) -> RetentionStrategy:
    strategies = {"current_only": CurrentOnly, "cumulative": Cumulative, "reactivation": Reactivation}
    if name not in strategies:
        raise ValueError(f"rubric retention policy must be one of {POLICIES}")
    return strategies[name]()
