"""Compare score LLMs (e.g. Jev vs GPT) on the same (turn, rubric) units."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Optional

from .rubric_scoring import LABELS, ScoreResult


def unit_labels(results: list[ScoreResult]) -> dict[tuple, Any]:
    return {
        (result.item_id, rubric_id): label
        for result in results
        if result.status == "scored"
        for rubric_id, label in result.labels.items()
    }


def cohen_kappa(a: list, b: list, categories: tuple) -> float:
    n = len(a)
    if not n:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb[c] for c in categories) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def quadratic_weighted_kappa(a: list[int], b: list[int], categories=(0, 1, 2)) -> float:
    n = len(a)
    if not n:
        return float("nan")
    k = len(categories) - 1
    index = {c: i for i, c in enumerate(categories)}
    weight = lambda x, y: ((index[x] - index[y]) / k) ** 2  # noqa: E731
    observed = sum(weight(x, y) for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[x] * cb[y] * weight(x, y) for x in categories for y in categories) / (n * n)
    return 1 - observed / expected if expected else float("nan")


def pair_agreement(left: dict[tuple, Any], right: dict[tuple, Any]) -> dict[str, Any]:
    shared = sorted(set(left) & set(right), key=str)
    a = [left[u] for u in shared]
    b = [right[u] for u in shared]
    both_applicable = [(x, y) for x, y in zip(a, b) if x != "NA" and y != "NA"]
    return {
        "shared_units": len(shared),
        "label_agreement": sum(x == y for x, y in zip(a, b)) / len(shared) if shared else float("nan"),
        "label_kappa": cohen_kappa(a, b, LABELS),
        "applicability_agreement": (
            sum((x == "NA") == (y == "NA") for x, y in zip(a, b)) / len(shared) if shared else float("nan")
        ),
        "both_applicable_units": len(both_applicable),
        "score_exact_agreement": (
            sum(x == y for x, y in both_applicable) / len(both_applicable) if both_applicable else float("nan")
        ),
        "score_quadratic_kappa": quadratic_weighted_kappa([x for x, _ in both_applicable], [y for _, y in both_applicable]),
    }


def scorer_summary(results: list[ScoreResult], expected_units: int) -> dict[str, Any]:
    labels = unit_labels(results)
    status = Counter(r.status for r in results)
    calls = [r.call for r in results if r.call and not r.call.get("cache_hit")]
    tokens = Counter()
    for call in calls:
        for key, value in (call.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                tokens[key] += value
    return {
        "items": len(results),
        "status_counts": dict(status),
        "unit_coverage": len(labels) / expected_units if expected_units else float("nan"),
        "label_distribution": {str(k): v for k, v in Counter(labels.values()).items()},
        "mean_latency_s": sum(c.get("latency_s", 0.0) for c in calls) / len(calls) if calls else None,
        "total_retries": sum(max(0, c.get("attempts", 1) - 1) for c in calls),
        "token_usage": dict(tokens),
    }


def compare_scorers(
    results: dict[str, list[ScoreResult]],
    expected_units: int,
    human: Optional[dict[tuple, Any]] = None,
) -> dict[str, Any]:
    labels = {name: unit_labels(rows) for name, rows in results.items()}
    report = {
        "scorers": {name: scorer_summary(rows, expected_units) for name, rows in results.items()},
        "pairwise": {f"{a}__vs__{b}": pair_agreement(labels[a], labels[b]) for a, b in combinations(sorted(labels), 2)},
    }
    if human:
        report["vs_human"] = {name: pair_agreement(human, labels[name]) for name in sorted(labels)}
        report["human_units"] = len(human)
    else:
        report["vs_human"] = None
        report["note"] = "No human audit labels: scorer agreement is not accuracy; choose a scorer only after a DEV audit."
    return report
