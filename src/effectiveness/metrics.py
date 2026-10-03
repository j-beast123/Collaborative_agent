"""Metrics for empirical value and Delta-V prediction."""

from __future__ import annotations

from collections import defaultdict
from math import log, sqrt
from typing import Hashable, Iterable


def mae(predictions: Iterable[float], targets: Iterable[float]) -> float:
    pairs = list(zip(predictions, targets))
    return sum(abs(prediction - target) for prediction, target in pairs) / len(pairs) if pairs else float("nan")


def rmse(predictions: Iterable[float], targets: Iterable[float]) -> float:
    pairs = list(zip(predictions, targets))
    return sqrt(sum((prediction - target) ** 2 for prediction, target in pairs) / len(pairs)) if pairs else float("nan")


def sign_agreement(predictions: Iterable[float], targets: Iterable[float], tolerance: float = 1e-8) -> float:
    pairs = list(zip(predictions, targets))
    if not pairs:
        return float("nan")

    def sign(value: float) -> int:
        if value > tolerance:
            return 1
        if value < -tolerance:
            return -1
        return 0

    return sum(sign(prediction) == sign(target) for prediction, target in pairs) / len(pairs)


def pairwise_ranking_accuracy(predictions: Iterable[float], targets: Iterable[float]) -> float:
    pairs = list(zip(predictions, targets))
    comparisons = [(i, j) for i in range(len(pairs)) for j in range(i + 1, len(pairs))]
    informative = [pair for pair in comparisons if pairs[pair[0]][1] != pairs[pair[1]][1]]
    if not informative:
        return float("nan")
    correct = sum(
        (pairs[i][0] - pairs[j][0]) * (pairs[i][1] - pairs[j][1]) > 0
        for i, j in informative
    )
    return correct / len(informative)


def grouped_pairwise_ranking(
    predictions: Iterable[float], targets: Iterable[float], groups: Iterable[Hashable]
) -> dict[str, float]:
    """Pairwise ranking only between candidates of the same history."""
    by_group = defaultdict(list)
    for prediction, target, group in zip(predictions, targets, groups):
        if group:
            by_group[group].append((prediction, target))
    correct = total = 0
    for rows in by_group.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i][1] == rows[j][1]:
                    continue
                total += 1
                agreement = (rows[i][0] - rows[j][0]) * (rows[i][1] - rows[j][1])
                correct += 1.0 if agreement > 0 else 0.5 if rows[i][0] == rows[j][0] else 0.0
    return {"accuracy": correct / total if total else float("nan"), "pairs": total}


def brier(predictions: Iterable[float], targets: Iterable[float]) -> float:
    pairs = list(zip(predictions, targets))
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else float("nan")


def binomial_nll(
    predictions: Iterable[float], successes: Iterable[int], counts: Iterable[int], eps: float = 1e-7
) -> float:
    """Per-rollout negative log likelihood of m successes out of K."""
    total_nll = total_rollouts = 0.0
    for p, m, k in zip(predictions, successes, counts):
        p = min(max(p, eps), 1 - eps)
        total_nll -= m * log(p) + (k - m) * log(1 - p)
        total_rollouts += k
    return total_nll / total_rollouts if total_rollouts else float("nan")


def expected_calibration_error(
    predictions: Iterable[float], targets: Iterable[float], weights: Iterable[float], bins: int = 10
) -> float:
    """Rollout-weighted ECE with equal-width probability bins."""
    buckets = defaultdict(lambda: [0.0, 0.0, 0.0])
    for p, y, w in zip(predictions, targets, weights):
        bucket = buckets[min(int(p * bins), bins - 1)]
        bucket[0] += w * p
        bucket[1] += w * y
        bucket[2] += w
    total = sum(bucket[2] for bucket in buckets.values())
    if not total:
        return float("nan")
    return sum(abs(b[0] - b[1]) for b in buckets.values()) / total


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for index in order[start : end + 1]:
            ranks[index] = (start + end) / 2
        start = end + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(_ranks(list(xs)), _ranks(list(ys)))
