"""Metrics for empirical value and Delta-V prediction."""

from __future__ import annotations

from math import sqrt
from typing import Iterable


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
