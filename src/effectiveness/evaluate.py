"""Held-out value and derived effectiveness metrics."""

from __future__ import annotations

import math
from typing import Iterable


def _pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def evaluate_predictions(predictions: Iterable[float], examples) -> dict[str, float]:
    predictions = list(predictions)
    targets = [example.empirical_value for example in examples]
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target lengths differ")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in predictions):
        raise ValueError("Predictions must be finite probabilities in [0,1]")
    if not predictions:
        return {"bce": float("nan"), "brier": float("nan"), "mae": float("nan"), "pearson": float("nan")}
    eps = 1e-6
    bce = sum(-(y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1 - p))) for p, y in zip(predictions, targets)) / len(targets)
    return {
        "bce": bce,
        "brier": sum((p - y) ** 2 for p, y in zip(predictions, targets)) / len(targets),
        "mae": sum(abs(p - y) for p, y in zip(predictions, targets)) / len(targets),
        "pearson": _pearson(predictions, targets),
    }


def evaluate_effectiveness(before_predictions, after_predictions, before_examples, after_examples, neutral_threshold=1e-8):
    if len(before_predictions) != len(before_examples):
        raise ValueError("Before prediction/example lengths differ")
    if len(after_predictions) != len(after_examples):
        raise ValueError("After prediction/example lengths differ")
    if len(before_predictions) != len(after_predictions):
        raise ValueError("Before/after lengths differ")
    if any(example.state_type != "before_agent" for example in before_examples):
        raise ValueError("Before examples must have state_type=before_agent")
    if any(example.state_type != "after_agent" for example in after_examples):
        raise ValueError("After examples must have state_type=after_agent")
    before_ids = [example.pair_id for example in before_examples]
    after_ids = [example.pair_id for example in after_examples]
    if len(before_ids) != len(set(before_ids)) or len(after_ids) != len(set(after_ids)):
        raise ValueError("Duplicate pair_id in effectiveness inputs")
    before = {example.pair_id: (prediction, example.empirical_value) for prediction, example in zip(before_predictions, before_examples)}
    after = {example.pair_id: (prediction, example.empirical_value) for prediction, example in zip(after_predictions, after_examples)}
    pairs = sorted(set(before) & set(after))
    if len(pairs) != len(before) or len(pairs) != len(after):
        raise ValueError("Before/after pair sets differ")
    predicted = [after[pair][0] - before[pair][0] for pair in pairs]
    target = [after[pair][1] - before[pair][1] for pair in pairs]
    return {
        "count": len(pairs),
        "mae": sum(abs(a - b) for a, b in zip(predicted, target)) / len(pairs) if pairs else float("nan"),
        "pearson": _pearson(predicted, target),
        "sign_accuracy": sum((a > neutral_threshold) == (b > neutral_threshold) and (a < -neutral_threshold) == (b < -neutral_threshold) for a, b in zip(predicted, target)) / len(pairs) if pairs else float("nan"),
        "predicted_distribution": {"min": min(predicted) if predicted else None, "max": max(predicted) if predicted else None, "mean": sum(predicted) / len(predicted) if predicted else None},
    }
