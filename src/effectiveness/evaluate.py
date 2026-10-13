"""DEV evaluation of the effectiveness model against rollout estimates."""

from __future__ import annotations

import math
from typing import Iterable

from .metrics import (
    binomial_nll,
    brier,
    expected_calibration_error,
    grouped_pairwise_ranking,
    mae,
    pearson,
    rmse,
    sign_agreement,
    spearman,
)


def _absolute(predictions: list[float], examples: list) -> dict[str, float]:
    targets = [e.empirical_value for e in examples]
    counts = [e.num_rollouts for e in examples]
    if not examples:
        return {"count": 0}
    return {
        "count": len(examples),
        "binomial_nll": binomial_nll(predictions, [e.num_success for e in examples], counts),
        "brier": brier(predictions, targets),
        "mae": mae(predictions, targets),
        "rmse": rmse(predictions, targets),
        "ece": expected_calibration_error(predictions, targets, counts),
        "pearson": pearson(predictions, targets),
        "spearman": spearman(predictions, targets),
    }


def evaluate_predictions(predictions: Iterable[float], examples) -> dict:
    predictions, examples = list(predictions), list(examples)
    if len(predictions) != len(examples):
        raise ValueError("Prediction and example lengths differ")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in predictions):
        raise ValueError("Predictions must be finite probabilities in [0,1]")
    by_kind = {kind: [(p, e) for p, e in zip(predictions, examples) if e.kind == kind] for kind in ("before", "after")}
    absolute = {"all": _absolute(predictions, examples)}
    for kind, rows in by_kind.items():
        absolute[kind] = _absolute([p for p, _ in rows], [e for _, e in rows])

    after = by_kind["after"]
    ranking = grouped_pairwise_ranking([p for p, _ in after], [e.empirical_value for _, e in after],
                                       [e.history_id for _, e in after])

    before = {e.history_id: (p, e.empirical_value) for p, e in by_kind["before"]}
    predicted_delta, rollout_delta = [], []
    for p, e in after:
        if e.history_id in before:
            predicted_delta.append(p - before[e.history_id][0])
            rollout_delta.append(e.empirical_value - before[e.history_id][1])
    nonzero = [(p, d) for p, d in zip(predicted_delta, rollout_delta) if d != 0]
    delta = {
        "count": len(predicted_delta),
        "mae": mae(predicted_delta, rollout_delta),
        "rmse": rmse(predicted_delta, rollout_delta),
        "pearson": pearson(predicted_delta, rollout_delta),
        "spearman": spearman(predicted_delta, rollout_delta),
        "nonzero_count": len(nonzero),
        "sign_agreement_nonzero": sign_agreement([p for p, _ in nonzero], [d for _, d in nonzero]),
    }
    return {"absolute": absolute,
            "ranking": {"within_history_pairwise_accuracy": ranking["accuracy"], "informative_pairs": ranking["pairs"]},
            "delta": delta}


def ordering_breakdown(predictions: Iterable[float], examples, group_of=None) -> dict:
    """Within-history pairwise ordering of AFTER predictions, beyond the pooled accuracy:"""
    by_history: dict[str, list] = {}
    for p, e in zip(predictions, examples):
        if e.kind == "after":
            by_history.setdefault(e.history_id, []).append((p, e.empirical_value))
    scores = {}
    for history_id, rows in by_history.items():
        result = grouped_pairwise_ranking([p for p, _ in rows], [t for _, t in rows], ["h"] * len(rows))
        if result["pairs"]:
            scores[history_id] = (result["accuracy"], result["pairs"])

    def pooled(ids) -> dict:
        pairs = sum(scores[h][1] for h in ids)
        return {"accuracy": sum(scores[h][0] * scores[h][1] for h in ids) / pairs if pairs else None,
                "pairs": pairs, "histories": len(ids)}

    out = {"overall": pooled(list(scores)),
           "informative_histories": {"accuracy": sum(a for a, _ in scores.values()) / len(scores) if scores else None,
                                     "histories": len(scores)}}
    if group_of:
        groups: dict[str, list[str]] = {}
        for history_id in scores:
            groups.setdefault(str(group_of(history_id)), []).append(history_id)
        out["by_group"] = {name: pooled(ids) for name, ids in sorted(groups.items())}
    return out
