"""DEV report: model vs baselines, bootstrap intervals, label diagnostics, noise floor."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Optional

from .evaluate import evaluate_predictions

KEY_METRICS = {
    "absolute.all.binomial_nll": ("absolute", "all", "binomial_nll"),
    "absolute.all.brier": ("absolute", "all", "brier"),
    "absolute.after.brier": ("absolute", "after", "brier"),
    "ranking.within_history_pairwise_accuracy": ("ranking", "within_history_pairwise_accuracy"),
    "delta.mae": ("delta", "mae"),
    "delta.sign_agreement_nonzero": ("delta", "sign_agreement_nonzero"),
}
HIGHER_IS_BETTER = {"ranking.within_history_pairwise_accuracy", "delta.sign_agreement_nonzero"}


def _get(metrics: dict, path: tuple):
    for key in path:
        metrics = metrics[key]
    return metrics


def _rate(examples) -> float:
    trials = sum(e.num_rollouts for e in examples)
    return sum(e.num_success for e in examples) / trials if trials else 0.5


def baselines(model_predictions: list[float], dev: list, train: list) -> dict[str, list[float]]:
    by_kind = {kind: _rate([e for e in train if e.kind == kind]) for kind in ("before", "after")}
    pooled = _rate(train)
    model_before = {e.history_id: p for p, e in zip(model_predictions, dev) if e.kind == "before"}
    return {
        "train_mean": [pooled for _ in dev],
        "train_mean_by_kind": [by_kind[e.kind] for e in dev],
        "no_response_effect": [model_before.get(e.history_id, p) if e.kind == "after" else p
                               for p, e in zip(model_predictions, dev)],
        "dev_mean_reference": [_rate(dev) for _ in dev],
    }


def label_diagnostics(examples: list) -> dict[str, Any]:
    """Is there signal in the labels at all?"""
    rates = defaultdict(list)
    for e in examples:
        rates[e.kind].append(e.empirical_value)
    after_by_history = defaultdict(list)
    for e in examples:
        if e.kind == "after":
            after_by_history[e.history_id].append(e.num_success / e.num_rollouts)
    with_pairs = [v for v in after_by_history.values() if len(v) > 1]
    per_kind = {
        kind: {"count": len(v), "mean": sum(v) / len(v) if v else None,
               "share_all_fail": sum(x == 0 for x in v) / len(v) if v else None,
               "share_all_success": sum(x == 1 for x in v) / len(v) if v else None}
        for kind, v in rates.items()
    }
    return {**per_kind, "histories_with_2plus_candidates": len(with_pairs),
            "histories_where_candidates_differ": sum(len(set(v)) > 1 for v in with_pairs)}


def brier_noise_floor(examples: list) -> float:
    terms = [(e.empirical_value * (1 - e.empirical_value)) / (e.num_rollouts - 1)
             for e in examples if e.num_rollouts > 1]
    return sum(terms) / len(terms) if terms else float("nan")


def _bootstrap(evaluate: Callable[[list[int]], dict], clusters: list[str], n_boot: int, seed: int) -> list[dict]:
    by_cluster = defaultdict(list)
    for index, cluster in enumerate(clusters):
        by_cluster[cluster].append(index)
    keys = sorted(by_cluster)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        indices = [i for key in (rng.choice(keys) for _ in keys) for i in by_cluster[key]]
        samples.append(evaluate(indices))
    return samples


def _interval(values: list[float]) -> dict[str, float]:
    values = sorted(v for v in values if v == v)  # drop NaN (e.g. no informative ranking pair)
    if len(values) < 20:
        return {"ci_low": None, "ci_high": None}
    return {"ci_low": values[int(0.025 * len(values))], "ci_high": values[int(0.975 * len(values)) - 1]}


NATURAL_PREFIXES = ("original", "sample_")


def is_natural(example) -> bool:
    """BEFORE states and responses written without a guideline (the trajectory's own response, plain samples)."""
    return example.kind == "before" or str(example.response_id).startswith(NATURAL_PREFIXES)


def subset_report(model_predictions: list[float], dev: list, train: list, natural: bool, **kwargs) -> Optional[dict]:
    """dev_report on natural candidates (what the predictor scores when it is used) or on controlled ones
    (stances, behaviors, advocates), each with the BEFORE states. None when DEV has no such AFTER example."""
    def keep(example):
        return example.kind == "before" or is_natural(example) == natural

    index = [i for i, e in enumerate(dev) if keep(e)]
    if not any(dev[i].kind == "after" for i in index):
        return None
    report = dev_report([model_predictions[i] for i in index], [dev[i] for i in index],
                        [e for e in train if keep(e)], **kwargs)
    return {key: report[key] for key in ("comparison", "label_diagnostics", "brier_noise_floor")}


SAME_ANSWER_PAIR = ("evidence_grounded_maintain", "unsupported_maintain")


def same_answer_behavior_ranking(model_predictions: list[float], dev: list) -> dict[str, Any]:
    """Within a history, the two maintain candidates argue the same option and differ in behavior.
    Ranking them correctly cannot come from which option the response states."""
    by_history = defaultdict(dict)
    for prediction, example in zip(model_predictions, dev):
        if example.kind == "after" and example.response_id in SAME_ANSWER_PAIR:
            by_history[example.history_id][example.response_id] = (prediction, example.empirical_value)
    correct = informative = tied = 0
    for found in by_history.values():
        if len(found) < 2:
            continue
        (p_evidence, y_evidence), (p_plain, y_plain) = (found[name] for name in SAME_ANSWER_PAIR)
        if y_evidence == y_plain:
            tied += 1
            continue
        informative += 1
        agreement = (p_evidence - p_plain) * (y_evidence - y_plain)
        correct += 1.0 if agreement > 0 else 0.5 if p_evidence == p_plain else 0.0
    return {"pair": list(SAME_ANSWER_PAIR), "accuracy": correct / informative if informative else None,
            "informative_pairs": informative, "pairs_with_equal_outcome": tied, "chance": 0.5}


def dev_report(model_predictions: list[float], dev: list, train: list, *, n_boot: int = 1000, seed: int = 0) -> dict:
    predictors = {"model": list(model_predictions), **baselines(model_predictions, dev, train)}
    full = {name: evaluate_predictions(preds, dev) for name, preds in predictors.items()}

    def evaluate_subset(indices):
        subset = [dev[i] for i in indices]
        return {name: evaluate_predictions([preds[i] for i in indices], subset) for name, preds in predictors.items()}

    boots = _bootstrap(evaluate_subset, [e.original_case_id for e in dev], n_boot, seed) if n_boot else []
    comparison = {}
    for label, path in KEY_METRICS.items():
        entry = {"model": _get(full["model"], path),
                 "model_ci": _interval([_get(b["model"], path) for b in boots])}
        for name in predictors:
            if name == "model":
                continue
            diffs = [_get(b["model"], path) - _get(b[name], path) for b in boots]
            entry[name] = {"value": _get(full[name], path),
                           "model_minus_baseline": _get(full["model"], path) - _get(full[name], path),
                           **_interval(diffs)}
        entry["better_is"] = "higher" if label in HIGHER_IS_BETTER else "lower"
        if label == "ranking.within_history_pairwise_accuracy":
            entry["chance"] = 0.5
        comparison[label] = entry
    return {
        "comparison": comparison,
        "full_metrics": full,
        "label_diagnostics": {"dev": label_diagnostics(dev), "train": label_diagnostics(train)},
        "brier_noise_floor": {"dev_all": brier_noise_floor(dev)},
        "bootstrap": {"n_boot": n_boot, "cluster": "original_case_id", "seed": seed},
        "reading_guide": ("Lower is better for NLL/Brier/MAE; higher for ranking and sign agreement. The model helps "
                          "where model_minus_baseline has the right sign and its 95% interval excludes 0. Ranking "
                          "chance is 0.5. Brier cannot go below the noise floor."),
    }
