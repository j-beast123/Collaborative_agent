"""Uncertainty utilities that respect the paired / clustered structure."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Hashable, Iterable, Optional


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = int(position), min(int(position) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def cluster_bootstrap_mean(
    values: Iterable[float],
    clusters: Iterable[Hashable],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Optional[float]]:
    """Mean with a percentile CI from resampling clusters (e.g. original cases)."""
    by_cluster = defaultdict(list)
    for value, cluster in zip(values, clusters):
        by_cluster[cluster].append(float(value))
    keys = sorted(by_cluster, key=str)
    n = sum(len(v) for v in by_cluster.values())
    if not n:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "clusters": 0}
    mean = sum(sum(v) for v in by_cluster.values()) / n
    if len(keys) < 2:
        return {"mean": mean, "ci_low": None, "ci_high": None, "n": n, "clusters": len(keys)}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(keys) for _ in keys]
        total = sum(sum(by_cluster[k]) for k in sample)
        count = sum(len(by_cluster[k]) for k in sample)
        boots.append(total / count)
    return {"mean": mean, "ci_low": _percentile(boots, alpha / 2), "ci_high": _percentile(boots, 1 - alpha / 2),
            "n": n, "clusters": len(keys)}


def paired_difference(
    pairs: Iterable[tuple[Hashable, float, float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Optional[float]]:
    """(cluster, a, b) triples -> mean(b - a) with a cluster bootstrap CI."""
    pairs = list(pairs)
    result = cluster_bootstrap_mean([b - a for _, a, b in pairs], [c for c, _, _ in pairs],
                                    n_boot=n_boot, seed=seed, alpha=alpha)
    return {"mean_difference": result["mean"], "ci_low": result["ci_low"], "ci_high": result["ci_high"],
            "n_pairs": result["n"], "clusters": result["clusters"]}
