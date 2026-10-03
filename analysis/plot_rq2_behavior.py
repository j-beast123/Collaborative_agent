"""Figures: RQ2 behavior appropriateness (two panels), RQ2 effectiveness, RQ3 bubble matrix.

Inputs are the tidy CSV/JSON outputs of scripts/run_eval.py. Final cosmetics
(series selection, typography) are for the paper stage; these renderings are
faithful to the data and use only rows with evaluation_status == scored.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

STATE_ORDER = ["Preserve", "Accept", "Challenge", "Recover"]
CONDITION_ORDER = ["correct_strong", "correct_weak", "incorrect_strong_looking", "incorrect_none"]
CONDITION_LABELS = {"correct_strong": "Correct +\nstrong", "correct_weak": "Correct +\nweak",
                    "incorrect_strong_looking": "Incorrect +\nstrong-looking", "incorrect_none": "Incorrect +\nno evidence"}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rates(rows, key):
    groups = defaultdict(list)
    for row in rows:
        if row["evaluation_status"] == "scored":
            groups[(f"{row['system']} ({row['checkpoint']})", row[key])].append(int(row["appropriateness_match"]))
    return {k: sum(v) / len(v) for k, v in groups.items()}, sorted({s for s, _ in groups})


def _panel(ax, rates, series, order, labels, title):
    width = 0.8 / max(len(series), 1)
    for i, name in enumerate(series):
        values = [rates.get((name, c), float("nan")) * 100 for c in order]
        ax.bar([x + i * width - 0.4 + width / 2 for x in range(len(order))], values, width, label=name)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels.get(c, c) for c in order])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Behavior appropriateness (%)")
    ax.set_title(title)


def plot_rq2_behavior(state_csv, controlled_csv, output_path, *, synthetic=False):
    """One figure, two panels: (A) collaboration states, (B) controlled evidence sensitivity."""
    for path in (state_csv, controlled_csv):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    state_rows, controlled_rows = _read_rows(state_csv), _read_rows(controlled_csv)
    if any((r["synthetic"] == "true") != synthetic for r in state_rows + controlled_rows):
        raise ValueError("Synthetic flag of the rows does not match the requested mode")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    rates, series = _rates(state_rows, "collaboration_state")
    _panel(axes[0], rates, series, STATE_ORDER, {}, "A. Collaboration-state appropriateness")
    rates, series_b = _rates(controlled_rows, "condition_id")
    _panel(axes[1], rates, series_b, CONDITION_ORDER, CONDITION_LABELS, "B. Controlled evidence sensitivity")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_effectiveness(rows_json, output_path):
    """Per-history downstream success of Base vs SERL (matched continuation)."""
    rows = json.loads(Path(rows_json).read_text(encoding="utf-8"))
    by_history = defaultdict(dict)
    for row in rows:
        if row.get("success_rate") is not None:
            by_history[row["history_id"]][row["system"]] = row["success_rate"]
    pairs = [(v["base"], v["serl"]) for v in by_history.values() if "base" in v and "serl" in v]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for a, b in pairs:
        ax.plot([0, 1], [a * 100, b * 100], color="0.7", linewidth=0.8)
    if pairs:
        ax.plot([0, 1], [sum(p[0] for p in pairs) / len(pairs) * 100, sum(p[1] for p in pairs) / len(pairs) * 100],
                color="black", linewidth=2.5, marker="o")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Base response", "SERL response"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Downstream success (%)")
    ax.set_title("Matched continuation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_bubble(bubble_json, output_path):
    """Rubric category x iteration bubble matrix (size = failure frequency)."""
    rows = json.loads(Path(bubble_json).read_text(encoding="utf-8"))
    categories = sorted({r["category_name"] for r in rows})
    iterations = sorted({r["iteration"] for r in rows})
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(1.2 * len(iterations) + 4, 0.45 * len(categories) + 1.5))
    for r in rows:
        if r["failure_count"]:
            ax.scatter(iterations.index(r["iteration"]), categories.index(r["category_name"]),
                       s=40 + 900 * r["proportion"], alpha=0.6)
    ax.set_xticks(range(len(iterations)))
    ax.set_xticklabels([f"Iter {i}" for i in iterations])
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_title("Rubric category frequency across iterations")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
