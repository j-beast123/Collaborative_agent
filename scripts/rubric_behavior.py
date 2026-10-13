"""Rubric behavior analysis (Experiment 3): per rubric category, how often it was discovered, how often the Agent
failed it, and how often satisfying it went with a drop in Effectiveness.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Optional

from scripts.main_experiments import MATRIX, load_matrix, write_versioned
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["category_id", "name", "rubric_id", "in_training_set", "removal_reason", "effectiveness_gap", "weight", "generation_count", "generation_frequency", "analysed_failures",
          "applicable_n", "scored_n", "score_0_n", "score_1_n", "score_2_n", "behavior_failure_n", "behavior_failure_rate",
          "compliant_effectiveness_n", "compliant_negative_effectiveness_n", "compliant_zero_effectiveness_n",
          "compliant_positive_effectiveness_n", "effectiveness_mismatch_rate"]


def rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def score_counts(log: list[dict], rubric_id: str, failure_scores: list[int], satisfied: int) -> dict:
    counts = {"scored_n": 0, "applicable_n": 0, "score_0_n": 0, "score_1_n": 0, "score_2_n": 0,
              "compliant_effectiveness_n": 0, "compliant_negative_effectiveness_n": 0, "compliant_zero_effectiveness_n": 0,
              "compliant_positive_effectiveness_n": 0}
    for row in log:
        for stage, scores in (row.get("rubric_scores") or {}).items():
            label = (scores.get("labels") or {}).get(rubric_id)
            if label is None:
                continue
            counts["scored_n"] += 1
            if label == "NA":
                continue
            counts["applicable_n"] += 1
            counts[f"score_{label}_n"] += 1
            effect = (row.get("effectiveness") or {}).get(stage) or {}
            if label == satisfied and effect.get("available") and effect.get("delta") is not None:
                counts["compliant_effectiveness_n"] += 1
                sign = "negative" if effect["delta"] < 0 else "positive" if effect["delta"] > 0 else "zero"
                counts[f"compliant_{sign}_effectiveness_n"] += 1
    counts["behavior_failure_n"] = sum(counts[f"score_{s}_n"] for s in failure_scores)
    counts["behavior_failure_rate"] = rate(counts["behavior_failure_n"], counts["applicable_n"])
    counts["effectiveness_mismatch_rate"] = rate(counts["compliant_negative_effectiveness_n"], counts["compliant_effectiveness_n"])
    return counts


def analyse(rubrics_dir: Path, log: list[dict], run_iteration: int, failure_scores: list[int], satisfied: int) -> list[dict]:
    rubric_set = json.loads((rubrics_dir / "rubric_set.json").read_text(encoding="utf-8"))
    registry_file = rubrics_dir / "categories.json"
    registry = json.loads(registry_file.read_text(encoding="utf-8")) if registry_file.is_file() else []
    selection = rubrics_dir / "failure_selection.json"
    analysed = json.loads(selection.read_text(encoding="utf-8")).get("analyzed") if selection.is_file() else None
    active = {r.get("category_id", r["rubric_id"]): r for r in rubric_set["rubrics"]}
    categories = {c["category_id"]: c for c in registry if c.get("status") != "merged"}
    rows = []
    for category_id in sorted(set(categories) | set(active)):
        category, rubric = categories.get(category_id, {}), active.get(category_id)
        count = (category.get("frequency_history") or {}).get(str(run_iteration))
        if count is None and rubric is not None:  # a set without a registry (unfiltered): what the rubric itself records
            count = rubric.get("failure_count") if rubric.get("discovered_iteration", run_iteration) == run_iteration else 0
        scored = score_counts(log, rubric["rubric_id"], failure_scores, satisfied) if rubric else {}
        rows.append({"category_id": category_id, "name": (rubric or category).get("name") or category.get("category_name"),
                     "rubric_id": rubric["rubric_id"] if rubric else None, "in_training_set": rubric is not None,
                     "removal_reason": category.get("removal_reason") or category.get("inactive_reason"),
                     "effectiveness_gap": (rubric or {}).get("effectiveness_gap"), "weight": (rubric or {}).get("weight"),
                     "generation_count": count or 0, "generation_frequency": rate(count or 0, analysed or 0),
                     "analysed_failures": analysed,
                     "source_rubric_candidates": (rubric or {}).get("source_rubric_candidates") or category.get("members") or [],
                     **{name: scored.get(name) for name in FIELDS if name.endswith(("_n", "_rate"))}})
    return sorted(rows, key=lambda r: (-r["generation_count"], r["category_id"]))


def plot(rows: list[dict], path: Path, title: str) -> None:
    """Rows = categories, columns = the three measures; circle area follows the rate, the cell shows n/N."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = [("Rubric frequency", "generation_count", "analysed_failures", "generation_frequency"),
               ("Behavior failure", "behavior_failure_n", "applicable_n", "behavior_failure_rate"),
               ("Effectiveness mismatch", "compliant_negative_effectiveness_n", "compliant_effectiveness_n", "effectiveness_mismatch_rate")]
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * max(len(rows), 1) + 1.8))
    for y, row in enumerate(rows):
        for x, (_, numerator, denominator, value) in enumerate(columns):
            if row[value] is None:
                ax.text(x, y, "n/a", ha="center", va="center", fontsize=8, color="gray")
                continue
            ax.scatter(x, y, s=60 + 1400 * row[value], alpha=0.45, color=f"C{x}")
            ax.text(x, y, f"{row[numerator]}/{row[denominator]}\n{100 * row[value]:.0f}%", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[0] for c in columns])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([str(r["name"])[:48] for r in rows], fontsize=8)
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=9, pad=24)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory, e.g. outputs/serl/<run id>")
    parser.add_argument("--iteration", type=int, required=True, help="table iteration: 1, 2 or 3")
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--matrix", default=MATRIX)
    args = parser.parse_args()
    settings = load_matrix(args.matrix)["rubric_behavior"]
    run_iteration = args.iteration - 1
    source = ROOT / args.run / f"iter{run_iteration}"
    log = read_jsonl(source / "reward_log.jsonl", missing_ok=True)
    rows = analyse(source / "rubrics", log, run_iteration, settings["failure_scores"], settings["fully_satisfied_score"])
    out, name = ROOT / args.output_dir, f"rubric_behavior_iter{args.iteration}"
    meta = {"run": args.run, "table_iteration": args.iteration, "run_iteration_index": run_iteration,
            "rubrics": str((source / "rubrics").relative_to(ROOT)), "population": settings["population"],
            "responses_in_population": len(log), "responses_with_effectiveness": sum(
                any((e or {}).get("available") for e in (r.get("effectiveness") or {}).values()) for r in log),
            "behavior_failure_scores": settings["failure_scores"], "behavior_failure_scores_status": settings["failure_scores_status"],
            "fully_satisfied_score": settings["fully_satisfied_score"], "effectiveness_mismatch": settings["mismatch"],
            "fixed_probe": "none exists in the repository; the population above is used"}
    write_versioned(out / f"{name}.json", json.dumps(dict(meta, rows=rows), indent=2, ensure_ascii=False) + "\n")
    buffer = io.StringIO()
    csv.writer(buffer).writerows([FIELDS] + [["" if row[f] is None else row[f] for f in FIELDS] for row in rows])
    write_versioned(out / f"{name}.csv", buffer.getvalue())
    pdf = out / f"{name}.pdf"
    if pdf.exists():
        (out / "history").mkdir(exist_ok=True)
        import time

        pdf.rename(out / "history" / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.pdf")
    plot(rows, pdf, f"Rubric behavior, Iter {args.iteration} (failure = score in {settings['failure_scores']}; "
                    f"mismatch = score {settings['fully_satisfied_score']} with Effectiveness delta < 0)")
    print(json.dumps(dict(meta, categories=len(rows), in_training_set=sum(r["in_training_set"] for r in rows)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
