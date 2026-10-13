"""Checkpoint comparison and post-hoc calibration of the Effectiveness predictor on DEV (no model needed)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from src.common.io import read_jsonl
from src.effectiveness import calibration
from src.effectiveness.evaluate import evaluate_predictions, ordering_breakdown
from src.effectiveness.metrics import mae, pearson, sign_agreement, spearman

ROOT = Path(__file__).resolve().parents[1]
STATES = ("Accept", "Challenge", "Preserve", "Recover")
IDENTITY = {"a": 1.0, "b": 0.0}


def examples_of(rows: list[dict]) -> list:
    return [SimpleNamespace(**row, original_case_id=row["case_id"], empirical_value=row["num_success"] / row["num_rollouts"])
            for row in rows]


def delta_rows(predictions: list[float], examples) -> list[dict]:
    """One row per (history, candidate): predicted and rollout delta."""
    before = {e.history_id: (p, e.empirical_value) for p, e in zip(predictions, examples) if e.kind == "before"}
    return [{"history_id": e.history_id, "predicted": p - before[e.history_id][0],
             "rollout": e.empirical_value - before[e.history_id][1]}
            for p, e in zip(predictions, examples) if e.kind == "after" and e.history_id in before]


def delta_metrics(rows: list[dict]) -> dict:
    predicted, rollout = [r["predicted"] for r in rows], [r["rollout"] for r in rows]
    nonzero = [r for r in rows if r["rollout"] != 0]
    return {"pairs": len(rows), "delta_mae": mae(predicted, rollout) if rows else None,
            "mean_abs_predicted_delta": sum(abs(p) for p in predicted) / len(predicted) if rows else None,
            "delta_pearson": pearson(predicted, rollout) if len(rows) > 1 else None,
            "delta_spearman": spearman(predicted, rollout) if len(rows) > 1 else None,
            "nonzero_pairs": len(nonzero),
            "delta_sign_accuracy": sign_agreement([r["predicted"] for r in nonzero], [r["rollout"] for r in nonzero])
            if nonzero else None}


def subset(predictions: list[float], examples, histories: set) -> tuple[list[float], list]:
    keep = [i for i, e in enumerate(examples) if e.history_id in histories]
    return [predictions[i] for i in keep], [examples[i] for i in keep]


def analyse(predictions: list[float], examples, state_of: dict) -> dict:
    full = evaluate_predictions(predictions, examples)
    ordering = ordering_breakdown(predictions, examples)
    deltas = delta_rows(predictions, examples)
    after = {}
    for e in examples:
        if e.kind == "after":
            after.setdefault(e.history_id, set()).add(e.empirical_value)
    informative = {h for h, values in after.items() if len(values) > 1}
    informative_predictions, informative_examples = subset(predictions, examples, informative)
    informative_ordering = ordering_breakdown(informative_predictions, informative_examples)
    result = {
        "all": {"examples": len(examples), "histories": len(after),
                "nll": full["absolute"]["all"]["binomial_nll"], "brier": full["absolute"]["all"]["brier"],
                "ece": full["absolute"]["all"]["ece"], "success_pearson": full["absolute"]["all"]["pearson"],
                **delta_metrics(deltas),
                "ordering": ordering["overall"]["accuracy"], "ordering_pairs": ordering["overall"]["pairs"],
                "ordering_informative_histories": ordering["informative_histories"]["accuracy"]},
        "informative_histories": {"histories": len(informative),
                                  **delta_metrics([r for r in deltas if r["history_id"] in informative]),
                                  "ordering": informative_ordering["overall"]["accuracy"],
                                  "ordering_pairs": informative_ordering["overall"]["pairs"]},
        "nonzero_delta_pairs": delta_metrics([r for r in deltas if r["rollout"] != 0]),
        "by_state": {},
    }
    for state in STATES:
        histories = {h for h in after if state_of.get(h) == state}
        state_predictions, state_examples = subset(predictions, examples, histories)
        state_ordering = ordering_breakdown(state_predictions, state_examples)
        result["by_state"][state] = {"histories": len(histories),
                                     "informative_histories": len(histories & informative),
                                     **delta_metrics([r for r in deltas if r["history_id"] in histories]),
                                     "ordering": state_ordering["overall"]["accuracy"],
                                     "ordering_pairs": state_ordering["overall"]["pairs"]}
    return result


def fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def table(header: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows])


OVERALL = ("nll", "brier", "ece", "success_pearson", "delta_mae", "delta_pearson", "delta_spearman",
           "delta_sign_accuracy", "mean_abs_predicted_delta", "ordering", "ordering_informative_histories")


def overall_row(name: str, m: dict) -> list:
    return [name] + [m["all"][k] for k in OVERALL] + [m["by_state"][state]["ordering"] for state in STATES]


OVERALL_HEADER = ["checkpoint", "NLL", "Brier", "ECE", "success r", "delta MAE", "delta r", "delta rho",
                  "delta sign", "mean abs predicted delta", "ordering", "ordering (per history)"] + [f"ordering {state}" for state in STATES]


def revision() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    files = ["scripts/predictor_checkpoint_report.py", "scripts/evaluate_predictor_epochs.py", "src/effectiveness/evaluate.py",
             "src/effectiveness/metrics.py", "src/effectiveness/calibration.py", "src/effectiveness/train.py",
             "configs/effectiveness/v5_predictor_frozen.json", "configs/effectiveness/v5_predictor_lora.json"]
    return {"git_head": git("rev-parse", "HEAD"),
            "working_tree": "has uncommitted changes (the file hashes below identify the code used)",
            "file_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16] for f in files}}


def calibrate(z: list[float], examples, state_of: dict, raw: dict) -> dict:
    """Temperature and Platt scaling of one checkpoint's logits: parameters fitted on all of DEV, metrics
    in-sample and cross-fitted (5 folds by case), and whether any ordering changed."""
    successes, trials = [e.num_success for e in examples], [e.num_rollouts for e in examples]
    fold_of = {case: i % 5 for i, case in enumerate(sorted({e.case_id for e in examples}))}
    parameters, metrics, checks = {}, {}, {}
    for method in calibration.METHODS:
        parameters[method] = calibration.fit_logits(z, successes, trials, method)
        crossed = [None] * len(examples)
        for fold in range(5):
            train = [i for i, e in enumerate(examples) if fold_of[e.case_id] != fold]
            fitted = calibration.fit_logits([z[i] for i in train], [successes[i] for i in train], [trials[i] for i in train], method)
            for i, e in enumerate(examples):
                if fold_of[e.case_id] == fold:
                    crossed[i] = calibration.apply_logits(fitted, [z[i]])[0]
        metrics[method] = {"in_sample": analyse(calibration.apply_logits(parameters[method], z), examples, state_of),
                           "cross_fitted": analyse(crossed, examples, state_of)}
        scaled = metrics[method]["in_sample"]
        checks[method] = {"ordering_unchanged": scaled["all"]["ordering"] == raw["all"]["ordering"],
                          "by_state_ordering_unchanged": all(scaled["by_state"][s]["ordering"] == raw["by_state"][s]["ordering"]
                                                             for s in STATES),
                          "delta_sign_raw": raw["all"]["delta_sign_accuracy"],
                          "delta_sign_calibrated": scaled["all"]["delta_sign_accuracy"]}
    return {"parameters": parameters, "metrics": metrics, "order_checks": checks,
            "best_method_by_cross_fitted_nll": min(calibration.METHODS, key=lambda m: metrics[m]["cross_fitted"]["all"]["nll"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=MODEL_DIR:CHECKPOINT")
    parser.add_argument("--calibrate", action="append", required=True, help="name of a checkpoint to calibrate (repeatable)")
    parser.add_argument("--logged-epochs", help="MODEL_DIR whose trainer_state.pt holds metrics of epochs without a saved state")
    parser.add_argument("--tag", default="", help="suffix of the output file names")
    args = parser.parse_args()
    run = Path(args.run_dir)
    suffix = f"_{args.tag}" if args.tag else ""
    state_by_case = {r["case_id"]: r["collaboration_state"] for r in read_jsonl(run / "prepare_diagnostics.jsonl")}

    logits, paths, examples = {}, {}, None
    for item in args.checkpoint:
        name, location = item.split("=", 1)
        model_dir, checkpoint = location.split(":", 1)
        dump = json.loads((run / model_dir / "dev_logits.json").read_text(encoding="utf-8"))
        rows = examples_of(dump["examples"])
        key = [(e.kind, e.history_id, e.response_id) for e in rows]
        if examples is not None and key != [(e.kind, e.history_id, e.response_id) for e in examples]:
            raise ValueError(f"{name} was evaluated on different DEV examples")
        examples = rows
        logits[name], paths[name] = dump["checkpoints"][checkpoint]["logits"], dump["checkpoints"][checkpoint]["path"]
    state_of = {e.history_id: state_by_case.get(e.case_id) for e in examples}
    cases = sorted({e.case_id for e in examples})

    metrics = {name: analyse(calibration.apply_logits(IDENTITY, z), examples, state_of) for name, z in logits.items()}
    trained = [n for n in metrics if n != "frozen"]
    best = {"best_nll": min(trained, key=lambda n: metrics[n]["all"]["nll"]),
            "best_ordering": max(trained, key=lambda n: metrics[n]["all"]["ordering"]),
            "best_delta_correlation": max(trained, key=lambda n: metrics[n]["all"]["delta_pearson"]),
            "best_delta_sign": max(trained, key=lambda n: metrics[n]["all"]["delta_sign_accuracy"])}
    calibrated = {target: calibrate(logits[target], examples, state_of, metrics[target]) for target in args.calibrate}

    logged = None
    if args.logged_epochs:
        import torch

        state = torch.load(run / args.logged_epochs / "trainer_state.pt", map_location="cpu", weights_only=False)
        logged = [{"epoch": h["epoch"], "nll": h["dev"]["absolute"]["all"]["binomial_nll"], "brier": h["dev"]["absolute"]["all"]["brier"],
                   "ece": h["dev"]["absolute"]["all"]["ece"], "success_pearson": h["dev"]["absolute"]["all"]["pearson"],
                   "delta_mae": h["dev"]["delta"]["mae"], "delta_pearson": h["dev"]["delta"]["pearson"],
                   "delta_spearman": h["dev"]["delta"]["spearman"], "delta_sign_accuracy": h["dev"]["delta"]["sign_agreement_nonzero"],
                   "ordering": h["dev"]["ranking"]["within_history_pairwise_accuracy"]} for h in state["history"]]

    first = next(iter(metrics.values()))
    provenance = {"dev_labels": str(run / "labels_dev.jsonl"), "dev_cases": cases, "dev_examples": len(examples),
                  "checkpoint_paths": paths, "revision": revision(),
                  "pair_counts": {"delta_pairs": first["all"]["pairs"], "nonzero_delta_pairs": first["all"]["nonzero_pairs"],
                                  "ordering_pairs": first["all"]["ordering_pairs"],
                                  "informative_histories": first["informative_histories"]["histories"]}}
    (run / f"checkpoint_metrics{suffix}.json").write_text(json.dumps({
        "provenance": provenance, "best_trained_checkpoints": best, "checkpoints": metrics,
        "logged_training_metrics_per_epoch": logged, "calibration": calibrated}, indent=2), encoding="utf-8")
    (run / f"calibration_params{suffix}.json").write_text(json.dumps({
        "fitted_on": f"DEV (all {len(cases)} cases); TEST not used",
        "formula": "p = sigmoid(a * z + b) on the raw success logit z; temperature: b = 0 and T = 1 / a",
        "checkpoints": {t: {"checkpoint_path": paths[t], "parameters": c["parameters"]} for t, c in calibrated.items()},
        "dev_cases": cases, "revision": provenance["revision"]}, indent=2), encoding="utf-8")

    lines = ["# Effectiveness predictor: checkpoint comparison on DEV", "",
             f"DEV: {len(cases)} cases, {len(examples)} examples ({first['all']['histories']} histories, {first['all']['pairs']} "
             f"(history, candidate) pairs, {first['all']['nonzero_pairs']} with a nonzero rollout delta, "
             f"{first['all']['ordering_pairs']} ordering pairs in {first['informative_histories']['histories']} informative histories).",
             "All rows are recomputed from the reloaded checkpoints. TEST is not used.", "",
             "## Overall", "", table(OVERALL_HEADER, [overall_row(n, m) for n, m in metrics.items()]), "",
             "ordering: pooled over candidate pairs with different rollout AFTER rates; (per history): mean of the "
             "per-history accuracy; ordering <state>: pooled within that collaboration state.", "",
             "## Which trained checkpoint is best on what", ""]
    lines += [f"- {key}: {value}" for key, value in best.items()]
    if logged:
        lines += ["", f"## Metrics logged during training ({args.logged_epochs}, every epoch)", "",
                  table(["epoch", "NLL", "Brier", "ECE", "success r", "delta MAE", "delta r", "delta rho", "delta sign", "ordering"],
                        [[r["epoch"]] + [r[k] for k in ("nll", "brier", "ece", "success_pearson", "delta_mae", "delta_pearson",
                                                        "delta_spearman", "delta_sign_accuracy", "ordering")] for r in logged])]
    lines += ["", "## A. Informative histories (candidates differ in rollout AFTER)", "",
              table(["checkpoint", "histories", "pairs", "delta MAE", "delta r", "delta rho", "delta sign", "ordering", "ordering pairs"],
                    [[n, m["informative_histories"]["histories"], m["informative_histories"]["pairs"]]
                     + [m["informative_histories"][k] for k in ("delta_mae", "delta_pearson", "delta_spearman", "delta_sign_accuracy",
                                                                "ordering", "ordering_pairs")] for n, m in metrics.items()]),
              "", "## B. Nonzero-delta pairs only", "",
              table(["checkpoint", "pairs", "delta MAE", "delta r", "delta rho", "delta sign"],
                    [[n, m["nonzero_delta_pairs"]["pairs"]] + [m["nonzero_delta_pairs"][k] for k in
                                                               ("delta_mae", "delta_pearson", "delta_spearman", "delta_sign_accuracy")]
                     for n, m in metrics.items()]),
              "", "## C. By collaboration state", ""]
    for state in STATES:
        lines += [f"### {state}", "",
                  table(["checkpoint", "histories", "informative histories", "ordering pairs", "ordering", "nonzero pairs",
                         "delta sign", "delta r", "delta MAE"],
                        [[n, m["by_state"][state]["histories"], m["by_state"][state]["informative_histories"],
                          m["by_state"][state]["ordering_pairs"], m["by_state"][state]["ordering"],
                          m["by_state"][state]["nonzero_pairs"], m["by_state"][state]["delta_sign_accuracy"],
                          m["by_state"][state]["delta_pearson"], m["by_state"][state]["delta_mae"]] for n, m in metrics.items()]), ""]
    final = [overall_row(n, m) for n, m in metrics.items()]
    for target, c in calibrated.items():
        final += [overall_row(f"{target} + temperature (cross-fitted)", c["metrics"]["temperature"]["cross_fitted"]),
                  overall_row(f"{target} + temperature (in-sample)", c["metrics"]["temperature"]["in_sample"])]
    lines += ["## Final comparison (raw and temperature-scaled)", "", table(OVERALL_HEADER, final)]
    (run / f"checkpoint_comparison{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = ["# Post-hoc calibration on DEV", "",
              "p = sigmoid(a z + b) on the raw success logit z, a > 0 (monotonic). Fitted on DEV by binomial NLL; TEST not used.",
              "in-sample: parameters fitted on the same DEV examples that are scored (optimistic). cross-fitted: each case is "
              "scored with parameters fitted on the other cases (5 folds by case).", ""]
    for target, c in calibrated.items():
        report += [f"## {target}", ""]
        report += [f"- {m}: a = {c['parameters'][m]['a']:.4f}, b = {c['parameters'][m]['b']:.4f}"
                   + (f" (T = {c['parameters'][m]['temperature']:.3f})" if m == "temperature" else "") for m in calibration.METHODS]
        versions = [("raw", metrics[target])] + [(f"{m} ({kind.replace('_', '-')})", c["metrics"][m][kind])
                                                 for m in calibration.METHODS for kind in ("in_sample", "cross_fitted")]
        report += ["", table(OVERALL_HEADER, [overall_row(name, m) for name, m in versions]), ""]
        for m in calibration.METHODS:
            k = c["order_checks"][m]
            report.append(f"- {m}: overall ordering unchanged: {k['ordering_unchanged']}; per-state ordering unchanged: "
                          f"{k['by_state_ordering_unchanged']}; delta sign accuracy {fmt(k['delta_sign_raw'])} -> {fmt(k['delta_sign_calibrated'])}")
        report.append("")
    (run / f"calibration_report{suffix}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print((run / f"checkpoint_comparison{suffix}.md").read_text(encoding="utf-8"))
    print((run / f"calibration_report{suffix}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
