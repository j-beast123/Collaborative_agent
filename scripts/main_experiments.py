"""The main table and the component ablation: conditions, runs, DEV evaluation, tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Optional

from src.common.config import read_config
from src.common.io import read_jsonl
from src.evaluation.stats import cluster_bootstrap_mean, paired_difference

ROOT = Path(__file__).resolve().parents[1]
MATRIX = "configs/experiments/main/matrix.json"
SYMBOL = {"outcome": "R_O", "appropriateness": "R_A,t", "effectiveness": "R_E,t"}
FIELDS = ["method", "condition", "iteration", "seed", "status", "checkpoint", "protocol_hash", "dataset_hash",
          "reward_components_enabled", "reward_formula", "rubric_strategy", "split", "cases", "collaborations",
          "task_success", "task_success_ci_low", "task_success_ci_high", "behavior_appropriateness", "effectiveness",
          "mean_turns", "early_stop_distribution"]


def load_matrix(path: str = MATRIX) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def formula(weights: dict) -> str:
    return " + ".join(f"{weights[name]:g} {SYMBOL[name]}" for name in SYMBOL if weights.get(name, 0) > 0) or "none"


def flatten(value, prefix: str = "") -> dict:
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in flatten(item, f"{prefix}{key}.").items()}
    return {prefix[:-1]: value}


def differences(reference: dict, other: dict, allowed: list[str]) -> list[str]:
    """Config values that differ between two conditions and are not on the allowed list (notes are documentation)."""
    a, b = flatten(reference), flatten(other)
    return sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key)
                  and not any(key == name or key.startswith(name + ".") for name in allowed)
                  and not key.split(".")[-1].endswith(("notes", "_status")))


def rubric_strategy(config: dict) -> str:
    if config["reward"]["weights"]["appropriateness"] == 0:
        return "none"
    return {"evolving": "filtered_evolving", "fixed": "static", "no_filtering": "no_filtering_cumulative"}[config["rubric_registry"]["schedule"]]


def plan(matrix: dict) -> dict:
    reference = read_config(ROOT / matrix["base_config"])
    rows, problems = [], []
    for name, condition in matrix["conditions"].items():
        config = read_config(ROOT / condition["config"])
        unexpected = differences(reference, config, matrix["allowed_differences"])
        if unexpected:
            problems.append(f"{name}: differs from {matrix['base_config']} in {unexpected}")
        weights, trained = config["reward"]["weights"], condition["training"]
        strategy = rubric_strategy(config) if trained else "none"
        reuse = next((row.get("reused_from") for row in matrix["ablation_table"]["rows"] if row["condition"] == name), None)
        share = condition.get("shares_initial_rubrics_with")
        launch = None
        if trained:
            env = [f"CONFIG={condition['config']}", "SEED=<seed>"]
            if strategy == "static":
                env.append(f"FIXED_RUBRIC_SET=outputs/serl/<{share} run>/iter0/rubrics/rubric_set.json")
            elif share:
                env.append(f"INITIAL_RUBRICS=outputs/serl/<{share} run>/iter0/rubrics")
            launch = " ".join(env) + " ITERATION=<0|1|2> [RUN_ID=<run id>] bash scripts/run_serl.sh"
        rows.append({
            "condition": name, "label": condition["label"],
            "reward_formula": f"R_t = {formula(weights)}" if trained else "no training",
            "reward_weights": weights if trained else None,
            "rubric_source": condition["rubric_source"], "rubric_strategy": strategy,
            "filtering": condition["filtering"], "evolving": condition["evolving"], "cumulative": condition["cumulative"],
            "regenerated_each_iteration": condition.get("regenerated_each_iteration"),
            "carries_over_earlier_rubrics": condition.get("carries_over_earlier_rubrics"), "weights": condition.get("weights"),
            "new_training_required": trained, "also_used_as": f"ablation row '- Rubric' ({reuse})" if reuse else None,
            "tables": [t for t, members in (("main", matrix["main_table"]), ("ablation", [r["condition"] for r in matrix["ablation_table"]["rows"]]))
                       if name in members],
            "existing_code_reused": {"none": "reward weights 0 skip rubric and predictor calls (existing Outcome-only path)",
                                     "filtered_evolving": "the main SERL pipeline (failure analysis, validation, consolidation; new: src/judges/effectiveness_gap.py)",
                                     "static": "rubric_registry.schedule = fixed (existing Fixed-vs-Evolving condition)",
                                     "no_filtering_cumulative": "failure analysis of the main pipeline; new: src/judges/raw_rubrics.py"}[strategy]
            if trained else "checkpoint_trajectories without an adapter",
            "config_path": condition["config"], "runner_path": "scripts/run_serl.sh" if trained else "scripts/main_experiments.py evaluate",
            "output_path": "outputs/serl/<run id>/iter{0,1,2}/" if trained else f"{matrix['output_dir']}/baseline/eval_dev/",
            "launch": launch, "iterations": matrix["iterations"] if trained else 0, "seeds": matrix["seeds"]})
    return {"matrix": matrix["name"], "model": matrix["model"], "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conditions": rows, "controls": {"reference_config": matrix["base_config"], "allowed_differences": matrix["allowed_differences"],
                                             "problems": problems},
            "open_decisions": [f"{key}: {value}" for section in (matrix, matrix["evaluation"], matrix["rubric_behavior"],
                                                                reference["rubric_registry"]["selection"])
                               for key, value in section.items() if isinstance(value, str) and value.startswith("OPEN DECISION")]}


def write_versioned(path: Path, text: str) -> None:
    """A table is rewritten as results arrive; the file it replaces is kept under history/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        history = path.parent / "history"
        history.mkdir(exist_ok=True)
        path.rename(history / f"{path.stem}_{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
    path.write_text(text, encoding="utf-8")


def plan_markdown(result: dict) -> str:
    columns = ["condition", "reward_formula", "rubric_source", "filtering", "evolving", "cumulative", "regenerated_each_iteration",
               "carries_over_earlier_rubrics", "weights", "new_training_required",
               "existing_code_reused", "config_path", "runner_path", "output_path"]
    lines = [f"# Experiment conditions: {result['matrix']} ({result['model']})", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join("-" if row[c] is None else str(row[c]) for c in columns) + " |" for row in result["conditions"]]
    lines += ["", "Open decisions:", *[f"- {item}" for item in result["open_decisions"]], "",
              "Control check: " + ("passed" if not result["controls"]["problems"] else "; ".join(result["controls"]["problems"])), ""]
    return "\n".join(lines)


def load_runs(out: Path) -> list[dict]:
    path = out / "runs.json"
    return json.loads(path.read_text(encoding="utf-8"))["runs"] if path.exists() else []


def register(matrix: dict, out: Path, condition: str, seed: int, run_id: str) -> dict:
    if condition not in matrix["conditions"] or not matrix["conditions"][condition]["training"]:
        raise SystemExit(f"{condition} is not a trained condition of the matrix")
    config = read_config(ROOT / matrix["conditions"][condition]["config"])
    run_dir = Path(config["output_root"]) / run_id
    if not (ROOT / run_dir).is_dir():
        raise SystemExit(f"{run_dir} does not exist")
    runs = load_runs(out)
    known = next((r for r in runs if (r["condition"], r["seed"]) == (condition, seed)), None)
    if known:
        if known["run_id"] != run_id:
            raise SystemExit(f"{condition} seed {seed} is already run {known['run_id']}")
        return known
    row = {"condition": condition, "seed": seed, "run_id": run_id, "run_dir": str(run_dir), "registered": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps({"runs": runs + [row]}, indent=2) + "\n", encoding="utf-8")
    return row


def eval_dir(matrix: dict, out: Path, runs: list[dict], condition: str, seed: int, table_iteration: int, split: str) -> Optional[Path]:
    if not matrix["conditions"][condition]["training"]:
        return out / "baseline" / f"eval_{split}"
    run = next((r for r in runs if (r["condition"], r["seed"]) == (condition, seed)), None)
    return ROOT / run["run_dir"] / f"iter{table_iteration - 1}" / f"eval_{split}" if run else None


def evaluate(matrix: dict, out: Path, condition: str, seed: int, table_iteration: Optional[int]) -> dict:
    """DEV collaborations of one checkpoint. Loads the policy; TEST is not an option here."""
    from src.common.main_config import load_main_config, open_run
    from src.grpo.trainer import checkpoint_trajectories

    split, settings = matrix["evaluation"]["split"], matrix["conditions"][condition]
    if split != "dev":
        raise SystemExit("this command evaluates on DEV only")
    config = load_main_config(ROOT / settings["config"], ROOT)
    if settings["training"]:
        run = next((r for r in load_runs(out) if (r["condition"], r["seed"]) == (condition, seed)), None)
        if run is None or table_iteration is None:
            raise SystemExit("a trained condition needs a registered run and --table-iteration")
        config, iteration = open_run(config, ROOT, run_id=run["run_id"], resume=True), table_iteration - 1
    else:
        config, iteration = dict(config, output_dir=str((out / "baseline").relative_to(ROOT)), run_id="baseline"), None
    target = eval_dir(matrix, out, load_runs(out), condition, seed, table_iteration or 1, split)
    target.mkdir(parents=True, exist_ok=True)
    return checkpoint_trajectories(config, ROOT, iteration, split=split, output=target / "trajectories.jsonl",
                                   repeats=matrix["evaluation"]["rollouts_per_case"])


def case_values(trajectories: list[dict]) -> dict[str, float]:
    """{case: share of its valid collaborations that ended in the correct diagnosis}."""
    by_case: dict[str, list] = {}
    for t in trajectories:
        if t["trajectory_status"] == "valid":
            by_case.setdefault(t["case_id"], []).append(float(bool(t["is_correct"])))
    return {case: mean(values) for case, values in by_case.items()}


def metrics(directory: Optional[Path], split: str) -> dict:
    path = directory / "trajectories.jsonl" if directory else None
    if path is None or not path.is_file():
        return {"status": "pending"}
    rows = read_jsonl(path)
    if any(t.get("split") not in (None, split) for t in rows):
        raise SystemExit(f"{path} holds cases outside {split}")
    valid = [t for t in rows if t["trajectory_status"] == "valid"]
    values = case_values(rows)
    interval = cluster_bootstrap_mean(values.values(), values.keys())

    def optional(name: str):
        file = directory / f"{name}.json"
        return json.loads(file.read_text(encoding="utf-8")).get("value") if file.is_file() else None

    return {"status": "evaluated", "split": split, "cases": len(values), "collaborations": len(valid), "invalid": len(rows) - len(valid),
            "task_success": interval["mean"], "task_success_ci_low": interval["ci_low"], "task_success_ci_high": interval["ci_high"],
            "behavior_appropriateness": optional("appropriateness"), "effectiveness": optional("effectiveness"),
            "mean_turns": mean(t["num_turns"] for t in valid) if valid else None,
            "early_stop_distribution": dict(sorted(Counter(f"A{t['num_turns']}" for t in valid).items())),
            "per_case": values}


def checkpoint_fields(run: Optional[dict], table_iteration: int) -> dict:
    path = ROOT / run["run_dir"] / f"iter{table_iteration - 1}" / "checkpoint.json" if run else None
    if path is None or not path.is_file():
        return {"checkpoint": None, "protocol_hash": None}
    saved = json.loads(path.read_text(encoding="utf-8"))
    return {"checkpoint": saved["checkpoint_id"], "protocol_hash": saved["protocol_identity"], "trained_reward": saved["reward"]["weights"],
            "rubric_set_version": saved.get("rubric_version")}


def dataset_hash(config: dict) -> Optional[str]:
    manifest = read_config(ROOT / config["environment"]["config"])["dataset"].get("manifest")
    return hashlib.sha256((ROOT / manifest).read_bytes()).hexdigest()[:16] if manifest else None


def table_rows(matrix: dict, out: Path, conditions: list[str], iterations: list[int]) -> list[dict]:
    runs, split, rows = load_runs(out), matrix["evaluation"]["split"], []
    for name in conditions:
        settings = matrix["conditions"][name]
        config = read_config(ROOT / settings["config"])
        weights = config["reward"]["weights"]
        for seed in matrix["seeds"] if settings["training"] else [None]:
            run = next((r for r in runs if (r["condition"], r["seed"]) == (name, seed)), None)
            for table_iteration in iterations:
                measured = metrics(eval_dir(matrix, out, runs, name, seed, table_iteration, split), split)
                row = {"method": settings["label"], "condition": name, "iteration": table_iteration, "seed": seed,
                       "split": split, "dataset_hash": dataset_hash(config),
                       "reward_components_enabled": [c for c in SYMBOL if weights[c] > 0] if settings["training"] else [],
                       "reward_formula": f"R_t = {formula(weights)}" if settings["training"] else "no training",
                       "rubric_strategy": rubric_strategy(config) if settings["training"] else "none",
                       **(checkpoint_fields(run, table_iteration) if settings["training"] else {"checkpoint": "base", "protocol_hash": None}),
                       **measured}
                if not settings["training"]:
                    row["same_checkpoint_in_every_iteration"] = True
                rows.append(row)
    return rows


def write_table(out: Path, name: str, rows: list[dict], fields: list[str], extra: dict) -> None:
    write_versioned(out / f"{name}.json", json.dumps(dict(extra, rows=rows), indent=2, ensure_ascii=False) + "\n")
    lines = [fields] + [["" if row.get(f) is None else json.dumps(row[f], ensure_ascii=False) if isinstance(row[f], (dict, list)) else row[f]
                         for f in fields] for row in rows]
    import io

    buffer = io.StringIO()
    csv.writer(buffer).writerows(lines)
    write_versioned(out / f"{name}.csv", buffer.getvalue())


def tables(matrix: dict, out: Path) -> dict:
    iterations = list(range(1, matrix["iterations"] + 1))
    main = table_rows(matrix, out, matrix["main_table"], iterations)
    note = {"matrix": matrix["name"], "split": matrix["evaluation"]["split"], "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": "table column Iter k = checkpoint of run iteration k - 1; the baseline is one checkpoint shown in every column",
            "test": "not evaluated: TEST is opened once by the final evaluation"}
    write_table(out, "main_table", main, FIELDS, note)
    settings = matrix["ablation_table"]
    final = settings["table_iteration"]
    rows = {row["condition"]: [r for r in table_rows(matrix, out, [row["condition"]], [final])] for row in settings["rows"]}
    reference = {r["seed"]: r for r in rows[settings["reference"]]}
    ablation = []
    for entry in settings["rows"]:
        evaluated = [r for r in rows[entry["condition"]] if r["status"] == "evaluated"]
        pairs = [(case, reference[r["seed"]]["per_case"][case], value) for r in evaluated
                 if reference.get(r["seed"], {}).get("status") == "evaluated"
                 for case, value in r["per_case"].items() if case in reference[r["seed"]]["per_case"]]
        difference = paired_difference(pairs) if pairs and entry["condition"] != settings["reference"] else None
        config = read_config(ROOT / matrix["conditions"][entry["condition"]]["config"])
        ablation.append({"method": entry["label"], "condition": entry["condition"], "iteration": final,
                         "reward_formula": f"R_t = {formula(config['reward']['weights'])}", "reused_from": entry.get("reused_from"),
                         "status": "evaluated" if evaluated else "pending", "seeds": [r["seed"] for r in evaluated], "split": note["split"],
                         "final_task_success": mean(r["task_success"] for r in evaluated) if evaluated else None,
                         "delta_vs_ours": 0.0 if evaluated and entry["condition"] == settings["reference"] else difference and difference["mean_difference"],
                         "paired_ci_low": difference and difference["ci_low"], "paired_ci_high": difference and difference["ci_high"],
                         "paired_cases": difference and difference["clusters"], "paired_n": difference and difference["n_pairs"]})
    write_table(out, "ablation_table", ablation,
                ["method", "condition", "iteration", "reward_formula", "status", "seeds", "split", "final_task_success", "delta_vs_ours",
                 "paired_ci_low", "paired_ci_high", "paired_cases", "paired_n", "reused_from"],
                dict(note, comparison="condition minus Ours on the same cases (and seeds); cluster bootstrap over cases, 95%"))
    for row in main:
        row.pop("per_case", None)
    return {"main_table_rows": len(main), "evaluated": sum(r["status"] == "evaluated" for r in main),
            "ablation_rows": len(ablation), "output_dir": str(out.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "register", "evaluate", "tables"))
    parser.add_argument("--matrix", default=MATRIX)
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--table-iteration", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    out = ROOT / matrix["output_dir"]
    if args.command == "plan":
        result = plan(matrix)
        write_versioned(out / "conditions.json", json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        write_versioned(out / "conditions.md", plan_markdown(result))
        if result["controls"]["problems"]:
            raise SystemExit("\n".join(result["controls"]["problems"]))
    elif args.command == "register":
        result = register(matrix, out, args.condition, args.seed, args.run_id)
    elif args.command == "evaluate":
        result = evaluate(matrix, out, args.condition, args.seed, args.table_iteration)
    else:
        result = tables(matrix, out)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
