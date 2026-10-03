"""RQ3 failure resolution on a FIXED probe set: the input of src/evaluation/rq3.py:probe_transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from scripts.main_experiments import eval_dir, load_matrix, load_runs, write_versioned
from src.agentclinic.selection import split_eligible_cases
from src.common.config import read_config
from src.common.io import read_jsonl
from src.common.main_config import load_main_config
from src.evaluation.rq3 import probe_transitions
from src.evaluation.stats import cluster_bootstrap_mean

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "dev"
SEEDS_PER_CASE = 3
EVALUATION_SEEDS = [0, 1, 2]
TRANSITIONS = ("fail_to_success", "fail_to_fail", "success_to_fail", "success_to_success")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def probe_dir(matrix: dict) -> Path:
    return ROOT / matrix["output_dir"] / "rq3_fixed_probe"


def freeze(matrix: dict) -> dict:
    target = probe_dir(matrix) / "manifest.json"
    if target.exists():
        raise SystemExit(f"{target} exists: the fixed probe set is not written again")
    config = load_main_config(ROOT / matrix["base_config"], ROOT)
    environment = read_config(ROOT / config["environment"]["config"])
    cases = split_eligible_cases(environment, ROOT, allowed_splits=(SPLIT,))
    histories = {h["case_id"]: h for h in read_jsonl(ROOT / config["environment"]["histories"]) if h["case_id"] in cases}
    if sorted(histories) != sorted(cases):
        raise SystemExit("every DEV case needs a frozen starting history")
    probes = [{"probe_id": cid, "case_id": cid, "history_id": histories[cid]["history_id"],
               "starting_history_sha256": digest(histories[cid]["history"])} for cid in sorted(cases)]
    manifest = {"matrix": matrix["name"], "split": SPLIT, "selection": "every DEV case of the dataset version; no outcome-based choice",
                "probes": probes, "probe_count": len(probes), "seeds_per_case": SEEDS_PER_CASE, "evaluation_seeds": EVALUATION_SEEDS,
                "probe_unit": "(case_id, evaluation seed)", "probe_units": len(probes) * SEEDS_PER_CASE,
                "uncertainty": "bootstrap over original cases (a case's units are resampled together)",
                "dataset_manifest": environment["dataset"]["manifest"],
                "dataset_manifest_sha256": hashlib.sha256((ROOT / environment["dataset"]["manifest"]).read_bytes()).hexdigest(),
                "starting_histories": config["environment"]["histories"],
                "protocol": environment["protocol"],
                "environment_protocol_identity": json.loads((ROOT / environment["output_dir"] / "protocol.frozen.json").read_text())["protocol_identity"],
                "run_identity_note": "a run's trajectories carry the run-level identity (environment, state prompt and the label "
                                     "protocol hash of its Effectiveness predictor), so runs created before and after the predictor "
                                     "was connected differ in it; every checkpoint file must hold one identity, recorded per row",
                "environment_config": config["environment"]["config"],
                "evaluation": {"procedure": "scripts/rq3_fixed_probes.py evaluate (the DEV evaluation of scripts/main_experiments.py: natural "
                                            "collaboration of the checkpoint from the starting history to the Doctor's final diagnosis, "
                                            "one pass per evaluation seed)",
                               "agent_decoding": environment["decoding"]["agent"], "doctor_decoding": environment["decoding"]["doctor"],
                               "max_attempts": config["rollout"]["max_attempts"],
                               "seed_note": "the random generators are set to the evaluation seed before each pass; concurrent "
                                            "collaborations share generation batches, so a pass is not reproducible token by token"},
                "checkpoints": "Base (iteration 0), Iter 1, Iter 2, Iter 3 of one condition and seed", "frozen": time.strftime("%Y-%m-%d %H:%M:%S")}
    manifest["manifest_sha256"] = digest(manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    target.chmod(0o444)
    return {"written": str(target.relative_to(ROOT)), "probes": len(probes), "manifest_sha256": manifest["manifest_sha256"]}


def probe_path(matrix: dict, out: Path, runs: list[dict], condition: str, seed: int, table_iteration: int) -> Optional[Path]:
    directory = eval_dir(matrix, out, runs, condition, seed, table_iteration, SPLIT)
    return directory.parent / f"probe_{SPLIT}" / "trajectories.jsonl" if directory else None


def evaluate(matrix: dict, condition: str, seed: int, table_iteration: Optional[int]) -> dict:
    """Every probe unit for one checkpoint. Loads the policy (GPU); DEV only."""
    import random

    import torch

    from src.agentclinic.runtime import build_clinic, check_identity, frozen_histories, load_policy, protocol_identity
    from src.agentclinic.trajectory import generate_trajectories, summary
    from src.common.main_config import open_run
    from src.grpo.trainer import iteration_dir, spec_of

    out = ROOT / matrix["output_dir"]
    manifest = load_manifest(matrix)
    settings = matrix["conditions"][condition]
    config = load_main_config(ROOT / settings["config"], ROOT)
    runs = load_runs(out)
    if settings["training"]:
        run = next((r for r in runs if (r["condition"], r["seed"]) == (condition, seed)), None)
        if run is None or table_iteration is None:
            raise SystemExit("a trained condition needs a registered run and --table-iteration")
        config, iteration = open_run(config, ROOT, run_id=run["run_id"], resume=True), table_iteration - 1
        checkpoint_id = json.loads((iteration_dir(config, ROOT, iteration) / "checkpoint.json").read_text(encoding="utf-8"))["checkpoint_id"]
        adapter = iteration_dir(config, ROOT, iteration) / "adapter"
    else:
        config, iteration, checkpoint_id, adapter = dict(config, output_dir=str((out / "baseline").relative_to(ROOT)), run_id="baseline"), None, "base", None
    target = probe_path(matrix, out, runs, condition, seed, table_iteration or 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_policy(config["policy"], adapter, trainable=False, device=config["devices"]["policy"])
    clinic = build_clinic(config, ROOT, model, tokenizer)
    identity = protocol_identity(config, ROOT)
    check_identity(read_jsonl(target, missing_ok=True), identity, target)
    wanted = {p["case_id"] for p in manifest["probes"]}
    with clinic.generation():
        histories = [row for row in frozen_histories(config, ROOT, clinic.cases) if row["case_id"] in wanted]
        if sorted(h["case_id"] for h in histories) != sorted(wanted):
            raise SystemExit("the frozen starting histories do not cover the probe set")
        for evaluation_seed in manifest["evaluation_seeds"]:
            random.seed(evaluation_seed)
            torch.manual_seed(evaluation_seed)
            torch.cuda.manual_seed_all(evaluation_seed)
            rows = generate_trajectories(clinic.cases, histories, clinic.runtime, clinic.judge, target,
                                         suffix=f"{checkpoint_id}:s{evaluation_seed}", workers=config["rollout"]["workers"],
                                         max_attempts=config["rollout"]["max_attempts"],
                                         iteration=0 if iteration is None else iteration + 1, checkpoint_id=checkpoint_id,
                                         protocol_identity=identity, spec=spec_of(clinic.source), rollout_index=evaluation_seed)
    clinic.close()
    return dict(summary(rows), checkpoint_id=checkpoint_id, probe_units=len(rows), output=str(target.relative_to(ROOT)))


def load_manifest(matrix: dict) -> dict:
    manifest = json.loads((probe_dir(matrix) / "manifest.json").read_text(encoding="utf-8"))
    if digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise SystemExit("the probe manifest was edited after it was frozen")
    return manifest


def checkpoint_rows(manifest: dict, trajectories: list[dict], iteration: int, where: str) -> list[dict]:
    """One row per probe unit (case, evaluation seed) for one checkpoint, after the checks against the manifest."""
    probes = {p["probe_id"]: p for p in manifest["probes"]}
    expected = {(cid, seed) for cid in probes for seed in manifest["evaluation_seeds"]}
    units = {}
    for t in trajectories:
        unit = (t["case_id"], t.get("rollout_index"))
        if unit in units:
            raise SystemExit(f"{where}: more than one collaboration for the probe unit {unit}")
        units[unit] = t
    if set(units) != expected:
        raise SystemExit(f"{where}: the evaluated units are not exactly the probe units "
                         f"(missing {sorted(expected - set(units), key=str)[:3]}, extra {sorted(set(units) - expected, key=str)[:3]})")
    if len({t.get("protocol_identity") for t in trajectories}) != 1:
        raise SystemExit(f"{where}: collaborations of more than one protocol identity")
    rows = []
    for (cid, seed), t in sorted(units.items()):
        opening = [{k: v for k, v in turn.items() if k in ("speaker", "stage", "text", "provisional_diagnosis")} for turn in t["turns"][:1]]
        if digest(opening) != probes[cid]["starting_history_sha256"]:
            raise SystemExit(f"{where}: {cid} did not start from the frozen starting history")
        rows.append({"probe_id": f"{cid}#s{seed}", "case_id": cid, "evaluation_seed": seed, "iteration": iteration,
                     "success": bool(t["is_correct"]) if t["trajectory_status"] == "valid" else None,
                     "final_answer": t.get("final_diagnosis"), "num_turns": t.get("num_turns"), "stop": t.get("stop"),
                     "trajectory_status": t["trajectory_status"], "collaboration_state": t.get("collaboration_state"),
                     "checkpoint": t.get("checkpoint_id"), "protocol_hash": t.get("protocol_identity"),
                     "dataset_hash": manifest["dataset_manifest_sha256"][:16]})
    return rows


def transition_shares(rows: list[dict]) -> list[dict]:
    """For category ALL: the share of each transition between consecutive checkpoints, with an interval from resampling cases."""
    by_unit = {}
    for row in rows:
        if row["category_id"] == "ALL":
            by_unit.setdefault(row["probe_id"], {})[row["iteration"]] = row
    iterations = sorted({r["iteration"] for r in rows})
    shares = []
    for before, after in zip(iterations, iterations[1:]):
        pairs = [(u[before], u[after]) for u in by_unit.values() if before in u and after in u]
        for name in TRANSITIONS:
            first, second = name.split("_to_")
            values = [float(a["success"] == (first == "success") and b["success"] == (second == "success")) for a, b in pairs]
            interval = cluster_bootstrap_mean(values, [a["case_id"] for a, _ in pairs])
            shares.append({"from_iteration": before, "to_iteration": after, "transition": name, "share": interval["mean"],
                           "ci_low": interval["ci_low"], "ci_high": interval["ci_high"], "units": interval["n"], "cases": interval["clusters"]})
    return shares


def collect(matrix: dict, condition: str, seed: int) -> dict:
    out = ROOT / matrix["output_dir"]
    manifest = load_manifest(matrix)
    runs = load_runs(out)
    sources = {0: probe_path(matrix, out, runs, "baseline", seed, 1)}
    for table_iteration in range(1, matrix["iterations"] + 1):
        sources[table_iteration] = probe_path(matrix, out, runs, condition, seed, table_iteration)
    per_checkpoint, pending = {}, []
    for iteration, path in sources.items():
        if path is None or not path.is_file():
            pending.append(iteration)
            continue
        per_checkpoint[iteration] = checkpoint_rows(manifest, read_jsonl(path), iteration, str(path.relative_to(ROOT)))
    if 0 not in per_checkpoint:
        raise SystemExit("the Base evaluation of the probes does not exist yet (rq3_fixed_probes evaluate --condition baseline)")
    base_state = {r["probe_id"]: r["collaboration_state"] for r in per_checkpoint[0]}
    rows, excluded = [], []
    for iteration, members in sorted(per_checkpoint.items()):
        for row in members:
            if row["success"] is None:  # an invalid collaboration is not a success or a failure
                excluded.append({"probe_id": row["probe_id"], "iteration": iteration, "status": row["trajectory_status"]})
                continue
            for category in ("ALL", f"state:{base_state[row['probe_id']]}"):
                rows.append({**{k: v for k, v in row.items() if k != "collaboration_state"}, "category_id": category})
    transitions = probe_transitions(rows)
    name = f"{condition}_seed{seed}"
    directory = probe_dir(matrix)
    write_versioned(directory / f"probe_results_{name}.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    result = {"condition": condition, "seed": seed, "probe_manifest_sha256": manifest["manifest_sha256"], "probes": manifest["probe_count"],
              "probe_units": manifest["probe_units"], "evaluation_seeds": manifest["evaluation_seeds"],
              "transition_shares_all_probes_case_bootstrap": transition_shares(rows),
              "checkpoints_evaluated": sorted(per_checkpoint), "checkpoints_pending": pending, "excluded_invalid": excluded,
              "success_by_checkpoint": {str(i): sum(bool(r["success"]) for r in members) for i, members in sorted(per_checkpoint.items())},
              "transitions": transitions, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "note": "iteration 0 is Base; transitions are between consecutive evaluated checkpoints, on the same probes"}
    write_versioned(directory / f"probe_transitions_{name}.json", json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("freeze", "evaluate", "collect"))
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int, default=0, help="the training seed of the condition's run")
    parser.add_argument("--table-iteration", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    if matrix["evaluation"]["split"] != SPLIT:
        raise SystemExit("the fixed probes of this script are the DEV cases")
    result = freeze(matrix) if args.command == "freeze" else collect(matrix, args.condition, args.seed) if args.command == "collect" \
        else evaluate(matrix, args.condition, args.seed, args.table_iteration)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
