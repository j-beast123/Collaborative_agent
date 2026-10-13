"""State of the running work before it is moved to another machine: progress, checkpoints, artifact counts, hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
from collections import Counter
from pathlib import Path

from src.common.config import read_config
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "outputs/main_experiments/launch"
ENV_CONFIG = "configs/effectiveness/v10_qwen4b_gemma12b_max7.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0


def grpo_runs() -> list[dict]:
    runs = []
    for id_file in sorted(LAUNCH.glob("*.run_id")):
        run = ROOT / "outputs/serl" / id_file.read_text().strip()
        manifest = json.loads((run / "manifest.json").read_text())
        iterations = []
        for directory in sorted(run.glob("iter*"), key=lambda p: int(p.name[4:])):
            checkpoints = sorted((directory / "trainer").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
            last = checkpoints[-1] if checkpoints else None
            state = json.loads((last / "trainer_state.json").read_text()) if last and (last / "trainer_state.json").is_file() else {}
            steps = [json.loads(line) for line in (directory / "train_log.jsonl").open()] if (directory / "train_log.jsonl").is_file() else []
            iterations.append({
                "iteration": int(directory.name[4:]), "finished": (directory / "checkpoint.json").is_file(),
                "steps_done": max((s.get("global_step", 0) for s in steps), default=0), "steps_total": state.get("max_steps"),
                "reward_rows": lines(directory / "reward_log.jsonl"),
                "last_trainer_checkpoint": str(last.relative_to(ROOT)) if last else None,
                "last_trainer_checkpoint_step": state.get("global_step"),
                "checkpoint_files": sorted(p.name for p in last.iterdir()) if last else [],
                "checkpoint_sha256": {name: sha(last / name) for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
                                      if last and (last / name).is_file()},
                "steps_after_the_last_checkpoint_would_be_trained_again": max((s.get("global_step", 0) for s in steps), default=0) - (state.get("global_step") or 0),
                "done_steps_of_the_runner": sorted(p.name[6:] for p in directory.glob(".done_*")),
                "next_iteration_trajectories": lines(directory / "trajectories.jsonl")})
        runs.append({"launch_script": str((LAUNCH / (id_file.stem + ".sh")).relative_to(ROOT)), "run_id": run.name,
                     "run_dir": str(run.relative_to(ROOT)), "config_hash": manifest["config_hash"], "seed": manifest["seed"],
                     "resolved_config_sha256": sha(run / "resolved_config.json"), "iterations": iterations,
                     "resume": f"bash {(LAUNCH / (id_file.stem + '.sh')).relative_to(ROOT)}   (in tmux; a finished step is skipped, "
                               "the train step continues from the last trainer checkpoint)"})
    return runs


def labels() -> dict:
    from scripts.run_agentclinic_turns import scope_histories, select_train_dev

    config = read_config(ROOT / ENV_CONFIG)
    out, k, names = ROOT / config["labels_dir"], config["k"], config["candidates"]["names"]
    selected, _ = select_train_dev(config)
    cases = {c["case_id"]: c for c in selected}
    counts = {}
    for scope in ("smoke", "full"):
        ids = {h["history_id"] for h in scope_histories(config, cases, scope)}
        compliance = [c for c in read_jsonl(out / "candidate_compliance.jsonl", missing_ok=True) if c["history_id"] in ids]
        compliant = sum(bool(c["behavior_compliant"]) for c in compliance)
        before = [r for r in read_jsonl(out / "before_rollouts.jsonl", missing_ok=True) if r["history_id"] in ids]
        after = [r for r in read_jsonl(out / "after_rollouts.jsonl", missing_ok=True) if r["history_id"] in ids]
        counts[scope] = {"histories": len(ids), "candidate_pairs_judged": len(compliance), "candidate_pairs_expected": len(ids) * len(names),
                         "compliant_pairs": compliant, "before_continuations": len(before), "before_expected": len(ids) * k,
                         "after_continuations": len(after), "after_expected_for_the_compliant_pairs_so_far": compliant * k,
                         "labels": sum(r["history_id"] in ids for r in read_jsonl(out / "labels.jsonl", missing_ok=True)),
                         "status": dict(Counter(r["status"] for r in before + after))}
    files = {p.name: {"rows": lines(p), "sha256": sha(p)} for p in sorted(out.glob("*.jsonl"))}
    return {"config": ENV_CONFIG, "labels_dir": str(out.relative_to(ROOT)), "progress": counts, "files": files,
            "smoke_report": (out / "smoke_report.json").is_file(), "labels_frozen": (out.parent / "labels.freeze.json").is_file(),
            "resume": "bash scripts/run_labels_predictor.sh   (in tmux; every written candidate, judgment and continuation is kept and "
                      "only the missing ones are made: keys are (history, candidate, attempt, rollout index) for AFTER and "
                      "(history, rollout index) for BEFORE, which the candidates of a history share)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output")
    args = parser.parse_args()
    out = ROOT / (args.output or f"outputs/audit/migration_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True)
    environment = read_config(ROOT / ENV_CONFIG)
    frozen = json.loads((ROOT / environment["output_dir"] / "protocol.frozen.json").read_text())
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines()
    manifest = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": socket.gethostname(), "gpus": gpu, "repository": str(ROOT),
        "protocol_identity": frozen["protocol_identity"], "protocol_frozen_sha256": sha(ROOT / environment["output_dir"] / "protocol.frozen.json"),
        "dataset_manifest_sha256": sha(ROOT / environment["dataset"]["manifest"]),
        "histories_sha256": frozen["histories_sha256"], "base_trajectories_sha256": frozen["base_trajectories_sha256"],
        "models": {"agent": environment["models"]["agent"], "doctor": environment["models"]["doctor"], "judge": environment["models"]["judge"],
                   "label_roles": {k: v.get("model_id") for k, v in environment["label_roles"].items() if isinstance(v, dict)}},
        "grpo_runs": grpo_runs(), "effectiveness_labels": labels(),
        "to_copy": ["the repository directory with outputs/ (run directories, label files, trainer checkpoints) and data/",
                    "the model directories <MODEL_ROOT>/Qwen3-4B and Qwen3-8B, or the same revisions at the same paths",
                    "the Python environment (<workspace>/.venv-qwen3) and <workspace>/.pydeps (trl, datasets, grpc)",
                    "<workspace>/.env with OPENROUTER_API_KEY (not part of the repository)"],
        "on_the_other_machine": ["the paths in the configs are absolute for the models and relative to the repository for everything else",
                                 "CUDA_VISIBLE_DEVICES is set in the launch scripts (7 here): change it there to the GPU of each job",
                                 "a continued run keeps its own resolved_config.json; a changed prompt file stops it",
                                 "a resume is recorded with host and GPU in iter<i>/resume_log.jsonl"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run_lines = [f"- `{r['run_id']}` (config hash {r['config_hash']}): " + "; ".join(
        f"iter{i['iteration']} {'finished' if i['finished'] else str(i['steps_done']) + '/' + str(i['steps_total']) + ' steps, last checkpoint step ' + str(i['last_trainer_checkpoint_step'])}"
        for i in r["iterations"]) for r in manifest["grpo_runs"]]
    progress = manifest["effectiveness_labels"]["progress"]
    (out / "README.md").write_text("\n".join([
        f"# State before a move ({manifest['written']}, host {manifest['host']})", "",
        f"Protocol identity {manifest['protocol_identity']}.", "", "GRPO runs:", *run_lines, "",
        "Effectiveness labels: " + "; ".join(f"{scope}: BEFORE {p['before_continuations']}/{p['before_expected']}, AFTER {p['after_continuations']}/"
                                             f"{p['after_expected_for_the_compliant_pairs_so_far']}, labels {p['labels']}" for scope, p in progress.items()), "",
        "To pause a GRPO iteration at a completed step: `bash scripts/pause_at_checkpoint.sh <tmux session> <run dir> <iteration>`.",
        "To pause the labels: stop the tmux session; continuations in progress are lost, written ones are kept.",
        "To continue: the commands under `resume` in manifest.json.", ""]), encoding="utf-8")
    print(json.dumps({"output": str(out.relative_to(ROOT)), "grpo": run_lines, "labels": progress}, indent=2))


if __name__ == "__main__":
    main()
