"""Isolated launcher binding CollabLLM scoring into the original five-GPU loop.

The original module is loaded in its own namespace. Only its experiment hook is
replaced; workers, tree building, checkpointing, evaluation and GRPO stay intact.
The hook runs before the original records digest, checkpoint and optimizer step.
Default invocation is a preflight only. --launch is required for live execution.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from src.online_rl.integrated.records import atomic_write_json, from_dict, sha256_json
from src.online_rl.tree import experiment
from src.online_rl.tree.records import Tree
from src.online_rl.tree.cache import CallCache
from baselines.collabllm.batch import prepare_batch
from baselines.collabllm.data import ROOT, audit_data
from baselines.collabllm.pilot import pilot_plan, summarize_pilot
from baselines.collabllm.reward import Cl100kCounter, InteractivityJudge, METHOD, score_tree
from baselines.common.protocol import apply_common, verify_sherpa_manifest, settings
from baselines.common.final_eval import evaluate_completed_run

SPEC_PATH = Path(__file__).with_name("adaptation_spec.json")
COMMON = ("policy", "data", "rollout", "schedule", "optimizer", "advantage_normalization", "five_gpu")


def common_audit():
    spec = json.loads(SPEC_PATH.read_text())
    fixed_path = spec["reference_sherpa_no_prompt_opt_config"]
    outcome_path = spec["reference_outcome_only_config"]
    fixed, outcome = (apply_common(experiment.load_config(path)) for path in (fixed_path, outcome_path))
    if fixed["prompt_update"]["enabled"] or fixed["experiment"]["prompt_mode"] != "p0_fixed":
        raise RuntimeError("The reference SHERPA config must have prompt optimization disabled")
    differs = [key for key in COMMON if fixed[key] != outcome[key]]
    if differs:
        raise RuntimeError(f"Common comparison conditions differ: {differs}")
    return {"scope": "current_checkout_not_a_remote_latest_claim", "reference_commit": spec["reference_commit"],
            "fixed_sherpa_config": fixed_path, "outcome_config": outcome_path,
            "common_sections": list(COMMON), "differences": differs,
            "common_sha256": sha256_json({k: fixed[k] for k in COMMON}),
            "prompt_optimization": False}


def source_hash():
    paths = list(Path(__file__).parent.glob("*.py")) + [SPEC_PATH, Path(__file__).with_name("interactivity.txt")]
    paths = [p for p in paths if not p.name.startswith("test_")]
    paths += [ROOT / "scripts/online_rl/run_five_gpu.py"]
    paths += [p for p in (ROOT / "baselines/common").glob("*.py") if not p.name.startswith("test_")] + [ROOT / "baselines/common/settings.json"]
    return sha256_json({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def frozen_identity(weights):
    copy_ = dict(weights)
    claimed = copy_.pop("sha256")
    if claimed != sha256_json(copy_) or weights.get("schema") != "collabllm_frozen_weights_v2":
        raise ValueError("Invalid frozen weight receipt")
    if weights["pilot_sha256"] != sha256_json(weights["pilot"]):
        raise ValueError("Frozen pilot was modified")
    if weights["pilot"].get("pilot_plan") != pilot_plan():
        raise ValueError("Frozen weights use a different pilot plan")
    if not weights["pilot"].get("stability", {}).get("ready_for_weight_review"):
        raise ValueError("Pilot expansion/review must precede a training launch")
    return claimed


def configuration(*, output, receipt, judge, counter, weights=None, pilot_cases=None,
                  initial_policy_sha256=None, base_model_revision=None):
    audit = common_audit()
    spec = json.loads(SPEC_PATH.read_text())
    cfg = apply_common(experiment.load_config(spec["reference_outcome_only_config"]))
    cfg["output_dir"] = str(Path(output).resolve())
    cfg["name"] = "collabllm_adapted_grpo"
    cfg["condition"] = "collabllm_adapted_grpo"
    # Keep the outcome-only experiment metadata as a transport validation rule
    # for workers; this parent hook supplies the actual method fingerprint.
    if weights is not None:
        digest = frozen_identity(weights)
        if any(weights["pilot"][k] != value for k, value in (("judge", judge.identity), ("tokenizer", counter.identity))):
            raise ValueError("Judge/tokenizer differs from the pilot used to choose lambda")
        if weights["pilot"]["initial_policy_sha256"] != initial_policy_sha256:
            raise ValueError("Initial policy differs from frozen pilot")
    else:
        digest = None
    if pilot_cases is not None:
        if pilot_cases not in pilot_plan()["stage_sizes"]:
            raise ValueError("Pilot must use a predeclared stage size")
        cfg["five_gpu"]["checkpoint_evaluation_enabled"] = False
    elif weights is None:
        raise ValueError("Training requires frozen pilot weights")
    # Remap paths only after checking original bytes. Hash values never change.
    for section, key, role in (("data", "train", "train"), ("five_gpu", "train_order_file", "train_order"),
                               ("five_gpu", "eval_file", "test")):
        cfg[section][key] = receipt["files"][role]["path"]
    cfg["collabllm"] = {"method_label": METHOD, "source_sha256": source_hash(),
        "common_sha256": audit["common_sha256"], "weights_sha256": digest,
        "judge": judge.identity, "tokenizer": counter.identity,
        "initial_policy_sha256": initial_policy_sha256, "base_model_revision": base_model_revision,
        "pilot_mode": pilot_cases is not None, "pilot_plan": pilot_plan(),
        "data_sha256": {k: v["actual_sha256"] for k, v in receipt["files"].items()}}
    return cfg, audit


class RewardHook:
    """Parent-only hook; the subprocess workers still produce original trees."""
    def __init__(self, cfg, receipt, judge, counter, weights=None):
        self.cfg, self.receipt, self.judge, self.counter, self.weights = cfg, receipt, judge, counter, weights
        self.out = Path(cfg["output_dir"])
        self.cache = CallCache(self.out / "collabllm/judge_cache.jsonl")
        self.fingerprint = sha256_json(cfg["collabllm"])

    def __getattr__(self, name):
        return getattr(experiment, name)

    def receipt_fields(self, cfg):
        return {"experiment_fingerprint": self.fingerprint, "method_label": METHOD,
                "collabllm_weights_sha256": cfg["collabllm"]["weights_sha256"]}

    def state_fields(self, cfg, theta0):
        return {**self.receipt_fields(cfg), "theta0_policy_sha256": theta0}

    def check_receipt(self, cfg, receipt, what):
        if any(receipt.get(k) != v for k, v in self.receipt_fields(cfg).items()):
            raise RuntimeError(f"{what}: CollabLLM weights/judge/tokenizer/source identity changed")

    def check_pfinal_source(self, cfg, theta0, protocol):
        if theta0 != cfg["collabllm"]["initial_policy_sha256"]:
            raise RuntimeError("Loaded initial adapter differs from the comparison/pilot adapter")

    def check_resume(self, cfg, out, state, theta0):
        self.check_receipt(cfg, state, "Run state")
        self.check_pfinal_source(cfg, theta0, None)
        if state.get("theta0_policy_sha256") != theta0:
            raise RuntimeError("Resume initial policy differs")
        checkpoint = Path(state["checkpoint"]).resolve()
        if checkpoint.parent != (Path(out) / "checkpoints").resolve():
            raise RuntimeError("Checkpoint is outside this run")
        self.check_receipt(cfg, json.loads((checkpoint / "COMPLETE.json").read_text()), "Checkpoint")

    def _score(self, record):
        tree = from_dict(Tree, record["tree"])
        path = self.out / "collabllm/scores" / f"case_{record['case_cursor']:06d}.json"
        if path.exists():
            report = json.loads(path.read_text())
            if (report["tree_sha256"] != sha256_json(record["tree"])
                    or report["judge"] != self.judge.identity or report["tokenizer"] != self.counter.identity):
                raise RuntimeError("Stored terminal scores belong to another tree/judge/tokenizer")
        else:
            report = score_tree(tree, judge=self.judge, cache=self.cache, token_counter=self.counter)
            atomic_write_json(path, report)
        return tree, report

    def check_batch(self, cfg, records):
        experiment.check_batch(cfg, records)  # validate the untouched workers' outcome-only records
        expected_policy = records[0]["policy_sha256"]
        pairs = [self._score(record) for record in records]
        if cfg["collabllm"]["pilot_mode"]:
            if expected_policy != cfg["collabllm"]["initial_policy_sha256"]:
                raise RuntimeError("Pilot changed the initial policy")
            for record, (_, report) in zip(records, pairs):
                record["collabllm"] = report
            return [{"case_id": t.case_id, "pilot_only": True, "components": r["leaves"]} for t, r in pairs]
        transformed = prepare_batch(records, [r for _, r in pairs], frozen_weights=self.weights,
            data_receipt=self.receipt, expected_policy=expected_policy)
        records[:] = transformed  # upstream digests and checkpoints the actual MR targets
        return [{"method_label": METHOD, "case_id": r["case_id"],
                 "weights_sha256": self.weights["sha256"], "nodes": r["agent_targets"]["nodes"]} for r in records]


def isolated_runner(hook):
    path = ROOT / "scripts/online_rl/run_five_gpu.py"
    spec = importlib.util.spec_from_file_location("collabllm_sherpa_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.experiment = hook
    if hook.cfg.get("collabllm", {}).get("pilot_mode"):
        module.update_batch = lambda *args: {"optimizer_step": False, "optimizer_steps": 0, "pilot_only": True}
    return module


def bind_run(cfg, *, create=False):
    """Reject changed resume identities before starting Doctors or API work."""
    path = Path(cfg["output_dir"]) / "collabllm/binding.json"
    value = {"identity": cfg["collabllm"], "sha256": sha256_json(cfg["collabllm"])}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("Resume binding differs: weights/judge/tokenizer/data/model/source changed")
    elif (Path(cfg["output_dir"]) / "run_state.json").exists():
        raise RuntimeError("Existing run has no CollabLLM binding")
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)


def reuse_first_pilot_batch(runner, hook, pilot_output):
    """Intercept only the first initial-policy rollout, using verified pilot records.

    All later submissions go directly to the original executor. No pilot trees
    from cases 5+ can enter the subsequently updated policy's training batches.
    """
    from concurrent.futures import Future
    root = Path(pilot_output)
    binding = json.loads((root / "collabllm/binding.json").read_text())
    expected = {**hook.cfg["collabllm"], "pilot_mode": True, "weights_sha256": None}
    if binding != {"identity": expected, "sha256": sha256_json(expected)}:
        raise RuntimeError("Pilot source differs from this training run")
    saved = []
    for index in range(5):
        record = json.loads((root / "cases" / f"case_{index:06d}.json").read_text())
        report = record.pop("collabllm")
        if (sha256_json(report) != hook.weights["pilot"]["report_sha256"][index]
                or sha256_json(record["tree"]) != report["tree_sha256"]
                or record["case_id"] != hook.receipt["train_case_order"][index]
                or record["policy_sha256"] != expected["initial_policy_sha256"]
                or record.get("batch_update", {}).get("optimizer_steps")):
            raise RuntimeError("Pilot record/report is inconsistent or already updated")
        record.pop("batch_update", None)
        saved.append(record)
        destination = hook.out / "collabllm/scores" / f"case_{index:06d}.json"
        if destination.exists() and json.loads(destination.read_text()) != report:
            raise RuntimeError("Training scores disagree with first pilot batch")
        atomic_write_json(destination, report)
    # Copying a score is not another paid judgment; audit keeps pilot provenance.
    atomic_write_json(hook.out / "collabllm/pilot_reuse.json", {
        "pilot_output": str(root.resolve()), "cases": 5,
        "report_sha256": hook.weights["pilot"]["report_sha256"][:5],
        "new_rollout_calls": 0, "new_judge_calls": 0})
    def wrap(factory):
        class Executor:
            def __init__(self, *args, **kwargs):
                self.inner = factory(*args, **kwargs)
            def submit(self, fn, *args, **kwargs):
                if (fn.__module__ == "src.online_rl.tree.five_gpu_worker" and fn.__name__ == "run_many"
                        and all(0 <= job["index"] < 5 for job in args[0])):
                    if args[2] != expected["initial_policy_sha256"]:
                        raise RuntimeError("First pilot batch cannot be replayed at a different policy")
                    rows = []
                    for job in args[0]:
                        row = copy.deepcopy(saved[job["index"]])
                        if row["case_id"] != job["case"]["case_id"]:
                            raise RuntimeError("Pilot job order differs")
                        row.update(rollout_gpu=job["worker"], dialogue_lane=job["lane"])
                        rows.append(row)
                    future = Future()
                    future.set_result(rows)
                    return future
                return self.inner.submit(fn, *args, **kwargs)
            def shutdown(self, *args, **kwargs):
                return self.inner.shutdown(*args, **kwargs)
        return Executor
    runner.ProcessPoolExecutor = wrap(runner.ProcessPoolExecutor)
    runner.ThreadPoolExecutor = wrap(runner.ThreadPoolExecutor)
    # Reuse all successful visible-input judgments, including expansion cases.
    # This reuses scores only; it never replays expansion trees after an update.
    source_cache = CallCache(root / "collabllm/judge_cache.jsonl")
    for row in source_cache.store.values():
        if row["key"] in hook.cache.store and hook.cache.store[row["key"]] != row:
            raise RuntimeError("Pilot and training judge caches conflict")
        hook.cache.call(row["kind"], row["identity"], lambda row=row: row["value"])


def complete_pilot(cfg, receipt, target_cases):
    pairs = []
    out = Path(cfg["output_dir"])
    for index in range(target_cases):
        record = json.loads((out / "cases" / f"case_{index:06d}.json").read_text())
        if record.get("batch_update", {}).get("optimizer_steps"):
            raise RuntimeError("Pilot contains optimizer updates")
        pairs.append((from_dict(Tree, record["tree"]), record["collabllm"]))
    summary = summarize_pilot(pairs, receipt, initial_policy_sha256=cfg["collabllm"]["initial_policy_sha256"])
    atomic_write_json(out / f"pilot_{target_cases:03d}.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/collabllm_train120")
    p.add_argument("--weights")
    p.add_argument("--sherpa-manifest")
    p.add_argument("--pilot-output", help="Verified pilot run to reuse for training batch 1")
    p.add_argument("--pilot-cases", type=int, choices=(5, 20, 50))
    p.add_argument("--judge-model", default="openai/gpt-5.6-luna")
    p.add_argument("--judge-revision")
    p.add_argument("--initial-policy-sha256")
    p.add_argument("--base-model-revision")
    for name in ("train", "train-order", "test"):
        p.add_argument("--" + name)
    p.add_argument("--launch", action="store_true")
    p.add_argument("--max-batches", type=int)
    a = p.parse_args()
    receipt = audit_data(train=a.train, train_order=a.train_order, test=a.test)
    audit = common_audit()
    blockers = []
    base_cfg = apply_common(experiment.load_config(json.loads(SPEC_PATH.read_text())["reference_outcome_only_config"]))
    blockers.extend(verify_sherpa_manifest(a.sherpa_manifest, base_cfg,
        initial_policy_sha256=a.initial_policy_sha256, base_model_revision=a.base_model_revision))
    if not receipt["all_verified"]:
        blockers.append("Original train120/train-order/test files have not all passed verification")
    for flag in ("judge_revision", "initial_policy_sha256", "base_model_revision"):
        if not getattr(a, flag):
            blockers.append(f"Missing --{flag.replace('_', '-')}")
    if (a.pilot_cases is None) == (a.weights is None):
        blockers.append("Choose either --pilot-cases or --weights")
    if a.weights and not a.pilot_output:
        blockers.append("Training requires --pilot-output for initial batch reuse")
    if a.max_batches is not None and a.max_batches < 1:
        blockers.append("--max-batches must be positive")
    import shutil
    if not (os.environ.get("OPENROUTER_API_KEY") or (ROOT / ".env").exists() or os.environ.get("SHERPA_ENV_FILE")):
        blockers.append("OPENROUTER_API_KEY/.env is not configured")
    if not shutil.which(os.environ.get("SHERPA_OLLAMA_BINARY", "ollama")):
        blockers.append("Pinned Ollama binary is unavailable; set SHERPA_OLLAMA_BINARY")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 5:
        blockers.append("Five CUDA GPUs are required for the unchanged live protocol")
    print(json.dumps({"method": METHOD, "common_settings": audit, "data": receipt,
                      "launch": a.launch, "blockers": blockers}, ensure_ascii=False, indent=2))
    if blockers:
        raise SystemExit(1)
    # Local-only resolution ensures SharedPolicy's default model revision is the pinned snapshot.
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache("Qwen/Qwen3-8B", "config.json")
    if not isinstance(cached, str) or f"/snapshots/{a.base_model_revision}/" not in cached:
        raise RuntimeError("Local Qwen3-8B main cache does not resolve to the comparison revision")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from baselines.common.backend import AuditedBackend
    backend = AuditedBackend(audit_path=Path(a.output) / "reward_api_attempts.jsonl", max_retries=2, base_url="https://openrouter.ai/api/v1", model=a.judge_model,
        api_key_env="OPENROUTER_API_KEY", revision=a.judge_revision)
    judge = InteractivityJudge(backend, Path(a.output) / "collabllm/judge_attempts.jsonl")
    counter = Cl100kCounter()
    weights = json.loads(Path(a.weights).read_text()) if a.weights else None
    cfg, _ = configuration(output=a.output, receipt=receipt, judge=judge, counter=counter, weights=weights,
        pilot_cases=a.pilot_cases, initial_policy_sha256=a.initial_policy_sha256, base_model_revision=a.base_model_revision)
    hook = RewardHook(cfg, receipt, judge, counter, weights)
    runner = isolated_runner(hook)
    bind_run(cfg)
    errors = runner.preflight(cfg)
    if errors:
        raise RuntimeError("; ".join(errors))
    if not a.launch:
        return
    bind_run(cfg, create=True)
    if weights:
        reuse_first_pilot_batch(runner, hook, a.pilot_output)
    from scripts.online_rl.run_tree import load_key
    os.environ["OPENROUTER_API_KEY"] = load_key()
    from scripts.online_rl.start_five_gpu_doctors import start
    start(cfg)
    limit = a.max_batches
    if a.pilot_cases is not None:
        state_path = Path(a.output) / "run_state.json"
        completed = json.loads(state_path.read_text())["case_cursor"] if state_path.exists() else 0
        if completed > a.pilot_cases:
            raise ValueError("Requested pilot stage is behind the completed pilot")
        limit = (a.pilot_cases - completed) // 5
    runner.run(cfg, max_batches=limit)
    if a.pilot_cases is not None:
        print(json.dumps(complete_pilot(cfg, receipt, a.pilot_cases)["stability"], indent=2))
    else:
        evaluate_completed_run(runner, cfg, method_identity=hook.fingerprint)


if __name__ == "__main__":
    main()
