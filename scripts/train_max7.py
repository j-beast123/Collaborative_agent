"""TRAIN-only max7 predictor: audit -> case-CV -> final TRAIN fit -> frozen DEV -> parity."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts import train_joint as joint
from src.common.config import local_model_path, read_config
from src.common.io import read_jsonl
from src.data.schema import digest
from src.effectiveness import calibration
from src.effectiveness.pair_data import load_pairs, to_examples, assert_manifest_splits
from src.data.split import read_manifest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/effectiveness/v8_agentclinic_max7"
TURNS = tuple(f"A{i}" for i in range(1, 8))


def workflow_settings(out):
    path = out / "workflow.json"
    settings = json.loads(path.read_text()) if path.exists() else {
        "environment_config": "configs/effectiveness/v8_labels_oof.json",
        "main_config": "configs/serl_agentclinic_max7.json"}
    return settings, read_config(ROOT / settings["environment_config"])


def frozen_identity(out):
    _, environment = workflow_settings(out)
    base = ROOT / environment["output_dir"]
    frozen = json.loads((base / "protocol.frozen.json").read_text())
    from scripts.run_base_collaborations import identity
    if frozen["protocol_identity"] != identity(environment):
        raise ValueError("environment changed after protocol freeze")
    return frozen["protocol_identity"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    """Atomic, immutable result: matching completed stages can be resumed."""
    path = Path(path)
    content = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"refusing to overwrite frozen artifact {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def config_for(out):
    c = read_config(ROOT / "configs/effectiveness/v5_predictor_lora.json")
    c["predictor"].update(input_format="turn_v1", supported_turns=list(TURNS))
    c["training"].update(init_from=None, turn_balanced=False, micro_batch_size=1,
                          cache_features=False, keep_epoch_checkpoints=True)
    c["training"]["selection"] = {"folds": 3, "seed": 0}
    c["data"] = {"train": str(out / "labels/labels_train.jsonl"), "dev": str(out / "labels/labels_dev.jsonl")}
    c["protocol_hash"] = frozen_identity(out)
    c.pop("continuation", None)
    c["frozen_environment"] = workflow_settings(out)[1]
    c["notes"] = "Fresh max7 joint predictor; no old labels/weights or DEV-based selection; plain pooled loss and TRAIN OOF global/identity calibration."
    return c


def verify(out):
    frozen = json.loads((out / "labels.freeze.json").read_text())
    for name, info in frozen["files"].items():
        if sha(out / "labels" / name) != info["sha256"]:
            raise ValueError(f"frozen labels changed: {name}")
    return frozen


def examples(c, split):
    pairs = load_pairs(c["data"][split], split=split, protocol_hash=c["protocol_hash"], expected_k=9)
    assert_manifest_splits({split: pairs}, read_manifest(ROOT / "data/splits/agentclinic_main_v2.split_manifest.json"),
                           "agentclinic_medqa_extended")
    return to_examples(pairs, "turn_v1")


def metrics(probabilities, rows):
    result = {"pooled": joint.turn_metrics(probabilities, rows)}
    for turn in TURNS:
        p, e = joint.subset(probabilities, rows, lambda e, t=turn: e.intervention_turn == t)
        result[turn] = joint.turn_metrics(p, e) if e else {"histories": 0, "status": "insufficient support"}
        if e and (result[turn]["ordering_pairs"] < 10 or result[turn]["nonzero_delta_pairs"] < 10):
            result[turn]["status"] = "insufficient support for reliable discrimination estimates"
    return result


def audit(out):
    from scripts.run_agentclinic_turns import select_train_dev, scope_histories, leakage_audit
    from src.agentclinic.protocol import protocol_spec
    c = workflow_settings(out)[1]
    protocol_hash = frozen_identity(out)
    cases, _ = select_train_dev(c)
    cases = {r["case_id"]: r for r in cases}
    histories = scope_histories(c, cases, "full")
    labels = read_jsonl(out / "labels/labels.jsonl")
    candidates = read_jsonl(out / "labels/candidates.jsonl")
    compliance = read_jsonl(out / "labels/candidate_compliance.jsonl")
    names = set(c["candidates"]["names"])
    expected = {(h["history_id"], n) for h in histories for n in names}
    generated = {(r["history_id"], r["candidate_id"]) for r in candidates}
    judged = {(r["history_id"], r["candidate_id"]) for r in compliance}
    eligible = {(r["history_id"], r["candidate_id"]) for r in compliance if r["behavior_compliant"]}
    labelled = {(r["history_id"], r["response_id"]) for r in labels}
    before = read_jsonl(out / "labels/before_rollouts.jsonl")
    after = read_jsonl(out / "labels/after_rollouts.jsonl")
    checks = {"all_five_candidates_generated_and_judged": generated == judged == expected,
              "every_compliant_pair_labelled": labelled == eligible,
              "every_reached_history_labelled": {r["history_id"] for r in labels} == {h["history_id"] for h in histories},
              "no_duplicate_label": len(labelled) == len(labels),
              "all_before_rollouts_complete": len(before) == len(histories) * 9 and all(r["status"] == "completed" and r["success"] is not None for r in before)}
    before_keys = Counter((r["history_id"], r["rollout_index"]) for r in before)
    checks["all_before_indices_exactly_once"] = set(before_keys) == {(h["history_id"], i) for h in histories for i in range(9)} and all(n == 1 for n in before_keys.values())
    chosen_attempts = {(r["history_id"], r["candidate_id"]): r.get("attempt", 1) for r in candidates}
    selected_after = [r for r in after if (r["history_id"], r["candidate_id"]) in eligible
                      and r.get("candidate_attempt", 1) == chosen_attempts[(r["history_id"], r["candidate_id"])]]
    after_keys = Counter((r["history_id"], r["candidate_id"], r["rollout_index"]) for r in selected_after)
    checks["all_after_indices_complete_exactly_once"] = (set(after_keys) == {(h, n, i) for h, n in eligible for i in range(9)}
        and all(n == 1 for n in after_keys.values()) and all(r["status"] == "completed" and r["success"] is not None for r in selected_after))
    checks["leakage"] = leakage_audit(out / "labels", cases, histories, protocol_spec(c["protocol"]))["passed"]
    for split in ("train", "dev"):
        load_pairs(out / f"labels/labels_{split}.jsonl", split=split,
                   protocol_hash=protocol_hash, expected_k=9)
    train_examples = to_examples([r for r in labels if r["split"] == "train"], "turn_v1")
    signal = metrics([e.empirical_value for e in train_examples], train_examples)
    report = {"checks": checks, "passed": all(checks.values()), "excluded_noncompliant_pairs": len(expected - eligible),
              "failed_rollouts": sum(r["status"] != "completed" for r in before + after),
              "retried_rollouts": sum(r.get("attempt", 1) > 1 for r in before + after),
              "train_signal_counts": {t: {k: m[k] for k in ("histories", "after_pairs", "ordering_pairs", "ordering_informative_histories", "nonzero_delta_pairs")}
                                      for t, m in signal.items() if "after_pairs" in m}, "train_by_turn": {}}
    for turn in TURNS:
        hs = [h for h in histories if h["intervention_turn"] == turn and cases[h["case_id"]]["split"] == "train"]
        rs = [r for r in labels if r["intervention_turn"] == turn and r["split"] == "train"]
        ids = {h["history_id"] for h in hs}
        b = {r["history_id"]: r["before"]["successes"] for r in rs}
        counts = Counter("negative" if r["delta_effectiveness"] < 0 else "positive" if r["delta_effectiveness"] > 0 else "zero" for r in rs)
        responses = Counter((r["history_id"], r["response"]) for r in candidates if r["history_id"] in ids)
        lengths = [r["turns"] for r in after if r["history_id"] in ids and r["status"] == "completed"]
        report["train_by_turn"][turn] = {"histories": len(hs), "labelled_histories": len(b), "pairs": len(rs),
            **{n: counts[n] for n in ("negative", "zero", "positive")},
            "nonzero_fraction": (counts["negative"] + counts["positive"]) / len(rs) if rs else None,
            "duplicate_candidate_texts": sum(n - 1 for n in responses.values()),
            "invalid_or_noncompliant_candidates": len(hs) * len(names) - len(rs),
            "before_success_counts": dict(Counter(b.values())), "after_success_counts": dict(Counter(r["after"]["successes"] for r in rs)),
            "before_extreme_fraction": sum(v in (0, 9) for v in b.values()) / len(b) if b else None,
            "after_extreme_fraction": sum(r["after"]["successes"] in (0, 9) for r in rs) / len(rs) if rs else None,
            "mean_final_length": sum(lengths) / len(lengths) if lengths else None}
    report["train_pooled"] = {n: sum(r[n] for r in report["train_by_turn"].values()) for n in
                              ("histories", "labelled_histories", "pairs", "negative", "zero", "positive", "duplicate_candidate_texts", "invalid_or_noncompliant_candidates")}
    write(out / "label_audit.json", report)
    if not report["passed"]:
        raise ValueError(f"label completeness audit failed: {checks}")
    files = {p.name: {"sha256": sha(p), "rows": sum(1 for _ in p.open())} for p in sorted((out / "labels").glob("*.jsonl"))}
    write(out / "labels.freeze.json", {"files": files, "protocol_hash": protocol_hash, "audit_sha256": sha(out / "label_audit.json")})
    c = config_for(out)
    # Determine context capacity on TRAIN only; never truncate away a late-turn response.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model_path(c["predictor"]["model_id"]), local_files_only=True)
    rows = examples(c, "train")
    sizes = [len(tokenizer(e.text)["input_ids"]) for e in rows]
    c["predictor"]["max_length"] = max(2048, math.ceil(max(sizes) / 512) * 512)
    write(out / "training.frozen.json", {**c, "train_max_tokens": max(sizes),
        "selection_rule": "lowest pooled raw OOF binomial NLL; ties earlier epoch; global T only if OOF NLL improves and ordering/sign do not regress"})


def training_config(out):
    verify(out)
    c = json.loads((out / "training.frozen.json").read_text())
    c["continuation_spec"] = SimpleNamespace(k=9, protocol_hash=lambda: c["protocol_hash"])
    return c


def cv(out, only_fold=None):
    """only_fold: fit that fold alone (its own directory and logits file) and stop, so that the folds, which share
    nothing, can run as separate processes on separate GPUs.
    """
    c = training_config(out)
    rows = examples(c, "train")
    folds = joint.folds_by_case(rows, 3, 0)
    oof = {epoch: [None] * len(rows) for epoch in range(1, c["training"]["epochs"] + 1)}
    for fold in range(3):
        if only_fold is not None and fold != only_fold:
            continue
        held = [i for i, e in enumerate(rows) if folds[e.original_case_id] == fold]
        path = out / "cv" / f"fold{fold}_logits.json"
        done = json.loads(path.read_text()) if path.exists() else {}
        if len(done) < len(oof):
            joint.run_fit(c, [e for e in rows if folds[e.original_case_id] != fold], [rows[i] for i in held],
                          out / "cv" / f"fold{fold}", c["training"]["epochs"], keep=False, logits_path=path)
            done = json.loads(path.read_text())
        for epoch in oof:
            if len(done[str(epoch)]) != len(held):
                raise ValueError("incomplete fold logits")
            for i, z in zip(held, done[str(epoch)]):
                oof[epoch][i] = z
    if only_fold is not None:
        return
    if any(z is None or not math.isfinite(z) for zs in oof.values() for z in zs):
        raise ValueError("missing/nonfinite OOF logits")
    epoch_metrics = {epoch: metrics([joint.sigmoid(z) for z in zs], rows) for epoch, zs in oof.items()}
    chosen = min(oof, key=lambda ep: (epoch_metrics[ep]["pooled"]["nll"], ep))
    zs = oof[chosen]
    fitted = calibration.fit_logits(zs, [e.num_success for e in rows], [e.num_rollouts for e in rows])
    raw = epoch_metrics[chosen]
    scaled = metrics(joint.calibrated(fitted, zs, rows), rows)
    use = scaled["pooled"]["nll"] < raw["pooled"]["nll"] - 1e-8
    for key in ("ordering", "delta_sign_accuracy"):
        a, b = raw["pooled"][key], scaled["pooled"][key]
        if a is not None and (b is None or b < a - 1e-12):
            use = False
    parameters = fitted if use else calibration.fit_logits([], [], [], "identity")
    parameters.update(fitted_on="TRAIN case-level OOF only", chosen_epoch=chosen,
                      protocol_hash=c["protocol_hash"], selection_reason="OOF NLL improvement without discrimination regression" if use else "identity: global scaling not selected")
    write(out / "calibration.json", parameters)
    write(out / "oof_logits.json", [{"kind": e.kind, "case_id": e.case_id, "history_id": e.history_id,
        "response_id": e.response_id, "intervention_turn": e.intervention_turn, "fold": folds[e.original_case_id],
        "raw_logit": z, "successes": e.num_success, "trials": e.num_rollouts} for e, z in zip(rows, zs)])
    counts = Counter(e.intervention_turn for e in rows)
    write(out / "cv_summary.json", {"chosen_epoch": chosen, "epochs": epoch_metrics, "raw": raw, "global_temperature": scaled,
        "calibration": parameters, "dev_used": False, "train_examples": {t: {"n": counts[t], "fraction": counts[t] / len(rows)} for t in TURNS}})


def final(out):
    c = training_config(out)
    summary = json.loads((out / "cv_summary.json").read_text())
    target = out / "joint_max7"
    if target.exists():
        from src.effectiveness.model import state_checksum, safe_torch_load
        saved = json.loads((target / "predictor_config.json").read_text())
        if saved["trainable_state_sha256"] != state_checksum(safe_torch_load(target / "trainable_state.pt", map_location="cpu")):
            raise ValueError("checkpoint hash mismatch")
    else:
        rows = examples(c, "train")
        folds = joint.folds_by_case(rows, 3, 0)
        epoch = summary["chosen_epoch"]
        joint.run_fit(c, rows, [e for e in rows if folds[e.original_case_id] == 0], out / "model", epoch, keep=True)
        staging = target.with_name(target.name + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(out / "model/epochs" / f"epoch{epoch}", staging)
        shutil.copyfile(out / "calibration.json", staging / "calibration.json")
        staging.replace(target)
    write(out / "checkpoint.freeze.json", {"checkpoint_sha256": sha(target / "trainable_state.pt"),
        "predictor_config_sha256": sha(target / "predictor_config.json"), "calibration_sha256": sha(target / "calibration.json"),
        "training_config_sha256": sha(out / "training.frozen.json"), "supported_turns": list(TURNS), "protocol_hash": c["protocol_hash"]})


def evaluate(out):
    c = training_config(out)
    frozen = json.loads((out / "checkpoint.freeze.json").read_text())
    if sha(out / "joint_max7/trainable_state.pt") != frozen["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")
    if (out / "dev_report.json").exists():
        return  # no second DEV evaluation
    rows = examples(c, "dev")
    from src.effectiveness.model import CollaborationValueModel
    model = CollaborationValueModel.from_pretrained(out / "joint_max7")
    sizes = [len(model.tokenizer(e.text)["input_ids"]) for e in rows]
    if max(sizes) > model.max_length:
        raise ValueError("frozen context length cannot hold DEV inputs; stop without tuning on DEV")
    log_path = out / "dev_logits.json"
    zs = json.loads(log_path.read_text()) if log_path.exists() else model.example_logits(rows, 1)
    write(log_path, zs)
    parameters = calibration.load(out / "calibration.json")
    selected = joint.calibrated(parameters, zs, rows)
    write(out / "dev_report.json", {"raw": metrics([joint.sigmoid(z) for z in zs], rows), "selected": metrics(selected, rows),
        "bootstrap_ordering": joint.bootstrap_ordering(selected, rows, 2000, 0), "settings_changed": False,
        "historical_v1_reference_only_noncomparable": {"ordering": 0.8169, "delta_sign": 0.7838, "nll": 0.599, "temperature": 2.96573009067666}})
    joint.release(model)


def parity(out):
    from src.effectiveness.service import LocalEffectiveness, GrpcEffectivenessClient, start_server
    c = training_config(out)
    if not (out / "dev_report.json").exists():
        raise ValueError("frozen DEV evaluation must precede parity")
    local = LocalEffectiveness.from_checkpoint(str(out / "joint_max7"), model_version="effectiveness-max7-oof-v1",
        calibration=str(out / "calibration.json"), device="cuda:0", supported_turns=TURNS,
        expected_protocol_hash=c["protocol_hash"])
    pairs = load_pairs(out / "labels/labels_train.jsonl", split="train", expected_k=9, protocol_hash=c["protocol_hash"])
    # Offline example_logits versus actual RPC (service uses unbatched serialization).
    fixed = [next(r for r in pairs if r["intervention_turn"] == t) for t in TURNS if any(r["intervention_turn"] == t for r in pairs)]
    ex = to_examples(fixed, "turn_v1")
    zs = local.model.example_logits(ex, 1)
    offline = {(e.history_id, e.kind): z for e, z in zip(ex, zs)}
    server, port = start_server(local, port=0, log_path=out / "grpc_logs/requests.jsonl")
    client = None
    try:
        client = GrpcEffectivenessClient(f"127.0.0.1:{port}", expected={k: local.identity()[k] for k in
            ("model_version", "checkpoint_id", "calibration_id", "continuation_protocol_hash", "supported_turns")})
        diffs = {k: 0.0 for k in ("logit_before", "logit_after", "p_before", "p_after", "delta")}
        for r in fixed:
            result = client.predict(r["history"], r["response"], r["intervention_turn"])
            b, a = offline[(r["history_id"], "before")], offline[(r["history_id"], "after")]
            pb, pa = calibration.apply_logits(local.calibration, [b, a], r["intervention_turn"])
            for key, value in {"logit_before": b, "logit_after": a, "p_before": pb, "p_after": pa, "delta": pa - pb}.items():
                diffs[key] = max(diffs[key], abs(getattr(result, key) - value))
        report = {"passed": max(diffs.values()) <= 1e-5, "max_absolute_difference": diffs, "turns": [r["intervention_turn"] for r in fixed], "identity": local.identity()}
        write(out / "parity.json", report)
        if not report["passed"]:
            raise ValueError("offline/service parity failed")
    finally:
        if client:
            client.close()
        server.stop(30).wait()
    # Config changes occur only after real parity passes; no training or rubric stages follow.
    main_config = workflow_settings(out)[0]["main_config"]
    path = ROOT / main_config
    main = json.loads(path.read_text())
    identity = local.identity()
    section = main["reward"]["effectiveness"]
    section.update(provider="grpc", endpoint="127.0.0.1:50557", device="cuda:0",
        checkpoint=str((out / "joint_max7").relative_to(ROOT)), calibration=str((out / "calibration.json").relative_to(ROOT)),
        model_version=identity["model_version"], expected_temperature=identity["calibration_temperature"],
        expected_checkpoint_id=identity["checkpoint_id"], expected_calibration_id=identity["calibration_id"],
        expected_continuation_protocol_hash=c["protocol_hash"], stages=list(TURNS), available={t: True for t in TURNS},
        notes="Frozen max7 joint predictor; TRAIN-only selection/calibration; DEV evaluated once; actual gRPC parity passed.")
    main["notes"] = "Main max7 early-stopping protocol; each reached A_t is a turn-local GRPO sample. Effectiveness joint predictor frozen using TRAIN only and connected after gRPC parity. No GRPO has been started by this workflow."
    path.write_text(json.dumps(main, indent=2) + "\n")
    write(out / "integration.json", {"config": str(path.relative_to(ROOT)), "identity": identity,
        "service_start_command": f"CUDA_VISIBLE_DEVICES=7 python -m scripts.serve_effectiveness --config {main_config} --device cuda:0 --port 50557 --log-dir {out.relative_to(ROOT)}/grpc_logs"})


def report(out):
    audit = json.loads((out / "label_audit.json").read_text())
    cv_result = json.loads((out / "cv_summary.json").read_text())
    dev = json.loads((out / "dev_report.json").read_text())
    rpc = json.loads((out / "parity.json").read_text())
    timing = read_jsonl(out / "timing.jsonl")
    cost = json.loads((out / "api_cost.json").read_text())
    def section(title, obj):
        return [f"## [{title}]", "", "```json", json.dumps(obj, indent=2, allow_nan=False), "```", ""]
    lines = ["# Max7 Effectiveness report", "", f"Protocol {frozen_identity(out)}. No rubric/GRPO/SERL/TEST stage executed.", ""]
    settings, _ = workflow_settings(out)
    wiring = {}
    for n in (1, 2, 5):
        if settings.get("e2e_runs", {}).get(str(n)):
            run = ROOT / "outputs/serl" / settings["e2e_runs"][str(n)]
            checks = json.loads((run / "checks.json").read_text())
            wiring[str(n)] = {"run_id": run.name, "passed": checks["passed"]}
    lines += section("E2E", wiring)
    lines += ["## [Label signal]", "", "| turn | histories | pairs | negative | zero | positive | nonzero % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for turn, row in audit["train_by_turn"].items():
        pct = f"{100 * row['nonzero_fraction']:.1f}" if row["nonzero_fraction"] is not None else "n/a"
        lines.append(f"| {turn} | {row['histories']} | {row['pairs']} | {row['negative']} | {row['zero']} | {row['positive']} | {pct} |")
    lines += [""] + section("Turn distribution in predictor training", cv_result["train_examples"])
    lines += section("TRAIN-CV discrimination and calibration comparison", {"raw_T1": cv_result["raw"],
        "global_T": cv_result["global_temperature"], "chosen_calibration": cv_result["calibration"]})
    lines += section("DEV frozen evaluation", dev)
    lines += section("Historical v1 reference", dev["historical_v1_reference_only_noncomparable"])
    lines += section("gRPC", rpc)
    lines += section("Timing", {"stages": timing, "label_generation": "UTC start/end in run.log", "calibration": "included in CV stage"})
    lines += section("Cost", cost)
    lines += section("Open issues", {"low_support_turns": [t for t in TURNS if dev["selected"][t].get("status")],
        "historical_reference": "different split/protocol, not a formal baseline", "gpu_runtime": "fresh 8B runs must complete before performance is claimed"})
    path = out / "FINAL_REPORT.md"
    content = "\n".join(lines)
    if not path.exists():
        path.write_text(content)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("audit", "cv", "final", "evaluate", "parity", "report"))
    p.add_argument("--output-dir", default="outputs/effectiveness/v8_agentclinic_max7/effectiveness_oof_v1")
    p.add_argument("--fold", type=int, choices=(0, 1, 2), help="cv: fit this fold only (one process per fold); cv without it then selects")
    args = p.parse_args()
    if args.fold is not None and args.stage != "cv":
        p.error("--fold goes with the stage cv")
    out = ROOT / args.output_dir
    started = time.time()
    if args.fold is not None:
        cv(out, only_fold=args.fold)
    else:
        globals()[args.stage](out)
    with (out / "timing.jsonl").open("a") as handle:
        handle.write(json.dumps({"stage": args.stage, **({"fold": args.fold} if args.fold is not None else {}),
                                 "started_unix": started, "elapsed_seconds": time.time() - started, "passed": True}) + "\n")


if __name__ == "__main__":
    main()
