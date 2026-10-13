"""Effectiveness v2: one turn-conditioned predictor for A1 and A2 interventions."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
from pathlib import Path
from statistics import mean

from src.common.io import read_jsonl
from src.data.schema import digest
from src.effectiveness import calibration
from src.effectiveness.evaluate import evaluate_predictions, ordering_breakdown
from src.effectiveness.metrics import binomial_nll, grouped_pairwise_ranking
from src.effectiveness.model import CollaborationValueModel, safe_torch_load
from src.effectiveness.pair_data import assert_case_disjoint, load_pairs, to_examples
from src.effectiveness.train import build_backbone, fit, load_predictor_config

ROOT = Path(__file__).resolve().parents[1]
TURNS = ("A1", "A2")


def load_examples(config: dict, split: str) -> list:
    """A1 (existing labels, replayed) and A2 examples of a split in the predictor's input format."""
    spec, data = config["continuation_spec"], config["data"]
    pairs = [row for turn in ("a1", "a2")
             for row in load_pairs(ROOT / data[f"{turn}_{split}"], split=split, protocol_hash=spec.protocol_hash(), expected_k=spec.k)]
    return to_examples(pairs, config["predictor"]["input_format"])


def check_splits(config: dict) -> None:
    """No case in both TRAIN and DEV, and every label's case is in the split manifest under its split (never TEST)."""
    from src.data.split import read_manifest
    from src.effectiveness.pair_data import assert_manifest_splits

    data = config["data"]
    rows = {split: [r for turn in ("a1", "a2") for r in read_jsonl(ROOT / data[f"{turn}_{split}"])] for split in ("train", "dev")}
    assert_case_disjoint(rows)
    assert_manifest_splits(rows, read_manifest(ROOT / data["split_manifest"]), data["source_dataset"])


def build_model(config: dict):
    """The v1 family, warm-started from training.init_from (v1's adapters and head), as a turn-conditioned predictor."""
    import torch

    predictor, training = config["predictor"], config["training"]
    torch.manual_seed(training["seed"])
    random.seed(training["seed"])
    model_id, revision, loader, options = build_backbone(predictor)
    backbone, tokenizer = loader(model_id, revision, **options)
    model = CollaborationValueModel(backbone, tokenizer, base_model_id=model_id, base_model_revision=revision,
                                    max_length=predictor["max_length"], training_mode=predictor["training_mode"],
                                    lora=predictor.get("lora"), backbone_options=options)
    model.input_format, model.supported_turns = predictor["input_format"], list(predictor["supported_turns"])
    if training.get("init_from"):
        model.load_trainable_state(safe_torch_load(ROOT / training["init_from"] / "trainable_state.pt", map_location="cpu"))
    if training.get("gradient_checkpointing"):
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
    return model


def release(model) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def folds_by_case(examples, folds: int, seed: int) -> dict[str, int]:
    cases = sorted({e.original_case_id for e in examples})
    random.Random(seed).shuffle(cases)
    return {case: i % folds for i, case in enumerate(cases)}


def run_fit(config: dict, train, held_out, output: Path, epochs: int, keep: bool, logits_path: Path | None = None) -> list[dict]:
    """One training run. With logits_path: the raw logits of held_out after every epoch are written there."""
    training = copy.deepcopy(config["training"])
    training.update(epochs=epochs, keep_epoch_checkpoints=keep)
    model = build_model(config)
    fingerprint = digest({"predictor": config["predictor"], "training": {k: v for k, v in training.items() if k != "epochs"},
                          "train": [(e.kind, e.history_id, e.response_id) for e in train],
                          "held_out": [(e.kind, e.history_id, e.response_id) for e in held_out]})
    kept = json.loads(logits_path.read_text(encoding="utf-8")) if logits_path and logits_path.is_file() else {}

    def on_epoch(epoch: int, current) -> None:
        if logits_path:
            kept[str(epoch)] = current.example_logits(held_out, training["batch_size"])
            logits_path.write_text(json.dumps(kept), encoding="utf-8")

    try:
        result = fit(model, train, held_out, training, output, fingerprint=fingerprint, resume=True, on_epoch=on_epoch,
                     checkpoint_metadata={"continuation_protocol_hash": config["continuation_spec"].protocol_hash(),
                                          "k": config["continuation_spec"].k, "input_format": model.input_format,
                                          "supported_turns": model.supported_turns,
                                          "warm_start": config["training"].get("init_from")})
    finally:
        release(model)
    return result["history"]


def cell(value, digits: int = 3, signed: bool = False) -> str:
    """A metric in a table; a metric that is undefined (e.g. no informative pair in the subset) is n/a."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def none_if_undefined(metrics: dict) -> dict:
    """NaN (a metric that has nothing to be computed on) as None, so that reports are valid JSON and say so."""
    return {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in metrics.items()}


def defined(values: list) -> list[float]:
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def fit_calibration(logits: list[float], examples, by_turn: bool) -> dict:
    def temperature(index: list[int]) -> dict:
        return calibration.fit_logits([logits[i] for i in index], [examples[i].num_success for i in index],
                                      [examples[i].num_rollouts for i in index], "temperature")

    if not by_turn:
        return temperature(list(range(len(examples))))
    return {"method": calibration.BY_TURN,
            "by_turn": {turn: temperature([i for i, e in enumerate(examples) if e.intervention_turn == turn]) for turn in TURNS}}


def calibrated(parameters: dict, logits: list[float], examples) -> list[float]:
    return [calibration.apply_logits(parameters, [z], e.intervention_turn)[0] for z, e in zip(logits, examples)]


def nll(probabilities: list[float], examples) -> float:
    return binomial_nll(probabilities, [e.num_success for e in examples], [e.num_rollouts for e in examples])


def subset(values: list, examples, keep) -> tuple[list, list]:
    index = [i for i, e in enumerate(examples) if keep(e)]
    return [values[i] for i in index], [examples[i] for i in index]


def turn_metrics(probabilities: list[float], examples) -> dict:
    """NLL, Brier, ECE, correlations, ordering and delta metrics of a set of examples (one turn or pooled)."""
    result = evaluate_predictions(probabilities, examples)
    after_p, after = subset(probabilities, examples, lambda e: e.kind == "after")
    before = {e.history_id: p for p, e in zip(probabilities, examples) if e.kind == "before"}
    history_only = [before[e.history_id] for e in after]  # the response ignored: P(success | h) for every candidate
    ordering = ordering_breakdown(probabilities, examples)
    return none_if_undefined({
        "examples": len(examples), "histories": len(before), "after_pairs": len(after),
        "nll": result["absolute"]["all"]["binomial_nll"], "brier": result["absolute"]["all"]["brier"],
        "ece": result["absolute"]["all"]["ece"], "after_nll": result["absolute"]["after"]["binomial_nll"],
        "before_nll": result["absolute"]["before"]["binomial_nll"],
        "success_rate_pearson": result["absolute"]["after"]["pearson"], "success_rate_spearman": result["absolute"]["after"]["spearman"],
        "ordering": result["ranking"]["within_history_pairwise_accuracy"], "ordering_pairs": result["ranking"]["informative_pairs"],
        "ordering_informative_histories": ordering["informative_histories"]["histories"],
        "delta_mae": result["delta"]["mae"], "delta_pearson": result["delta"]["pearson"], "delta_spearman": result["delta"]["spearman"],
        "delta_sign_accuracy": result["delta"]["sign_agreement_nonzero"], "nonzero_delta_pairs": result["delta"]["nonzero_count"],
        "history_only_after_nll": nll(history_only, after),
        "after_nll_gain_from_reading_the_response": nll(history_only, after) - nll(after_p, after),
        "mean_abs_predicted_delta": mean(abs(p - before[e.history_id]) for p, e in zip(after_p, after)),
        "mean_abs_label_delta": mean(abs(e.empirical_value - b.empirical_value) for e in after
                                     for b in examples if b.kind == "before" and b.history_id == e.history_id),
    })


def by_turn_metrics(probabilities: list[float], examples) -> dict:
    out = {turn: turn_metrics(*subset(probabilities, examples, lambda e, t=turn: e.intervention_turn == t)) for turn in TURNS}
    out["pooled"] = turn_metrics(probabilities, examples)
    return out


def cross_validate(config: dict, out: Path) -> dict:
    selection, epochs = config["training"]["selection"], config["training"]["epochs"]
    examples = load_examples(config, "train")
    fold_of = folds_by_case(examples, selection["folds"], selection["seed"])
    (out / "cv").mkdir(parents=True, exist_ok=True)
    out_of_fold = {epoch: [None] * len(examples) for epoch in range(1, epochs + 1)}
    for fold in range(selection["folds"]):
        held = [i for i, e in enumerate(examples) if fold_of[e.original_case_id] == fold]
        logits_path = out / "cv" / f"fold{fold}_logits.json"
        done = json.loads(logits_path.read_text(encoding="utf-8")) if logits_path.is_file() else {}
        if len(done) < epochs:
            run_dir = out / "cv" / f"fold{fold}"
            run_fit(config, [e for i, e in enumerate(examples) if fold_of[e.original_case_id] != fold],
                    [examples[i] for i in held], run_dir, epochs, keep=False, logits_path=logits_path)
            shutil.rmtree(run_dir)  # the weights of a cross-validation run are not needed, only its held-out logits
            done = json.loads(logits_path.read_text(encoding="utf-8"))
        for epoch in range(1, epochs + 1):
            for i, z in zip(held, done[str(epoch)]):
                out_of_fold[epoch][i] = z
        print(f"fold {fold}: held-out logits of {len(held)} examples for {epochs} epochs", flush=True)

    candidates = []
    for epoch, logits in out_of_fold.items():
        pooled = fit_calibration(logits, examples, by_turn=False)
        metrics = by_turn_metrics(calibrated(pooled, logits, examples), examples)
        candidates.append({"epoch": epoch, "pooled_temperature": pooled["temperature"],
                           "mean_turn_after_nll": mean(metrics[t]["after_nll"] for t in TURNS),
                           # a turn without any informative pair in the held-out folds has no ordering and is left out
                           "mean_turn_ordering": mean(defined([metrics[t]["ordering"] for t in TURNS]) or [0.5]),
                           "metrics": metrics})
    floor = min(c["mean_turn_after_nll"] for c in candidates)
    admissible = [c for c in candidates if c["mean_turn_after_nll"] <= floor + selection["nll_tolerance"]]
    chosen = max(admissible, key=lambda c: (c["mean_turn_ordering"], -c["epoch"]))

    # calibration: one temperature or one per turn, compared on cross-fitted out-of-fold probabilities
    logits = out_of_fold[chosen["epoch"]]
    cross = {}
    for name, by_turn in (("single", False), ("by_turn", True)):
        fitted = [None] * len(examples)
        for fold in range(selection["folds"]):
            train = [i for i, e in enumerate(examples) if fold_of[e.original_case_id] != fold]
            test = [i for i, e in enumerate(examples) if fold_of[e.original_case_id] == fold]
            parameters = fit_calibration([logits[i] for i in train], [examples[i] for i in train], by_turn)
            for i, p in zip(test, calibrated(parameters, [logits[i] for i in test], [examples[i] for i in test])):
                fitted[i] = p
        cross[name] = {"nll": nll(fitted, examples),
                       **{turn: nll(*subset(fitted, examples, lambda e, t=turn: e.intervention_turn == t)) for turn in TURNS}}
    use_by_turn = cross["single"]["nll"] - cross["by_turn"]["nll"] > selection["calibration_gain_for_turn_specific"]
    parameters = fit_calibration(logits, examples, by_turn=use_by_turn)
    parameters.update(fitted_on="out-of-fold TRAIN logits of the chosen epoch (cross-validation models); DEV not used",
                      chosen_epoch=chosen["epoch"], cross_fitted_nll=cross,
                      choice_reason=("one temperature per turn lowers the cross-fitted out-of-fold NLL by more than "
                                     if use_by_turn else "one temperature per turn does not lower the cross-fitted out-of-fold NLL by more than ")
                      + f"{selection['calibration_gain_for_turn_specific']} ({cross['single']['nll']:.4f} single, {cross['by_turn']['nll']:.4f} by turn)")
    if not use_by_turn:  # the plain temperature format of calibration.load
        parameters = {**parameters, "method": "temperature"}
    (out / "calibration.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    raw = by_turn_metrics([sigmoid(z) for z in logits], examples)
    summary = {"folds": selection["folds"], "fold_seed": selection["seed"], "train_cases": len(fold_of),
               "train_examples": {turn: sum(e.intervention_turn == turn for e in examples) for turn in TURNS},
               "rule": selection["rule"], "candidates": candidates, "chosen_epoch": chosen["epoch"],
               "uncalibrated_out_of_fold_at_chosen_epoch": raw, "calibration": parameters, "dev_used": False}
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Joint Effectiveness predictor: cross-validation over TRAIN cases (DEV not read)", "",
             f"{selection['folds']} folds over {len(fold_of)} TRAIN cases; examples A1 {summary['train_examples']['A1']}, A2 {summary['train_examples']['A2']}.", "",
             "Out-of-fold metrics after one pooled temperature per epoch:", "",
             "| epoch | T | AFTER NLL A1 | AFTER NLL A2 | ordering A1 (pairs) | ordering A2 (pairs) | delta sign A1 | delta sign A2 | chosen |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in candidates:
        m = c["metrics"]
        lines.append(f"| {c['epoch']} | {c['pooled_temperature']:.2f} | {cell(m['A1']['after_nll'])} | {cell(m['A2']['after_nll'])} | "
                     f"{cell(m['A1']['ordering'])} ({m['A1']['ordering_pairs']}) | {cell(m['A2']['ordering'])} ({m['A2']['ordering_pairs']}) | "
                     f"{cell(m['A1']['delta_sign_accuracy'])} ({m['A1']['nonzero_delta_pairs']}) | "
                     f"{cell(m['A2']['delta_sign_accuracy'])} ({m['A2']['nonzero_delta_pairs']}) | {'<-' if c is chosen else ''} |")
    lines += ["", f"Calibration: {parameters['choice_reason']}.", ""]
    (out / "cv_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return summary


def final(config: dict, out: Path) -> dict:
    summary = json.loads((out / "cv_summary.json").read_text(encoding="utf-8"))
    selection, epochs = config["training"]["selection"], summary["chosen_epoch"]
    examples = load_examples(config, "train")
    fold_of = folds_by_case(examples, selection["folds"], selection["seed"])
    monitor = [e for e in examples if fold_of[e.original_case_id] == 0]  # seen in training; monitoring only
    run_fit(config, examples, monitor, out / "model", epochs, keep=True)
    target = out / "joint_v2"
    if target.exists():
        raise SystemExit(f"{target} exists; the final checkpoint is written once")
    shutil.copytree(out / "model" / "epochs" / f"epoch{epochs}", target)
    shutil.copyfile(out / "calibration.json", target / "calibration.json")
    result = {"epochs": epochs, "train_examples": len(examples), "checkpoint": str(target.relative_to(ROOT)),
              "calibration": str((target / "calibration.json").relative_to(ROOT)),
              "checkpoint_id": json.loads((target / "predictor_config.json").read_text())["trainable_state_sha256"][:16],
              "monitoring_set": "TRAIN fold 0 (in-sample)", "dev_used": False}
    (out / "final_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def bootstrap_ordering(probabilities: list[float], examples, n: int, seed: int, other: list[float] | None = None) -> dict:
    """Bootstrap over histories of the pooled within-history ordering accuracy (and, with other, of the difference
    probabilities - other on the same histories)."""
    by_history: dict[str, list[int]] = {}
    for i, e in enumerate(examples):
        if e.kind == "after":
            by_history.setdefault(e.history_id, []).append(i)

    def score(values: list[float], histories: list[str]) -> float | None:
        correct = pairs = 0.0
        for index, history in enumerate(histories):
            rows = by_history[history]
            result = grouped_pairwise_ranking([values[i] for i in rows], [examples[i].empirical_value for i in rows], [index] * len(rows))
            if result["pairs"]:
                correct += result["accuracy"] * result["pairs"]
                pairs += result["pairs"]
        return correct / pairs if pairs else None

    histories, rng = sorted(by_history), random.Random(seed)
    samples, differences, undefined = [], [], 0
    for _ in range(n):
        drawn = [histories[rng.randrange(len(histories))] for _ in histories]
        value = score(probabilities, drawn)
        if value is None:  # the resample holds no history with an informative pair
            undefined += 1
            continue
        samples.append(value)
        if other is not None:
            reference = score(other, drawn)
            differences.append(value - reference)

    def interval(values: list[float]):
        ordered = sorted(values)
        return [ordered[int(0.025 * len(ordered))], ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))]] if ordered else None

    estimate = score(probabilities, histories)
    informative = sum(len({examples[i].empirical_value for i in rows}) > 1 for rows in by_history.values())
    return {"estimate": estimate, "ci95": interval(samples), "histories": len(histories),
            "histories_with_an_informative_pair": informative, "resamples_without_an_informative_pair": undefined,
            **({"difference": estimate - score(other, histories) if estimate is not None else None,
                "difference_ci95": interval(differences)} if other is not None else {})}


def evaluate(config: dict, out: Path) -> dict:
    import torch

    if (out / "dev_report.json").exists():
        raise SystemExit("DEV has already been evaluated for this run (dev_report.json exists); it is read once")
    gate, baseline = config["gate"], config["baseline"]
    examples = load_examples(config, "dev")
    checkpoint = out / "joint_v2"
    loader = build_backbone(config["predictor"])[2]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CollaborationValueModel.from_pretrained(checkpoint, backbone_loader=lambda model_id, revision, **options: loader(
        model_id, revision, **{**options, "device": device}))
    logits = model.example_logits(examples, config["training"]["batch_size"])
    release(model)
    # the calibration is the one fitted on the out-of-fold TRAIN logits of the cross-validation, frozen before the
    # final model existed; it is never refitted on the final model's predictions or on DEV
    if (checkpoint / "calibration.json").read_bytes() != (out / "calibration.json").read_bytes():
        raise SystemExit("joint_v2/calibration.json is not the calibration written by the cross-validation")
    parameters = calibration.load(checkpoint / "calibration.json")
    if "out-of-fold TRAIN" not in json.loads((checkpoint / "calibration.json").read_text(encoding="utf-8")).get("fitted_on", ""):
        raise SystemExit("the calibration does not record that it was fitted on out-of-fold TRAIN logits")
    probabilities = calibrated(parameters, logits, examples)
    (out / "dev_logits.json").write_text(json.dumps({
        "checkpoint": str(checkpoint.relative_to(ROOT)), "input_format": config["predictor"]["input_format"],
        "examples": [{"kind": e.kind, "history_id": e.history_id, "response_id": e.response_id,
                      "intervention_turn": e.intervention_turn, "num_success": e.num_success, "num_rollouts": e.num_rollouts}
                     for e in examples], "logits": logits, "calibrated": probabilities}), encoding="utf-8")
    joint = by_turn_metrics(probabilities, examples)
    joint_raw = by_turn_metrics([sigmoid(z) for z in logits], examples)

    # Effectiveness v1 on the same A1 DEV examples, from its stored offline logits and its temperature
    stored = json.loads((ROOT / baseline["v1_dev_logits"]).read_text(encoding="utf-8"))
    v1_logit = {(e["kind"], e["history_id"], e["response_id"]): z
                for e, z in zip(stored["examples"], stored["checkpoints"]["epoch3"]["logits"])}
    a1_p, a1 = subset(probabilities, examples, lambda e: e.intervention_turn == "A1")
    v1_parameters = calibration.load(ROOT / baseline["v1_calibration"])
    v1_p = calibration.apply_logits(v1_parameters, [v1_logit[(e.kind, e.history_id, e.response_id)] for e in a1])
    v1 = turn_metrics(v1_p, a1)

    a2_p, a2 = subset(probabilities, examples, lambda e: e.intervention_turn == "A2")
    # diagnostic only (the gate uses all A2 candidates): A2 without request_evidence, which at the Agent's last turn is a
    # message of questions that nothing answers. Shows whether a missing A2 signal is an artifact of that candidate.
    core_p, core = subset(a2_p, a2, lambda e: e.response_id != "request_evidence")
    boot = {"A1_joint": bootstrap_ordering(a1_p, a1, gate["bootstrap"], 0, other=v1_p),
            "A1_v1": bootstrap_ordering(v1_p, a1, gate["bootstrap"], 0),
            "A2_joint": bootstrap_ordering(a2_p, a2, gate["bootstrap"], 0),
            "A2_joint_without_request_evidence": bootstrap_ordering(core_p, core, gate["bootstrap"], 0)}
    sign, nonzero = joint["A2"]["delta_sign_accuracy"], joint["A2"]["nonzero_delta_pairs"]
    checks = {  # a metric that cannot be computed (no informative pair) does not pass
        "a2_ordering_ci_lower_bound_above_0.5": bool(boot["A2_joint"]["ci95"]) and boot["A2_joint"]["ci95"][0] > 0.5,
        "a2_after_nll_below_history_only": joint["A2"]["after_nll"] < joint["A2"]["history_only_after_nll"],
        "a2_delta_sign_accuracy_above_0.5_on_enough_nonzero_pairs": nonzero >= gate["a2_min_nonzero_delta_pairs"]
        and sign is not None and sign > 0.5,
        "a1_ordering_within_margin_of_v1": joint["A1"]["ordering"] >= v1["ordering"] - gate["a1_ordering_margin"],
        "a1_after_nll_within_margin_of_v1": joint["A1"]["after_nll"] <= v1["after_nll"] + gate["a1_nll_margin"],
    }
    decision = {"a2_response_sensitive": checks["a2_ordering_ci_lower_bound_above_0.5"] and checks["a2_after_nll_below_history_only"]
                and checks["a2_delta_sign_accuracy_above_0.5_on_enough_nonzero_pairs"],
                "a1_preserved": checks["a1_ordering_within_margin_of_v1"] and checks["a1_after_nll_within_margin_of_v1"]}
    decision["usable_for_reward"] = all(decision.values())
    report = {"checkpoint": str(checkpoint.relative_to(ROOT)), "calibration": parameters, "dev_read": "once",
              "joint": joint, "joint_uncalibrated": joint_raw, "v1_on_A1": v1, "ordering_bootstrap": boot,
              "diagnostic_A2_without_request_evidence": turn_metrics(core_p, core),
              "gate": {"criteria": {k: gate[k] for k in ("a2_response_sensitive", "a2_delta_sign", "a1_preserved", "a1_ordering_margin",
                                                         "a1_nll_margin", "a2_min_nonzero_delta_pairs")},
                       "checks": checks, **decision}}
    (out / "dev_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "gate.json").write_text(json.dumps(report["gate"], indent=2), encoding="utf-8")

    def row(name: str, m: dict) -> str:
        return (f"| {name} | {m['histories']} / {m['after_pairs']} | {cell(m['nll'])} | {cell(m['after_nll'])} | {cell(m['brier'])} | {cell(m['ece'])} | "
                f"{cell(m['success_rate_pearson'], 2)} | {cell(m['ordering'])} ({m['ordering_pairs']}) | {cell(m['delta_mae'])} | "
                f"{cell(m['delta_pearson'], 2)} / {cell(m['delta_spearman'], 2)} | {cell(m['delta_sign_accuracy'])} ({m['nonzero_delta_pairs']}) | "
                f"{cell(m['after_nll_gain_from_reading_the_response'], 3, True)} |")

    def interval(entry: dict, key: str = "estimate", ci: str = "ci95") -> str:
        return f"{cell(entry.get(key))} {entry.get(ci)}"

    lines = ["# Joint Effectiveness predictor (v2) on DEV", "",
             "| model / subset | histories / pairs | NLL | AFTER NLL | Brier | ECE | success r | ordering (pairs) | delta MAE | delta r / rho | delta sign (pairs) | AFTER NLL gain over history-only |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|",
             row("v1 on A1", v1), row("joint on A1", joint["A1"]), row("joint on A2", joint["A2"]), row("joint pooled", joint["pooled"]),
             row("diagnostic: joint on A2 without request_evidence", report["diagnostic_A2_without_request_evidence"]), "",
             f"Ordering, bootstrap over histories (95%): v1 A1 {interval(boot['A1_v1'])}; joint A1 {interval(boot['A1_joint'])}; "
             f"joint - v1 on A1 {interval(boot['A1_joint'], 'difference', 'difference_ci95')}; joint A2 {interval(boot['A2_joint'])} "
             f"({boot['A2_joint']['histories_with_an_informative_pair']} of {boot['A2_joint']['histories']} A2 histories have an informative pair).", "",
             f"Gate: {json.dumps(report['gate']['checks'])}; usable for the reward: {decision['usable_for_reward']}.", ""]
    (out / "dev_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("cv", "final", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_predictor_config(args.config)
    check_splits(config)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    {"cv": cross_validate, "final": final, "evaluate": evaluate}[args.command](config, out)


if __name__ == "__main__":
    main()
