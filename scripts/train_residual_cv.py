"""Choose the ranking weight of the residual Effectiveness predictor on TRAIN only, then train the final model."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from pathlib import Path

from src.data.schema import digest
from src.effectiveness.model import CollaborationValueModel, safe_torch_load
from src.effectiveness.pair_data import load_pairs, to_examples
from src.effectiveness.train import build_backbone, fit, load_predictor_config


def build_model(config: dict):
    import torch

    predictor, training = config["predictor"], config["training"]
    torch.manual_seed(training["seed"])
    random.seed(training["seed"])
    model_id, revision, loader, options = build_backbone(predictor)
    backbone, tokenizer = loader(model_id, revision, **options)
    lora = predictor.get("lora") if predictor["training_mode"] == "lora_scalar_head" else None
    model = CollaborationValueModel(backbone, tokenizer, base_model_id=model_id, base_model_revision=revision,
                                    max_length=predictor["max_length"], training_mode=predictor["training_mode"],
                                    lora=lora, backbone_options=options, architecture=predictor.get("architecture", "single"))
    head = safe_torch_load(Path(training["init_head_from"]) / "trainable_state.pt", map_location="cpu")
    model.value_head.load_state_dict({name.split("value_head.", 1)[1]: value for name, value in head.items()
                                      if name.startswith("value_head.")})
    if lora and training.get("gradient_checkpointing"):
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


def run_fit(config: dict, train, held_out, output: Path, lam: float, epochs: int, keep: bool) -> list[dict]:
    """One training run; returns the held-out metrics after every epoch."""
    training = copy.deepcopy(config["training"])
    training["ranking"] = dict(training["ranking"], **{"lambda": lam})
    training.update(epochs=epochs, keep_epoch_checkpoints=keep, frozen_before_baseline_logits=None)
    for name in ("lambda_candidates",):
        training["ranking"].pop(name, None)
    model = build_model(config)
    fingerprint = digest({"predictor": config["predictor"], "training": {k: v for k, v in training.items() if k != "epochs"},
                          "train": [(e.kind, e.history_id, e.response_id) for e in train],
                          "held_out": [(e.kind, e.history_id, e.response_id) for e in held_out]})
    try:
        result = fit(model, train, held_out, training, output, fingerprint=fingerprint, resume=True,
                     checkpoint_metadata={"continuation_protocol_hash": config["continuation_spec"].protocol_hash(),
                                          "k": config["continuation_spec"].k, "lambda_rank": lam})
    finally:
        release(model)
    return [{"epoch": h["epoch"], "after_nll": h["dev"]["absolute"]["after"]["binomial_nll"],
             "after_count": h["dev"]["absolute"]["after"]["count"], "ece": h["dev"]["absolute"]["all"]["ece"],
             "ordering": h["dev"]["ranking"]["within_history_pairwise_accuracy"],
             "ordering_pairs": h["dev"]["ranking"]["informative_pairs"],
             "delta_sign": h["dev"]["delta"]["sign_agreement_nonzero"], "nonzero_pairs": h["dev"]["delta"]["nonzero_count"],
             "delta_pearson": h["dev"]["delta"]["pearson"], "delta_mae": h["dev"]["delta"]["mae"]} for h in result["history"]]


def pooled(rows: list[dict]) -> dict:
    """Held-out metrics of one (lambda, epoch) over the folds: pair-weighted ordering and sign, example-weighted NLL."""
    def weighted(value: str, weight: str):
        usable = [r for r in rows if r[value] is not None and r[weight]]
        total = sum(r[weight] for r in usable)
        return sum(r[value] * r[weight] for r in usable) / total if total else None

    def mean(value: str):
        values = [r[value] for r in rows if isinstance(r[value], (int, float)) and r[value] == r[value]]
        return sum(values) / len(values) if values else None

    return {"after_nll": weighted("after_nll", "after_count"), "ordering": weighted("ordering", "ordering_pairs"),
            "ordering_pairs": sum(r["ordering_pairs"] for r in rows), "delta_sign": weighted("delta_sign", "nonzero_pairs"),
            "nonzero_pairs": sum(r["nonzero_pairs"] for r in rows), "ece": mean("ece"),
            "delta_pearson": mean("delta_pearson"), "delta_mae": mean("delta_mae"),
            "ordering_by_fold": [r["ordering"] for r in rows]}


def choose(candidates: list[dict], nll_tolerance: float) -> dict:
    floor = min(c["after_nll"] for c in candidates)
    admissible = [c for c in candidates if c["after_nll"] <= floor + nll_tolerance]
    return max(admissible, key=lambda c: (c["ordering"] or 0.0, c["delta_sign"] or 0.0, -c["after_nll"]))


def cross_validate(config: dict, examples, out: Path) -> dict:
    selection = config["training"]["selection"]
    folds, epochs = selection["folds"], config["training"]["epochs"]
    fold_of = folds_by_case(examples, folds, selection["seed"])
    candidates = []
    for lam in config["training"]["ranking"]["lambda_candidates"]:
        per_fold = []
        for fold in range(folds):
            result_path = out / "cv" / f"lambda{lam}" / f"fold{fold}.json"
            if not result_path.is_file():
                train = [e for e in examples if fold_of[e.original_case_id] != fold]
                held_out = [e for e in examples if fold_of[e.original_case_id] == fold]
                run_dir = result_path.with_suffix("")
                rows = run_fit(config, train, held_out, run_dir, lam, epochs, keep=False)
                result_path.write_text(json.dumps({"lambda_rank": lam, "fold": fold, "train_examples": len(train),
                                                   "held_out_examples": len(held_out), "epochs": rows}, indent=2), encoding="utf-8")
                shutil.rmtree(run_dir)  # the weights of a cross-validation run are not needed, only its metrics
            per_fold.append(json.loads(result_path.read_text(encoding="utf-8"))["epochs"])
            print(f"lambda {lam} fold {fold}: " + json.dumps(per_fold[-1]), flush=True)
        for epoch in range(epochs):
            candidates.append({"lambda_rank": lam, "epoch": epoch + 1, **pooled([rows[epoch] for rows in per_fold])})
    chosen = choose(candidates, selection["nll_tolerance"])
    summary = {"folds": folds, "fold_seed": selection["seed"], "train_cases": len(fold_of),
               "rule": f"admissible: held-out AFTER NLL <= min + {selection['nll_tolerance']}; then highest ordering, "
                       "then delta sign accuracy, then lower AFTER NLL", "candidates": candidates, "chosen": chosen,
               "dev_used": False}
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# TRAIN cross-validation of lambda_rank (held-out TRAIN folds; DEV not read)", "",
             f"{folds} folds by case over {len(fold_of)} TRAIN cases. Rule: {summary['rule']}.", "",
             "| lambda_rank | epoch | AFTER NLL | ECE | ordering (pairs) | ordering per fold | delta sign (pairs) | delta r | delta MAE | chosen |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in candidates:
        by_fold = ", ".join("n/a" if v is None else f"{v:.2f}" for v in c["ordering_by_fold"])
        cell = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"| {c['lambda_rank']} | {c['epoch']} | {cell(c['after_nll'])} | {cell(c['ece'])} | {cell(c['ordering'])} ({c['ordering_pairs']}) | "
                     f"{by_fold} | {cell(c['delta_sign'])} ({c['nonzero_pairs']}) | {cell(c['delta_pearson'])} | {cell(c['delta_mae'])} | "
                     f"{'<-' if c is chosen else ''} |")
    (out / "cv_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return summary


def final(config: dict, examples, out: Path) -> dict:
    chosen = json.loads((out / "cv_summary.json").read_text(encoding="utf-8"))["chosen"]
    selection = config["training"]["selection"]
    fold_of = folds_by_case(examples, selection["folds"], selection["seed"])
    monitor = [e for e in examples if fold_of[e.original_case_id] == 0]  # seen in training; monitoring only
    rows = run_fit(config, examples, monitor, out / "final", chosen["lambda_rank"], chosen["epoch"], keep=True)
    result = {"lambda_rank": chosen["lambda_rank"], "epochs": chosen["epoch"], "train_examples": len(examples),
              "checkpoint": str(out / "final" / "epochs" / f"epoch{chosen['epoch']}"),
              "monitoring_set": "TRAIN fold 0 (in-sample)", "monitoring_metrics": rows, "dev_used": False}
    (out / "final_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("cv", "final"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_predictor_config(args.config)
    spec = config["continuation_spec"]
    examples = to_examples(load_pairs(args.train_data, split="train", protocol_hash=spec.protocol_hash(), expected_k=spec.k))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (cross_validate if args.command == "cv" else final)(config, examples, out)


if __name__ == "__main__":
    main()
