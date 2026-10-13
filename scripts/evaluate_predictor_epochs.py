"""DEV metrics of every saved epoch of an Effectiveness predictor run, side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.io import read_jsonl
from src.effectiveness import calibration
from src.effectiveness.evaluate import evaluate_predictions, ordering_breakdown
from src.effectiveness.model import CollaborationValueModel, safe_torch_load
from src.effectiveness.pair_data import load_pairs, to_examples
from src.effectiveness.train import build_backbone, load_predictor_config


def checkpoints(run_dir: Path) -> dict[str, Path]:
    """{name: file with the trainable state}; epoch<N> from epochs/ when present, else from epoch_states/."""
    found = {}
    for path in sorted(run_dir.glob("epoch_states/epoch*.pt")):
        found[path.stem] = path
    for path in sorted(run_dir.glob("epochs/epoch*/trainable_state.pt")):
        found[path.parent.name] = path
    for name in ("best", "best_ordering"):
        if (run_dir / name / "trainable_state.pt").is_file():
            found[name] = run_dir / name / "trainable_state.pt"
    return dict(sorted(found.items(), key=lambda kv: (not kv[0].startswith("epoch"), int(kv[0][5:]) if kv[0][5:].isdigit() else 0, kv[0])))


def trainable_state(path: Path) -> dict:
    state = safe_torch_load(path, map_location="cpu")
    return state["trainable"] if "trainable" in state else state


def metrics(predictions: list[float], examples, state_of) -> dict:
    result = evaluate_predictions(predictions, examples)
    ordering = ordering_breakdown(predictions, examples, state_of)
    absolute, delta = result["absolute"]["all"], result["delta"]
    return {"nll": absolute["binomial_nll"], "brier": absolute["brier"], "ece": absolute["ece"],
            "delta_mae": delta["mae"], "delta_pearson": delta["pearson"], "delta_spearman": delta["spearman"],
            "delta_sign_accuracy": delta["sign_agreement_nonzero"], "delta_nonzero_pairs": delta["nonzero_count"],
            "ordering_overall": ordering["overall"]["accuracy"], "ordering_pairs": ordering["overall"]["pairs"],
            "ordering_informative_histories": ordering["informative_histories"]["accuracy"],
            "informative_histories": ordering["informative_histories"]["histories"],
            "ordering_by_state": ordering.get("by_group", {})}


def table(rows: dict[str, dict], states: list[str]) -> str:
    columns = ["nll", "brier", "ece", "delta_mae", "delta_pearson", "delta_spearman", "delta_sign_accuracy",
               "ordering_overall", "ordering_informative_histories"]
    lines = ["| checkpoint | " + " | ".join(columns + [f"ordering {s}" for s in states]) + " |",
             "|" + "---|" * (len(columns) + len(states) + 1)]
    for name, row in rows.items():
        cells = [f"{row[c]:.3f}" if row[c] is not None else "n/a" for c in columns]
        for state in states:
            value = row["ordering_by_state"].get(state)
            cells.append(f"{value['accuracy']:.3f} ({value['pairs']})" if value and value["accuracy"] is not None else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dev-data", required=True)
    parser.add_argument("--states")
    parser.add_argument("--calibration", choices=calibration.METHODS)
    parser.add_argument("--save-calibration", action="append", default=[], metavar="CHECKPOINT=FILE")
    parser.add_argument("--only", action="append", default=[], metavar="CHECKPOINT",
                        help="evaluate only these checkpoints (e.g. epoch3), so DEV is read once for a chosen model")
    args = parser.parse_args()
    config = load_predictor_config(args.config)
    predictor, training, spec = config["predictor"], config["training"], config["continuation_spec"]
    examples = to_examples(load_pairs(args.dev_data, split="dev", protocol_hash=spec.protocol_hash(), expected_k=spec.k))
    case_state = {r["case_id"]: r["collaboration_state"] for r in read_jsonl(args.states)} if args.states else {}
    case_of = {e.history_id: e.original_case_id for e in examples}
    state_of = (lambda history_id: case_state.get(case_of[history_id])) if case_state else None

    model_id, revision, loader, options = build_backbone(predictor)
    backbone, tokenizer = loader(model_id, revision, **options)
    lora = predictor.get("lora") if predictor["training_mode"] == "lora_scalar_head" else None
    model = CollaborationValueModel(backbone, tokenizer, base_model_id=model_id, base_model_revision=revision,
                                    max_length=predictor["max_length"], training_mode=predictor["training_mode"],
                                    lora=lora, backbone_options=options,
                                    architecture=predictor.get("architecture", "single"))
    run_dir = Path(args.run_dir)
    save = dict(item.split("=", 1) for item in args.save_calibration)
    raw, calibrated, dump = {}, {}, {}
    for name, path in checkpoints(run_dir).items():
        if args.only and name not in args.only:
            continue
        model.load_trainable_state(trainable_state(path))
        logits = model.example_logits(examples, training["batch_size"])
        predictions = calibration.apply_logits({"a": 1.0, "b": 0.0}, logits)
        dump[name] = {"path": str(path), "logits": logits}
        raw[name] = metrics(predictions, examples, state_of)
        if args.calibration:
            calibrated[name] = metrics(calibration.cross_fitted(predictions, examples, args.calibration), examples, state_of)
        if name in save:
            calibration.save(calibration.fit(predictions, [e.num_success for e in examples],
                                             [e.num_rollouts for e in examples], args.calibration or "temperature"), save[name])
        print(name, json.dumps({k: v for k, v in raw[name].items() if k != "ordering_by_state"}), flush=True)

    # raw success logits of every checkpoint on the DEV examples, for analysis without the model
    (run_dir / "dev_logits.json").write_text(json.dumps({
        "config": args.config, "dev_data": args.dev_data,
        "examples": [{"kind": e.kind, "history_id": e.history_id, "response_id": e.response_id,
                      "case_id": e.original_case_id, "num_success": e.num_success, "num_rollouts": e.num_rollouts}
                     for e in examples],
        "checkpoints": dump}), encoding="utf-8")
    states = sorted({s for row in raw.values() for s in row["ordering_by_state"]})
    result = {"dev_examples": len(examples), "checkpoints": raw,
              "calibrated_cross_fitted": {"method": args.calibration, "checkpoints": calibrated} if args.calibration else None}
    (run_dir / "epoch_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = "# DEV metrics per checkpoint\n\nordering <state>: pooled accuracy (informative pairs)\n\n" + table(raw, states)
    if args.calibration:
        text += f"\n\n## After {args.calibration} calibration (case-level cross-fitted on DEV)\n\n" + table(calibrated, states)
    (run_dir / "epoch_metrics.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
