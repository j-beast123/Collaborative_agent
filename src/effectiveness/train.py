"""Minimal supervised trainer for rollout-derived value targets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .dataset import load_value_examples
from .model import CollaborationValueModel
from .evaluate import evaluate_effectiveness, evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data")
    parser.add_argument("--dev-data")
    parser.add_argument("--test-data")
    parser.add_argument("--model", default="/home/kai/models/Qwen3-8B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--training-mode", choices=["lora", "head-only"], default="lora")
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not all((args.train_data, args.dev_data, args.test_data)):
        raise ValueError("--train-data, --dev-data, and --test-data are required")
    splits = {
        "train": load_value_examples(args.train_data),
        "dev": load_value_examples(args.dev_data),
        "test": load_value_examples(args.test_data),
    }
    case_sets = {name: {example.case_id for example in values} for name, values in splits.items()}
    if case_sets["train"] & case_sets["dev"] or case_sets["train"] & case_sets["test"] or case_sets["dev"] & case_sets["test"]:
        raise ValueError("Train/dev/test case sets overlap")
    if not splits["train"] or not splits["dev"] or not splits["test"]:
        raise ValueError("Train, dev, and test splits must all be non-empty")
    examples = splits["train"]
    model = CollaborationValueModel(args.model, max_length=args.max_length, use_lora=args.training_mode == "lora", freeze_backbone=args.training_mode == "head-only")
    optimizer = torch.optim.AdamW([p for p in model.trainable_parameters() if p.requires_grad], lr=args.lr)
    model.backbone.train()
    model.value_head.train()
    def predict_batched(current_model, rows, batch_size):
        values = []
        current_model.eval()
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                values.extend(float(value) for value in current_model.probabilities([e.text for e in rows[start:start + batch_size]]).cpu())
        model.train()
        return values
    best_dev = float("inf")
    history = []
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            logits = model.forward([example.text for example in batch])
            successes = torch.tensor([example.num_success for example in batch], device=model.device)
            counts = torch.tensor([example.num_rollouts for example in batch], device=model.device)
            loss = model.binomial_nll(logits, successes, counts)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        dev_predictions = predict_batched(model, splits["dev"], args.batch_size)
        dev_metrics = evaluate_predictions(dev_predictions, splits["dev"])
        history.append({"epoch": epoch + 1, "dev_metrics": dev_metrics})
        if dev_metrics["bce"] < best_dev:
            best_dev = dev_metrics["bce"]
            model.save_pretrained(args.output_dir)
    model.eval()
    from .model import CollaborationValueModel
    best_model = CollaborationValueModel.from_pretrained(args.output_dir)
    dev_predictions = predict_batched(best_model, splits["dev"], args.batch_size)
    test_predictions = predict_batched(best_model, splits["test"], args.batch_size)
    metrics = {
        "dev": evaluate_predictions(dev_predictions, splits["dev"]),
        "test": evaluate_predictions(test_predictions, splits["test"]),
    }
    for name, rows in (("dev", splits["dev"]), ("test", splits["test"])):
        before = [(index, row) for index, row in enumerate(rows) if row.state_type == "before_agent"]
        after = [(index, row) for index, row in enumerate(rows) if row.state_type == "after_agent"]
        before_rows = [row for _, row in before]
        after_rows = [row for _, row in after]
        if before_rows and after_rows:
            before_predictions = predict_batched(best_model, before_rows, args.batch_size)
            after_predictions = predict_batched(best_model, after_rows, args.batch_size)
            metrics[name]["effectiveness"] = evaluate_effectiveness(
                before_predictions, after_predictions, before_rows, after_rows
            )
    if args.verify_checkpoint:
        reloaded = CollaborationValueModel.from_pretrained(args.output_dir)
        probe = splits["dev"][: min(2, len(splits["dev"]))]
        first = predict_batched(best_model, probe, args.batch_size)
        second = predict_batched(reloaded, probe, args.batch_size)
        if any(abs(a - b) > 1e-6 for a, b in zip(first, second)):
            raise RuntimeError("Checkpoint round-trip predictions differ")
        metrics["checkpoint_round_trip"] = "pass"
    print(json.dumps({"saved": args.output_dir, "train_examples": len(examples), "dev_examples": len(splits["dev"]), "test_examples": len(splits["test"]), "history": history, "metrics": metrics}))


if __name__ == "__main__":
    main()
