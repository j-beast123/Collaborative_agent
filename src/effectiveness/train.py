"""Minimal supervised trainer for rollout-derived value targets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .dataset import load_value_examples
from .model import CollaborationValueModel
from .evaluate import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.data and any((args.train_data, args.dev_data, args.test_data)):
        raise ValueError("Use either --data or explicit train/dev/test files, not both")
    if args.data:
        raise ValueError("Generate train/dev/test with write_split_rows, then pass explicit split files")
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
    if not splits["train"] or not splits["dev"]:
        raise ValueError("Need at least two case_ids to create non-empty train and dev splits")
    examples = splits["train"]
    model = CollaborationValueModel(args.model, max_length=args.max_length, use_lora=args.training_mode == "lora", freeze_backbone=args.training_mode == "head-only")
    optimizer = torch.optim.AdamW([p for p in model.trainable_parameters() if p.requires_grad], lr=args.lr)
    model.backbone.train()
    model.value_head.train()
    for _ in range(args.epochs):
        random.shuffle(examples)
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            logits = model.forward([example.text for example in batch])
            successes = torch.tensor([example.num_success for example in batch], device=model.device)
            counts = torch.tensor([example.num_rollouts for example in batch], device=model.device)
            loss = model.binomial_nll(logits, successes, counts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.save_pretrained(args.output_dir)
    model.eval()
    def predict_batched(rows):
        values = []
        with torch.no_grad():
            for start in range(0, len(rows), args.batch_size):
                values.extend(float(value) for value in model.probabilities([e.text for e in rows[start:start + args.batch_size]]).cpu())
        return values
    dev_predictions = predict_batched(splits["dev"])
    test_predictions = predict_batched(splits["test"])
    print(json.dumps({"saved": args.output_dir, "train_examples": len(examples), "dev_examples": len(splits["dev"]), "test_examples": len(splits["test"]), "dev_metrics": evaluate_predictions(dev_predictions, splits["dev"]), "test_metrics": evaluate_predictions(test_predictions, splits["test"])}))


if __name__ == "__main__":
    main()
