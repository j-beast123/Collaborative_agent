"""Minimal supervised trainer for rollout-derived value targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import load_value_examples
from .model import CollaborationValueModel
from .evaluate import evaluate_predictions
from .value_dataset import split_by_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="/home/kai/models/Qwen3-8B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import torch

    torch.manual_seed(args.seed)
    examples = load_value_examples(args.data)
    splits = split_by_case(examples, seed=args.seed)
    if not splits["train"] or not splits["dev"]:
        raise ValueError("Need at least two case_ids to create non-empty train and dev splits")
    examples = splits["train"]
    model = CollaborationValueModel(args.model, max_length=args.max_length, use_lora=not args.no_lora)
    optimizer = torch.optim.AdamW([p for p in model.trainable_parameters() if p.requires_grad], lr=args.lr)
    model.backbone.train()
    model.value_head.train()
    for _ in range(args.epochs):
        import random
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
    with torch.no_grad():
        dev_predictions = [float(value) for value in model.probabilities([e.text for e in splits["dev"]]).cpu()]
    print(json.dumps({"saved": args.output_dir, "train_examples": len(examples), "dev_examples": len(splits["dev"]), "dev_metrics": evaluate_predictions(dev_predictions, splits["dev"])}))


if __name__ == "__main__":
    main()
