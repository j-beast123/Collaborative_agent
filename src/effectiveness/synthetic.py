"""SYNTHETIC smoke-test components. Never valid for research results."""

from __future__ import annotations

import hashlib
import random
from types import SimpleNamespace
from typing import Any

import torch


SYNTHETIC_MODEL_ID = "synthetic-tiny-backbone"
SYNTHETIC_REVISION = "v1"


def _token_id(word: str, vocab_size: int) -> int:
    return 1 + int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % (vocab_size - 1)


class HashingTokenizer:
    pad_token = "<pad>"
    eos_token = "<pad>"
    pad_token_id = 0

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.padding_side = "right"

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors="pt"):
        ids = [[_token_id(w, self.vocab_size) for w in text.lower().split()] or [0] for text in texts]
        if truncation and max_length:
            ids = [row[:max_length] for row in ids]
        width = max(len(row) for row in ids)
        return {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in ids]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in ids]),
        }


class _Config(dict):
    """Attribute and dict access (peft reads backbone.config with .get)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class TinyBackbone(torch.nn.Module):
    """Causal running mean of seeded embeddings; identical weights on every load."""

    def __init__(self, vocab_size: int = 512, hidden_size: int = 16, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.config = _Config(hidden_size=hidden_size, model_type="synthetic_tiny")
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        # Identity projection: leaves the frozen output unchanged and gives LoRA a module to adapt ("q").
        self.q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab_size, hidden_size, generator=generator))
            self.q.weight.copy_(torch.eye(hidden_size))

    def forward(self, input_ids, attention_mask, **_):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (self.q(self.embed(input_ids)) * mask).cumsum(dim=1)
        hidden = summed / mask.cumsum(dim=1).clamp_min(1)
        return SimpleNamespace(last_hidden_state=hidden)


def load_synthetic_backbone(model_id, revision, **_options):
    if (model_id, revision) != (SYNTHETIC_MODEL_ID, SYNTHETIC_REVISION):
        raise ValueError("Synthetic loader only provides the synthetic tiny backbone")
    return TinyBackbone(), HashingTokenizer()


RESPONSES = {
    "supported": ("Given the strong evidence, I revise my answer and explain why.", 0.85),
    "unsupported": ("I simply agree with you without checking anything.", 0.2),
}
BEFORE_PROBABILITY = 0.5


def synthetic_pair_rows(num_histories: int, k: int, protocol_hash: str, seed: int, prefix: str,
                        split: str = "train") -> list[dict[str, Any]]:
    """Toy label rows in the labels.py schema: Binomial(k, p) BEFORE / AFTER counts."""
    rng = random.Random(seed)
    rows = []
    for index in range(num_histories):
        case_id = f"{prefix}-case-{index}"
        history = {
            "history_id": f"{case_id}:traj:turn3", "case_id": case_id, "original_case_id": case_id,
            "source_trajectory_id": f"{case_id}:traj", "question": f"SYNTHETIC question {index}",
            "options": {"A": "alpha", "B": "beta"}, "scenario": "SYNTHETIC smoke-test scenario",
            "claim_order": "doctor_first", "belief_enabled": True, "round": 2,
            "turns": [{"speaker": "doctor", "text": f"Synthetic collaborator claim {index}.", "round": 1},
                      {"speaker": "agent", "text": "My initial answer is A.", "round": 1},
                      {"speaker": "doctor", "text": "Are you sure? Consider B.", "round": 2}],
        }
        before = sum(rng.random() < BEFORE_PROBABILITY for _ in range(k))
        for response_id, (text, probability) in RESPONSES.items():
            rows.append({
                "case_id": case_id, "original_case_id": case_id, "history_id": history["history_id"],
                "response_id": response_id, "response_source": "synthetic", "split": split,
                "history": history, "response": text,
                "before": {"successes": before, "trials": k, "rollout_ids": []},
                "after": {"successes": sum(rng.random() < probability for _ in range(k)), "trials": k,
                          "rollout_ids": []},
                "metadata": {"continuation_protocol_hash": protocol_hash, "synthetic": True},
            })
    return rows
