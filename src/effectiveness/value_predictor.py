"""A simple text-conditioned value predictor V_phi(h) -> [0, 1]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValuePrediction:
    state_id: str
    case_id: str
    predicted_value: float


class ValuePredictor:
    """Thin interface for a learned value model.

    The default implementation uses a transformer encoder when torch and
    transformers are available. The interface keeps the predictor independent
    from the rollout policy and from appropriateness scoring.
    """

    def __init__(self, model_name: str, *, max_length: int = 2048) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)
        self.max_length = max_length

    def forward(self, texts: list[str]) -> Any:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        output = self.encoder(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.torch.sigmoid(self.value_head(pooled)).squeeze(-1)

    def loss(
        self,
        predicted: Any,
        *,
        successes: Any,
        rollout_counts: Any,
        objective: str = "binomial_nll",
    ) -> Any:
        """Compute binomial NLL or MSE against empirical value targets."""
        torch = self.torch
        if objective == "mse":
            targets = successes.float() / rollout_counts.float().clamp_min(1)
            return torch.nn.functional.mse_loss(predicted, targets)
        if objective != "binomial_nll":
            raise ValueError("objective must be 'binomial_nll' or 'mse'")
        eps = torch.finfo(predicted.dtype).eps
        p = predicted.clamp(eps, 1.0 - eps)
        failures = rollout_counts - successes
        return -(successes * torch.log(p) + failures * torch.log1p(-p)).mean()

    def effectiveness(self, before: Any, after: Any) -> Any:
        """Compute Eff_t from two predicted state values."""
        return after - before
