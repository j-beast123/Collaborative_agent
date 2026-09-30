"""Inference interface for state value and turn effectiveness."""

from __future__ import annotations

from .value_dataset import build_state_text


def predict_success_probability(model, state: dict) -> float:
    model.backbone.eval()
    model.value_head.eval()
    with model.torch.no_grad():
        return float(model.probabilities([build_state_text(state)])[0].cpu())


def compute_effectiveness(model, state_before: dict, state_after: dict) -> float:
    before = predict_success_probability(model, state_before)
    after = predict_success_probability(model, state_after)
    return max(-1.0, min(1.0, after - before))
