"""Effectiveness providers and reward transforms."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

from src.common.config import IntegrationUnavailable

from .history import format_input

TRANSFORMS: dict[str, Callable[[float], float]] = {
    "identity": lambda delta: delta,
    "affine_unit": lambda delta: (delta + 1.0) / 2.0,
    "clip_unit": lambda delta: min(1.0, max(0.0, delta)),
}


def build_transform(name_or_callable) -> Callable[[float], float]:
    if callable(name_or_callable):
        return name_or_callable
    if name_or_callable not in TRANSFORMS:
        raise ValueError(f"effectiveness transform must be one of {sorted(TRANSFORMS)} or a callable")
    return TRANSFORMS[name_or_callable]


class EffectivenessProvider(ABC):
    name = "abstract"

    @abstractmethod
    def predict_before(self, history: dict) -> float:
        """P(final success | history)."""

    @abstractmethod
    def predict_after(self, history: dict, response: str) -> float:
        """P(final success | history, fixed response)."""

    def predict_delta(self, history: dict, response: str) -> float:
        return self.predict_after(history, response) - self.predict_before(history)

    def identity(self) -> dict[str, Any]:
        return {"provider": self.name}


class LearnedPredictorProvider(EffectivenessProvider):
    """One effectiveness model for both modes (history only / history + response)."""

    name = "learned_predictor"

    def __init__(self, model, checkpoint: Optional[str] = None, calibration: Optional[dict] = None) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.calibration = calibration  # monotonic map fitted on DEV (src/effectiveness/calibration.py), or None

    @classmethod
    def from_checkpoint(cls, checkpoint: Optional[str], *, expected_protocol_hash: Optional[str] = None,
                        backbone_loader=None, calibration: Optional[str] = None) -> "LearnedPredictorProvider":
        """A valid checkpoint is a required dependency; anything missing fails here, not mid-training."""
        if not checkpoint:
            raise IntegrationUnavailable("Effectiveness provider learned_predictor needs a checkpoint path")
        path = Path(checkpoint)
        for name in ("predictor_config.json", "trainable_state.pt"):
            if not (path / name).is_file():
                raise IntegrationUnavailable(f"Effectiveness checkpoint {path} is missing {name}")
        metadata = json.loads((path / "predictor_config.json").read_text(encoding="utf-8")).get("metadata", {})
        if expected_protocol_hash and metadata.get("continuation_protocol_hash") != expected_protocol_hash:
            raise IntegrationUnavailable("Effectiveness checkpoint was trained under a different continuation protocol")
        from .model import CollaborationValueModel

        kwargs = {"backbone_loader": backbone_loader} if backbone_loader else {}
        from .calibration import load as load_calibration

        return cls(CollaborationValueModel.from_pretrained(path, **kwargs), checkpoint=str(path),
                   calibration=load_calibration(calibration) if calibration else None)

    def _predict(self, text: str) -> float:
        probability = self.model.predict([text], batch_size=1)[0]
        if self.calibration:
            from .calibration import apply

            probability = apply(self.calibration, [probability])[0]
        return probability

    def predict_before(self, history: dict) -> float:
        if getattr(self.model, "architecture", "single") == "residual_frozen_before":  # the frozen baseline path
            from types import SimpleNamespace

            example = SimpleNamespace(kind="before", history_id="h", text=format_input(history))
            probability = self.model.predict_examples([example], 1)[0]
            if self.calibration:
                from .calibration import apply

                probability = apply(self.calibration, [probability])[0]
            return probability
        return self._predict(format_input(history))

    def predict_after(self, history: dict, response: str) -> float:
        if getattr(self.model, "architecture", "single") != "single":  # sigmoid(B(h) + G(h, r))
            probability = self.model.residual_after_probability(format_input(history), format_input(history, response))
            if self.calibration:
                from .calibration import apply

                probability = apply(self.calibration, [probability])[0]
            return probability
        return self._predict(format_input(history, response))

    def identity(self) -> dict[str, Any]:
        return {"provider": self.name, "checkpoint": self.checkpoint, "calibration": self.calibration}


class RolloutProvider(EffectivenessProvider):
    """Monte Carlo estimate: K continuations from the history / after the fixed response."""

    name = "rollout"

    def __init__(self, runtime, spec, *, set_seed: Optional[Callable[[int], None]] = None,
                 max_rollout_attempts: int = 1) -> None:
        if spec.k is None or spec.max_rounds is None or spec.final_decision_role is None \
                or spec.seed_policy.mode is None or spec.seed_policy.base_seed is None:
            raise IntegrationUnavailable("Rollout provider needs ContinuationSpec k, max_rounds, final_decision_role, seeds")
        self.runtime, self.spec, self.set_seed = runtime, spec, set_seed
        self.max_rollout_attempts = max_rollout_attempts
        self._before_cache: dict[str, float] = {}

    def _rate(self, history: dict, response: Optional[str], state_id: str) -> float:
        from src.collaboration.rerun import continue_collaboration
        from src.collaboration.runtime import run_with_retries

        from .history import continuation_trajectory

        if not (history.get("ground_truth") or {}).get("correct_option"):
            raise ValueError("RolloutProvider needs history['ground_truth'] to judge outcomes")
        trajectory, index = continuation_trajectory(history, history["ground_truth"], max_rounds=self.spec.max_rounds,
                                                    final_decision_role=self.spec.final_decision_role,
                                                    response=response)
        outcomes = []
        def rollout(seed):
            if self.set_seed:
                self.set_seed(seed)
            return continue_collaboration(trajectory, index, self.runtime, fixed_response=response,
                                          allow_initial_turn=True)

        for seed in self.spec.rollout_seeds(history["history_id"], state_id):
            result = run_with_retries(lambda: rollout(seed), format_attempts=self.max_rollout_attempts)
            if result["status"] != "completed":
                raise RuntimeError(f"continuation failed ({result['status']}) for {state_id}")
            outcomes.append(bool(result["correct"]))
        return sum(outcomes) / len(outcomes)

    def predict_before(self, history: dict) -> float:
        # The history's BEFORE estimate is shared by all of its candidate responses.
        if history["history_id"] not in self._before_cache:
            self._before_cache[history["history_id"]] = self._rate(history, None, f"{history['history_id']}:before")
        return self._before_cache[history["history_id"]]

    def predict_after(self, history: dict, response: str) -> float:
        from src.data.schema import digest

        return self._rate(history, response, f"{history['history_id']}:after:{digest(response)[:12]}")

    def identity(self) -> dict[str, Any]:
        return {"provider": self.name, "k": self.spec.k, "continuation_protocol_hash": self.spec.protocol_hash()}


def build_effectiveness_provider(section: dict, weight: float, *, factories: Optional[dict] = None
                                 ) -> Optional[EffectivenessProvider]:
    """Provider for the effectiveness reward component."""
    if weight == 0:
        return None
    kind = (section or {}).get("provider")
    if kind is None:
        raise IntegrationUnavailable("w_E > 0 requires reward.effectiveness.provider (learned_predictor or rollout)")
    factories = factories or {}
    if kind in factories:
        return factories[kind](section)
    if kind == "learned_predictor":
        return LearnedPredictorProvider.from_checkpoint(
            section.get("checkpoint"), expected_protocol_hash=section.get("expected_continuation_protocol_hash"),
            calibration=section.get("calibration"))
    if kind == "rollout":
        raise IntegrationUnavailable("rollout provider needs a continuation runtime; pass it through factories['rollout']")
    raise ValueError(f"unknown effectiveness provider {kind!r}")
