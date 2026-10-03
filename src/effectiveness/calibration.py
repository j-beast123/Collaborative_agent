"""Post-hoc monotonic calibration of the Effectiveness Model's probabilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

METHODS = ("temperature", "platt")  # methods that fit parameters; identity fits nothing
BY_TURN = "temperature_by_turn"  # one checkpoint, one temperature per intervention turn
EPS = 1e-6
LOG_A_BOUND = 8.0  # a = 1 / T is kept in [e^-8, e^8]: labels that carry no signal would otherwise drive T to infinity


def _logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def apply(parameters: dict, probabilities: list[float]) -> list[float]:
    return apply_logits(parameters, [_logit(p) for p in probabilities])


def for_turn(parameters: dict, turn: Optional[str] = None) -> dict:
    """The {a, b, temperature} that applies to a prediction of an intervention turn. One model can carry one
    temperature for all turns (temperature) or one per turn (temperature_by_turn: {"by_turn": {turn: {...}}})."""
    if parameters.get("method") != BY_TURN:
        return parameters
    if turn not in parameters["by_turn"]:
        raise ValueError(f"no calibration for intervention turn {turn!r}")
    return parameters["by_turn"][turn]


def apply_logits(parameters: dict, logits: list[float], turn: Optional[str] = None) -> list[float]:
    """Calibrated probabilities from the model's raw success logits (turn: needed for temperature_by_turn)."""
    chosen = for_turn(parameters, turn)
    return [_sigmoid(chosen["a"] * z + chosen["b"]) for z in logits]


def fit(probabilities: list[float], successes: list[int], trials: list[int], method: str = "temperature") -> dict:
    """{"method", "a", "b"} minimizing the binomial NLL of the calibrated probabilities."""
    return fit_logits([_logit(p) for p in probabilities], successes, trials, method)


def fit_logits(logits: list[float], successes: list[int], trials: list[int], method: str = "temperature") -> dict:
    """fit() on the model's raw success logits z: temperature gives sigmoid(z / T) with T = 1 / a."""
    import torch

    if method not in METHODS + ("identity",):
        raise ValueError(f"calibration method must be one of {METHODS}")
    if method == "identity":
        return {"method": "identity", "a": 1.0, "b": 0.0, "temperature": 1.0, "fitted_on": 0}
    z = torch.tensor(logits, dtype=torch.float64)
    m, n = torch.tensor(successes, dtype=torch.float64), torch.tensor(trials, dtype=torch.float64)
    log_a = torch.zeros((), dtype=torch.float64, requires_grad=True)  # a = exp(log_a) > 0 keeps the map increasing
    b = torch.zeros((), dtype=torch.float64, requires_grad=method == "platt")
    optimizer = torch.optim.LBFGS([log_a] + ([b] if method == "platt" else []), lr=0.5, max_iter=200)

    def closure():
        optimizer.zero_grad()
        logits = torch.exp(torch.clamp(log_a, -LOG_A_BOUND, LOG_A_BOUND)) * z + b
        loss = -(m * torch.nn.functional.logsigmoid(logits) + (n - m) * torch.nn.functional.logsigmoid(-logits)).sum() / n.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    a = float(torch.exp(torch.clamp(log_a.detach(), -LOG_A_BOUND, LOG_A_BOUND)))
    return {"method": method, "a": a, "b": float(b), "temperature": 1 / a, "fitted_on": len(logits)}


def cross_fitted(probabilities: list[float], examples, method: str = "temperature", folds: int = 5) -> list[float]:
    """Calibrated probabilities where each example's parameters were fitted on the other folds; folds are
    by case, so a case never calibrates itself."""
    cases = sorted({e.original_case_id for e in examples})
    fold_of = {case: i % folds for i, case in enumerate(cases)}
    calibrated: list[Optional[float]] = [None] * len(examples)
    for fold in range(folds):
        train = [i for i, e in enumerate(examples) if fold_of[e.original_case_id] != fold]
        test = [i for i, e in enumerate(examples) if fold_of[e.original_case_id] == fold]
        if not test:
            continue
        parameters = fit([probabilities[i] for i in train], [examples[i].num_success for i in train],
                         [examples[i].num_rollouts for i in train], method)
        for i, value in zip(test, apply(parameters, [probabilities[i] for i in test])):
            calibrated[i] = value
    return calibrated


def save(parameters: dict, path) -> None:
    Path(path).write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")


def load(path) -> dict:
    parameters = json.loads(Path(path).read_text(encoding="utf-8"))
    if parameters.get("method") == BY_TURN:
        if not isinstance(parameters.get("by_turn"), dict) or not parameters["by_turn"]:
            raise ValueError(f"{path}: temperature_by_turn needs by_turn: {{turn: temperature parameters}}")
        for turn, own in parameters["by_turn"].items():
            _check(own, f"{path} [{turn}]")
        return parameters
    return _check(parameters, path)


def _check(parameters: dict, path) -> dict:
    if parameters.get("method") not in METHODS + ("identity",) or not parameters.get("a", 0) > 0:
        raise ValueError(f"{path} is not a monotonic calibration (method in {METHODS}, a > 0)")
    if not all(isinstance(parameters.get(k), (int, float)) and math.isfinite(parameters[k]) for k in ("a", "b")):
        raise ValueError("calibration coefficients must be finite")
    if parameters["method"] == "identity" and (parameters["a"] != 1.0 or parameters["b"] != 0.0 or parameters.get("temperature") != 1.0):
        raise ValueError("identity calibration must use T=1, a=1, b=0")
    if parameters["method"] == "temperature":
        t = parameters.get("temperature")
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 or parameters["b"] != 0 or not math.isclose(parameters["a"] * t, 1.0, rel_tol=1e-10):
            raise ValueError("temperature calibration must apply sigmoid(logit / temperature)")
    return parameters
