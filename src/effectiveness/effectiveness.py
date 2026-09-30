"""Turn-level effectiveness and rubric intervention gains."""

from __future__ import annotations


def effectiveness(value_before: float, value_after: float) -> float:
    """Return dense state-transition effectiveness in [-1, 1]."""
    if not 0.0 <= value_before <= 1.0 or not 0.0 <= value_after <= 1.0:
        raise ValueError("Values must be in [0, 1]")
    return value_after - value_before


def intervention_gain(original_after: float, ideal_after: float) -> float:
    """Return G_k for an ideal rubric intervention.

    This is a rubric validation statistic, not a reward multiplier.
    """
    if not 0.0 <= original_after <= 1.0 or not 0.0 <= ideal_after <= 1.0:
        raise ValueError("Values must be in [0, 1]")
    return ideal_after - original_after
