"""Clinician-model API compatibility layer.

The implementation remains shared with the legacy generator while the
generation pipeline is being decomposed into independently testable modules.
"""

from .generator import doctor_call

__all__ = ["doctor_call"]
