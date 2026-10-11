from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compact_quality_gate_v3_3_1 import GateExecutionError  # noqa: E402
from public_evidence_v4_1 import SCHEMA_PATH, validate_wire_behavior  # noqa: E402


def packet(operation="mention", target="A", *, performed=False):
    return {
        "requested_operation": {
            "operation_type": "compatibility_evaluation",
            "dimension": "timing",
            "definition": "Evaluate compatibility with the current presentation."
        },
        "requested_target": "A",
        "observed_operations": [{
            "operation_type": operation,
            "dimension": "timing",
            "target": target,
            "evidence_quote": "A presents early."
        }],
        "target_covered": target == "A",
        "operation_performed": performed,
        "missing_component": "Compatibility was not evaluated." if not performed else ""
    }


def test_provider_schema_has_no_unique_items_or_duplicate_quote_ledger():
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    raw = json.dumps(schema)
    assert "uniqueItems" not in raw
    assert "evidence_quotes" not in raw


def test_quotes_are_derived_once_from_observed_operations():
    result = validate_wire_behavior(packet(), target_reply="A presents early.")
    assert result["evidence_quotes"] == ["A presents early."]
    assert result["operation_performed"] is False


def test_requested_operation_still_needs_matching_typed_observation():
    value = packet(performed=True)
    with pytest.raises(GateExecutionError):
        validate_wire_behavior(value, target_reply="A presents early.")


def test_requested_operation_and_target_can_pass():
    value = packet(operation="compatibility_evaluation", performed=True)
    result = validate_wire_behavior(value, target_reply="A presents early.")
    assert result["operation_performed"] is True


def test_semantically_same_target_does_not_require_exact_label_text():
    value = packet()
    value["requested_target"] = "the user's favored exclusion option, A"
    value["target_covered"] = True
    result = validate_wire_behavior(value, target_reply="A presents early.")
    assert result["target_covered"] is True


def test_covered_target_still_requires_literal_observation():
    value = packet()
    value["observed_operations"] = []
    value["target_covered"] = True
    with pytest.raises(GateExecutionError):
        validate_wire_behavior(value, target_reply="A presents early.")
