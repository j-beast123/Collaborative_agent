"""Nonredundant wire contract and local normalization for evidence v4.1."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from compact_quality_gate_v3_3_1 import GateExecutionError, validate_behavior_contract

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "contracts/public_evidence_contract_v4_1.schema.json"
CONTRACT_VERSION = "public_evidence_contract_v4_1"


def validate_wire_behavior(value: dict, *, target_reply: str) -> dict:
    """Validate v4.1 and derive the old flat quote view locally, once."""
    schema = json.loads(SCHEMA_PATH.read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(value),
                    key=lambda error: list(error.path))
    if errors:
        raise GateExecutionError("Behavior v4.1 schema failure: " + errors[0].message)
    normalized = copy.deepcopy(value)
    normalized["evidence_quotes"] = list(dict.fromkeys(
        row["evidence_quote"] for row in normalized["observed_operations"]
    ))
    # Reuse the already-tested structural invariants after deriving, rather
    # than asking the model to maintain two equal ledgers.
    validate_behavior_contract(normalized, target_reply=target_reply)
    return normalized
