from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compact_quality_gate_v3_3_1 import (  # noqa: E402
    GateExecutionError,
    evaluation_state,
    validate_applicability_contract,
    validate_behavior_contract,
)


def behavior(*, requested="comparison", observed="mention", requested_target="A",
             observed_target="A", covered=True, performed=False, quote="A was mentioned"):
    return {
        "requested_operation": {"operation_type": requested, "dimension": "clinical relation",
                                "definition": "Perform the requested operation."},
        "requested_target": requested_target,
        "observed_operations": [{"operation_type": observed, "dimension": "clinical relation",
                                 "target": observed_target, "evidence_quote": quote}],
        "target_covered": covered,
        "operation_performed": performed,
        "evidence_quotes": [quote],
        "missing_component": "The requested relation was not performed." if not performed else "",
    }


def app_clause(requirement_type, status, *, scope="active_history", quote=None,
               inspected=None, satisfied=False):
    return {"clause_id": "c0", "clause": "Frozen clause", "requirement_type": requirement_type,
            "status": status, "evidence_scope": scope, "evidence_quote": quote,
            "inspected_turn_ids": inspected or [], "satisfied": satisfied}


PREFIX = [{"turn_id": "p0", "text": "Doctor states the premise."},
          {"turn_id": "p1", "text": "Agent gives an earlier answer."}]


def test_01_target_covered_but_requested_operation_missing():
    value = behavior()
    assert validate_behavior_contract(value, target_reply="A was mentioned")["operation_performed"] is False


def test_02_operation_on_wrong_target_is_not_target_coverage():
    value = behavior(observed="comparison", observed_target="B", covered=False, performed=False,
                     quote="B was compared")
    assert validate_behavior_contract(value, target_reply="B was compared")["target_covered"] is False


def test_03_mention_does_not_become_comparison():
    value = behavior(requested="comparison", observed="mention")
    assert validate_behavior_contract(value, target_reply="A was mentioned")["operation_performed"] is False


def test_04_cause_mention_does_not_become_causal_explanation():
    value = behavior(requested="causal_explanation", observed="mention", quote="Cause A")
    assert validate_behavior_contract(value, target_reply="Cause A")["operation_performed"] is False


def test_05_presence_supported():
    clause = app_clause("presence", "supported", quote={"turn_id": "p0", "quote": "states the premise"},
                        inspected=["p0"], satisfied=True)
    assert validate_applicability_contract({"clauses": [clause], "applicable": True},
                                          visible_prefix=PREFIX)["applicable"]


def test_06_presence_absent_is_unsatisfied_not_zero():
    clause = app_clause("presence", "absent", inspected=["p0", "p1"], satisfied=False)
    result = validate_applicability_contract({"clauses": [clause], "applicable": False},
                                             visible_prefix=PREFIX)
    assert result["applicable"] is False


def test_07_absence_confirmed_after_complete_prefix():
    clause = app_clause("absence", "absent", scope="complete_visible_prefix",
                        inspected=["p0", "p1"], satisfied=True)
    assert validate_applicability_contract({"clauses": [clause], "applicable": True},
                                          visible_prefix=PREFIX)["applicable"]


def test_08_absence_unknown_is_unsatisfied():
    clause = app_clause("absence", "unknown", scope="complete_visible_prefix",
                        inspected=["p0", "p1"], satisfied=False)
    assert not validate_applicability_contract({"clauses": [clause], "applicable": False},
                                              visible_prefix=PREFIX)["applicable"]


def test_09_target_reply_leak_is_execution_error():
    clause = app_clause("presence", "supported", quote={"turn_id": "source", "quote": "future"},
                        inspected=["source"], satisfied=True)
    with pytest.raises(GateExecutionError):
        validate_applicability_contract({"clauses": [clause], "applicable": True},
                                        visible_prefix=PREFIX, forbidden_turn_ids={"source"})


def test_10_literal_quote_does_not_force_semantic_operation_true():
    value = behavior(requested="compatibility_evaluation", observed="mention", quote="Timing is early")
    checked = validate_behavior_contract(value, target_reply="Timing is early")
    assert checked["operation_performed"] is False


def test_11_matching_target_with_insufficient_operation():
    value = behavior(requested="exception_analysis", observed="mention", quote="A is an exception topic")
    assert validate_behavior_contract(value, target_reply="A is an exception topic")["target_covered"]
    assert not value["operation_performed"]


def test_12_matching_operation_with_wrong_target():
    value = behavior(requested="decision_linkage", observed="decision_linkage", observed_target="B",
                     covered=False, performed=False, quote="B supports the decision")
    checked = validate_behavior_contract(value, target_reply="B supports the decision")
    assert checked["target_covered"] is False


def test_performed_operation_requires_matching_typed_observation():
    value = behavior(requested="comparison", observed="mention", performed=True)
    value["missing_component"] = ""
    with pytest.raises(GateExecutionError):
        validate_behavior_contract(value, target_reply="A was mentioned")


def test_statuses_are_not_interchangeable():
    states = {
        evaluation_state(gate_decision="pass"),
        evaluation_state(gate_decision="reject"),
        evaluation_state(applicable=True, level="UNSCORABLE"),
        evaluation_state(execution_error=True),
        evaluation_state(applicable=False),
        evaluation_state(applicable=True, level="0"),
    }
    assert states == {"PASS", "REJECT", "UNSCORABLE", "EXECUTION_ERROR", "N/A", "SCORE_0"}
