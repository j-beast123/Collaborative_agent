"""Independent, isolated Gate v3.3.1 contracts and final assembler.

This module is not imported by the active trainer.  It does not import the
v3.2/v3.3 assembler and it never turns an execution failure into a rubric
decision or score.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

PROTOCOL = "public_mc_gate_compact_v3_3_1"
ASSEMBLER_VERSION = "gate_v3_3_1_isolated_assembler_v1"
BEHAVIOR_CONTRACT_VERSION = "public_evidence_contract_v4"
APPLICABILITY_CONTRACT_VERSION = "applicability_contract_v3_3_1"
HERE = Path(__file__).resolve().parent
BEHAVIOR_SCHEMA_PATH = HERE / "contracts/public_evidence_contract_v4.schema.json"
APP_SCHEMA_PATH = HERE / "contracts/applicability_contract_v3_3_1.schema.json"

OPERATION_TYPES = {
    "mention",
    "comparison",
    "compatibility_evaluation",
    "causal_explanation",
    "exception_analysis",
    "decision_linkage",
}

EVALUATION_STATES = {
    "PASS", "REJECT", "UNSCORABLE", "EXECUTION_ERROR", "N/A",
    "SCORE_0", "SCORE_1", "SCORE_2",
}


class GateExecutionError(ValueError):
    """A malformed, contradictory, truncated, or leaked execution record."""


def evaluation_state(*, applicable: bool | None = None, level: str | None = None,
                     gate_decision: str | None = None, execution_error: bool = False) -> str:
    """Keep rubric decisions, score zero, N/A, and technical failure disjoint."""
    if execution_error:
        return "EXECUTION_ERROR"
    if gate_decision is not None:
        mapping = {"pass": "PASS", "reject": "REJECT", "unscorable": "UNSCORABLE"}
        if gate_decision not in mapping:
            raise ValueError("Unknown Gate decision")
        return mapping[gate_decision]
    if applicable is False:
        return "N/A"
    if applicable is not True:
        raise ValueError("Applicability must be resolved before producing a score state")
    if level == "UNSCORABLE":
        return "UNSCORABLE"
    if level in ("0", "1", "2"):
        return "SCORE_" + level
    raise ValueError("Missing or invalid score level")


def sha256_json(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assembler_sha256() -> str:
    return file_sha256(Path(__file__))


def validate_behavior_contract(value: dict, *, target_reply: str) -> dict:
    """Validate structure and literal evidence, not semantic truth by substring."""
    errors = sorted(Draft202012Validator(json.loads(BEHAVIOR_SCHEMA_PATH.read_text())).iter_errors(value),
                    key=lambda error: list(error.path))
    if errors:
        raise GateExecutionError("Behavior v4 schema failure: " + errors[0].message)

    requested = value["requested_operation"]
    if requested["operation_type"] not in OPERATION_TYPES:
        raise GateExecutionError("Unknown requested operation")
    for quote in value["evidence_quotes"]:
        if quote not in target_reply:
            raise GateExecutionError("Behavior evidence quote is not literal target-reply evidence")
    for observed in value["observed_operations"]:
        if observed["evidence_quote"] not in target_reply:
            raise GateExecutionError("Observed-operation quote is not literal target-reply evidence")
        if observed["evidence_quote"] not in value["evidence_quotes"]:
            raise GateExecutionError("Observed-operation quote is absent from the evidence ledger")

    target_observations = [row for row in value["observed_operations"]
                           if row["target"] == value["requested_target"]]
    if value["target_covered"] != bool(target_observations):
        raise GateExecutionError("target_covered contradicts the observed target ledger")
    matching_operations = [row for row in target_observations
                           if row["operation_type"] == requested["operation_type"]]
    if value["operation_performed"] and not value["target_covered"]:
        raise GateExecutionError("Operation cannot be performed when the requested target is uncovered")
    if value["operation_performed"] and not matching_operations:
        raise GateExecutionError("Performed operation lacks a matching typed observation for the requested target")
    if value["operation_performed"] and value["missing_component"].strip():
        raise GateExecutionError("Performed operation cannot retain a missing component")
    if not value["operation_performed"] and not value["missing_component"].strip():
        raise GateExecutionError("Unperformed operation needs a concrete missing component")
    return copy.deepcopy(value)


def validate_applicability_contract(value: dict, *, visible_prefix: list[dict],
                                    forbidden_turn_ids: set[str] | None = None) -> dict:
    errors = sorted(Draft202012Validator(json.loads(APP_SCHEMA_PATH.read_text())).iter_errors(value),
                    key=lambda error: list(error.path))
    if errors:
        raise GateExecutionError("Applicability v3.3.1 schema failure: " + errors[0].message)
    catalog = {row["turn_id"]: row["text"] for row in visible_prefix}
    complete_ids = [row["turn_id"] for row in visible_prefix]
    forbidden = set(forbidden_turn_ids or ())
    local_satisfied = []
    for clause in value["clauses"]:
        quote = clause["evidence_quote"]
        if quote is not None:
            turn_id = quote["turn_id"]
            if turn_id in forbidden or turn_id not in catalog:
                raise GateExecutionError("Applicability cites the target reply or a non-prefix turn")
            if quote["quote"] not in catalog[turn_id]:
                raise GateExecutionError("Applicability quote is not literal prefix evidence")
        rtype, status = clause["requirement_type"], clause["status"]
        if rtype == "presence":
            satisfied = status == "supported"
            if satisfied and quote is None:
                raise GateExecutionError("Supported presence clause lacks prefix evidence")
        elif rtype == "absence":
            satisfied = (
                status == "absent"
                and clause["evidence_scope"] == "complete_visible_prefix"
                and clause["inspected_turn_ids"] == complete_ids
            )
            if status == "absent" and quote is not None:
                raise GateExecutionError("Confirmed absence cannot cite positive evidence")
        else:
            satisfied = status == "not_required"
            if quote is not None:
                raise GateExecutionError("Not-required clause cannot cite evidence")
        if clause["satisfied"] != satisfied:
            raise GateExecutionError("Applicability clause satisfaction contradicts polarity rules")
        local_satisfied.append(satisfied)
    computed = all(local_satisfied)
    if value["applicable"] != computed:
        raise GateExecutionError("Aggregate applicability contradicts clause satisfaction")
    return copy.deepcopy(value)


def _validate_quality(quality: dict) -> tuple[dict, list[str]]:
    checks = quality.get("checks")
    codes = quality.get("failure_codes")
    if not isinstance(checks, dict) or not checks or any(type(v) is not bool for v in checks.values()):
        raise GateExecutionError("Quality stage lacks a complete boolean check map")
    if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
        raise GateExecutionError("Quality stage lacks failure codes")
    if quality.get("decision") == "pass" and (codes or not all(checks.values())):
        raise GateExecutionError("Quality pass contradicts its checks or failure codes")
    if quality.get("decision") != "pass" and not codes:
        raise GateExecutionError("Quality rejection lacks a failure code")
    return copy.deepcopy(checks), list(codes)


def _validate_source(source: dict, target_reply: str) -> dict:
    level = source.get("independent_level")
    if level not in ("0", "1", "2", "UNSCORABLE"):
        raise GateExecutionError("Source level is missing")
    if level != "UNSCORABLE" and source.get("score_unique") is not True:
        raise GateExecutionError("Numeric source level is not unique")
    if level == "UNSCORABLE" and source.get("score_unique") is not False:
        raise GateExecutionError("UNSCORABLE source cannot be unique")
    for row in source.get("evidence", []):
        if row.get("quote") not in target_reply:
            raise GateExecutionError("Source evidence is not literal target-reply evidence")
    behavior_v4 = source.get("behavior_v4")
    if behavior_v4 is not None:
        validate_behavior_contract(behavior_v4, target_reply=target_reply)
    return copy.deepcopy(source)


def _validate_intervention(intervention: dict) -> tuple[dict, list[str]]:
    quality = intervention.get("quality")
    if not isinstance(quality, dict) or set(quality) != {
        "instruction_aligned", "no_unsupported_context_premise", "no_conclusion_prescribed"
    } or any(type(v) is not bool for v in quality.values()):
        raise GateExecutionError("Intervention stage lacks its three independent booleans")
    code_map = {
        "instruction_aligned": "INSTRUCTION_ALIGNMENT",
        "no_unsupported_context_premise": "UNSUPPORTED_CONTEXT_PREMISE",
        "no_conclusion_prescribed": "CONCLUSION_PRESCRIBED",
    }
    return copy.deepcopy(quality), [code for key, code in code_map.items() if not quality[key]]


def assemble_v3_3_1(*, source_outcome: str, state: dict, quality: dict,
                    applicability: dict, source_behavior: dict, intervention: dict,
                    provenance: dict) -> dict:
    """Compute one final decision from independently validated stage records."""
    if source_outcome not in ("success", "failure"):
        raise GateExecutionError("Invalid source outcome")
    qflags, qcodes = _validate_quality(quality)
    app = validate_applicability_contract(
        applicability,
        visible_prefix=state["dialogue_before_target_turn"],
        forbidden_turn_ids={state["source_agent_reply"]["turn_id"], "source"},
    )
    behavior = _validate_source(source_behavior, state["source_agent_reply"]["text"])
    iflags, icodes = _validate_intervention(intervention)
    codes = list(dict.fromkeys(qcodes + ([] if app["applicable"] else ["APPLICABILITY_UNSUPPORTED"]) + icodes))
    expected = "2" if source_outcome == "success" else "0"
    actual = behavior["independent_level"]
    if actual == "UNSCORABLE":
        codes.extend(code for code in ("SOURCE_ANCHOR_MISMATCH", "SOURCE_UNSCORABLE") if code not in codes)
        decision = "unscorable"
    else:
        if actual != expected and "SOURCE_ANCHOR_MISMATCH" not in codes:
            codes.append("SOURCE_ANCHOR_MISMATCH")
        decision = "reject" if codes else "pass"
    if decision == "pass" and not (
        app["applicable"] and behavior["score_unique"] and all(qflags.values()) and all(iflags.values())
    ):
        raise GateExecutionError("Pass lacks a mandatory grounded check")

    contract_hashes = {
        "behavior_schema_sha256": file_sha256(BEHAVIOR_SCHEMA_PATH),
        "applicability_schema_sha256": file_sha256(APP_SCHEMA_PATH),
        "assembler_sha256": assembler_sha256(),
    }
    result = {
        "protocol": PROTOCOL,
        "assembler_version": ASSEMBLER_VERSION,
        "decision": decision,
        "passed": decision == "pass",
        "source_level": None if actual == "UNSCORABLE" else int(actual),
        "failure_codes": codes,
        "stage_summary": {
            "quality": qflags,
            "applicability": app,
            "source_behavior": behavior,
            "intervention_quality": iflags,
        },
        "provenance": {**copy.deepcopy(provenance), **contract_hashes,
                       "behavior_contract_version": BEHAVIOR_CONTRACT_VERSION,
                       "applicability_contract_version": APPLICABILITY_CONTRACT_VERSION},
    }
    result["cache_identity_sha256"] = sha256_json({
        "protocol": PROTOCOL,
        "assembler_version": ASSEMBLER_VERSION,
        "contract_hashes": contract_hashes,
        "provenance": result["provenance"],
        "state_sha256": sha256_json(state),
    })
    return result
