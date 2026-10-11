#!/usr/bin/env python3
"""Read-only six-case replay and C3 request preview for isolated v3.3.1."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from compact_quality_gate_v3_3_1 import (
    APPLICABILITY_CONTRACT_VERSION,
    ASSEMBLER_VERSION,
    BEHAVIOR_CONTRACT_VERSION,
    PROTOCOL,
    assemble_v3_3_1,
    assembler_sha256,
    file_sha256,
    sha256_json,
)
from legacy_v33_stage_adapter_v1 import (
    ADAPTER_VERSION,
    adapt_applicability,
    adapter_sha256,
)

HERE = Path(__file__).resolve().parent
AUDIT_ROOT = HERE.parent
OLD_ROOT = AUDIT_ROOT / "gate_v3_3_pilot_20261010"
OLD_RESULTS = OLD_ROOT / "results/candidates"
FROZEN = AUDIT_ROOT / "gate_v3_2_remaining_20261011/frozen_all29/frozen_manifest.json"
V32_MANIFEST = AUDIT_ROOT / "gate_v3_2_remaining_20261011/v32_results/gemini_gate_v3_2_all29_manifest.json"
RESULTS = HERE / "results/candidates"
PREVIEW = HERE / "api_preview"
SIX = ("C5/0.0", "C4/0.0", "C1/1.0", "C5/1.2", "C3/0.0", "C4/0.3")
COMPLETED = SIX[:4] + ("C4/0.3",)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def state_of(row: dict) -> dict:
    return json.loads(row["requests"]["compact_v2"]["messages"][1]["content"])["state"]


def _source_from_gate(old: dict) -> dict:
    value = copy.deepcopy(old["audit_checks"]["source_behavior"])
    return {
        "independent_level": value["independent_level"],
        "score_unique": value["score_unique"],
        "anchor_matches": value["anchor_matches"],
        "evidence": value["evidence"],
        "legacy_task_frame": value.get("task_frame"),
        "behavior_v4": None,
        "legacy_source_record_sha256": sha256_json(value),
    }


def _source_from_stage(packet: dict, state: dict) -> dict:
    answer = packet["answer"]
    stage = answer["stage2"]
    level = stage["level"]
    validation = answer["evidence_validation"]
    decisions = validation.get("decisions", [])
    parsed = decisions[-1]["parsed"]["source_criterion"] if decisions else None
    if not isinstance(parsed, dict) or parsed.get("level") != level:
        raise ValueError("Saved blind source lacks a matching validated raw decision")
    matches = parsed.get("anchor_matches", [])
    unique = level != "UNSCORABLE" and matches == [level] and parsed.get("criterion_status") == "coherent"
    quotes = list(dict.fromkeys(
        quote
        for requirement in stage.get("requirements", [])
        for quote in requirement.get("response_quotes", [])
    ))
    if any(quote not in state["source_agent_reply"]["text"] for quote in quotes):
        raise ValueError("Saved blind-source quote is not literal")
    return {
        "independent_level": level,
        "score_unique": unique,
        "anchor_matches": matches,
        "evidence": [{"turn_id": state["source_agent_reply"]["turn_id"], "quote": q} for q in quotes],
        "legacy_task_frame": parsed.get("task_frame"),
        "behavior_v4": None,
        "legacy_source_record_sha256": sha256_json(answer),
    }


def _intervention_from_gate(old: dict) -> dict:
    value = copy.deepcopy(old["audit_checks"]["intervention_quality"])
    return {"quality": value["quality"], "legacy_intervention_sha256": sha256_json(value)}


def _intervention_from_stage(packet: dict) -> dict:
    answer = packet["answer"]["answer"]
    quality = {
        "instruction_aligned": answer["instruction_aligned"],
        "no_unsupported_context_premise": not answer["unsupported_context_premise"],
        "no_conclusion_prescribed": not answer["conclusion_prescribed"],
    }
    return {"quality": quality, "legacy_intervention_sha256": sha256_json(packet["answer"])}


def _quality(packet: dict) -> dict:
    return copy.deepcopy(packet["answer"])


def _provenance(row: dict, state: dict, old_dir: Path, app_provenance: dict) -> dict:
    old_provenance = json.loads((old_dir / "provenance.json").read_text())
    stage_files = {
        name: old_dir / f"{name}.json"
        for name in ("quality", "applicability", "blind_source", "instruction")
        if (old_dir / f"{name}.json").exists()
    }
    return {
        "stage_prompt_sha256": old_provenance["stage_prompt_sha256"],
        "stage_record_sha256": {name: file_sha256(path) for name, path in stage_files.items()},
        "json_schema_sha256": {
            "behavior": file_sha256(HERE / "contracts/public_evidence_contract_v4.schema.json"),
            "applicability": file_sha256(HERE / "contracts/applicability_contract_v3_3_1.schema.json"),
        },
        "behavior_contract": {"version": BEHAVIOR_CONTRACT_VERSION,
                              "hash": file_sha256(HERE / "contracts/public_evidence_contract_v4.schema.json")},
        "applicability_contract": {"version": APPLICABILITY_CONTRACT_VERSION,
                                   "hash": file_sha256(HERE / "contracts/applicability_contract_v3_3_1.schema.json")},
        "legacy_adapter": {"version": ADAPTER_VERSION, "hash": adapter_sha256(),
                           "migration": app_provenance},
        "final_assembler": {"version": ASSEMBLER_VERSION, "hash": assembler_sha256()},
        "rubric_content_sha256": sha256_json(state["candidate_rubric"]),
        "strict_prefix_sha256": sha256_json({
            "case_context": state["case_context"],
            "dialogue_before_target_turn": state["dialogue_before_target_turn"],
        }),
        "target_reply_sha256": sha256_json(state["source_agent_reply"]),
        "model_settings": old_provenance["model_settings"],
        "source_record_tag": row["tag"],
        "source_outcome_used_only_by_local_assembly": row["source_outcome"],
        "existing_v3_3_records_read_only": True,
    }


def replay_one(row: dict) -> dict:
    tag = row["tag"]
    old_dir = OLD_RESULTS / tag.replace("/", "_")
    state = state_of(row)
    app_packet = json.loads((old_dir / "applicability.json").read_text())
    app, app_provenance = adapt_applicability(
        app_packet, visible_prefix=state["dialogue_before_target_turn"]
    )
    gate_path = old_dir / "gate_result.json"
    if gate_path.exists():
        old_gate = json.loads(gate_path.read_text())
        source = _source_from_gate(old_gate)
        intervention = _intervention_from_gate(old_gate)
    else:
        source = _source_from_stage(json.loads((old_dir / "blind_source.json").read_text()), state)
        intervention = _intervention_from_stage(json.loads((old_dir / "instruction.json").read_text()))
    quality = _quality(json.loads((old_dir / "quality.json").read_text()))
    provenance = _provenance(row, state, old_dir, app_provenance)
    result = assemble_v3_3_1(
        source_outcome=row["source_outcome"],
        state=state,
        quality=quality,
        applicability=app,
        source_behavior=source,
        intervention=intervention,
        provenance=provenance,
    )
    write_json(RESULTS / tag.replace("/", "_") / "gate_result.json", result)
    if tag == "C4/0.3":
        write_json(RESULTS / tag.replace("/", "_") / "legacy_conversion_provenance.json", app_provenance)
    return result


def build_c3_preview(row: dict) -> dict:
    state = state_of(row)
    old_error = json.loads((OLD_RESULTS / "C3_0.0/execution_error.json").read_text())
    old_request = old_error["api_audit"]["requests"][0]
    schema = json.loads((HERE / "contracts/public_evidence_contract_v4.schema.json").read_text())
    response_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["level", "criterion_status", "anchor_matches", "operation_evidence", "reason"],
        "properties": {
            "level": {"enum": ["0", "1", "2", "UNSCORABLE"]},
            "criterion_status": {"enum": ["coherent", "ambiguous", "contradictory"]},
            "anchor_matches": {"type": "array", "items": {"enum": ["0", "1", "2"]}, "uniqueItems": True},
            "operation_evidence": schema,
            "reason": {"type": "string", "minLength": 1},
        },
    }
    payload = {
        "purpose": "failed_blind_source_stage_only",
        "provider": "openai_direct",
        "model": "gpt-6-luna",
        "messages": [
            {"role": "system", "content": (HERE / "prompts/blind_source_v3_3_1.txt").read_text()},
            old_request["messages"][1],
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "public_evidence_contract_v4_response",
            "strict": True,
            "schema": response_schema,
        }},
        "decoding": {
            "temperature": "provider_default_structured_decision",
            "max_output_tokens": 4096,
        },
        "technical_retry_policy": {
            "initial_attempts": 1,
            "maximum_additional_attempts": 1,
            "allowed_reasons": ["transport_error", "provider_schema_error", "local_schema_error"],
            "semantic_disagreement_is_retryable": False,
            "preserve_every_attempt": True,
        },
        "frozen_hashes": {
            "system_prompt_sha256": file_sha256(HERE / "prompts/blind_source_v3_3_1.txt"),
            "json_schema_sha256": sha256_json(response_schema),
            "source_prefix_sha256": sha256_json({
                "case_context": state["case_context"],
                "dialogue_before_target_turn": state["dialogue_before_target_turn"],
            }),
            "target_reply_sha256": sha256_json(state["source_agent_reply"]),
            "rubric_sha256": sha256_json(state["candidate_rubric"]),
        },
        "estimated_usage": {
            "basis": "previous failed stage used 5,094 prompt tokens and 2,497 completion tokens on its first attempt",
            "expected_prompt_tokens": 5100,
            "maximum_output_tokens": 4096,
            "currency_estimate": "not computed: no pinned direct-API price table is stored with the frozen run",
        },
        "not_executed": True,
    }
    payload["request_payload_sha256"] = sha256_json({k: v for k, v in payload.items()
                                                      if k != "request_payload_sha256"})
    write_json(PREVIEW / "C3_0.0_blind_source_request.json", payload)
    return payload


def main() -> None:
    rows = {row["tag"]: row for row in json.loads(FROZEN.read_text())["sample"]}
    baseline = {
        "v32_manifest": file_sha256(V32_MANIFEST),
        "v33_runner": file_sha256(OLD_ROOT / "run_gate_v33.py"),
        "v33_results_manifest": file_sha256(OLD_ROOT / "results/manifest.json"),
        "v33_offline_regression": file_sha256(OLD_ROOT / "offline_regression.json"),
    }
    results = {tag: replay_one(rows[tag]) for tag in COMPLETED}
    preview = build_c3_preview(rows["C3/0.0"])
    c3_record = {
        "protocol": PROTOCOL,
        "assembler_version": ASSEMBLER_VERSION,
        "assembler_sha256": assembler_sha256(),
        "decision": "execution_error",
        "passed": False,
        "source_level": None,
        "failure_codes": [],
        "reason": "Stored blind-source execution error remains unresolved; the v4 request is frozen but not executed.",
        "request_preview_sha256": preview["request_payload_sha256"],
    }
    write_json(RESULTS / "C3_0.0/execution_error.json", c3_record)
    results["C3/0.0"] = c3_record
    after = {
        "v32_manifest": file_sha256(V32_MANIFEST),
        "v33_runner": file_sha256(OLD_ROOT / "run_gate_v33.py"),
        "v33_results_manifest": file_sha256(OLD_ROOT / "results/manifest.json"),
        "v33_offline_regression": file_sha256(OLD_ROOT / "offline_regression.json"),
    }
    manifest = {
        "schema": "gate_v3_3_1_offline_replay_v1",
        "protocol": PROTOCOL,
        "assembler_version": ASSEMBLER_VERSION,
        "assembler_sha256": assembler_sha256(),
        "records": [{
            "tag": tag,
            "decision": results[tag]["decision"],
            "source_level": results[tag]["source_level"],
            "failure_codes": results[tag]["failure_codes"],
            "result_path": str(RESULTS / tag.replace("/", "_") /
                               ("execution_error.json" if tag == "C3/0.0" else "gate_result.json")),
        } for tag in SIX],
        "single_assembler_for_all_six": all(
            result["assembler_version"] == ASSEMBLER_VERSION
            and (result.get("provenance", {}).get("final_assembler", {}).get("hash", assembler_sha256()) == assembler_sha256())
            for result in results.values()
        ),
        "original_hashes_before": baseline,
        "original_hashes_after": after,
        "originals_unchanged": baseline == after,
        "full29_authorized": False,
        "active_promotion_authorized": False,
    }
    write_json(HERE / "results/manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
