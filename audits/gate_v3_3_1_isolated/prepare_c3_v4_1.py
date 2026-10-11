#!/usr/bin/env python3
"""Freeze the nonredundant C3 v4.1 request without calling an API."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from compact_quality_gate_v3_3_1 import file_sha256, sha256_json

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "api_preview/C3_0.0_blind_source_request_technical_retry.json"
OUTPUT = HERE / "api_preview/C3_0.0_blind_source_request_v4_1.json"


def main() -> None:
    prior = json.loads(SOURCE.read_text())
    behavior = json.loads((HERE / "contracts/public_evidence_contract_v4_1.schema.json").read_text())
    response_schema = copy.deepcopy(prior["response_format"]["json_schema"]["schema"])
    response_schema["properties"]["operation_evidence"] = behavior
    response_schema["properties"]["anchor_matches"].pop("uniqueItems", None)

    payload = copy.deepcopy(prior)
    payload["messages"][0]["content"] = (HERE / "prompts/blind_source_v3_3_1_1.txt").read_text()
    payload["response_format"]["json_schema"].update(
        name="public_evidence_contract_v4_1_response", schema=response_schema
    )
    payload["frozen_hashes"]["system_prompt_sha256"] = file_sha256(
        HERE / "prompts/blind_source_v3_3_1_1.txt"
    )
    payload["frozen_hashes"]["json_schema_sha256"] = sha256_json(response_schema)
    payload["technical_retry_policy"] = {
        "initial_attempts": 1,
        "maximum_additional_attempts": 0,
        "semantic_disagreement_is_retryable": False,
        "preserve_every_provider_response_before_local_validation": True,
    }
    payload["contract_revision"] = {
        "version": "public_evidence_contract_v4_1",
        "changes": [
            "observed_operations is the sole model-produced evidence ledger",
            "flat evidence_quotes is derived locally",
            "provider response is persisted before parsing and local validation",
        ],
    }
    payload["supersedes_failed_request"] = prior["request_payload_sha256"]
    payload["not_executed"] = True
    payload.pop("technical_retry_of", None)
    payload.pop("technical_correction", None)
    payload.pop("request_payload_sha256", None)
    payload["request_payload_sha256"] = sha256_json(payload)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "request_payload_sha256": payload["request_payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
