#!/usr/bin/env python3
"""Execute frozen C3 v4.1 once; persist provider output before validation."""
from __future__ import annotations

import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator
from openai import OpenAI

from compact_quality_gate_v3_3_1 import GateExecutionError
from public_evidence_v4_1 import validate_wire_behavior

HERE = Path(__file__).resolve().parent
SECRET_FILE = Path.home() / ".config/sherpa/openai.env"
PREVIEW = HERE / "api_preview/C3_0.0_blind_source_request_v4_1.json"
CALL_DIR = HERE / "results/candidates/C3_0.0/api_call_v4_1"
RAW_PROVIDER = CALL_DIR / "provider_response_raw.json"
VALIDATED = CALL_DIR / "validated_response.json"
ERROR = CALL_DIR / "execution_error.json"


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def private_key() -> str:
    for line in SECRET_FILE.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY=") and line.split("=", 1)[1].strip():
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENAI_API_KEY is missing from the private env file")


def target_reply_from_user_message(message: str) -> str:
    """Read the frozen structured user payload without text-marker guessing."""
    try:
        payload = json.loads(message)
        reply = payload["state"]["target_agent_reply"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GateExecutionError("Frozen request lacks state.target_agent_reply") from exc
    if not isinstance(reply, str) or not reply.strip():
        raise GateExecutionError("Frozen state.target_agent_reply is empty or non-string")
    return reply


def main() -> None:
    if RAW_PROVIDER.exists() or VALIDATED.exists():
        raise RuntimeError("A v4.1 provider response already exists; refusing a repeated semantic call")
    preview = json.loads(PREVIEW.read_text())
    schema = preview["response_format"]["json_schema"]["schema"]
    completion = OpenAI(api_key=private_key(), max_retries=0, timeout=300.0).chat.completions.create(
        model=preview["model"],
        messages=preview["messages"],
        max_completion_tokens=preview["decoding"]["max_output_tokens"],
        response_format=preview["response_format"],
    )

    # This write deliberately precedes finish-reason, JSON, schema, and local
    # semantic validation so every provider response remains auditable.
    raw = {
        "schema": "c3_blind_source_provider_response_v4_1",
        "request_payload_sha256": preview["request_payload_sha256"],
        "provider_response": completion.model_dump(mode="json"),
        "api_key_stored": False,
    }
    write_atomic(RAW_PROVIDER, raw)

    try:
        choice = completion.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise GateExecutionError(f"Incomplete completion: finish_reason={choice.finish_reason!r}")
        parsed = json.loads(choice.message.content)
        Draft202012Validator(schema).validate(parsed)
        behavior = validate_wire_behavior(
            parsed["operation_evidence"],
            target_reply=target_reply_from_user_message(preview["messages"][1]["content"]),
        )
        if parsed["level"] == "UNSCORABLE":
            if parsed["criterion_status"] == "coherent":
                raise GateExecutionError("UNSCORABLE contradicts coherent criterion status")
        elif not (
            parsed["criterion_status"] == "coherent"
            and parsed["anchor_matches"] == [parsed["level"]]
        ):
            raise GateExecutionError("Numeric level lacks a unique coherent anchor")
        result = {
            "schema": "c3_blind_source_validated_v4_1",
            "request_payload_sha256": preview["request_payload_sha256"],
            "response_id": completion.id,
            "finish_reason": choice.finish_reason,
            "usage": completion.usage.model_dump() if completion.usage else None,
            "parsed": parsed,
            "normalized_operation_evidence": behavior,
            "raw_provider_response": str(RAW_PROVIDER),
        }
        write_atomic(VALIDATED, result)
        print(json.dumps({"status": "validated", "result": str(VALIDATED)}, indent=2))
    except Exception as exc:
        write_atomic(ERROR, {
            "schema": "c3_blind_source_execution_error_v4_1",
            "status": "execution_error",
            "request_payload_sha256": preview["request_payload_sha256"],
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "raw_provider_response_preserved": str(RAW_PROVIDER),
            "converted_to_score_or_gate_decision": False,
        })
        raise


if __name__ == "__main__":
    main()
