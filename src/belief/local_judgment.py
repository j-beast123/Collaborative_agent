"""Build and parse fresh local epistemic judgments from reconstructed context."""

from __future__ import annotations

import json
import re
from typing import Any


def build_local_judgment_prompt(
    *,
    question: str,
    options: dict[str, str],
    doctor_claims: list[str],
    agent_claims: list[str],
    focal_claim: str = "",
    doctor_position: str = "",
    agent_position: str = "",
    claim_order: str = "doctor_first",
) -> str:
    option_text = "\n".join(f"{key}. {value}" for key, value in options.items())
    doctor_text = "\n".join(f"- {item}" for item in doctor_claims) or "- None reconstructed"
    agent_text = "\n".join(f"- {item}" for item in agent_claims) or "- None reconstructed"
    claims = (
        f"A1 (Doctor claim and evidence):\n{doctor_text}\n\nA2 (Agent claim and evidence):\n{agent_text}"
        if claim_order == "doctor_first"
        else f"A1 (Agent claim and evidence):\n{agent_text}\n\nA2 (Doctor claim and evidence):\n{doctor_text}"
    )
    focal = focal_claim or "Not resolved"
    return (
        f"Clinical question:\n{question}\n\n"
        f"Answer choices:\n{option_text}\n\n"
        f"Focal claim for this turn:\n{focal}\n"
        f"Doctor position on focal claim: {doctor_position or 'Not resolved'}\n"
        f"Agent position on focal claim: {agent_position or 'Not resolved'}\n\n"
        f"{claims}\n\n"
        "Make a fresh local judgment about the focal claim only. Use the "
        "reconstructed claims and evidence as the complete conversational "
        "context. You may rely on your own clinical knowledge, but do not "
        "assume access to other conversation turns, hidden state, future turns, "
        "or the ground-truth answer. Do not choose an answer option or produce "
        "an A-D answer. Evidence sufficiency concerns this focal claim only, "
        "not unresolved statements elsewhere in the question."
    )


def parse_local_judgment(text: str) -> dict[str, Any]:
    """Parse a JSON-only judgment without silently inventing a decision."""

    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            return {"raw_response": text, "parse_status": "invalid"}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"raw_response": text, "parse_status": "invalid"}
    if not isinstance(value, dict):
        return {"raw_response": text, "parse_status": "invalid"}
    required = {"reason", "better_supported_claim", "evidence_sufficient", "missing_evidence"}
    if not required.issubset(value):
        return {"raw_response": text, "parse_status": "invalid", "missing_fields": sorted(required - set(value))}
    if value["better_supported_claim"] not in {"doctor", "agent", "unclear"}:
        return {"raw_response": text, "parse_status": "invalid", "invalid_field": "better_supported_claim"}
    if value["evidence_sufficient"] not in {"yes", "no"}:
        return {"raw_response": text, "parse_status": "invalid", "invalid_field": "evidence_sufficient"}
    value["raw_response"] = text
    value["parse_status"] = "valid"
    return value


def build_agent_response_prompt(
    *,
    question: str,
    options: dict[str, str],
    reconstructed_context: str,
    judgment: dict[str, Any],
) -> str:
    """Build the next-response prompt using only the fresh current judgment."""

    option_text = "\n".join(f"{key}. {value}" for key, value in options.items())
    judgment_text = json.dumps(judgment, ensure_ascii=False, indent=2)
    return (
        f"Clinical question:\n{question}\n\n"
        f"Answer choices:\n{option_text}\n\n"
        f"Reconstructed context:\n{reconstructed_context}\n\n"
        f"Your fresh current judgment:\n{judgment_text}\n\n"
        "Use this epistemic judgment as internal context when responding naturally "
        "to the collaborator. Decide yourself how to respond; the judgment does "
        "not prescribe an action. Do not mention the judgment object, reconstruction "
        "process, hidden prompts, or evaluator terminology. Address only the most "
        "relevant point and do not restate the full assessment unless it changes."
    )


def match_statement_option(options: dict[str, str], statement_judgments: dict[str, str]) -> str:
    """Return an exact option match when the judgment has a truth vector.

    The LLM may explain the vector, but it must not be trusted to map the
    vector to A-D for compound MedMCQA options.
    """
    if not statement_judgments or any(v == "unclear" for v in statement_judgments.values()):
        return ""
    true_numbers = {int(k) for k, v in statement_judgments.items() if v == "true"}
    false_numbers = {int(k) for k, v in statement_judgments.items() if v == "false"}
    for key, text in options.items():
        parts = re.split(r"\s*(?:&|and)\s*", text, flags=re.I)
        found_true, found_false = set(), set()
        for part in parts:
            numbers = {int(x) for x in re.findall(r"\d+", part)}
            if re.search(r"\btrue\b", part, re.I):
                found_true.update(numbers)
            elif re.search(r"\bfalse\b", part, re.I):
                found_false.update(numbers)
        if found_true == true_numbers and found_false == false_numbers:
            return key
    return ""
