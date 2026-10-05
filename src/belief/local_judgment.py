"""Prompt builders and strict parsers for turn-local epistemic judgments."""

from __future__ import annotations

import json
from typing import Any


def _json_only(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    return json.loads(stripped)


def build_local_judgment_prompt(*, question: str, options: dict[str, str], doctor_claims: list[str] | None = None, agent_claims: list[str] | None = None, focal_claim: str = "", doctor_position: str = "", agent_position: str = "", reconstruction: dict[str, Any] | None = None, scenario: str = "", claim_order: str = "doctor_first") -> str:
    source = reconstruction or {"focal_claim": focal_claim, "clinician_current_position": doctor_position, "agent_current_position": agent_position, "clinician_claims_so_far": doctor_claims or [], "agent_claims_so_far": agent_claims or []}
    # Deliberately remove speaker identity from the judgment model's input.
    if claim_order == "doctor_first":
        first, second = "clinician", "agent"
    else:
        first, second = "agent", "clinician"
    def pick(prefix: str) -> dict[str, Any]:
        return {
            "position": source.get(f"{prefix}_current_position", ""),
            "claims": source.get(f"{prefix}_claims_so_far", []),
            "evidence": source.get(f"{prefix}_evidence_so_far", []),
        }
    state = {
        "focal_claim": source.get("focal_claim", ""),
        "A1": pick(first),
        "A2": pick(second),
        "shared_evidence": source.get("shared_evidence", []),
        "resolved_points": source.get("resolved_points", []),
        "unresolved_points": source.get("unresolved_points", []),
        "unsupported_case_details": source.get("unsupported_case_details", []),
    }
    option_text = "\n".join(f"{key}. {value}" for key, value in options.items())
    return ("Make a fresh, stateless turn-local epistemic judgment.\n\n"
        f"Scenario:\n{scenario or 'No separate patient vignette is provided.'}\n\n"
        f"Question:\n{question}\n\nAnswer choices:\n{option_text}\n\n"
        "Current cumulative reconstruction (the complete conversational context):\n"
        f"{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
        "Do not use raw dialogue, previous reconstructions, previous judgments, future turns, ground truth, correctness, final outcome, or rubric information.\n"
        "The two displayed positions are anonymized as A1 and A2. Their participant "
        "identities are intentionally withheld. Evaluate only claims and supporting evidence.\n"
        "Q1: Which displayed position is better supported on the focal claim: A1, A2, aligned, or unclear? Use aligned when both positions substantively support the same claim; do not choose a side merely because one response is more detailed.\n"
        "Q2: Is the currently available evidence, together with established clinical knowledge, sufficient to make, maintain, or revise a decision on the focal claim?\n\n"
        "Answer each independently. Give a separate concise, evidence-grounded reason for each. Do not solve the multiple-choice item or prescribe an action.\n"
        'Return JSON only: {"q1":{"answer":"A1"|"A2"|"aligned"|"unclear","reason":"..."},"q2":{"answer":"Y"|"N","reason":"..."}}')


def map_judgment_to_roles(judgment: dict[str, Any], claim_order: str) -> dict[str, Any]:
    """Expose role semantics to the response agent without leaking them to the judge."""
    mapped = {"q1": dict(judgment["q1"]), "q2": dict(judgment["q2"])}
    display_to_role = {"A1": "clinician", "A2": "agent"} if claim_order == "doctor_first" else {"A1": "agent", "A2": "clinician"}
    display_answer = mapped["q1"]["answer"]
    mapped["q1"]["display_answer"] = display_answer
    mapped["q1"]["better_supported_position"] = display_to_role.get(display_answer, display_answer)
    return mapped


def parse_local_judgment(text: str) -> dict[str, Any]:
    try:
        value = _json_only(text)
    except (json.JSONDecodeError, IndexError):
        return {"raw_response": text, "parse_status": "invalid"}
    if not isinstance(value, dict) or set(value) != {"q1", "q2"}:
        return {"raw_response": text, "parse_status": "invalid"}
    for name in ("q1", "q2"):
        item = value[name]
        if not isinstance(item, dict) or set(item) != {"answer", "reason"}:
            return {"raw_response": text, "parse_status": "invalid", "invalid_field": name}
        allowed = {"A1", "A2", "aligned", "unclear"} if name == "q1" else {"Y", "N"}
        if item["answer"] not in allowed:
            return {"raw_response": text, "parse_status": "invalid", "invalid_field": f"{name}.answer"}
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            return {"raw_response": text, "parse_status": "invalid", "invalid_field": f"{name}.reason"}
    value["raw_response"] = text
    value["parse_status"] = "valid"
    return value


def build_initial_agent_response_prompt(*, scenario: str, question: str, options: dict[str, str], clinician_message: str) -> str:
    option_text = "\n".join(f"{key}. {value}" for key, value in options.items())
    return (f"Scenario:\n{scenario or 'No separate patient vignette is provided.'}\n\n"
        f"Question:\n{question}\n\nAnswer choices:\n{option_text}\n\n"
        f"Clinician's opening message:\n{clinician_message}\n\n"
        "Respond naturally as a clinical collaborator. Use only the source problem, this message, and established domain knowledge. Do not invent patient or event details. Do not mention reconstruction or judgment.")


def build_plain_agent_response_prompt(*, scenario: str, question: str, options: dict[str, str], dialogue_history: str, current_round: int, max_rounds: int, is_final_round: bool = False) -> str:
    option_text = "\n".join(f"{key}. {value}" for key, value in options.items())
    final = "This is the final allowed round; state one final option explicitly.\n" if is_final_round else ""
    return (f"Scenario:\n{scenario or 'No separate patient vignette is provided.'}\n\n"
        f"Question:\n{question}\n\nAnswer choices:\n{option_text}\n\n"
        f"Complete visible dialogue history:\n{dialogue_history}\n\n"
        f"This is collaboration round {current_round} of {max_rounds}.\n{final}"
        "Respond directly to the clinician's latest message as a clinical collaborator. "
        "Use the full visible dialogue and established domain knowledge. Do not mention "
        "reconstruction, judgment, Q1, Q2, rubrics, or hidden prompts. Do not invent "
        "patient or event details. Keep the response concise, no more than five sentences.")


def build_agent_response_prompt(*, question: str, options: dict[str, str], reconstructed_context: str, judgment: dict[str, Any], dialogue_history: str = "", scenario: str = "", current_round: int | None = None, max_rounds: int | None = None, is_final_round: bool = False) -> str:
    option_text = "\n".join(f"{key}. {value}" for key, value in options.items())
    final = "This is the final allowed round; state one final option explicitly.\n" if is_final_round else ""
    return (f"Scenario:\n{scenario or 'No separate patient vignette is provided.'}\n\n"
        f"Question:\n{question}\n\nAnswer choices:\n{option_text}\n\n"
        f"Complete visible dialogue history:\n{dialogue_history or '(No previous dialogue.)'}\n\n"
        f"Current reconstructed interaction state:\n{reconstructed_context}\n\n"
        f"Private current Q1/Q2 judgment:\n{json.dumps(judgment, ensure_ascii=False, indent=2)}\n\n"
        f"This is collaboration round {current_round or '?'} of {max_rounds or '?'}.\n{final}"
        "Continue naturally. Respond directly to the clinician's latest message and use the history for continuity. Use the reconstruction and judgment only as private decision support. Do not mention Q1, Q2, labels, JSON, reconstruction, private judgment, or evaluator terminology. Do not quote or mechanically paraphrase the judgment rationales. If Q2 is N, do not present the focal decision as settled; state the relevant uncertainty or compare the unresolved alternatives directly. Speak in first person, do not call yourself the agent, do not invent details, do not use unsupported_case_details as evidence, and do not merely repeat the previous answer. Keep it concise, no more than five sentences.")
