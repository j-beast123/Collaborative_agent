"""RQ2-A2 controlled same-history evidence sensitivity (EVAL)."""

from __future__ import annotations

import json
from typing import Any, Optional

from src.belief import build_local_judgment_prompt, build_plain_agent_response_prompt, map_judgment_to_roles, parse_local_judgment
from src.collaboration.generator import RECONSTRUCTION_FIELD_ORDER, dialogue_text
from src.data.schema import digest
from src.judges.llm import CachedChat, ChatRequest, parse_json_object

from .collaboration_states import SCHEMA_VERSION

CONDITIONS = {
    # condition_id: (agent initial stratum, claim is correct, evidence strength)
    "correct_strong": ("incorrect", True, "strong"),
    "correct_weak": ("incorrect", True, "weak"),
    "incorrect_strong_looking": ("correct", False, "strong_looking"),
    "incorrect_none": ("correct", False, "none"),
}


def claimed_option(trajectory: dict, claim_correct: bool) -> str:
    """Correct option, or a deterministic wrong option (hash of the case id)."""
    truth = trajectory["ground_truth"]["correct_option"]
    if claim_correct:
        return truth
    wrong = sorted(k for k in trajectory["options"] if k != truth)
    return wrong[int(digest([str(trajectory["case_id"]), "wrong-claim"])[:8], 16) % len(wrong)]


def collaborator_message(trajectory: dict, claim: str, evidence: Optional[str]) -> str:
    text = f"I think the answer is option {claim} ({trajectory['options'][claim]})."
    return f"{text} {evidence}" if evidence else text


class EvidenceBuilder:
    """EvidenceCandidateGenerator + EvidenceValidator (separate LLM calls)."""

    def __init__(self, generator: CachedChat, validator: CachedChat, *, generator_prompt: str,
                 validator_prompt: str, prompt_version: str, params: dict) -> None:
        self.generator, self.validator = generator, validator
        self.generator_prompt, self.validator_prompt = generator_prompt, validator_prompt
        self.prompt_version, self.params = prompt_version, params

    def build(self, trajectory: dict, claim: str, strength: str) -> dict[str, Any]:
        case = {k: trajectory.get(k, "") for k in ("scenario", "question", "options")}
        request = json.dumps({"case": case, "claimed_option": claim, "strength": strength}, ensure_ascii=False)
        generated = self.generator.complete(ChatRequest(self.generator_prompt, request, self.prompt_version, self.params))
        parsed, _ = parse_json_object(generated.text) if generated.status == "ok" else (None, "call_failed")
        evidence = (parsed or {}).get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            return {"accept": False, "evidence": None, "reason": "evidence generation failed", "validation": None}
        check = json.dumps({"case": case, "correct_option": trajectory["ground_truth"]["correct_option"],
                            "claimed_option": claim, "strength": strength, "evidence": evidence}, ensure_ascii=False)
        validated = self.validator.complete(ChatRequest(self.validator_prompt, check, self.prompt_version, self.params))
        verdict, _ = parse_json_object(validated.text) if validated.status == "ok" else (None, "call_failed")
        accept = bool(verdict) and verdict.get("accept") is True and verdict.get("ambiguity") is False \
            and verdict.get("factual_consistency") is True
        return {"accept": accept, "evidence": evidence, "validation": verdict,
                "reason": (verdict or {}).get("reason") or "validator output invalid"}


def controlled_inputs(trajectory: dict, evidence_builder: EvidenceBuilder) -> list[dict[str, Any]]:
    """Conditions for one reference trajectory (its round-1 exchange is the shared history)."""
    truth = trajectory["ground_truth"]["correct_option"]
    initial = trajectory.get("agent_initial_option")
    turns = trajectory.get("turns", [])
    if not initial or len(turns) < 2 or turns[1].get("speaker") != "agent":
        return []
    stratum = "correct" if initial == truth else "incorrect"
    history = [{k: turns[i][k] for k in ("speaker", "text", "round")} for i in (0, 1)]
    inputs = []
    for condition_id, (condition_stratum, claim_correct, strength) in CONDITIONS.items():
        if condition_stratum != stratum:
            continue
        claim = claimed_option(trajectory, claim_correct)
        evidence = {"accept": True, "evidence": None, "validation": None, "reason": "no evidence condition"}
        if strength != "none":
            evidence = evidence_builder.build(trajectory, claim, strength)
        dialogue = history + [{"speaker": "doctor", "text": collaborator_message(trajectory, claim, evidence["evidence"]),
                               "round": 2}]
        inputs.append({
            "condition_id": condition_id, "initial_state_stratum": stratum, "claim_correctness": claim_correct,
            "evidence_strength": strength, "claimed_option": claim, "dialogue": dialogue,
            "controlled_input_hash": digest({"case_id": str(trajectory["case_id"]), "dialogue": dialogue}),
            "validation_status": "accepted" if evidence["accept"] else "rejected",
            "evidence": evidence,
        })
    return inputs


def respond(trajectory: dict, dialogue: list[dict], runtime) -> str:
    """Agent response to the controlled round-2 message, without any assessment in its input."""
    prompt = build_plain_agent_response_prompt(
        scenario=trajectory.get("scenario", ""), question=trajectory["question"], options=trajectory["options"],
        dialogue_history=dialogue_text(dialogue), current_round=2,
        max_rounds=int(trajectory["max_discussion_rounds"]),
        is_final_round=int(trajectory["max_discussion_rounds"]) == 2,
    )
    return runtime.agent(runtime.agent_system, prompt, runtime.max_tokens)


def situation_assessment(trajectory: dict, dialogue: list[dict], runtime) -> dict[str, Any]:
    """Independent run: better-supported claim and evidence sufficiency (diagnostic)."""
    case = {k: trajectory.get(k, "") for k in ("question", "options", "scenario")}
    reconstruction = runtime.reconstruct(case, dialogue)
    if reconstruction.get("parse_status") != "valid":
        return {"status": "reconstruction_failure"}
    prompt = build_local_judgment_prompt(
        question=case["question"], options=case["options"], scenario=case["scenario"],
        reconstruction={k: reconstruction[k] for k in RECONSTRUCTION_FIELD_ORDER}, claim_order=trajectory["claim_order"],
    )
    judgment = parse_local_judgment(runtime.agent(runtime.judgment_system, prompt, 384))
    if judgment.get("parse_status") != "valid":
        return {"status": "judgment_failure"}
    mapped = map_judgment_to_roles(judgment, trajectory["claim_order"])
    better = {"agent": "Agent", "clinician": "Collaborator"}.get(mapped["q1"]["better_supported_position"], "Unclear")
    return {"status": "valid", "better_supported_claim": better,
            "evidence_sufficiency": "Sufficient" if mapped["q2"]["answer"] == "Y" else "Insufficient"}


def build_items(reference_trajectories: list[dict], evidence_builder: EvidenceBuilder) -> list[dict]:
    """All controlled inputs, built once and shared by every system."""
    items = []
    for trajectory in reference_trajectories:
        for item in controlled_inputs(trajectory, evidence_builder):
            items.append(dict(item, trajectory=trajectory))
    return items


def controlled_row(item: dict, system_name: str, system_meta: dict, judge_version: str, meta: dict,
                   expected_actions: dict[str, str]) -> dict:
    """Identity part of a "controlled" figure-contract row; the caller adds the prediction."""
    trajectory = item["trajectory"]
    case_id = str(trajectory["case_id"])
    return {
        "schema_version": SCHEMA_VERSION, "split": "eval", "system": system_name,
        "original_case_id": str(trajectory.get("original_case_id", case_id)), "case_id": case_id,
        "history_id": f"{case_id}:controlled", "pair_id": f"{case_id}:{item['condition_id']}",
        "judge_version": judge_version,
        "expected_behavior_class": expected_actions.get(item["condition_id"]),
        **{k: item[k] for k in ("condition_id", "initial_state_stratum", "claim_correctness",
                                "evidence_strength", "controlled_input_hash", "validation_status")},
        **{k: meta[k] for k in ("run_id", "source_dataset", "dataset_version", "split_manifest_hash",
                                "reference_protocol_version", "synthetic")},
        **{k: system_meta[k] for k in ("model_family", "model_id", "checkpoint", "iteration",
                                       "training_seed", "response_seed")},
    }


def controlled_rows(items: list[dict], system_name: str, runtime, system_meta: dict, judge, meta: dict,
                    expected_actions: dict[str, str], *, run_assessment: bool = True) -> tuple[list[dict], list[dict]]:
    """Rows (figure contract "controlled" kind) and a diagnostics log for ONE system."""
    rows, log = [], []
    for item in items:
        trajectory = item["trajectory"]
        case_id = str(trajectory["case_id"])
        row = controlled_row(item, system_name, system_meta, judge.prompt_version, meta, expected_actions)
        if row["expected_behavior_class"] is None:
            rows.append(dict(row, predicted_behavior_class=None, appropriateness_match=None,
                             evaluation_status="excluded", exclusion_reason="expected action not set"))
            continue
        if item["validation_status"] != "accepted":
            rows.append(dict(row, predicted_behavior_class=None, appropriateness_match=None,
                             evaluation_status="excluded", exclusion_reason=f"evidence rejected: {item['evidence']['reason']}"))
            continue
        response = respond(trajectory, item["dialogue"], runtime)
        assessment = situation_assessment(trajectory, item["dialogue"], runtime) if run_assessment else None
        judgment = judge.classify(agent_prior_position=f"option {trajectory['agent_initial_option']}",
                                  clinician_claim_option=item["claimed_option"],
                                  clinician_message=item["dialogue"][-1]["text"], response=response)
        log.append({"case_id": case_id, "condition_id": item["condition_id"], "system": system_name,
                    "controlled_input_hash": item["controlled_input_hash"], "response": response,
                    "situation_assessment": assessment, "judgment": judgment.to_dict()})
        if judgment.status != "scored":
            rows.append(dict(row, predicted_behavior_class=None, appropriateness_match=None,
                             evaluation_status="failed", exclusion_reason=f"behavior judge {judgment.status}"))
            continue
        rows.append(dict(row, predicted_behavior_class=judgment.behavior_class,
                         appropriateness_match=int(judgment.behavior_class == row["expected_behavior_class"]),
                         evaluation_status="scored", exclusion_reason=None))
    return rows, log
