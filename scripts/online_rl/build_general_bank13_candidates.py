#!/usr/bin/env python3
"""Build a schema-valid, general-purpose 13-rubric candidate bank.

This artifact is deliberately not a cumulative KEEP snapshot.  Candidates must
still pass source applicability, quality-gate, and rerun witness validation
before any of them may be admitted to the executable rubric bank.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.judges.rubric_schema import build_levels, validate_rubric
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cumulative_bank import content_hash

DEFAULT_PARENT = ROOT / "data/interim/medmcqa/revised_bank13_20261012/revised_bank13_snapshot.json"
DEFAULT_OUTPUT = (
    ROOT / "data/interim/medmcqa/general_bank13_candidates_v2_20261012/"
    "general_bank13_candidates.json"
)
CRITERION_VERSION = "public_behavior_mc_semantic_v3"


def rubric(name: str, user: str, state: str, required: str, minimum: str,
           completion: str, zero: str, one: str, two: str) -> dict:
    applicability = {"user_side_context": user, "agent_side_state": state}
    return {
        "name": name,
        "required_behavior": required,
        "minimum_performance": minimum,
        "completion_condition": completion,
        "criterion_version": CRITERION_VERSION,
        "applicability": applicability,
        "scores": build_levels(user, state, {"0": zero, "1": one, "2": two}),
    }


def candidates() -> list[dict]:
    clinical = "The Agent is composing its next clinical reply from the visible case and dialogue."
    return [
        rubric(
            "direct_request_fulfillment",
            "The Doctor's latest turn contains a concrete clinical question or requested task.", clinical,
            "Address the Doctor's concrete request directly.",
            "The reply answers the main request rather than only discussing adjacent facts.",
            "The request is fully answered with a clear, case-relevant response.",
            "The reply omits, evades, or contradicts the requested task.",
            "The reply addresses part of the request but leaves a material part unresolved or unclear.",
            "The reply directly and fully answers every material part of the request.",
        ),
        rubric(
            "evidence_grounded_clinical_reasoning",
            "The Doctor asks for or presents a clinical assessment that should be supported by the supplied case facts.", clinical,
            "Connect the assessment to decision-relevant facts actually present in the case or dialogue.",
            "At least one relevant supplied fact is correctly linked to the assessment.",
            "The central assessment is supported by the most discriminating supplied facts without inventing evidence.",
            "The assessment is unsupported, rests on invented facts, or misuses the supplied evidence.",
            "The reply cites a relevant fact, but the link is incomplete or misses a major conflicting fact.",
            "The reply uses the decisive supplied facts coherently and accounts for material conflicting evidence.",
        ),
        rubric(
            "candidate_comparison",
            "The visible dialogue considers two or more diagnoses, mechanisms, tests, or management options.", clinical,
            "Compare the live candidates using discriminating evidence.",
            "The reply states at least one relevant distinction between the live candidates.",
            "The reply explains which candidate is favored and why the alternatives are less consistent.",
            "The reply lists or selects candidates without a valid comparison, or uses a false distinction.",
            "The reply gives a valid distinction but incompletely weighs the main alternatives.",
            "The reply compares the main candidates on decisive features and reaches a supported preference.",
        ),
        rubric(
            "premise_error_correction",
            "The Doctor's latest turn contains a clinically consequential premise that conflicts with the visible evidence or established medical knowledge.", clinical,
            "Identify and correct the consequential premise before relying on it.",
            "The reply clearly rejects or qualifies the faulty premise and supplies a correct replacement.",
            "The reply corrects the premise and explains how the correction changes the clinical conclusion.",
            "The reply accepts, repeats, or builds its conclusion on the faulty premise.",
            "The reply signals doubt or gives a correction, but the replacement or consequence remains unclear.",
            "The reply explicitly corrects the premise, supports the correction, and updates the conclusion accordingly.",
        ),
        rubric(
            "uncertainty_calibration",
            "The visible evidence leaves meaningful uncertainty, an unresolved differential, or a stated information gap.", clinical,
            "Express uncertainty in proportion to the evidence while still making the most useful supported judgment.",
            "The reply distinguishes what is supported from what remains uncertain.",
            "The reply gives a calibrated leading judgment and identifies the specific uncertainty that could change it.",
            "The reply is unjustifiably certain, indiscriminately vague, or treats missing information as known.",
            "The reply acknowledges uncertainty but does not localize it or provide a useful supported judgment.",
            "The reply states the best-supported judgment, its confidence boundary, and the decisive unresolved issue.",
        ),
        rubric(
            "no_invented_case_facts",
            "The clinical reply must remain grounded in the finite facts supplied in the case and visible dialogue.", clinical,
            "Do not assert unprovided patient findings, test results, history, or prior decisions as facts.",
            "Any added statement is clearly framed as general knowledge, a possibility, or a proposed next step.",
            "Every case-specific assertion is traceable to the visible evidence and hypothetical additions are labeled.",
            "The reply presents one or more unprovided case-specific details as established facts.",
            "The reply is mostly grounded but includes an ambiguous or weakly implied case-specific assertion.",
            "The reply cleanly separates supplied case facts from general knowledge, hypotheses, and recommendations.",
        ),
        rubric(
            "mechanism_to_conclusion_link",
            "The Doctor asks why a finding occurs or invokes a mechanism to support a clinical conclusion.", clinical,
            "State a medically coherent causal link from the relevant mechanism to the conclusion.",
            "The reply provides a correct mechanism that is relevant to the conclusion.",
            "The reply traces the key causal steps and shows why they favor the stated conclusion.",
            "The mechanism is absent, medically incorrect, or unrelated to the conclusion.",
            "The mechanism is broadly correct but skips a key link or does not clearly support the conclusion.",
            "The reply gives a concise, correct causal chain that directly supports the clinical conclusion.",
        ),
        rubric(
            "decision_relevant_constraint_use",
            "The case supplies a decision-relevant constraint such as age, timing, anatomy, exposure, laboratory pattern, or treatment history.", clinical,
            "Use the supplied constraint when evaluating the clinical question.",
            "The reply correctly incorporates at least one decisive supplied constraint.",
            "The reply uses all materially decisive constraints and explains their effect on the conclusion.",
            "The reply ignores or contradicts a decisive supplied constraint.",
            "The reply mentions a constraint but incompletely applies it or overlooks another decisive constraint.",
            "The reply integrates the material constraints and shows how they narrow or change the conclusion.",
        ),
        rubric(
            "evidence_responsive_position_update",
            "The Agent previously expressed a position and the Doctor has since supplied a new reason, correction, or objection relevant to that position.", clinical,
            "Reassess the prior position in light of the Doctor's new contribution.",
            "The reply explicitly engages the new contribution and either revises or justifiably maintains the position.",
            "The reply updates the reasoning and conclusion to the exact extent warranted by the new contribution.",
            "The reply ignores the new contribution or repeats the prior position without addressing it.",
            "The reply acknowledges the contribution but does not resolve its effect on the prior position.",
            "The reply evaluates the contribution and makes a proportionate, clearly justified update or maintenance decision.",
        ),
        rubric(
            "objection_resolution",
            "The Doctor raises a specific objection, counterexample, or inconsistency concerning the Agent's clinical reasoning.", clinical,
            "Resolve the Doctor's specific objection with relevant reasoning.",
            "The reply addresses the objection's central claim with a clinically relevant reason.",
            "The reply determines whether the objection succeeds and reconciles it with the final conclusion.",
            "The reply bypasses, misstates, or merely dismisses the objection.",
            "The reply responds to the objection but leaves its central implication unresolved.",
            "The reply accurately resolves the objection and clearly states its effect on the conclusion.",
        ),
        rubric(
            "decision_focused_conciseness",
            "The Doctor's latest turn calls for a bounded clinical answer rather than an exhaustive review.", clinical,
            "Keep the response focused on information that changes or supports the requested decision.",
            "The main answer and its relevant support are easy to identify.",
            "Every substantive statement advances the requested decision, with no distracting detour.",
            "The main answer is obscured by irrelevant, repetitive, or contradictory material.",
            "The main answer is identifiable but contains avoidable detours or repetition.",
            "The reply is concise, coherent, and limited to the answer and decision-relevant support.",
        ),
        rubric(
            "explicit_supported_conclusion",
            "The dialogue has reached a decision point where the Agent should select, endorse, reject, or rank a clinical conclusion.", clinical,
            "State the selected conclusion explicitly and support it with the decisive reason.",
            "The reply makes an unambiguous selection and gives at least one relevant supporting reason.",
            "The selection follows from the decisive evidence and is consistent with the reply's reasoning.",
            "The reply gives no clear selection, selects inconsistently, or supplies no relevant support.",
            "The reply makes a selection, but its support is incomplete, ambiguous, or partly inconsistent.",
            "The reply states one clear conclusion and justifies it with the decisive case evidence.",
        ),
        rubric(
            "actionable_information_limit",
            "A material ambiguity or missing datum prevents a fully reliable clinical conclusion and the Doctor asks how to proceed.", clinical,
            "Name the missing information that would most change the decision and explain its directional value.",
            "The reply identifies one decision-relevant missing datum or discriminating next step.",
            "The reply identifies the highest-value missing datum and explains how each plausible result would change the conclusion.",
            "The reply claims unsupported certainty or requests information that would not resolve the material ambiguity.",
            "The reply identifies relevant missing information but does not explain how it would affect the decision.",
            "The reply names the decisive information and states how its possible results would update the clinical judgment.",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    parent = json.loads(args.parent.read_text())
    rows = []
    for item in candidates():
        problems = validate_rubric(item)
        if problems:
            raise ValueError(f"{item['name']}: {'; '.join(problems)}")
        digest = content_hash(item)
        rows.append({"candidate_id": f"general-{digest[:16]}", "content_sha256": digest, "rubric": item})
    if len(rows) != 13 or len({row["content_sha256"] for row in rows}) != 13:
        raise ValueError("Expected exactly 13 unique candidates")
    payload = {
        "protocol": "manual_general_bank_candidates_v1",
        "version": 1,
        "status": "DRAFT_REQUIRES_SOURCE_GATE_AND_RERUN_WITNESS",
        "parent_snapshot_sha256": parent["sha256"],
        "admission_requirements": [
            "source_applicability_pass",
            "public_mc_gate_v3_pass",
            "independent_rerun_witness_keep",
        ],
        "candidates": rows,
    }
    artifact = {**payload, "sha256": sha256_json(payload)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "candidates": len(rows), "sha256": artifact["sha256"]}))


if __name__ == "__main__":
    main()
