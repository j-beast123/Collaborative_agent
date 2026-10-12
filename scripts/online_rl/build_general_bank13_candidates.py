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
            "The Doctor's latest turn contains an explicit clinical question or request with a specific named target, comparison, or relationship.", clinical,
            "Answer the exact requested target, comparison, or relationship before adding adjacent reasoning.",
            "The reply gives a recognizable answer to the explicitly named target.",
            "The reply resolves every requested part and makes the answer's relationship to the named target explicit.",
            "The reply omits the named target, substitutes a different question, or contradicts the requested task.",
            "The reply answers the named target but leaves a requested comparison, relationship, or material subpart unresolved.",
            "The reply directly resolves the named target and every material requested relation or subpart.",
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
            "The Doctor explicitly identifies a particular unresolved ambiguity, conflict, or missing datum that could change the current choice.", clinical,
            "Resolve what the visible evidence can resolve, and confine uncertainty to the Doctor's particular unresolved issue.",
            "The reply gives the best-supported current judgment and identifies the specific unresolved issue without generic hedging.",
            "The reply explains whether the named issue can change the choice and what concrete evidence would do so, while still answering the current question.",
            "The reply ignores the named uncertainty, treats missing information as known, or uses vague uncertainty to avoid a supported judgment.",
            "The reply names the specific uncertainty and gives a current judgment, but does not explain its decision impact or resolution condition.",
            "The reply gives a supported current judgment and precisely states how resolution of the named issue would preserve or change it.",
        ),
        rubric(
            "hypothetical_fact_boundary",
            "The Doctor's latest turn asks the Agent to reason beyond the supplied case facts, proposes an unverified patient-specific detail, or discusses a hypothetical finding or result.", clinical,
            "Keep unverified or hypothetical case details explicitly conditional rather than treating them as observed facts.",
            "The reply marks the relevant added detail as hypothetical, possible, unverified, or proposed.",
            "The reply preserves the conditional boundary and explains separately how each relevant possibility would affect the judgment.",
            "The reply presents an unverified or hypothetical patient-specific detail as an established fact.",
            "The reply marks the detail as uncertain but blurs whether subsequent reasoning is conditional on it.",
            "The reply keeps every unverified detail conditional and clearly states the decision consequence under each relevant condition.",
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
            "nonredundant_dialogue_progression",
            "The Agent has already given a position or reason, and the Doctor's latest turn asks for clarification, challenges that reason, or adds a new consideration.", clinical,
            "Advance the discussion by addressing the Doctor's new contribution rather than merely repeating the prior Agent reply.",
            "The reply engages the new contribution and adds at least one relevant clarification, distinction, or consequence.",
            "The reply resolves the new contribution with reasoning that materially advances or appropriately updates the prior position.",
            "The reply substantially repeats the prior position without engaging the Doctor's new contribution.",
            "The reply acknowledges the new contribution but adds only a partial clarification or leaves its consequence unresolved.",
            "The reply directly resolves the new contribution and makes a substantive, justified progression from the prior position.",
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
            "The Doctor explicitly asks which missing fact or discriminating information would resolve a named ambiguity; the task still requires a current best-supported answer from the available evidence.", clinical,
            "Give the current best-supported answer and identify only information capable of changing that answer.",
            "The reply states a current judgment and identifies one genuinely decision-relevant missing datum.",
            "The reply states the current answer, identifies the highest-value missing datum, and explains how its plausible results would change or preserve that answer.",
            "The reply defers the requested answer, claims unsupported certainty, or requests information that would not resolve the named ambiguity.",
            "The reply gives a current judgment and relevant missing information but does not explain its directional effect.",
            "The reply answers now and precisely states how the decisive missing information could update the choice without inventing its result.",
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
