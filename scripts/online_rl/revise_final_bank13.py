from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

RUN = Path("outputs/offline_bank90_from_c30_overlap9_v2_20261012/r1_o0__p0fixed__gemma3doctor__seed0__offline_bank90_from_c30_overlap9_v2")
SOURCE = RUN / "batches/batch_000012/bank_frozen.json"
OUTPUT = RUN / "bank_revised_mc_v1.json"


def scores(context: str, state: str, minimum: str, completion: str) -> dict:
    return {
        "0": {
            "user_side_context": context,
            "agent_side_state": state,
            "behavior": f"The response does not satisfy the minimum performance: {minimum}",
        },
        "1": {
            "user_side_context": context,
            "agent_side_state": state,
            "behavior": f"The response satisfies the minimum performance but does not satisfy the completion condition. Minimum: {minimum} Completion still missing: {completion}",
        },
        "2": {
            "user_side_context": context,
            "agent_side_state": state,
            "behavior": f"The response satisfies both the minimum performance and completion condition. Minimum: {minimum} Completion: {completion}",
        },
    }


REVISIONS = {
    "Queried Feature Phenotype Evaluation": {
        "name": "queried_feature_membership_evaluation_v2",
        "required_behavior": "Evaluate whether the queried feature belongs to the relevant set or phenotype using a concrete defining relationship, then connect that supported membership judgment to the exception criterion.",
        "minimum_performance": "State whether the queried feature belongs to the relevant set or phenotype and provide at least one concrete defining property or relationship supporting that judgment; a bare yes/no declaration is insufficient.",
        "completion_condition": "Connect the supported membership judgment to whether the feature is the exception or a non-exception under the prompt.",
    },
    "clinical_feature_etiology_evaluation": {
        "name": "clinical_feature_etiology_evaluation_v2",
        "required_behavior": "Relate objective findings to the candidate etiologies and use those relationships to evaluate their relative viability without presupposing that one or several must remain plausible.",
        "minimum_performance": "Relate at least one documented finding to the support it provides for or against at least one candidate etiology; merely naming the finding or candidate is insufficient.",
        "completion_condition": "Use the finding-to-etiology relationship to state which candidates remain supported, weakened, excluded, or unresolved under the available information, allowing one, several, or none to remain viable.",
    },
    "diagnostic_framework_component_evaluation": {
        "name": "diagnostic_framework_component_evaluation_v2",
        "required_behavior": "Compare the queried item with the relevant definition or parameter in the identified framework and connect that comparison to membership, deviation, or uncertainty.",
        "minimum_performance": "State the relevant framework definition or parameter and concretely compare the queried item with it; a bare confirmation or rejection is insufficient.",
        "completion_condition": "Use that comparison to justify inclusion, exclusion, deviation, or a specific unresolved ambiguity. Enumerating unrelated framework components is not required.",
    },
    "alternative_option_exception_classification": {
        "name": "queried_factor_exception_evaluation_v2",
        "required_behavior": "Evaluate the queried factor against the stated set or criterion and connect that supported status to the exception decision, comparing another option only when needed to resolve the selection.",
        "minimum_performance": "Characterize the queried factor using at least one concrete property tied to the relevant set or criterion; a bare label of risk factor, associated finding, exception, or non-exception is insufficient.",
        "completion_condition": "Explain whether that supported characterization makes the queried factor the exception or a non-exception, or state the specific comparison still needed to decide.",
    },
}


def main() -> None:
    source = json.loads(SOURCE.read_text())
    output = {
        "protocol": "cumulative_keep_bank_mc_revision_v1",
        "source_bank_sha256": source["sha256"],
        "source_bank_version": source["version"],
        "entries": [],
    }
    for entry in source["entries"]:
        old = entry["rubric"]
        public = copy.deepcopy(old["rubric"])
        change = REVISIONS.get(public["name"])
        if change is None:
            output["entries"].append({
                "status": "unchanged_keep",
                "inherits_keep": True,
                "source_bank_id": entry["bank_id"],
                "rubric_id": old["rubric_id"],
                "rubric": public,
                "source_leaf_id": old["leaf_id"],
            })
            continue
        public.update(change)
        context = public["applicability"]["user_side_context"]
        state = public["applicability"]["agent_side_state"]
        public["scores"] = scores(context, state, public["minimum_performance"], public["completion_condition"])
        digest = hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        output["entries"].append({
            "status": "revised_requires_source_score_and_rerun",
            "inherits_keep": False,
            "source_bank_id": entry["bank_id"],
            "supersedes_rubric_id": old["rubric_id"],
            "rubric_id": f"mc_v2-{digest[:12]}",
            "rubric": public,
            "source_leaf_id": old["leaf_id"],
            "revision_reason": {
                "Queried Feature Phenotype Evaluation": "M previously allowed a bare membership declaration.",
                "clinical_feature_etiology_evaluation": "C previously prescribed that multiple etiologies remain viable.",
                "diagnostic_framework_component_evaluation": "M previously required an unnecessary full-framework enumeration.",
                "alternative_option_exception_classification": "M previously required an unrequested alternative option.",
            }[old["rubric"]["name"]],
        })
    body = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    output["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
