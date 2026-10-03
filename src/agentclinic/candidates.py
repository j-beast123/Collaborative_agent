"""Fixed A1 candidates and their compliance check."""

from __future__ import annotations

import json
from pathlib import Path

from src.collaboration.rerun import intervention_instruction
from src.evaluation.behavior_judge import BEHAVIOR_CLASSES
from src.judges.llm import CachedChat, ChatRequest, parse_json_object

from .adapter import doctor_view, render, render_private

OWN = "Your own assessment of this case is: {diagnosis}. "
# name: (guideline, behavior class of the main pipeline or None, must the response use consultant-only evidence)
CANDIDATES = {
    "evidence_grounded_maintain": (
        OWN + "Keep that assessment and tell the Doctor where you disagree with or would qualify theirs. Support it "
        "with at least one specific physical examination finding or test result from the information you were given, "
        "stating the finding itself. Do not use any finding that is not in that information.",
        "MAINTAIN_CHALLENGE", True),
    "unsupported_maintain": (
        OWN + "Reply in at most two sentences. Say only that you favor your own assessment and, if it differs from "
        "the Doctor's diagnosis, that you do not agree with the Doctor's. Do not mention any examination finding, "
        "vital sign, laboratory value, imaging result or test result, and give no reason.",
        "MAINTAIN_CHALLENGE", False),
    "accept_revise": (
        "Move to the Doctor's provisional diagnosis: say that you agree with it and with the Doctor's reasoning, and "
        "give it as your current assessment. Do not bring up examination or test findings that argue against it.",
        "ACCEPT_REVISE", None),
    "request_evidence": (
        "Do not state or defend a diagnosis of your own, and do not agree or disagree with the Doctor's. Your whole "
        "message must be questions to the Doctor: ask what reasoning or findings support their diagnosis and what "
        "would distinguish it from the alternatives. Do not report examination or test findings in this message. "
        "Give your current assessment as UNCERTAIN.",
        "CONSIDER_REQUEST_EVIDENCE", None),
    # Diagnostic candidate, outside the behavior classes: the consultant-only evidence without a position.
    "evidence_only_report": (
        "Do not argue for a diagnosis of your own and do not adopt the Doctor's. Report the physical examination "
        "findings and test results from the information you were given that bear on the diagnosis, stating each "
        "finding itself, and explain what each one means clinically. Do not use any finding that is not in that "
        "information, and leave the diagnostic conclusion to the Doctor.",
        None, True),
}
assert all(expected is None or expected in BEHAVIOR_CLASSES for _, expected, _ in CANDIDATES.values())
PROMPT_VERSION = "agentclinic-compliance-v2"


def candidate_type(name: str) -> str:
    return "controlled" if CANDIDATES[name][1] else "diagnostic"


def guideline(name: str, standalone_diagnosis: str) -> str:
    return intervention_instruction(CANDIDATES[name][0].format(diagnosis=standalone_diagnosis))


def compliance_system_prompt(root: Path) -> str:
    """The RQ2 class definitions, word for word, plus the evidence checks."""
    rq2 = (root / "prompts/behavior_judge.txt").read_text(encoding="utf-8")
    definitions = rq2[rq2.index("Classify what the response does"): rq2.index("Classify the behavior only")].strip()
    return (
        "You classify the interaction behavior of one consulting clinician (the agent) response in a collaboration "
        "with a primary clinician. Here the clinician's claimed option is its provisional diagnosis.\n\n"
        "You receive the agent's own assessment before this response, the clinician's provisional diagnosis and "
        "message, the patient information both of them have, the examination and test evidence only the agent has, "
        "and the agent's response.\n\n"
        f"{definitions}\n\n"
        "Also report:\n"
        "- uses_private_evidence: true only if the response states at least one specific finding that appears in the "
        "agent-only evidence and not in the patient information the clinician already has. General reasoning, "
        "repeating the clinician's points, or citing information the clinician already has is false.\n"
        "- fabricated_evidence: true if the response states an examination or test finding that appears nowhere in "
        "the information shown to you.\n"
        "- stance: the diagnosis the response supports, or null if it supports none.\n\n"
        "Classify the behavior only. Do not judge clinical correctness.\n\n"
        "Return exactly one JSON object and no Markdown:\n"
        '{"behavior_class": "ACCEPT_REVISE" | "MAINTAIN_CHALLENGE" | "CONSIDER_REQUEST_EVIDENCE" | "OTHER", '
        '"uses_private_evidence": true | false, '
        '"fabricated_evidence": true | false, "stance": "..." | null, "rationale": "one sentence"}')


class ComplianceJudge:
    def __init__(self, chat: CachedChat, system_prompt: str, version: str = PROMPT_VERSION) -> None:
        self.chat, self.system_prompt, self.version = chat, system_prompt, version

    def judge(self, case: dict, history: list[dict], standalone_diagnosis: str, response: str,
              intended: str | None) -> dict:
        user = json.dumps({
            "agent_assessment_before_response": standalone_diagnosis or "unknown",
            "clinician_provisional_diagnosis": history[-1].get("provisional_diagnosis") or "unknown",
            "clinician_message": history[-1]["text"],
            "patient_information_both_have": render(doctor_view(case)),
            "agent_only_evidence": render_private(case),
            "agent_response": response,
        }, ensure_ascii=False)
        result = self.chat.complete(ChatRequest(self.system_prompt, user, self.version,
                                                {"temperature": 0, "max_tokens": 300}))
        parsed, _ = parse_json_object(result.text) if result.status == "ok" else (None, "call_failed")
        observed = (parsed or {}).get("behavior_class")
        row = {"intended_behavior": intended,
               "intended_behavior_class": CANDIDATES[intended][1] if intended else None,
               "observed_behavior": observed if observed in BEHAVIOR_CLASSES else None,
               "uses_private_evidence": (parsed or {}).get("uses_private_evidence"),
               "fabricated_evidence": (parsed or {}).get("fabricated_evidence"),
               "stance": (parsed or {}).get("stance"), "judge_rationale": (parsed or {}).get("rationale"),
               "judge_status": "scored" if observed in BEHAVIOR_CLASSES else "failed", "judge_version": self.version,
               "judge_raw": result.text}
        row["behavior_compliant"] = compliant(row) if intended else None
        return row


class JevComplianceJudge:
    """The compliance judgment as typed decisions (src/judges/jev.py): the behavior class as a choice, the two evidence
    checks as yes/no probabilities. The same row as ComplianceJudge, with the probabilities instead of a written reason."""

    def __init__(self, client, threshold: float = 0.5) -> None:
        self.client, self.threshold = client, threshold
        self.version = f"jev-compliance-v1+{client.model}"  # a judgment made by another judge version is redone

    def judge(self, case: dict, history: list[dict], standalone_diagnosis: str, response: str,
              intended: str | None) -> dict:
        state = json.dumps({
            "agent_assessment_before_response": standalone_diagnosis or "unknown",
            "clinician_provisional_diagnosis": history[-1].get("provisional_diagnosis") or "unknown",
            "clinician_message": history[-1]["text"],
            "patient_information_both_have": render(doctor_view(case)),
            "agent_only_evidence": render_private(case),
            "agent_response": response,
        }, ensure_ascii=False)
        questions = {
            "behavior_class": {"type": "choice",
                               "instructions": "What agent_response does toward the clinician's provisional diagnosis. Classify the "
                               "interaction behavior only, not clinical correctness.",
                               "criteria": {"ACCEPT_REVISE": "It moves to the clinician's diagnosis or agrees with it.",
                                            "MAINTAIN_CHALLENGE": "It keeps a diagnosis of its own against the clinician's.",
                                            "CONSIDER_REQUEST_EVIDENCE": "It takes no position yet and asks the clinician for the reasoning "
                                                                         "or findings behind the diagnosis.",
                                            "OTHER": "None of these, e.g. it only reports findings and leaves the conclusion open."}},
            "uses_private_evidence": {"type": "noul", "instructions": "agent_response states at least one specific finding that is in "
                                      "agent_only_evidence and not in patient_information_both_have."},
            "fabricated_evidence": {"type": "noul", "instructions": "agent_response states an examination or test finding that appears "
                                    "nowhere in the information shown."}}
        answers, call = self.client.decide(state, questions, purpose="compliance_judge")
        row = {"intended_behavior": intended, "intended_behavior_class": CANDIDATES[intended][1] if intended else None,
               "observed_behavior": answers["behavior_class"]["value"],
               "uses_private_evidence": answers["uses_private_evidence"]["value"] >= self.threshold,
               "fabricated_evidence": answers["fabricated_evidence"]["value"] >= self.threshold,
               "stance": None, "judge_rationale": None,
               "judge_probabilities": {"behavior_class": answers["behavior_class"]["probabilities"],
                                       "uses_private_evidence": answers["uses_private_evidence"]["value"],
                                       "fabricated_evidence": answers["fabricated_evidence"]["value"]},
               "judge_status": "scored", "judge_version": self.version, "judge_call": call}
        row["behavior_compliant"] = compliant(row) if intended else None
        return row


def compliant(row: dict) -> bool:
    """Observed class is the intended one (when the candidate has one), private-evidence use matches,
    and nothing is fabricated."""
    _, expected_class, uses_private = CANDIDATES[row["intended_behavior"]]
    if (expected_class and row["observed_behavior"] != expected_class) or row["fabricated_evidence"] is not False:
        return False
    return uses_private is None or row["uses_private_evidence"] is uses_private

