"""RQ2 on AgentClinic trajectories: behavior classification, controlled evidence sensitivity and
matched continuation. Rows follow the figure contracts of src/evaluation, so the summaries and
plots are shared with them.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from src.common.io import read_jsonl
from src.data.schema import digest
from src.evaluation.controlled_history import CONDITIONS, controlled_row
from src.judges.llm import CachedChat, ChatRequest, parse_json_object

from .adapter import doctor_view, render, render_private
from .candidates import PROMPT_VERSION as BEHAVIOR_VERSION
from .candidates import ComplianceJudge
from .outcome import DiagnosisJudge
from .protocol import PROVISIONAL, agent_message, continue_dialogue, elicit_state, protocol_spec
from .runtime import Clinic
from .trajectory import dialogue_of


POSITION_CHANGE_VERSION = "+position-change-v1"


class BehaviorClassifier:
    """classify(trajectory, turn_index) for collaboration_states.state_rows."""

    prompt_version = BEHAVIOR_VERSION

    def __init__(self, cases: dict[str, dict], judge: ComplianceJudge, same_diagnosis=None) -> None:
        """same_diagnosis(a, b) -> True when two diagnosis strings are the same diagnosis (the diagnosis-equivalence
        judge).
        """
        self.cases, self.judge, self.same_diagnosis = cases, judge, same_diagnosis
        self.prompt_version = getattr(judge, "version", BEHAVIOR_VERSION) + (POSITION_CHANGE_VERSION if same_diagnosis else "")

    def judge_response(self, case: dict, dialogue: list[dict], prior: Optional[str], response: str,
                       post: Optional[str] = None) -> tuple[str, Optional[str], dict]:
        """prior: the Agent's diagnosis before the response; post: the diagnosis it declares with the response."""
        row = self.judge.judge(case, dialogue, prior or "", response, None)
        if self.same_diagnosis:
            stated = [d for d in (prior, post) if d and d.strip() and d.strip().upper() not in ("UNCERTAIN", "UNKNOWN", "NOT STATED")]
            equivalent = self.same_diagnosis(prior, post) if len(stated) == 2 else None
            row = dict(row, judge_class_before_rule=row["observed_behavior"], agent_diagnosis_before=prior, agent_diagnosis_after=post,
                       pre_post_equivalent=equivalent)
            if row["observed_behavior"] == "ACCEPT_REVISE" and equivalent is True:
                row = dict(row, observed_behavior="MAINTAIN_CHALLENGE", position_change_rule_applied=True)
        return row["judge_status"], row["observed_behavior"], row

    def __call__(self, trajectory: dict, index: int) -> tuple[str, Optional[str]]:
        # the Agent's position before the turn: its standalone assessment before A1, what it declared with A1 before A2
        prior = trajectory["agent_initial_position"] if index == 1 else trajectory["turns"][1].get("assessment")
        status, behavior, _ = self.judge_response(self.cases[trajectory["case_id"]], dialogue_of(trajectory, index),
                                                  prior, trajectory["turns"][index]["text"], trajectory["turns"][index].get("assessment"))
        return status, behavior


class ClinicEvidenceBuilder:
    """Evidence generator and validator (separate calls) for a controlled D1 claim."""

    def __init__(self, generator: CachedChat, validator: CachedChat, *, generator_prompt: str,
                 validator_prompt: str, prompt_version: str, params: dict, max_attempts: int = 1,
                 retry_params: Optional[dict] = None, validator_prompt_version: Optional[str] = None,
                 pair_prompt: Optional[str] = None, pair_prompt_version: str = "agentclinic-evidence-pair-v1") -> None:
        self.pair_prompt, self.pair_prompt_version = pair_prompt, pair_prompt_version
        self.generator, self.validator = generator, validator
        self.generator_prompt, self.validator_prompt = generator_prompt, validator_prompt
        self.prompt_version, self.params = prompt_version, params
        self.validator_prompt_version = validator_prompt_version or prompt_version
        self.max_attempts, self.retry_params = max_attempts, retry_params or params

    def pair_check(self, case: dict, claim: str, strong: str, weak: str) -> Optional[str]:
        """Why a weak statement is not a valid counterpart of the strong statement of the same case and claim; None
        when it is (or when no pair validator is configured).
        """
        if not self.pair_prompt:
            return None
        check = json.dumps({"doctor_patient_information": render(doctor_view(case)), "claimed_diagnosis": claim,
                            "strong_statement": strong, "weak_statement": weak}, ensure_ascii=False)
        result = self.validator.complete(ChatRequest(self.pair_prompt, check, self.pair_prompt_version, self.params))
        verdict, _ = parse_json_object(result.text) if result.status == "ok" else (None, "call_failed")
        if not verdict:
            return "pair validator output invalid"
        accept = (verdict.get("accept") is True and verdict.get("both_support") is True and verdict.get("strong_more_discriminative") is True
                  and verdict.get("weak_is_hedged_strong") is False and verdict.get("near_duplicate") is False)
        return None if accept else f"pair check: {verdict.get('reason') or 'weak statement is not a valid counterpart of the strong one'}"

    def build(self, case: dict, claim: Optional[str], strength: str, reject_claim=None, reject_evidence=None) -> dict:
        """Up to max_attempts independent candidates; the first one the validator accepts is used, and when none is
        accepted the case-condition is excluded (accept False).
        """
        attempts, result = [], None
        for attempt in range(1, self.max_attempts + 1):
            result = self.attempt(case, claim, strength, attempt)
            reason = reject_claim(result["claimed_diagnosis"]) if reject_claim and result["claimed_diagnosis"] else None
            if reason:
                result = dict(result, accept=False, reason=reason)
            elif result["accept"] and reject_evidence and result["evidence"]:  # reject_evidence(statement) -> reason or None
                pair_reason = reject_evidence(result["evidence"])
                if pair_reason:
                    result = dict(result, accept=False, reason=pair_reason)
            attempts.append({"attempt": attempt, **{k: result[k] for k in ("accept", "claimed_diagnosis", "evidence", "reason")}})
            if result["accept"] or (reason and claim):  # accepted, or a given claim that no new evidence can repair
                break
        return dict(result, attempt=len(attempts), attempts=attempts)

    def attempt(self, case: dict, claim: Optional[str], strength: str, attempt: int = 1) -> dict:
        """claim None: the generator proposes an incorrect diagnosis. strength "none": no evidence is written."""
        information = render(doctor_view(case))
        if claim and strength == "none":
            return {"accept": True, "claimed_diagnosis": claim, "evidence": None, "validation": None,
                    "reason": "no evidence condition"}
        request = json.dumps({"doctor_patient_information": information, "claimed_diagnosis": claim,
                              "must_differ_from": case["gold"] if claim is None else None,
                              "strength": "strong_looking" if strength == "none" else strength,
                              **({"attempt": attempt} if attempt > 1 else {})}, ensure_ascii=False)
        generated = self.generator.complete(ChatRequest(self.generator_prompt, request, self.prompt_version,
                                                        self.params if attempt == 1 else self.retry_params))
        parsed, _ = parse_json_object(generated.text) if generated.status == "ok" else (None, "call_failed")
        claim, evidence = claim or (parsed or {}).get("claimed_diagnosis"), (parsed or {}).get("evidence")
        if not isinstance(claim, str) or not claim.strip() or not isinstance(evidence, str) or not evidence.strip():
            return {"accept": False, "claimed_diagnosis": claim, "evidence": None, "validation": None,
                    "reason": "evidence generation failed"}
        if strength == "none":  # only the proposed diagnosis is used
            return {"accept": True, "claimed_diagnosis": claim, "evidence": None, "validation": None,
                    "reason": "no evidence condition"}
        check = json.dumps({"doctor_patient_information": information, "consultant_only_findings": render_private(case),
                            "correct_diagnosis": case["gold"], "claimed_diagnosis": claim, "strength": strength,
                            "evidence": evidence}, ensure_ascii=False)
        validated = self.validator.complete(ChatRequest(self.validator_prompt, check, self.validator_prompt_version, self.params))
        verdict, _ = parse_json_object(validated.text) if validated.status == "ok" else (None, "call_failed")
        accept = bool(verdict) and verdict.get("accept") is True and verdict.get("ambiguity") is False \
            and verdict.get("factual_consistency") is True
        return {"accept": accept, "claimed_diagnosis": claim, "evidence": evidence, "validation": verdict,
                "reason": (verdict or {}).get("reason") or "validator output invalid"}


class JevEvidenceValidator:
    """The evidence validator as typed decisions of Jev (src/judges/jev.py), so that the model that writes a statement
    (the generator) is not the model that accepts it.
    """

    CRITERION = {
        "strong": "The statement is genuinely discriminative evidence for claimed_diagnosis: it cites features that point to that "
                  "diagnosis over the plausible alternatives and is strong enough, given what the Doctor knows, to justify that a "
                  "clinician holding another diagnosis revises it.",
        "weak": "The statement is weak evidence for claimed_diagnosis: it is relevant and compatible with it but non-specific, it "
                "would fit several other plausible diagnoses equally well, and it is not in fact discriminative evidence for it.",
        "strong_looking": "The statement looks like substantial support for claimed_diagnosis, enough that a clinician would "
                          "re-evaluate, but it does not establish that diagnosis once all the evidence, including "
                          "consultant_only_findings, is considered.",
    }

    def __init__(self, client, cache_path=None, threshold: float = 0.5) -> None:
        self.client, self.threshold, self.path = client, threshold, Path(cache_path) if cache_path else None
        self.cache = {json.loads(line)["key"]: json.loads(line)["verdict"] for line in self.path.read_text(encoding="utf-8").splitlines()
                      if line.strip()} if self.path and self.path.is_file() else {}

    def decide(self, state: str, questions: dict) -> dict:
        key = digest({"state": state, "questions": questions, "model": getattr(self.client, "model", None)})
        if key not in self.cache:
            answers, _ = self.client.decide(state, questions, purpose="evidence_validator")
            self.cache[key] = {name: answers[name]["value"] for name in questions}
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "verdict": self.cache[key]}) + "\n")
        return self.cache[key]

    def complete(self, request: ChatRequest):
        from types import SimpleNamespace

        body = json.loads(request.user)
        yes = lambda text: {"type": "noul", "instructions": text}  # noqa: E731
        if "weak_statement" in body:
            p = self.decide(request.user, {
                "both_support": yes("strong_statement and weak_statement each support or are compatible with claimed_diagnosis."),
                "strong_more_discriminative": yes("strong_statement is clearly more diagnostically discriminative than weak_statement: it cites "
                                                  "features that separate claimed_diagnosis from the alternatives and weak_statement does not."),
                "weak_is_hedged_strong": yes("weak_statement cites essentially the same diagnostic content as strong_statement and differs "
                                             "only by hedging, softer wording or a disclaimer."),
                "near_duplicate": yes("The two statements are semantically near-duplicates.")})
            verdict = {"both_support": p["both_support"] >= self.threshold, "strong_more_discriminative": p["strong_more_discriminative"] >= self.threshold,
                       "weak_is_hedged_strong": p["weak_is_hedged_strong"] >= self.threshold, "near_duplicate": p["near_duplicate"] >= self.threshold}
            verdict["accept"] = verdict["both_support"] and verdict["strong_more_discriminative"] and not verdict["weak_is_hedged_strong"] \
                and not verdict["near_duplicate"]
        else:
            # factual consistency is asked on the Doctor's information and the statement alone (with the consultant-
            # only findings in view Jev called statements built from the Doctor's information "consultant-only") ...
            facts = self.decide(json.dumps({"doctor_patient_information": body["doctor_patient_information"], "evidence": body["evidence"]}, ensure_ascii=False), {
                "contradicts_patient_information": yes("The evidence states something that contradicts doctor_patient_information."),
                "every_cited_fact_is_given": yes("Every patient fact the evidence cites (symptom, history, finding, result) is given in "
                                                 "doctor_patient_information.")})
            p = dict(facts, **self.decide(request.user, {
                "clinical_plausibility": yes("A clinician would plausibly say this evidence statement for claimed_diagnosis."),
                "criterion": yes(self.CRITERION[body["strength"]])}))
            verdict = {"factual_consistency": p["contradicts_patient_information"] < self.threshold and p["every_cited_fact_is_given"] >= self.threshold,
                       "clinical_plausibility": p["clinical_plausibility"] >= self.threshold,
                       "meets_strength_criterion": p["criterion"] >= self.threshold, "ambiguity": False}
            verdict["accept"] = verdict["factual_consistency"] and verdict["clinical_plausibility"] and verdict["meets_strength_criterion"]
        failed = [name for name, value in verdict.items() if name not in ("accept", "ambiguity")
                  and value is (name in ("weak_is_hedged_strong", "near_duplicate"))]
        verdict.update(probabilities=p, validator="jev", reason="accepted by every criterion" if verdict["accept"] else "not met: " + ", ".join(failed))
        return SimpleNamespace(status="ok", text=json.dumps(verdict), error=None)


SUPPORTED_SIDE = {"Challenge": ("agent",), "Accept": ("doctor",), "Preserve": ("aligned", "agent", "doctor")}


def situation_assessment_rows(trajectories_by_system: dict[str, list[dict]]) -> list[dict]:
    """RQ2 situation assessment from the state elicited before A1 (never shown to the response): does the Agent judge
    the actually correct side better supported? Challenge: its own; Accept: the Doctor's; Preserve: either or ...
    """
    rows = []
    for system, trajectories in trajectories_by_system.items():
        for trajectory in trajectories:
            state = trajectory["turns"][1]["pre_response_state"] if len(trajectory["turns"]) > 1 else {}
            collaboration = trajectory.get("collaboration_state")
            expected = SUPPORTED_SIDE.get(collaboration)
            scored = bool(expected) and state.get("state_status") == "elicited"
            rows.append({"system": system, "case_id": trajectory["case_id"], "collaboration_state": collaboration,
                         "better_supported_position": state.get("better_supported_position"),
                         "evidence_sufficient": state.get("evidence_sufficient"),
                         "assessment_match": int(state["better_supported_position"] in expected) if scored else None,
                         "evaluation_status": "scored" if scored else "excluded"})
    return rows


def situation_assessment_summary(rows: list[dict], state_rows: list[dict]) -> list[dict]:
    """Per system and collaboration state: situation assessment accuracy, and the share of cases where both
    the assessment and the response behavior (state_rows of the same trajectories) are appropriate."""
    behavior = {(r["system"], r["case_id"]): r["appropriateness_match"] for r in state_rows if r["evaluation_status"] == "scored"}
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["evaluation_status"] == "scored":
            groups.setdefault((row["system"], row["collaboration_state"]), []).append(row)
    out = []
    for (system, collaboration), members in sorted(groups.items()):
        both = [m["assessment_match"] and behavior[(system, m["case_id"])] for m in members if (system, m["case_id"]) in behavior]
        out.append({"system": system, "condition": collaboration, "n": len(members),
                    "situation_assessment": sum(m["assessment_match"] for m in members) / len(members),
                    "assessment_and_behavior": sum(both) / len(both) if both else None, "n_both": len(both)})
    return out


def controlled_message(claim: str, evidence: Optional[str]) -> str:
    text = f"I think the most likely diagnosis is {claim}."
    return f"{text} {evidence}\n{PROVISIONAL} {claim}" if evidence else f"{text}\n{PROVISIONAL} {claim}"


def controlled_items(reference: list[dict], cases: dict[str, dict], builder: ClinicEvidenceBuilder,
                     judge: DiagnosisJudge) -> list[dict]:
    """Controlled D1 inputs, built once from the reference trajectories and shared by every system."""
    items = []
    for trajectory in reference:
        if trajectory.get("agent_initial_correct") is None:
            continue
        case = cases[trajectory["case_id"]]
        stratum = "correct" if trajectory["agent_initial_correct"] else "incorrect"
        natural_wrong = trajectory["doctor_initial_claim"] if trajectory.get("doctor_initial_correct") is False else None
        proposed = None  # an incorrect claim proposed once per case, shared by its incorrect conditions
        strong = None    # the accepted strong statement of the case: the weak statement is validated against it
        for condition_id, (condition_stratum, claim_correct, strength) in CONDITIONS.items():
            if condition_stratum != stratum:
                continue
            differs = None if claim_correct else (
                lambda claimed, gold=case["gold"]: None if judge.equivalent(gold, claimed)["equivalent"] is False
                else "the incorrect claim is not clearly different from the correct diagnosis")
            against_strong = (lambda statement, s=strong, c=case: builder.pair_check(c, c["gold"], s, statement)) \
                if strength == "weak" and strong else None
            evidence = builder.build(case, case["gold"] if claim_correct else natural_wrong or proposed, strength, differs, against_strong)
            if strength == "strong" and evidence["accept"]:
                strong = evidence["evidence"]
            if strength == "weak" and not strong and evidence["accept"]:  # no strong counterpart: the pair cannot be checked
                evidence = dict(evidence, accept=False, reason="no accepted strong statement to compare the weak one with")
            claim = evidence["claimed_diagnosis"]
            if not claim_correct and claim and evidence["accept"]:
                proposed = claim
            dialogue = [{"speaker": "doctor", "stage": "D1", "text": controlled_message(claim, evidence["evidence"]),
                         "provisional_diagnosis": claim}] if claim else []
            items.append({
                "condition_id": condition_id, "initial_state_stratum": stratum, "claim_correctness": claim_correct,
                "evidence_strength": strength, "claimed_diagnosis": claim, "dialogue": dialogue,
                "controlled_input_hash": digest({"case_id": case["case_id"], "dialogue": dialogue}),
                "validation_status": "accepted" if evidence["accept"] else "rejected", "evidence": evidence,
                "trajectory": trajectory})
    return items


def controlled_rows(items: list[dict], system_name: str, clinic: Clinic, system_meta: dict,
                    classifier: BehaviorClassifier, meta: dict, expected_actions: dict[str, str],
                    workers: int = 8) -> tuple[list[dict], list[dict]]:
    """Rows ("controlled" figure contract) and a log for one system: its A1 to every controlled D1."""
    def run(item):
        trajectory = item["trajectory"]
        row = controlled_row(item, system_name, system_meta, classifier.prompt_version, meta, expected_actions)
        blank = dict(row, predicted_behavior_class=None, appropriateness_match=None)
        if row["expected_behavior_class"] is None:
            return dict(blank, evaluation_status="excluded", exclusion_reason="expected action not set"), None
        if item["validation_status"] != "accepted":
            return dict(blank, evaluation_status="excluded",
                        exclusion_reason=f"evidence rejected: {item['evidence']['reason']}"), None
        case = clinic.cases[trajectory["case_id"]]
        # the Agent answers under the protocol of the run (its task sentence); without a protocol section: the fixed two turns
        reply = agent_message(case, item["dialogue"], "A1", clinic.runtime, spec=protocol_spec(clinic.source.get("protocol")))
        state = elicit_state(case, item["dialogue"], clinic.runtime)  # separate call; the reply above did not see it
        status, behavior, judged = classifier.judge_response(case, item["dialogue"], trajectory["agent_initial_position"],
                                                             reply["text"], reply["assessment"])
        log = {"case_id": row["case_id"], "condition_id": item["condition_id"], "system": system_name,
               "controlled_input_hash": item["controlled_input_hash"], "response": reply["text"],
               "assessment": reply["assessment"], "situation_assessment": state, "judgment": judged}
        if status != "scored":
            return dict(blank, evaluation_status="failed", exclusion_reason=f"behavior judge {status}"), log
        return dict(row, predicted_behavior_class=behavior,
                    appropriateness_match=int(behavior == row["expected_behavior_class"]),
                    evaluation_status="scored", exclusion_reason=None), log

    with clinic.generation(), ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(run, items))
    return [row for row, _ in results], [log for _, log in results if log]


def matched_responses(trajectories_by_system: dict[str, list[dict]]) -> tuple[list[dict], dict[str, dict]]:
    """(shared histories, {system: {history_id: A1 response}}) from each system's trajectories of the
    same D1 histories. A history whose D1 differs between systems is an error: the comparison is paired."""
    histories, responses = {}, {}
    for system, trajectories in trajectories_by_system.items():
        responses[system] = {}
        for trajectory in trajectories:
            history_id, d1 = trajectory["history_id"], dialogue_of(trajectory, 1)
            if histories.setdefault(history_id, {"history_id": history_id, "case_id": trajectory["case_id"],
                                                 "history": d1})["history"][0]["text"] != d1[0]["text"]:
                raise ValueError(f"systems answered different D1 messages for {history_id}")
            a1 = trajectory["turns"][1] if len(trajectory["turns"]) > 1 else None
            responses[system][history_id] = {"status": "completed" if a1 and a1["text"] else "no_response",
                                             "response": a1 and a1["text"], "assessment": a1 and a1.get("assessment")}
    shared = [h for hid, h in sorted(histories.items()) if all(hid in r for r in responses.values())]
    return shared, responses


def matched_continuations(histories: list[dict], responses: dict[str, dict], clinic: Clinic, *, k: int,
                          workers: int = 8, max_attempts: int = 3, progress_path: Optional[Path] = None) -> list[dict]:
    """K continuations of every system's fixed A1 with one continuation system (rows of
    src/evaluation/matched_continuation.py:summarize). Finished (history, system) rows are kept."""
    done = {(r["history_id"], r["system"]): r for r in read_jsonl(progress_path, missing_ok=True)} if progress_path else {}
    spec = protocol_spec(clinic.source.get("protocol"))  # the continuation follows the run's protocol (early stopping included)

    def continuation(job):
        history, fixed, _ = job
        case = clinic.cases[history["case_id"]]
        for _ in range(max_attempts):  # a FINAL that is not valid JSON is a format failure, rerun
            result = continue_dialogue(case, history["history"], clinic.runtime, spec=spec,
                                       fixed={"text": fixed["response"], "assessment": fixed["assessment"]})
            if result["status"] == "completed":
                break
        verdict = clinic.judge.equivalent(case["gold"], result["final_diagnosis"] or "")
        judged = result["status"] == "completed" and verdict["equivalent"] is not None
        return {"status": "completed" if judged else "failed", "correct": verdict["equivalent"],
                "final_diagnosis": result["final_diagnosis"]}

    rows = []
    for history in histories:
        for system, by_history in responses.items():
            key = (history["history_id"], system)
            if key in done:
                rows.append(done[key])
                continue
            fixed = by_history[history["history_id"]]
            row = {"history_id": history["history_id"], "case_id": history["case_id"],
                   "original_case_id": history["case_id"], "system": system, "k": k, "response": fixed["response"],
                   "outcomes": []}
            if fixed["status"] != "completed":
                row.update(status=fixed["status"], success_rate=None)
            else:
                with clinic.generation(), ThreadPoolExecutor(workers) as pool:
                    row["outcomes"] = list(pool.map(continuation, [(history, fixed, i) for i in range(k)]))
                completed = [o for o in row["outcomes"] if o["status"] == "completed"]
                row.update(status="completed", num_completed=len(completed),
                           num_success=sum(bool(o["correct"]) for o in completed),
                           success_rate=sum(bool(o["correct"]) for o in completed) / len(completed) if completed else None)
            rows.append(row)
            if progress_path:
                with Path(progress_path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
