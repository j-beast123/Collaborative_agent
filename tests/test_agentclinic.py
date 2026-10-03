"""v5: AgentClinic-MedQA adapter, two-turn protocol, candidates, K rollouts, outcome judge, split."""

import json
from pathlib import Path

import pytest

from src.agentclinic.adapter import (agent_view, assert_no_leakage, doctor_view, evaluator_view, load_cases, render)
from src.agentclinic.candidates import CANDIDATES, ComplianceJudge, compliance_system_prompt, compliant, guideline
from src.agentclinic.outcome import DiagnosisJudge, normalize
from src.agentclinic.pipeline import build_labels, generate_candidates, prepare
from src.agentclinic.protocol import (ASSESSMENT, PROVISIONAL, ClinicRuntime, agent_prompt, continue_from_a1,
                                      doctor_prompt, final_diagnosis, parse_agent, standalone_prompt, stated)
from src.agentclinic.report import build_report
from src.agentclinic.selection import eligible_cases, sample_pilot, split_cases
from src.common.io import read_jsonl
from src.judges.llm import CachedChat, ScriptedBackend

ROOT = Path(__file__).resolve().parents[1]
GOLD = "Myasthenia gravis"


def osce(i: int, tests=None) -> dict:
    return {"OSCE_Examination": {
        "Objective_for_Doctor": f"Assess patient {i} presenting with double vision.",
        "Patient_Actor": {"Demographics": "35-year-old female", "History": f"History of case {i}: weakness after activity.",
                          "Symptoms": {"Primary_Symptom": "Double vision", "Secondary_Symptoms": ["Fatigue"]},
                          "Past_Medical_History": "None.", "Social_History": "Non-smoker.", "Review_of_Systems": "No fever."},
        "Physical_Examination_Findings": {"Neurological_Examination": {"Cranial_Nerves": f"PTOSIS-FINDING-{i} worsening on upward gaze"}},
        "Test_Results": {"Blood_Tests": {"AChR_Antibodies": f"ANTIBODY-RESULT-{i} elevated"},
                         "Electromyography": {"Findings": "Decrement on repetitive stimulation"}} if tests is None else tests,
        "Management_and_Follow_Up": "SECRET-MANAGEMENT-PLAN",
        "Correct_Diagnosis": GOLD}}


@pytest.fixture
def cases(tmp_path):
    path = tmp_path / "agentclinic.jsonl"
    path.write_text("\n".join(json.dumps(osce(i)) for i in range(6)) + "\n" + json.dumps(osce(6, tests={})) + "\n")
    return load_cases(path)


def fake_runtime(doctor_final=lambda dialogue: "Viral illness", calls=None):
    """Doctor: D1 says 'Viral illness'; FINAL comes from doctor_final(dialogue text). Agent echoes its guideline."""
    calls = calls if calls is not None else []

    def doctor(system, user):
        calls.append(("doctor", user))
        if "Give your final diagnosis" in user:
            return json.dumps({"final_diagnosis": doctor_final(user), "brief_reason": "r"})
        return f"I think this is viral.\n{PROVISIONAL} Viral illness"

    def agent(system, user):
        calls.append(("agent", user))
        if "No one has commented" in user:
            return f"{ASSESSMENT} {GOLD}"
        if "stating the finding itself" in user:
            return f"{ASSESSMENT} {GOLD}\nResponse: The antibody result is elevated, which supports MG."
        if "Reply in at most two sentences" in user:
            return f"{ASSESSMENT} {GOLD}\nResponse: I do not agree."
        if "Move to the Doctor's provisional diagnosis" in user:
            return f"{ASSESSMENT} Viral illness\nResponse: I agree with you."
        if "Your whole message must be questions" in user:
            return f"{ASSESSMENT} UNCERTAIN\nResponse: What supports that?"
        if "leave the diagnostic conclusion to the Doctor" in user:
            return f"{ASSESSMENT} {GOLD}\nResponse: Findings only: the antibody result is elevated."
        return f"{ASSESSMENT} Viral illness\nResponse: NATURAL reply."

    return ClinicRuntime(doctor=doctor, agent=agent, doctor_system="D", agent_system="A"), calls


def diagnosis_judge(cache=None, backend=None):
    backend = backend or ScriptedBackend(lambda request: json.dumps({"equivalent": False, "reason": "different"}))
    return DiagnosisJudge(CachedChat(backend, cache), "E"), backend


def compliance_judge(behave=None):
    def scripted(request):
        response = json.loads(request.user)["agent_response"]
        if "Findings only" in response:
            verdict = ("OTHER", True)
        elif "antibody result" in response:
            verdict = ("MAINTAIN_CHALLENGE", True)
        elif "I do not agree" in response:
            verdict = ("MAINTAIN_CHALLENGE", False)
        elif "I agree" in response:
            verdict = ("ACCEPT_REVISE", False)
        else:
            verdict = ("CONSIDER_REQUEST_EVIDENCE", False)
        if behave:
            verdict = behave(response, verdict)
        return json.dumps({"behavior_class": verdict[0], "uses_private_evidence": verdict[1], "fabricated_evidence": False,
                           "stance": None, "rationale": "r"})
    return ComplianceJudge(CachedChat(ScriptedBackend(scripted), None), compliance_system_prompt(ROOT))


def _agent_missing(case):
    """A broken case: the patient block the Agent gets is not the one the Doctor sees."""
    return dict(case, private_evidence={"Physical_Examination_Findings": {}, "Test_Results": {},
                                        "Patient_Actor": {"History": "a different patient block"}})


# 1-4: parsing, views, leakage
def test_adapter_views_and_leakage(cases):
    case = cases[0]
    assert case["case_id"] == "agentclinic_medqa_extended_0000" and case["gold"] == GOLD and len(cases) == 7
    doctor, agent = render(doctor_view(case)), render(agent_view(case))
    assert "History of case 0" in doctor and "Double vision" in doctor
    for private in ("PTOSIS-FINDING-0", "ANTIBODY-RESULT-0", "Decrement on repetitive stimulation"):
        assert private not in doctor and private in agent  # consultant-only evidence
    assert agent.startswith(doctor)  # the Agent sees everything the Doctor sees
    for text in (doctor, agent, json.dumps(doctor_view(case)), json.dumps(agent_view(case))):
        assert GOLD not in text and "Correct_Diagnosis" not in text and "SECRET-MANAGEMENT-PLAN" not in text
    assert evaluator_view(case) == {"case_id": case["case_id"], "Correct_Diagnosis": GOLD}
    for c in cases:
        assert_no_leakage(c)
    # the Doctor's view is built from the objective and the patient-reported block only: nothing else can enter it
    polluted = dict(case, doctor_view=dict(case["doctor_view"], Test_Results=case["private_evidence"]["Test_Results"],
                                           Correct_Diagnosis=GOLD))
    assert set(doctor_view(polluted)) == {"case_id", "Objective_for_Doctor", "Patient_Actor"}
    assert "ANTIBODY-RESULT-0" not in render(doctor_view(polluted)) and GOLD not in render(doctor_view(polluted))
    with pytest.raises(AssertionError, match="everything the Doctor sees"):
        assert_no_leakage(_agent_missing(case))


def test_prompts_never_contain_the_diagnosis_or_the_other_sides_evidence(cases):
    case = cases[1]
    history = [{"speaker": "doctor", "stage": "D1", "text": "D1 text"}]
    for stage in ("D1", "D2", "FINAL"):
        prompt = doctor_prompt(case, history, stage)
        assert "ANTIBODY-RESULT-1" not in prompt and "PTOSIS-FINDING-1" not in prompt and GOLD not in prompt
    for prompt in (agent_prompt(case, history, "A1"), agent_prompt(case, history, "A2"), standalone_prompt(case),
                   agent_prompt(case, history, "A1", guideline("evidence_grounded_maintain", "some diagnosis"))):
        assert "ANTIBODY-RESULT-1" in prompt and GOLD not in prompt
    # the same prompts for another ground truth: nothing depends on it
    other = dict(case, gold="Something else")
    assert doctor_prompt(other, history, "D2") == doctor_prompt(case, history, "D2")
    assert agent_prompt(other, history, "A1") == agent_prompt(case, history, "A1")


# 5: interaction order
def test_two_turn_order_and_fixed_a1(cases):
    runtime, calls = fake_runtime()
    history = [{"speaker": "doctor", "stage": "D1", "text": f"x\n{PROVISIONAL} Viral illness", "provisional_diagnosis": "Viral illness"}]
    natural = continue_from_a1(cases[0], history, runtime)
    assert [m["stage"] for m in natural["dialogue"]] == ["D1", "A1", "D2", "A2", "FINAL"]
    assert [m["speaker"] for m in natural["dialogue"]] == ["doctor", "agent", "doctor", "agent", "doctor"]
    assert [who for who, _ in calls] == ["agent", "doctor", "agent", "doctor"]
    assert natural["status"] == "completed" and natural["final_diagnosis"] == "Viral illness"
    calls.clear()
    fixed = continue_from_a1(cases[0], history, runtime, fixed_a1="FIXED text", fixed_assessment=GOLD)
    assert [who for who, _ in calls] == ["doctor", "agent", "doctor"]  # A1 is not generated
    a1 = fixed["dialogue"][1]
    assert a1["fixed"] and a1["text"] == "FIXED text" and a1["assessment"] == GOLD
    assert "FIXED text" in calls[0][1] and "Do not reverse your position solely because the Doctor disagrees" in calls[1][1]
    assert natural["dialogue"][1] == {"speaker": "agent", "stage": "A1", "text": "NATURAL reply.", "assessment": "Viral illness",
                                      "raw": f"{ASSESSMENT} Viral illness\nResponse: NATURAL reply."}
    assert ASSESSMENT not in calls[0][1].split("Conversation so far:")[1].split("Respond to")[0]  # the Doctor sees the message only
    assert stated(f"a\n{PROVISIONAL} **X.**", PROVISIONAL) == "X" and stated(f"{ASSESSMENT} UNCERTAIN", ASSESSMENT) == ""
    assert parse_agent(f"{ASSESSMENT} Syringomyelia\nResponse: The MRI shows a syrinx.") == ("Syringomyelia", "The MRI shows a syrinx.")
    assert parse_agent(f"{ASSESSMENT} uncertain\nResponse: What supports that?") == ("", "What supports that?")
    assert parse_agent(f"Free text.\n{ASSESSMENT} X") == ("X", "Free text.")  # format not followed: still usable
    for prompt in (agent_prompt(cases[0], history, "A1"), standalone_prompt(cases[0])):
        assert "one disease name, without alternatives or qualifiers" in prompt
    assert final_diagnosis('```json\n{"final_diagnosis": "MG", "brief_reason": "r"}\n```') == "MG" and final_diagnosis("MG") == ""


# 6-7: candidate mapping and compliance schema
def test_candidate_mapping_and_compliance_rules():
    assert {name: spec[1] for name, spec in CANDIDATES.items()} == {
        "evidence_grounded_maintain": "MAINTAIN_CHALLENGE", "unsupported_maintain": "MAINTAIN_CHALLENGE",
        "accept_revise": "ACCEPT_REVISE", "request_evidence": "CONSIDER_REQUEST_EVIDENCE",
        "evidence_only_report": None}  # diagnostic candidate: not one of the behavior classes
    from src.agentclinic.candidates import candidate_type
    assert candidate_type("evidence_only_report") == "diagnostic" and candidate_type("accept_revise") == "controlled"
    rq2 = (ROOT / "prompts/behavior_judge.txt").read_text()
    definitions = rq2[rq2.index("- ACCEPT_REVISE"): rq2.index("Classify the behavior only")].strip()
    assert definitions in compliance_system_prompt(ROOT)  # the RQ2 class definitions, word for word
    assert GOLD not in guideline("evidence_grounded_maintain", "Lambert-Eaton syndrome")

    def row(name, observed, private, fabricated=False):
        return {"intended_behavior": name, "observed_behavior": observed, "uses_private_evidence": private,
                "fabricated_evidence": fabricated}

    assert compliant(row("evidence_grounded_maintain", "MAINTAIN_CHALLENGE", True))
    assert not compliant(row("evidence_grounded_maintain", "MAINTAIN_CHALLENGE", False))  # no consultant-only evidence
    assert not compliant(row("evidence_grounded_maintain", "MAINTAIN_CHALLENGE", True, fabricated=True))
    assert compliant(row("unsupported_maintain", "MAINTAIN_CHALLENGE", False))
    assert not compliant(row("unsupported_maintain", "MAINTAIN_CHALLENGE", True))  # it cited consultant-only evidence
    assert not compliant(row("accept_revise", "MAINTAIN_CHALLENGE", False))
    assert compliant(row("accept_revise", "ACCEPT_REVISE", True))  # evidence use is not constrained here
    assert compliant(row("request_evidence", "CONSIDER_REQUEST_EVIDENCE", False))
    for observed in ("OTHER", "MAINTAIN_CHALLENGE"):  # no behavior class is required of the diagnostic candidate
        assert compliant(row("evidence_only_report", observed, True))
    assert not compliant(row("evidence_only_report", "OTHER", False))


# 8-11: K rollouts, BEFORE cache, delta, outcome judge cache; plus the report
def test_pilot_pipeline_k9_labels_and_report(cases, tmp_path):
    k = 9
    pilot = cases[:2]
    by_id = {c["case_id"]: c for c in pilot}
    for c in pilot:
        c["split"] = "train"
    # the Doctor reaches the right diagnosis only when the fixed A1 communicated the antibody result
    runtime, calls = fake_runtime(doctor_final=lambda user: GOLD if "antibody result" in user else "Viral illness")
    judge, backend = diagnosis_judge(tmp_path / "judge_cache.jsonl")
    out = tmp_path / "run"
    assert prepare(pilot, runtime, judge, out, workers=4) == {"histories": 2}
    histories = read_jsonl(out / "histories.jsonl")
    assert all(h["agent_standalone_diagnosis"] == GOLD and h["doctor_provisional_diagnosis"] == "Viral illness"
               and [m["stage"] for m in h["natural"]["dialogue"]] == ["D1", "A1", "D2", "A2", "FINAL"] for h in histories)
    from src.agentclinic.pipeline import prepare_diagnostics
    summary = prepare_diagnostics(out, judge)
    # Agent standalone = gold (exact match), Doctor D1 = 'Viral illness' (judged different) -> Challenge
    assert summary["agent_only_correct (Challenge)"] == 2 and summary["doctor_d1_correct"] == 0
    assert summary["d1_and_agent_disagree"] == 2 and summary["natural_final_by_state"]["Challenge"] == "0/2"
    per_case = read_jsonl(out / "prepare_diagnostics.jsonl")
    assert [r["collaboration_state"] for r in per_case] == ["Challenge", "Challenge"] and per_case[0]["d1_agent_agree"] is False

    from src.agentclinic.pipeline import select_label_histories
    selection = select_label_histories(histories, judge, n=1, seed=0)  # D1 'Viral illness' vs standalone gold: both differ
    assert len(selection["pool_case_ids"]) == 2 and len(selection["selected_case_ids"]) == 1
    assert selection == select_label_histories(list(reversed(histories)), judge, n=1, seed=0)
    assert select_label_histories([dict(h, agent_standalone_diagnosis=h["doctor_provisional_diagnosis"]) for h in histories],
                                  judge, n=1, seed=0)["pool_case_ids"] == []  # agreement is not in the pool
    first = histories[0]["history_id"]
    # the judge never accepts request_evidence in the first history: it is generated max_attempts times and stays out
    compliance = compliance_judge()
    plain_judge = compliance.judge

    def judge_with_one_failure(case, history, standalone, response, intended):
        row = plain_judge(case, history, standalone, response, intended)
        if intended == "request_evidence" and f"{case['case_id']}:A1" == first:
            row["observed_behavior"], row["behavior_compliant"] = "MAINTAIN_CHALLENGE", False
        return row

    compliance.judge = judge_with_one_failure
    made = generate_candidates(by_id, histories, runtime, compliance, out, workers=4, max_attempts=3)
    assert made["candidates"] == 10 and made["generations"] == 12  # 2 extra tries of the one that never complies
    assert made["compliant"]["request_evidence"] == "1/2" and made["compliant"]["evidence_only_report"] == "2/2"
    judged = {(r["history_id"], r["candidate_id"]): r for r in read_jsonl(out / "candidate_compliance.jsonl")}
    assert all(set(r) >= {"intended_behavior", "intended_behavior_class", "observed_behavior", "behavior_compliant",
                          "uses_private_evidence", "fabricated_evidence", "stance", "attempt"} for r in judged.values())
    assert judged[(first, "request_evidence")]["attempt"] == 3 and judged[(first, "accept_revise")]["attempt"] == 1
    assert judged[(first, "evidence_only_report")]["intended_behavior_class"] is None
    assert len(read_jsonl(out / "candidate_attempts.jsonl")) == 12 and len(read_jsonl(out / "candidates.jsonl")) == 10
    # running the stage again generates nothing new
    n_agent_calls = sum(who == "agent" for who, _ in calls)
    assert generate_candidates(by_id, histories, runtime, compliance, out, workers=4, max_attempts=3)["generations"] == 12
    assert sum(who == "agent" for who, _ in calls) == n_agent_calls

    doctor_calls_before = sum(who == "doctor" for who, _ in calls)
    counts = build_labels(by_id, histories, runtime, judge, out, k=k, workers=8, require_compliance=True,
                          metadata={"split": "train", "k": k})
    assert counts == {"before_rollouts": 2 * k, "after_rollouts": 9 * k, "labels": 9, "incomplete_pairs": 0}
    assert sum(who == "doctor" for who, _ in calls) - doctor_calls_before == (2 + 9) * k * 2  # D2 + FINAL each
    labels = {(r["history_id"], r["response_id"]): r for r in read_jsonl(out / "labels.jsonl")}
    assert (first, "request_evidence") not in labels  # non-compliant candidates are never labelled
    grounded = labels[(first, "evidence_grounded_maintain")]
    assert grounded["before"] == {"successes": 0, "trials": k} and grounded["after"] == {"successes": k, "trials": k}
    assert (grounded["before_rate"], grounded["after_rate"], grounded["delta_effectiveness"]) == (0.0, 1.0, 1.0)
    assert labels[(first, "unsupported_maintain")]["delta_effectiveness"] == 0.0
    only = labels[(first, "evidence_only_report")]  # the findings alone also move the Doctor
    assert (only["candidate_type"], only["delta_effectiveness"], only["uses_private_evidence"]) == ("diagnostic", 1.0, True)
    assert grounded["before"] == labels[(first, "accept_revise")]["before"]  # BEFORE is shared within a history
    index = int(first.split("_")[-1].split(":")[0])
    assert GOLD not in json.dumps(grounded["history"]) and f"ANTIBODY-RESULT-{index}" in grounded["history"]["scenario"]
    after = read_jsonl(out / "after_rollouts.jsonl")
    assert sorted(r["rollout_index"] for r in after if (r["history_id"], r["candidate_id"]) == (first, "accept_revise")) == list(range(k))
    assert all(r["dialogue"][1].get("fixed") and r["gold"] == GOLD for r in after)

    # resumable: nothing is rerun, nothing is duplicated
    n_calls = len(calls)
    assert build_labels(by_id, histories, runtime, judge, out, k=k, workers=8, require_compliance=True,
                        metadata={"split": "train", "k": k})["labels"] == 9 and len(calls) == n_calls
    # outcome judge: exact matches need no call; each distinct (gold, prediction) pair is judged once
    pairs = [(json.loads(c.user)["reference_diagnosis"], json.loads(c.user)["predicted_diagnosis"]) for c in backend.calls]
    assert sorted(pairs) == [(GOLD, "Viral illness"), ("Viral illness", GOLD)]  # outcome; D1 vs standalone (diagnostics)

    # the label rows feed the existing predictor unchanged
    from src.effectiveness.pair_data import load_pairs, to_examples
    examples = to_examples(load_pairs(out / "labels.jsonl", split="train", expected_k=k))
    assert sum(e.kind == "before" for e in examples) == 2 and all(e.num_rollouts == k for e in examples)
    assert "[AGENT RESPONSE]" in next(e.text for e in examples if e.kind == "after")

    report = build_report(out, {"source_cases": 7, "eligible": 6, "sampled": 2, "ineligible_reasons": {}, "selection": "s",
                                "gold_named_in_private_evidence": 0}, judge, lambda dialogue, index: False)
    assert report["signal"]["histories_with_nonzero_delta"] == 1.0 and report["signal"]["histories_all_candidates_identical"] == 0
    assert report["behavior_comparisons"]["evidence_grounded_vs_unsupported"]["mean_difference"] == 1.0
    assert report["doctor_responsiveness"]["evidence_grounded_maintain"]["doctor_revised"] == 1.0
    assert report["doctor_responsiveness"]["accept_revise"]["doctor_revised"] == 0.0
    assert report["continuation_agent"]["evidence_grounded_maintain"]["a2_reversed"] == 1.0  # A2 went back to 'Viral illness'
    assert report["outcome_direction"]["agent_correct_before_a1"]["histories"] == 2
    challenge = report["signal_by_state"]["Challenge"]
    assert (challenge["histories"], challenge["with_nonzero_delta"], challenge["all_candidates_identical"]) == (2, 2, 0)
    assert report["delta_by_behavior"]["evidence_grounded_maintain"]["distribution"] == {1.0: 2}
    text = (out / "signal_report.md").read_text()
    for heading in ("Dataset", "Candidate compliance", "Outcome distribution", "Effectiveness signal", "Delta by behavior",
                    "Signal by collaboration state", "Behavior comparisons",
                    "Doctor responsiveness", "Continuation Agent", "Outcome direction"):
        assert f"## {heading}" in text
    assert (out / "doctor_revision_examples.jsonl").exists() and (out / "agent_reversal_examples.jsonl").exists()


def test_outcome_judge_exact_match_cache_and_blindness(tmp_path):
    seen = []
    backend = ScriptedBackend(lambda request: (seen.append(request.user), json.dumps({"equivalent": True, "reason": "synonym"}))[1])
    judge = DiagnosisJudge(CachedChat(backend, tmp_path / "cache.jsonl"), "E")
    assert normalize("Progressive multifocal encephalopathy (PML)") == "progressive multifocal encephalopathy"
    assert judge.equivalent("Myasthenia gravis", "myasthenia gravis.")["method"] == "exact" and not seen
    assert judge.equivalent("Myasthenia gravis", "")["equivalent"] is False and not seen
    assert judge.equivalent("Myasthenia gravis", "MG") == {"equivalent": True, "method": "judge", "reason": "synonym"}
    judge.equivalent("Myasthenia gravis", "MG")
    assert len(seen) == 1 and set(json.loads(seen[0])) == {"reference_diagnosis", "predicted_diagnosis"}  # nothing else
    again = DiagnosisJudge(CachedChat(ScriptedBackend(lambda r: None), tmp_path / "cache.jsonl"), "E")
    assert again.equivalent("Myasthenia gravis", "MG")["equivalent"] is True  # served from the cache file


# 12: split
def test_case_level_split_and_pilot_sampling(cases):
    eligible, reasons = eligible_cases(cases, min_test_findings=2)
    assert len(eligible) == 6 and reasons == {"no Test_Results": 1}
    named = dict(cases[0], private_evidence={"Physical_Examination_Findings": {"x": "y"},
                                             "Test_Results": {"Biopsy": f"consistent with {GOLD.lower()}", "CT": "normal"}})
    kept, why = eligible_cases(cases[1:6] + [named], min_test_findings=2, exclude_named_diagnosis=True)
    assert len(kept) == 5 and why == {"diagnosis named in consultant-only evidence": 1}
    assert len(eligible_cases(cases[1:6] + [named], min_test_findings=2)[0]) == 6
    ids = [f"case_{i:03d}" for i in range(100)]
    fractions = {"train": 0.6, "dev": 0.2, "test": 0.2}
    splits = split_cases(ids, 0, fractions)
    assert sorted(splits) == sorted(ids) and [sum(v == s for v in splits.values()) for s in ("train", "dev", "test")] == [60, 20, 20]
    assert split_cases(list(reversed(ids)), 0, fractions) == splits and split_cases(ids, 1, fractions) != splits
    case_splits = split_cases([c["case_id"] for c in eligible], 0, fractions)
    pilot = sample_pilot(eligible, case_splits, split="train", n=2, seed=0)
    assert len(pilot) == 2 and all(case_splits[c["case_id"]] == "train" for c in pilot)
    assert pilot == sample_pilot(eligible, case_splits, split="train", n=2, seed=0)


def test_full_labelling_is_gated_and_split_manifest_is_valid():
    from scripts import run_agentclinic
    from src.agentclinic.selection import split_manifest
    from src.common.config import IntegrationUnavailable, read_config
    from src.data.schema import identity
    from src.data.split import validate_manifest
    from src.effectiveness.train import load_predictor_config

    pilot = read_config(ROOT / "configs/effectiveness/v5_agentclinic_pilot.json")
    full = read_config(ROOT / "configs/effectiveness/v5_agentclinic_full.json")
    assert pilot["run_full_labeling"] is False and full["run_full_labeling"] is True and pilot["k"] == full["k"] == 9
    frozen, lora = (load_predictor_config(ROOT / f"configs/effectiveness/v5_predictor_{n}.json") for n in ("frozen", "lora"))
    assert frozen["continuation_spec"].k == 9 and frozen["continuation_spec"].max_rounds == 2
    assert frozen["continuation_spec"].protocol_hash() == lora["continuation_spec"].protocol_hash()
    assert (frozen["predictor"]["training_mode"], lora["predictor"]["training_mode"]) == (
        "frozen_backbone_scalar_head", "lora_scalar_head")
    assert lora["predictor"]["lora"]["r"] == 16 and lora["predictor"]["lora"]["alpha"] == 32
    assert run_agentclinic.protocol_identity(full) == {"continuation_protocol_hash": frozen["continuation_spec"].protocol_hash()}

    splits = {f"case_{i}": split for i, split in enumerate(["train"] * 6 + ["dev"] * 2 + ["test"] * 2)}
    manifest = split_manifest(splits, source_dataset="ds", dataset_version="sha256:x", seed=0,
                              fractions={"train": 0.6, "dev": 0.2, "test": 0.2})
    assert validate_manifest(manifest)["group_counts"] == {"train": 6, "dev": 2, "eval": 2}
    assert manifest["members"][identity("ds", "case_9")]["split"] == "eval"  # TEST is held out under the manifest's name
    with pytest.raises(ValueError, match="cannot be regenerated"):
        validate_manifest(manifest, cases=[])

    if not (ROOT / pilot["dataset"]["path"]).exists():
        pytest.skip("AgentClinic source file not downloaded")
    with pytest.raises(IntegrationUnavailable, match="full labelling is disabled"):
        run_agentclinic.select(dict(full, run_full_labeling=False), "full")
    cases, summary = run_agentclinic.select(full, "full")
    assert {c["split"] for c in cases} == {"train", "dev"} and len(cases) == 137  # TEST cases are never selected
    assert all(summary["splits"][c["case_id"]] == c["split"] for c in cases)
    with pytest.raises(SystemExit):  # screening runs prepare only
        import sys
        sys.argv = ["x", "--config", str(ROOT / "configs/effectiveness/v5_agentclinic_screen_base.json"), "--scope",
                    "screen", "labels"]
        run_agentclinic.main()


# 13: earlier versions
def test_v3_and_v4_configs_still_load():
    from src.effectiveness.train import load_predictor_config

    hashes = {name: load_predictor_config(ROOT / f"configs/effectiveness/{name}.json")["continuation_spec"].protocol_hash()
              for name in ("predictor_pilot", "v3_choice_conditioned", "predictor_frozen", "diagnostic_last_turn")}
    assert hashes["predictor_pilot"] == hashes["v3_choice_conditioned"] != hashes["predictor_frozen"]
    assert hashes["predictor_frozen"] == hashes["diagnostic_last_turn"]


def test_freeze_rescore_and_rejudge_reuse_existing_generations(cases, tmp_path):
    from src.agentclinic.pipeline import check_frozen, freeze_histories, rescore
    from src.common.config import IntegrationUnavailable

    pilot = cases[:1]
    pilot[0]["split"] = "train"
    by_id = {c["case_id"]: c for c in pilot}
    runtime, calls = fake_runtime()  # the Doctor always ends on 'Viral illness'
    judge, _ = diagnosis_judge()     # ... which this judge calls different from the gold
    out = tmp_path / "run"
    prepare(pilot, runtime, judge, out, workers=2)
    freeze_histories(out)
    with pytest.raises(IntegrationUnavailable, match="frozen"):
        prepare(pilot, runtime, judge, out, workers=2)  # histories are never sampled again
    check_frozen(out)
    histories = read_jsonl(out / "histories.jsonl")
    generate_candidates(by_id, histories, runtime, compliance_judge(), out, workers=2)
    build_labels(by_id, histories, runtime, judge, out, k=2, workers=4, require_compliance=True, metadata={"split": "train"})
    assert {r["after"]["successes"] for r in read_jsonl(out / "labels.jsonl")} == {0}

    # a revised equivalence judge: the same stored dialogues, judged again, no model call
    lenient = DiagnosisJudge(CachedChat(ScriptedBackend(lambda r: json.dumps({"equivalent": True, "reason": "same"})), None), "E2")
    n_calls = len(calls)
    changed = rescore(out, lenient)
    assert changed["after_rollouts.jsonl"] == {"continuations": 10, "outcome_changed": 10} and len(calls) == n_calls
    assert not (out / "labels.jsonl").exists() and (out / "labels.before_rescore.jsonl").exists()
    build_labels(by_id, histories, runtime, lenient, out, k=2, workers=4, require_compliance=True, metadata={"split": "train"})
    assert len(calls) == n_calls and {r["after"]["successes"] for r in read_jsonl(out / "labels.jsonl")} == {2}

    # a revised compliance judge re-reads the stored responses; a candidate that no longer complies is
    # generated again, and its continuations belong to the new generation
    def strict(response, verdict):
        return ("ACCEPT_REVISE", False) if "I do not agree" in response else verdict

    attempts_before = {(a["candidate_id"], a["attempt"]) for a in read_jsonl(out / "candidate_attempts.jsonl")}
    rows = [dict(a, judge_version="older") for a in read_jsonl(out / "candidate_attempts.jsonl")]
    (out / "candidate_attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    made = generate_candidates(by_id, histories, runtime, compliance_judge(strict), out, workers=2, max_attempts=2)
    assert made["compliant"]["unsupported_maintain"] == "0/1" and made["generations"] == len(attempts_before) + 1
    chosen = {c["candidate_id"]: c["attempt"] for c in read_jsonl(out / "candidates.jsonl")}
    assert chosen["unsupported_maintain"] == 2 and chosen["accept_revise"] == 1
    (out / "histories.jsonl").write_text((out / "histories.jsonl").read_text() + "\n")
    with pytest.raises(RuntimeError, match="frozen snapshot"):
        check_frozen(out)


def test_restricted_doctor_view_keeps_the_agent_view_complete(tmp_path):
    path = tmp_path / "agentclinic.jsonl"
    path.write_text(json.dumps(osce(0)) + "\n")
    full = load_cases(path)[0]
    for fields, visible, hidden in (
            (["Demographics", "History"], ["35-year-old female", "History of case 0"], ["Double vision\n", "Fatigue", "Non-smoker"]),
            (["Demographics", "History", "Symptoms.Primary_Symptom"], ["35-year-old female", "Primary Symptom: Double vision"],
             ["Fatigue", "Non-smoker", "No fever"])):
        case = load_cases(path, doctor_patient_fields=fields)[0]
        doctor = render(doctor_view(case))
        assert all(text in doctor for text in visible) and not any(text in doctor for text in hidden)
        assert "Objective: Assess patient 0" in doctor and "ANTIBODY-RESULT-0" not in doctor and GOLD not in doctor
        assert render(agent_view(case)) == render(agent_view(full))  # the Agent still sees everything the patient reports
        assert_no_leakage(case)
        assert doctor in doctor_prompt(case, [], "D1") and "Fatigue" in agent_prompt(case, [], "A1")
    assert render(doctor_view(full)) == render(doctor_view(load_cases(path, doctor_patient_fields=None)[0]))

    from scripts import run_agentclinic
    from src.common.config import read_config
    base = read_config(ROOT / "configs/effectiveness/v5_agentclinic_pilot.json")
    for name, fields in (("a", ["Demographics", "History"]), ("b", ["Demographics", "History", "Symptoms.Primary_Symptom"])):
        variant = read_config(ROOT / f"configs/effectiveness/v5_agentclinic_view_{name}.json")
        assert variant["doctor_view"]["patient_fields"] == fields and variant["run_full_labeling"] is False
        assert variant["output_dir"] != base["output_dir"] and variant["k"] == base["k"]
        if (ROOT / variant["pilot"]["same_cases_as"] / "cases.jsonl").exists() and (ROOT / variant["dataset"]["path"]).exists():
            cases, _ = run_agentclinic.select(variant, "pilot")
            assert sorted(c["case_id"] for c in cases) == sorted(c["case_id"] for c in run_agentclinic.select(base, "pilot")[0])
            assert all(set(doctor_view(c)["Patient_Actor"]) <= {"Demographics", "History", "Symptoms"} for c in cases)


def test_screening_runs_share_the_agent_standalone_assessment(cases, tmp_path):
    judge, _ = diagnosis_judge()
    shared = tmp_path / "screen" / "agent_standalone.jsonl"
    standalone_calls = []
    for name in ("base", "view_a"):
        runtime, calls = fake_runtime()
        prepare(cases[:3], runtime, judge, tmp_path / "screen" / name, workers=3, standalone_file=shared)
        standalone_calls.append(sum(who == "agent" and "No one has commented" in user for who, user in calls))
    assert standalone_calls == [3, 0]  # generated once, read by the second run
    first, second = (read_jsonl(tmp_path / "screen" / name / "histories.jsonl") for name in ("base", "view_a"))
    assert {h["case_id"]: h["agent_standalone_diagnosis"] for h in first} == \
        {h["case_id"]: h["agent_standalone_diagnosis"] for h in second} == {c["case_id"]: GOLD for c in cases[:3]}
    from src.agentclinic.pipeline import prepare_diagnostics
    summary = prepare_diagnostics(tmp_path / "screen" / "base", judge)
    assert summary["natural_final_minus_d1"] == 0 and summary["gained_by_collaboration (D1 wrong, final right)"] == 0


def test_integrity_check_blocks_training_on_broken_labels(tmp_path):
    from src.agentclinic.pipeline import integrity_check

    def label(case, split, candidate, before, after, k=9):
        return {"case_id": case, "history_id": f"{case}:A1", "response_id": candidate, "split": split, "response": "text",
                "before": {"successes": before, "trials": k}, "after": {"successes": after, "trials": k},
                "before_rate": before / k, "after_rate": after / k, "delta_effectiveness": (after - before) / k,
                "behavior_compliant": True}

    def write(train, dev, compliant=10, generated=10):
        (tmp_path / "labels_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
        (tmp_path / "labels_dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dev))
        (tmp_path / "candidate_compliance.jsonl").write_text(
            "".join(json.dumps({"behavior_compliant": i < compliant}) + "\n" for i in range(generated)))

    splits = {"t1": "train", "t2": "train", "d1": "dev", "x1": "test"}
    limits = {"min_train_pairs": 2, "min_dev_pairs": 2, "min_dev_nonzero_delta_pairs": 1, "min_compliant_share": 0.6}
    train = [label("t1", "train", "accept_revise", 5, 9), label("t2", "train", "accept_revise", 0, 0)]
    dev = [label("d1", "dev", "accept_revise", 9, 9), label("d1", "dev", "unsupported_maintain", 9, 4)]
    write(train, dev)
    good = integrity_check(tmp_path, k=9, splits=splits, thresholds=limits)
    assert good["passed"] and good["numbers"]["dev_nonzero_delta_pairs"] == 1
    assert json.loads((tmp_path / "integrity_report.json").read_text())["failures"] == []

    def failures(train, dev, **kwargs):
        write(train, dev, **kwargs)
        return integrity_check(tmp_path, k=9, splits=splits, thresholds=limits)["failures"]

    assert any("exactly_k" in f for f in failures(train + [label("t2", "train", "request_evidence", 0, 3, k=8)], dev))
    assert "no_case_in_both_train_and_dev" in failures(train + [label("d1", "train", "request_evidence", 9, 9)], dev)
    assert "every_label_case_is_in_its_split_and_none_is_test" in failures(train + [label("x1", "train", "accept_revise", 1, 1)], dev)
    assert "before_is_shared_within_a_history" in failures(train, dev + [label("d1", "dev", "request_evidence", 8, 8)])
    assert any("dev_nonzero_delta" in f for f in failures(train, [label("d1", "dev", "accept_revise", 9, 9)] * 1 + [label("d1", "dev", "request_evidence", 9, 9)]))
    assert any("compliant_share" in f for f in failures(train, dev, compliant=3))
    assert "train_and_dev_label_files_not_empty" in failures(train, [])


def test_frozen_equivalence_gives_one_decision_per_normalized_pair(tmp_path):
    """The judge may answer the same pair differently from call to call; with the frozen file the first decision stays,
    for every capitalization of the pair and for every process that opens the file."""
    from src.agentclinic.outcome import FrozenEquivalence

    answers = iter([True, False, False, False])
    backend = ScriptedBackend(lambda request: json.dumps({"equivalent": next(answers), "reason": "r"}))
    path = tmp_path / "frozen.jsonl"
    judge = DiagnosisJudge(CachedChat(backend, None), "E", frozen=FrozenEquivalence(path))
    first = judge.equivalent("Polycystic kidney disease", "ADPKD type 1")
    assert first["equivalent"] is True and first["method"] == "judge" and len(backend.calls) == 1
    again = judge.equivalent("polycystic Kidney Disease", "adpkd  TYPE 1.")  # the same normalized pair: no new call
    assert again["equivalent"] is True and again["method"] == "frozen" and len(backend.calls) == 1
    other = DiagnosisJudge(CachedChat(backend, None), "E", frozen=FrozenEquivalence(path))  # another process
    assert other.equivalent("POLYCYSTIC KIDNEY DISEASE", "ADPKD type 1")["equivalent"] is True and len(backend.calls) == 1
    assert other.equivalent("Polycystic kidney disease", "renal cysts")["equivalent"] is False and len(backend.calls) == 2
    assert judge.equivalent("Polycystic kidney disease", "Renal cysts")["method"] == "frozen" and len(backend.calls) == 2  # read from the file
    assert judge.equivalent("X", "x")["method"] == "exact" and judge.equivalent("X", "")["method"] == "empty"
    assert len(path.read_text().splitlines()) == 2
    failing = DiagnosisJudge(CachedChat(ScriptedBackend(lambda request: "not json"), None), "E", frozen=FrozenEquivalence(path))
    assert failing.equivalent("A", "B")["equivalent"] is None and len(path.read_text().splitlines()) == 2  # a failed call freezes nothing


def test_majority_policy_two_agreeing_votes_decide_and_a_split_takes_five():
    from src.agentclinic.outcome import MAJORITY_POLICY

    def judge_with(answers):
        it = iter(answers)
        backend = ScriptedBackend(lambda request: json.dumps({"equivalent": next(it), "reason": "r"}))
        return DiagnosisJudge(CachedChat(backend, None), "E", policy=MAJORITY_POLICY), backend

    judge, backend = judge_with([True, True])
    verdict = judge.equivalent("Atrial myxoma", "Possible atrial myxoma")
    assert verdict["equivalent"] is True and verdict["votes"] == [True, True] and len(backend.calls) == 2
    judge, backend = judge_with([True, False, False, True, False])
    verdict = judge.equivalent("A", "B")
    assert verdict["equivalent"] is False and verdict["votes"] == [True, False, False, True, False] and len(backend.calls) == 5
    judge, backend = judge_with([False, True, True, True, False])
    assert judge.equivalent("A", "B")["equivalent"] is True
    single, backend = diagnosis_judge()
    assert single.equivalent("A", "B")["equivalent"] is False and len(backend.calls) == 1  # without the policy: one call, as before
    with pytest.raises(ValueError):
        DiagnosisJudge(CachedChat(ScriptedBackend(lambda request: "{}"), None), "E", policy="other")
