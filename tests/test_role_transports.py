"""Sol as the writer of the Effectiveness candidates and Jev as the maker of bounded decisions (scores, compliance):
schema tests with a synthetic transport. No call leaves the machine."""

import json
import urllib.error
from pathlib import Path

import pytest

from scripts import run_agentclinic
from src.agentclinic.candidates import BEHAVIOR_CLASSES, ComplianceJudge, JevComplianceJudge
from src.agentclinic.pipeline import generate_candidates, turn_histories
from src.agentclinic.protocol import ASSESSMENT, ProtocolSpec, continue_dialogue
from src.agentclinic.trajectory import collaborate
from src.common.config import IntegrationUnavailable, read_config
from src.judges.jev import JevClient, parse_answers
from src.judges.rubric_scoring import JevRubricScorer, ScoringItem
from test_agentclinic import GOLD, cases, diagnosis_judge, fake_runtime  # noqa: F401 (cases is a fixture)
from test_judges import schema_rubric

ROOT = Path(__file__).resolve().parents[1]
MAX7 = ProtocolSpec(max_turns=7, early_stopping=True)


def client(answer, sent=None, **extra):
    """A Jev client whose transport is `answer(payload) -> response body`."""
    def post(url, payload, key, timeout_s):
        if sent is not None:
            sent.append({"url": url, "payload": payload, "key": key})
        return answer(payload)
    return JevClient(endpoint="https://jev.invalid/v1/systemone", model="jev-test", api_key_env="JEV_TEST_KEY", post=post,
                     sleep=lambda s: None, **extra)


def typed(payload, choose):
    """A response in the documented shape: an answer of the asked type for every question."""
    answers = {}
    for qid, question in payload["questions"].items():
        value = choose(qid, question)
        answers[qid] = ({"type": "choice", "choice": value, "probabilities": {o: (0.7 if o == value else 0.1) for o in question["criteria"]},
                         "confidence": 0.7} if question["type"] == "choice" else {"type": "noul", "noul": value})
    return {"model": "typesafe/jev-test-0001", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0, "cost": 1e-6},
            "id": "gen-dec-test", "provider": "TypeSafe"}


def test_jev_request_and_response_schema(monkeypatch):
    monkeypatch.setenv("JEV_TEST_KEY", " secret ")
    sent = []
    questions = {"q1": {"type": "choice", "criteria": {"a": "first", "b": "second"}, "instructions": "pick"},
                 "q2": {"type": "noul", "instructions": "yes?"}}
    jev = client(lambda p: typed(p, lambda qid, q: "b" if q["type"] == "choice" else 0.25), sent)
    answers, call = jev.decide("state text", questions)
    assert sent[0]["payload"] == {"model": "jev-test", "state": "state text", "questions": questions} and sent[0]["key"] == "secret"
    assert answers["q1"] == {"value": "b", "probabilities": {"a": 0.1, "b": 0.7}, "confidence": 0.7} and answers["q2"]["value"] == 0.25
    assert call["provider"] == "jev" and call["attempts"] == 1 and jev.calls == 1 and call["model"] == "typesafe/jev-test-0001"
    # a response the transport does not recognize is an error, never a decision
    for body, message in (({"text": "b"}, "no `answers`"), ({"answers": {"q1": {"choice": "b"}}}, "no noul answer for question q2"),
                          ({"answers": {"q1": {"choice": "c"}, "q2": {"noul": 0.5}}}, "not one of its options"),
                          ({"answers": {"q1": {"choice": "a"}, "q2": {"noul": "yes"}}}, "a probability was expected")):
        with pytest.raises(IntegrationUnavailable, match=message):
            parse_answers(body, questions)
    with pytest.raises(ValueError):
        jev.decide("s", {"q": {"type": "essay"}})  # only typed questions exist
    # a transport error is retried; a missing key or a missing setting stops
    failures = iter([urllib.error.URLError("down"), urllib.error.URLError("down")])

    def flaky(payload):
        try:
            raise next(failures)
        except StopIteration:
            return typed(payload, lambda qid, q: "a" if q["type"] == "choice" else 1.0)
    assert client(flaky).decide("s", questions)[1]["attempts"] == 3
    with pytest.raises(IntegrationUnavailable, match="failed after"):
        client(lambda p: (_ for _ in ()).throw(urllib.error.URLError("down")), max_retries=1).decide("s", questions)
    monkeypatch.delenv("JEV_TEST_KEY")
    with pytest.raises(IntegrationUnavailable, match="JEV_TEST_KEY is not set"):
        jev.decide("s", questions)
    with pytest.raises(IntegrationUnavailable, match="not configured"):
        JevClient(endpoint=None, model=None, api_key_env=None)


def test_jev_rubric_scorer_returns_labels_and_probabilities_without_text(monkeypatch):
    monkeypatch.setenv("JEV_TEST_KEY", "k")
    rubrics = tuple(dict(schema_rubric(f"Rubric {i}"), rubric_id=f"r{i}") for i in range(3))
    item = ScoringItem("i", "case", "t", 1, ({"turn_id": 0, "speaker": "doctor", "text": "D says X"},
                                             {"turn_id": 1, "speaker": "agent", "text": "A answers"}), rubrics,
                       pre_response_state={"better_supported_position": "agent"})
    sent = []
    scorer = JevRubricScorer("jev", {}, client=client(lambda p: typed(p, lambda qid, q: {"r0": "2", "r1": "NA", "r2": "0"}[qid]), sent))
    result = scorer.score(item)
    assert result.status == "scored" and result.labels == {"r0": 2, "r1": "NA", "r2": 0} and result.rationales == {}
    assert result.probabilities["r0"]["2"] == 0.7 and set(result.probabilities["r1"]) == {"NA", "0", "1", "2"}
    payload = sent[0]["payload"]
    assert set(payload["questions"]) == {"r0", "r1", "r2"} and list(payload["questions"]["r0"]["criteria"]) == ["NA", "0", "1", "2"]
    assert "level 2" in payload["questions"]["r0"]["criteria"]["2"] and "clinician repeats a claim" in payload["questions"]["r0"]["instructions"]
    # many rubrics go in several calls of max_rubrics_per_call (Jev's context is limited); the labels are the same
    split = JevRubricScorer("jev", {"max_rubrics_per_call": 2}, client=client(lambda p: typed(p, lambda qid, q: {"r0": "2", "r1": "NA", "r2": "0"}[qid]), sent))
    assert split.score(item).labels == result.labels and [len(s["payload"]["questions"]) for s in sent[1:]] == [2, 1]
    state = json.loads(payload["state"])  # what the scorer sees: the dialogue up to the target turn and the pre-turn state
    assert [t["text"] for t in state["dialogue_through_target_turn"]] == ["D says X", "A answers"] and "rubrics" not in state
    assert state["agent_pre_response_state"] == {"better_supported_position": "agent"}
    assert not any(word in sent[0]["payload"]["state"] for word in ("final_diagnosis", "Correct_Diagnosis", GOLD))
    # an unconfigured scorer does not score
    with pytest.raises(IntegrationUnavailable, match="not configured"):
        JevRubricScorer("jev", {"endpoint": None, "model_id": None, "api_key_env": None}).score(item)


def test_jev_compliance_judge_gives_the_same_row_as_the_written_judge(cases, monkeypatch):
    monkeypatch.setenv("JEV_TEST_KEY", "k")
    case = cases[0]
    history = [{"speaker": "doctor", "stage": "D1", "text": "I think this is viral.", "provisional_diagnosis": "Viral illness"}]

    def answer(behavior, private, fabricated):
        return lambda p: typed(p, lambda qid, q: {"behavior_class": behavior, "uses_private_evidence": private, "fabricated_evidence": fabricated}[qid])

    sent = []
    judge = JevComplianceJudge(client(answer("MAINTAIN_CHALLENGE", 0.9, 0.05), sent))
    row = judge.judge(case, history, "MG", "The antibody result is elevated.", "evidence_grounded_maintain")
    assert (row["observed_behavior"], row["uses_private_evidence"], row["fabricated_evidence"], row["behavior_compliant"]) == (
        "MAINTAIN_CHALLENGE", True, False, True)
    assert row["judge_rationale"] is None and row["judge_probabilities"]["uses_private_evidence"] == 0.9 and row["judge_status"] == "scored"
    assert row["judge_version"] == judge.version == "jev-compliance-v1+jev-test"
    questions = sent[0]["payload"]["questions"]
    assert list(questions["behavior_class"]["criteria"]) == list(BEHAVIOR_CLASSES) and questions["fabricated_evidence"]["type"] == "noul"
    assert "agent_only_evidence" in json.loads(sent[0]["payload"]["state"]) and GOLD not in sent[0]["payload"]["state"]
    # the other decisions: wrong behavior, fabricated finding, missing evidence
    for behavior, private, fabricated, name, expected in (("ACCEPT_REVISE", 0.9, 0.05, "evidence_grounded_maintain", False),
                                                           ("MAINTAIN_CHALLENGE", 0.9, 0.8, "evidence_grounded_maintain", False),
                                                           ("MAINTAIN_CHALLENGE", 0.2, 0.05, "evidence_grounded_maintain", False),
                                                           ("MAINTAIN_CHALLENGE", 0.2, 0.05, "unsupported_maintain", True),
                                                           ("ACCEPT_REVISE", 0.1, 0.0, "accept_revise", True)):
        got = JevComplianceJudge(client(answer(behavior, private, fabricated))).judge(case, history, "MG", "r", name)
        assert got["behavior_compliant"] is expected, (behavior, private, fabricated, name)


def test_label_roles_select_the_generator_and_the_judge_and_do_not_touch_the_protocol_identity(cases, tmp_path, monkeypatch):
    from scripts.run_base_collaborations import identity
    from src.judges.llm import ChatResult

    config = read_config(ROOT / "configs/effectiveness/v10_qwen4b_gemma12b_max7.json")
    roles = config["label_roles"]
    assert (roles["candidate_generator"]["role"], roles["compliance_judge"]["role"]) == ("Sol", "Jev")
    assert roles["candidate_generator"]["model_id"] == "openai/gpt-6.1-sol"  # the pinned version, not the latest alias
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")
    assert callable(run_agentclinic.candidate_generator(config))
    unnamed = dict(config, label_roles=dict(roles, candidate_generator=dict(roles["candidate_generator"], model_id=None)))
    assert run_agentclinic.candidate_generator(unnamed) is None  # without a named model the Agent model writes them
    # Jev is the compliance judge: the direct decision model on OpenRouter's Decisions API, not the Jev router
    assert (roles["compliance_judge"]["model_id"], roles["compliance_judge"]["endpoint"]) == ("typesafe/jev-1.13", "https://openrouter.ai/api/alpha/decisions")
    assert isinstance(run_agentclinic.compliance_judge(config, tmp_path), JevComplianceJudge)
    without = dict(config, label_roles=dict(roles, compliance_judge=dict(roles["compliance_judge"], model_id=None)))
    assert isinstance(run_agentclinic.compliance_judge(without, tmp_path), ComplianceJudge)
    named = dict(config, label_roles={"candidate_generator": dict(roles["candidate_generator"], model_id="vendor/sol-test"),
                                      "compliance_judge": dict(roles["compliance_judge"], endpoint="https://jev.invalid", model_id="jev-test",
                                                               api_key_env="JEV_TEST_KEY")})
    assert identity(named) == identity(config) == "2c07b9ef7b0ab2cb"  # who writes and judges candidates is not the environment
    assert isinstance(run_agentclinic.compliance_judge(named, tmp_path), JevComplianceJudge)

    seen = []

    class Backend:
        def complete(self, request):
            seen.append(request)
            text = (f"{ASSESSMENT} {GOLD}\nResponse: The antibody result is elevated, which supports MG." if "stating the finding itself" in request.user
                    else f"{ASSESSMENT} Viral illness\nResponse: I agree with your assessment.")
            return ChatResult("ok", text, "fake", "vendor/sol-test", None, {}, 0.0, 1)

    monkeypatch.setattr("src.judges.llm.build_chat_backend", lambda settings: Backend())
    generator = run_agentclinic.candidate_generator(named)
    # the generator writes the candidates from the Agent's own prompt (its information, the history, the behavior instruction);
    # the Agent model writes nothing at that stage and still writes every continuation
    from dataclasses import replace

    calls = []
    runtime, _ = fake_runtime(calls=calls)
    judge, _ = diagnosis_judge(tmp_path / "judge.jsonl")
    by_id = {cases[0]["case_id"]: dict(cases[0], split="train")}
    case = by_id[cases[0]["case_id"]]
    d1 = continue_dialogue(case, [], runtime)["dialogue"][:1]
    trajectory = collaborate(case, d1, runtime, judge, trajectory_id="t:A1:base", standalone="MG", history_id=f"{case['case_id']}:A1")
    histories = turn_histories([trajectory])[:1]
    calls.clear()
    monkeypatch.setenv("JEV_TEST_KEY", "k")
    compliance = JevComplianceJudge(client(lambda p: typed(p, lambda qid, q: {
        "behavior_class": "MAINTAIN_CHALLENGE" if "antibody" in json.loads(p["state"])["agent_response"] else "ACCEPT_REVISE",
        "uses_private_evidence": 0.9 if "antibody" in json.loads(p["state"])["agent_response"] else 0.1, "fabricated_evidence": 0.0}[qid])))
    summary = generate_candidates(by_id, histories, replace(runtime, agent=generator), compliance, tmp_path, workers=1, max_attempts=2,
                                  stage="A1", names=("evidence_grounded_maintain", "accept_revise"))
    assert summary["compliant"] == {"evidence_grounded_maintain": "1/1", "accept_revise": "1/1"}
    assert [who for who, _ in calls] == [] and len(seen) == 2  # the Agent model wrote no candidate
    assert all(r.system == runtime.agent_system and "Clinical information available to you" in r.user and GOLD not in r.system for r in seen)
    stored = [json.loads(line) for line in (tmp_path / "candidate_attempts.jsonl").read_text().splitlines()]
    assert {s["judge_version"] for s in stored} == {"jev-compliance-v1+jev-test"} and all(s["judge_probabilities"] for s in stored)
    # a second pass keeps the judgments of the same judge version: no new generation, no new judgment
    before = len(seen)
    generate_candidates(by_id, histories, replace(runtime, agent=generator), compliance, tmp_path, workers=1, max_attempts=2,
                        stage="A1", names=("evidence_grounded_maintain", "accept_revise"))
    assert len(seen) == before and compliance.client.calls == 2
