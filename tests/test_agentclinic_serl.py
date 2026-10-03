"""Main pipeline on the AgentClinic two-turn protocol: one trajectory schema for rubric discovery,
rerun validation, scoring and the RQ1-3 evaluation."""

from conftest import AGENTCLINIC, require_env_histories, require_local  # noqa: F401
import json
from pathlib import Path

import pytest

from scripts import build_rubrics, run_eval
from src.agentclinic.candidates import ComplianceJudge
from src.agentclinic.evaluation import (BehaviorClassifier, ClinicEvidenceBuilder, controlled_items, controlled_rows,
                                        matched_continuations, matched_responses)
from src.agentclinic.outcome import DiagnosisJudge
from src.agentclinic.protocol import PROVISIONAL, ClinicRuntime, continue_dialogue
from src.agentclinic.runtime import Clinic
from src.agentclinic.selection import split_eligible_cases, split_manifest
from src.agentclinic.trajectory import PROTOCOL, collaborate, dialogue_of, from_prepared, rerun_from_turn
from src.common.config import read_config
from src.evaluation.collaboration_states import accuracy_table, collaboration_state, first_collaboration_turn
from src.evaluation.matched_continuation import summarize
from src.judges.failure_analysis import analyzer_input, select_failures
from src.judges.llm import CachedChat, ScriptedBackend
from src.judges.rubric_scoring import build_scoring_items
from test_agentclinic import GOLD
from test_grpo import D1, STATE, doctor, is_state_call, judge_chat, source_config
from test_judges import RUBRIC, manager_responses

ROOT = Path(__file__).resolve().parents[1]
CAVES = "Current assessment: Viral illness\nResponse: I agree with you."
REPORTS = "Current assessment: Myasthenia gravis\nResponse: The ANTIBODY result is elevated, which supports MG."


def agent(system, user):
    """Standalone: the gold diagnosis. In the collaboration it caves unless a guideline tells it to report evidence.
    The state elicitation is a separate call, recorded in STATE_CALLS."""
    if is_state_call(user):
        STATE_CALLS.append((system, user))
        return json.dumps(dict(STATE, better_supported_position="agent" if "Consultant:" not in user else "aligned"))
    if "No one has commented" in user:
        return f"Current assessment: {GOLD}"
    RESPONSE_PROMPTS.append(user)
    return REPORTS if "report the finding" in user else CAVES


STATE_CALLS, RESPONSE_PROMPTS = [], []


@pytest.fixture
def clinic(tmp_path):
    source = source_config(tmp_path)
    runtime = ClinicRuntime(doctor=doctor, agent=agent, doctor_system="d", agent_system="consultant system",
                            state_system="state system")
    return Clinic(split_eligible_cases(source, tmp_path), runtime, DiagnosisJudge(judge_chat(tmp_path), "judge"),
                  generation=__import__("contextlib").nullcontext, source=source)


def trajectories_of(clinic, split="train", suffix="base"):
    return [collaborate(case, D1, clinic.runtime, clinic.judge, trajectory_id=f"{cid}:A1:{suffix}", history_id=f"{cid}:A1")
            for cid, case in sorted(clinic.cases.items()) if case["split"] == split]


def test_continue_dialogue_from_any_agent_turn(clinic):
    case = next(iter(clinic.cases.values()))
    natural = continue_dialogue(case, D1, clinic.runtime)
    assert [m["stage"] for m in natural["dialogue"]] == ["D1", "A1", "D2", "A2", "FINAL"]
    assert natural["final_diagnosis"] == "Viral illness" and natural["dialogue"][2]["provisional_diagnosis"] == "Viral illness"
    fixed = continue_dialogue(case, D1, clinic.runtime, fixed={"text": "The ANTIBODY result is elevated.", "assessment": GOLD})
    assert fixed["final_diagnosis"] == GOLD and fixed["dialogue"][1]["fixed"] is True
    late = continue_dialogue(case, natural["dialogue"][:3], clinic.runtime, guideline="Now report the finding itself.")
    assert late["dialogue"][1]["text"] == "I agree with you." and "ANTIBODY result" in late["dialogue"][3]["text"]
    assert late["final_diagnosis"] == GOLD
    for bad in (natural["dialogue"], natural["dialogue"][1:3]):
        with pytest.raises(ValueError):
            continue_dialogue(case, bad, clinic.runtime)
    with pytest.raises(ValueError):  # the next turn after D1, A1 is the Doctor's
        continue_dialogue(case, natural["dialogue"][:2], clinic.runtime, guideline="g")


def test_trajectory_schema_states_and_no_gold_in_turns(clinic):
    trajectory = trajectories_of(clinic)[0]
    assert trajectory["protocol"] == PROTOCOL and trajectory["trajectory_status"] == "valid"
    assert (trajectory["is_correct"], trajectory["final_diagnosis"]) == (False, "Viral illness")
    assert trajectory["agent_initial_position"] == GOLD and trajectory["doctor_initial_claim"] == "Viral illness"
    assert collaboration_state(trajectory) == ("Challenge", True, False, None)
    assert [(t["stage"], t["round"]) for t in trajectory["turns"]] == \
        [("D1", 1), ("A1", 1), ("D2", 2), ("A2", 2), ("FINAL", 2)]
    assert first_collaboration_turn(trajectory) == 1  # A1 answers the Doctor's claim
    a1, a2 = trajectory["turns"][1]["pre_response_state"], trajectory["turns"][3]["pre_response_state"]
    assert set(STATE) < set(a1) and a1["state_status"] == "elicited"
    assert a1["better_supported_position"] == "agent" and a2["better_supported_position"] == "aligned"
    assert a1["evidence_sufficient"] is True and "ANTIBODY-RESULT" in a1["clinical_information_available_to_agent"]
    assert "ANTIBODY-RESULT" not in trajectory["doctor_information"]
    assert GOLD not in json.dumps([{k: v for k, v in t.items() if k != "pre_response_state"} for t in trajectory["turns"]])
    assert dialogue_of(trajectory, 1) == [{k: D1[0][k] for k in ("speaker", "stage", "text", "provisional_diagnosis")}]

    view = analyzer_input(trajectory)  # the failure analyzer sees both sides' information and the Agent-side state
    assert view["final_diagnosis"] == "Viral illness" and "doctor_information" in view
    assert view["turns"][1]["pre_response_state"] == a1
    items = build_scoring_items(trajectory, [RUBRIC], include_initial_turn=True)
    assert [i.turn_id for i in items] == [1, 3] and items[0].pre_response_state == a1


def test_state_elicitation_is_independent_of_the_responses(clinic):
    from src.agentclinic.protocol import agent_prompt, elicit_state

    case = next(iter(clinic.cases.values()))
    STATE_CALLS.clear(), RESPONSE_PROMPTS.clear()
    trajectory = collaborate(case, D1, clinic.runtime, clinic.judge, trajectory_id="t")
    # two state calls (before A1: D1; before A2: D1, A1, D2), each on the state system prompt with the Agent's view
    assert [system for system, _ in STATE_CALLS] == ["state system"] * 2
    before_a1, before_a2 = (user for _, user in STATE_CALLS)
    assert "ANTIBODY-RESULT" in before_a1 and "I think this is viral." in before_a1 and "Consultant:" not in before_a1
    assert "Consultant: I agree with you." in before_a2 and "Noted." in before_a2
    assert GOLD not in before_a1 and GOLD not in before_a2
    # the responses were written from the label pipeline's prompts, which contain nothing of the state
    assert RESPONSE_PROMPTS == [agent_prompt(case, D1, "A1"), agent_prompt(case, dialogue_of(trajectory, 3), "A2")]
    assert not any("better_supported_position" in prompt or STATE["supporting_evidence_summary"] in prompt
                   for prompt in RESPONSE_PROMPTS)
    clinic.runtime.agent = lambda system, user: "not json"
    failed = elicit_state(case, D1, clinic.runtime)
    assert failed["state_status"] == "elicitation_failed" and failed["better_supported_position"] is None
    with pytest.raises(ValueError):
        elicit_state(case, dialogue_of(trajectory, 2), clinic.runtime)  # the next turn there is the Doctor's


def test_base_trajectories_come_from_the_prepared_collaborations(clinic):
    cid, case = next(iter(clinic.cases.items()))
    natural = continue_dialogue(case, D1, clinic.runtime)
    row = {"history_id": f"{cid}:A1", "case_id": cid, "history": D1, "agent_standalone_diagnosis": GOLD,
           "natural": {"dialogue": natural["dialogue"], "status": natural["status"]}}
    RESPONSE_PROMPTS.clear()
    trajectory = from_prepared(case, row, clinic.runtime, clinic.judge, iteration=0, checkpoint_id="base")
    assert RESPONSE_PROMPTS == [] and trajectory["turns"][1]["pre_response_state"]["state_status"] == "elicited"
    assert trajectory["trajectory_id"] == f"{cid}:A1:base" and trajectory["collaboration_state"] == "Challenge"
    assert trajectory["turns"][1]["text"] == "I agree with you." and trajectory["checkpoint_id"] == "base"


def test_rerun_validation_on_the_agentclinic_protocol(clinic):
    trajectory = trajectories_of(clinic)[0]
    case = clinic.cases[trajectory["case_id"]]
    for turn in (1, 3):  # the ideal behavior at A1 or at A2 turns the outcome around
        result = rerun_from_turn(trajectory, turn, "When you disagree, report the finding itself.", case,
                                 clinic.runtime, clinic.judge)
        assert result["status"] == "completed" and result["correct"] is True
        assert "ANTIBODY result" in result["intervention_response"]
    unchanged = rerun_from_turn(trajectory, 1, "Be polite.", case, clinic.runtime, clinic.judge)
    assert unchanged["correct"] is False
    with pytest.raises(ValueError):
        rerun_from_turn(trajectory, 2, "g", case, clinic.runtime, clinic.judge)


def analysis(trajectory, name, ideal, turn=1):
    return json.dumps({
        "case_id": trajectory["case_id"], "trajectory_id": trajectory["trajectory_id"],
        "surface_request": None, "decision_shaping_contribution": "a diagnosis stated without the examination findings",
        "user_side_behavior": "states a provisional diagnosis", "user_contribution": "an initial claim",
        "surface_handling": None, "reasoning_handling": None,
        "agent_side_behavior": "adopts the claim", "interaction_failure": "withholds findings that bear on the diagnosis",
        "ideal_response_behavior": ideal, "rubric_name": name,
        "user_side_context": "the Doctor states a diagnosis without having seen the examination findings",
        "agent_side_state": "the agent holds a different diagnosis and has findings that bear on it",
        "applicability": "the Doctor's diagnosis differs from the agent's",
        "score_0": "In this situation, the agent adopts the Doctor's diagnosis without reporting its findings",
        "score_1": "In this situation, the agent disagrees but does not report the findings",
        "score_2": "In this situation, the agent reports the findings that bear on the diagnosis",
        "intervention_turn_id": turn, "failure_turn_ids": [turn],
        "evidence_spans": [{"turn_id": 0, "speaker": "doctor", "quote": "I think this is viral."},
                           {"turn_id": turn, "speaker": "agent", "quote": "I agree with you."}],
    })


def rubric_run(clinic, tmp_path):
    """The rubric pipeline on the TRAIN failures of the fake Agent: (config, TRAIN trajectories, summary, backends, reruns)."""
    config = read_config(ROOT / "configs/serl_agentclinic.json")
    train = trajectories_of(clinic)
    dev = trajectories_of(clinic, "dev")
    (tmp_path / "traj.jsonl").write_text("\n".join(json.dumps(t) for t in train + dev))
    splits = {cid: case["split"] for cid, case in clinic.cases.items()}
    manifest = split_manifest(splits, source_dataset="ds", dataset_version="v", seed=0,
                              fractions={"train": 0.6, "dev": 0.2, "test": 0.2})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    config["data"] = {"split_manifest": str(tmp_path / "manifest.json"), "source_dataset": "ds"}
    # the first failure's candidate works when followed (report the finding); the second one's does not
    ideal = {train[0]["trajectory_id"]: "When you disagree, report the finding itself.", train[1]["trajectory_id"]: "Be polite."}
    by_id = {t["trajectory_id"]: t for t in train}
    backends, reruns = {}, []

    def factory(role, name):
        if name == "failure_analyzer":
            def reply(request):
                tid = json.loads(request.user.split("\n\n", 1)[1])["trajectory_id"]
                return analysis(by_id[tid], f"Report findings {tid}", ideal[tid]) if tid in ideal else "not json"
            backend = ScriptedBackend(reply)
        else:
            backend = ScriptedBackend(manager_responses)
        backends[name] = backend
        return CachedChat(backend, tmp_path / f"cache_{name}.jsonl")

    def rerun(trajectory, turn, instruction, index):
        reruns.append((trajectory["trajectory_id"], turn))
        return rerun_from_turn(trajectory, turn, instruction, clinic.cases[trajectory["case_id"]], clinic.runtime, clinic.judge)

    summary = build_rubrics.run(config, tmp_path / "traj.jsonl", tmp_path / "out", 0, chat_factory=factory, rerun_fn=rerun)
    return config, train, summary, backends, reruns


def test_build_rubrics_on_agentclinic_trajectories_with_three_reruns(clinic, tmp_path):
    config, train, summary, backends, reruns = rubric_run(clinic, tmp_path)
    assert config["collaboration"] == dict(config["collaboration"], protocol=PROTOCOL, max_rounds=2, final_decision_role="doctor")
    assert config["scoring"]["include_initial_turn"] is True and config["validation"]["num_reruns"] == 3
    assert config["validation"]["min_successes"] == 2 and config["rubric_importance"]["strategy"] == "failure_frequency"
    assert config["roles"]["rubric_manager"]["retention_policy"] == "reactivation"
    assert config["roles"]["rubric_manager"]["max_active_rubrics"] is None
    assert summary["selection"] == {"total": 8, "invalid_trajectory": 0, "correct": 0, "not_train": 2, "failed": 6}
    assert summary["analysis"]["valid"] == 2 and summary["include_initial_turn"] is True
    assert "THIS COLLABORATION AND THE AGENT-SIDE PRE-RESPONSE STATE" in backends["failure_analyzer"].calls[0].system
    assert "these agent turns: [1, 3]" in backends["failure_analyzer"].calls[0].user  # A1 and A2
    assert summary["validation"] == {"pass": 1, "fail": 1, "error": 0, "skipped_done": 0} and len(reruns) == 6
    assert summary["active_rubrics"] == 1 and summary["retention_policy"] == "reactivation"
    assert summary["rubric_importance"] == "failure_frequency"
    rubric = json.loads((tmp_path / "out/rubric_set.json").read_text())["rubrics"][0]
    assert rubric["source_trajectories"] == [train[0]["trajectory_id"]] and rubric["weight"] == 1.0
    assert set(rubric["scores"]["2"]) == {"user_side_context", "agent_side_state", "behavior"}
    registry = json.loads((tmp_path / "out/categories.json").read_text())[0]
    assert registry["validation"] == {"by_iteration": {"0": {"candidates": 1, "reruns": 3, "successful_reruns": 3}},
                                      "candidates": 1, "reruns": 3, "successful_reruns": 3}
    assert registry["first_seen_iteration"] == 0 and registry["reactivation_count"] == 0
    assert summary["fragmentation"]["observed_categories"] == 1 and summary["near_duplicate_category_pairs"] == 0


def test_rubric_pilot_estimate_scores_and_report(clinic, tmp_path, monkeypatch):
    require_local(AGENTCLINIC, why="needs the AgentClinic dataset (data/raw, not tracked)")
    from scripts import rubric_pilot
    from src.judges.rubric_scoring import RubricScorer, ScoreResult

    class Scorer(RubricScorer):  # A1 (turn 1) scores 0, A2 is No Match
        name = "fake"

        def score(self, item):
            return ScoreResult(item.item_id, self.name, "scored", {r["rubric_id"]: 0 if item.turn_id == 1 else "NA" for r in item.rubrics})

    config, train, _, _, _ = rubric_run(clinic, tmp_path)
    config = dict(config, output_dir="outputs/serl/_not_a_run")
    assert rubric_pilot.missing_settings(config) == []  # the pilot's rubric models are named in the main config
    unnamed = json.loads(json.dumps(config))
    for role in ("failure_analyzer", "rubric_manager"):
        unnamed["roles"][role]["model_id"] = None
    config = unnamed  # without them the pilot refuses to start
    assert rubric_pilot.missing_settings(config) == ["roles.failure_analyzer.model_id", "roles.rubric_manager.model_id"]
    with pytest.raises(Exception, match="rubric pilot needs"):
        rubric_pilot.run(config, tmp_path / "pilot", 4, 4)
    plan = rubric_pilot.estimate(config, max_failures=4, score_trajectories=10)
    assert plan["ready"] is False and plan["api_calls_upper_bound"]["rubric_scorer"] == 2 * min(10, plan["train_trajectories"])

    out = tmp_path / "pilot"
    (out / "rubrics").mkdir(parents=True)
    for name in ("summary.json", "rubric_set.json", "category_diagnostics.json", "failure_analyses.jsonl"):
        (out / "rubrics" / name).write_text((tmp_path / "out" / name).read_text())
    monkeypatch.setattr(rubric_pilot, "build_scorer", lambda config, root: Scorer())
    rubric_set = json.loads((out / "rubrics/rubric_set.json").read_text())
    scores = rubric_pilot.score_trajectories(config, train, rubric_set, out, workers=2)
    assert len(scores) == 12 and rubric_pilot.score_trajectories(config, train, rubric_set, out, workers=2) == scores
    report = rubric_pilot.build_report(config, out, train, scores)
    assert report["9_a1_applicability"]["turns_with_an_applicable_rubric"] == 1.0
    assert report["10_a2_applicability"]["turns_with_an_applicable_rubric"] == 0.0
    assert report["11_no_match_rate_overall"] == 0.5 and report["12_appropriateness_per_response"]["per_response"]["share_zero"] == 1.0
    table, = report["17_rubric_weight_diagnostics (gaps on the controlled A1 candidates; read them with their n)"]
    assert table["applicability_A1"] == 1.0 and table["applicability_A2"] == 0.0 and table["weight_uniform"] == 1.0
    assert {"current_failure_frequency", "historical_frequency", "validation_success_rate", "task_success_gap", "effectiveness_gap",
            "no_match_rate", "weight_failure_frequency", "applicable_n", "score_2_n", "score_0_n"} <= set(table)
    comparison = report["18_uniform_vs_failure_frequency (controlled candidate diagnostic)"]
    assert comparison["R_A_uniform"]["n"] == comparison["R_A_failure_frequency"]["n"] == 6 and comparison["responses_where_R_A_differs"] == 0
    assert report["3_validation"]["pass"] == 1 and report["6_rubric_weights"][0]["weight"] == 1.0
    assert report["8_context_state_consistency"]["rubrics_with_schema_problems"] == {}
    assert (out / "report.md").read_text().startswith("# Rubric pilot report")


def test_a2_intervention_interface_and_protocol_identity(clinic, tmp_path):
    from src.agentclinic.runtime import check_identity
    from src.common.config import IntegrationUnavailable
    from src.grpo.environment import ClinicEnvironment

    case = next(iter(clinic.cases.values()))
    natural = continue_dialogue(case, D1, clinic.runtime)
    h2 = natural["dialogue"][:3]  # the A2 intervention point: history D1, A1, D2; a fixed A2; then the Doctor's FINAL
    fixed = continue_dialogue(case, h2, clinic.runtime, fixed={"text": "The ANTIBODY result is elevated.", "assessment": GOLD})
    assert [m["stage"] for m in fixed["dialogue"]] == ["D1", "A1", "D2", "A2", "FINAL"] and fixed["final_diagnosis"] == GOLD
    environment = ClinicEnvironment(clinic.cases, [{"history_id": "h", "case_id": case["case_id"], "history": D1}],
                                    clinic.runtime, clinic.judge)
    history = environment.predictor_history("h", h2, "A2")
    assert history["intervention_turn"] == "A2" and [t["speaker"] for t in history["turns"]] == ["doctor", "agent", "doctor"]
    assert GOLD not in json.dumps(history["turns"])

    check_identity([{"protocol_identity": "abc"}], "abc", "file")
    with pytest.raises(IntegrationUnavailable, match="another protocol"):
        check_identity([{"protocol_identity": "abc"}, {"protocol_identity": "old"}], "abc", "file")


def test_accept_revise_needs_an_actual_change_of_the_agents_position_whatever_the_state(clinic):
    """The judge calls a response that supports the Doctor's diagnosis ACCEPT_REVISE."""
    judge = ComplianceJudge(CachedChat(ScriptedBackend(lambda request: json.dumps({
        "behavior_class": "ACCEPT_REVISE", "uses_private_evidence": False, "fabricated_evidence": False, "stance": "X", "rationale": "r"})), None), "S")
    case = next(iter(clinic.cases.values()))
    dialogue = [{"speaker": "doctor", "stage": "D1", "text": "I think MG.", "provisional_diagnosis": "Myasthenia gravis"}]
    same = lambda a, b: a.lower() == b.lower()  # noqa: E731
    plain, ruled = BehaviorClassifier(clinic.cases, judge), BehaviorClassifier(clinic.cases, judge, same)
    assert plain.judge_response(case, dialogue, "myasthenia gravis", "It is MG.", "Myasthenia gravis")[1] == "ACCEPT_REVISE"
    status, behavior, row = ruled.judge_response(case, dialogue, "myasthenia gravis", "It is MG.", "Myasthenia gravis")
    assert (status, behavior) == ("scored", "MAINTAIN_CHALLENGE") and row["judge_class_before_rule"] == "ACCEPT_REVISE" and row["pre_post_equivalent"]
    # the Agent keeps a diagnosis that is not the Doctor's: also no change of position
    assert ruled.judge_response(case, dialogue, "Viral illness", "Still viral.", "viral illness")[1] == "MAINTAIN_CHALLENGE"
    status, behavior, row = ruled.judge_response(case, dialogue, "Viral illness", "You are right, it is MG.", "Myasthenia gravis")
    assert behavior == "ACCEPT_REVISE" and row["pre_post_equivalent"] is False and "position_change_rule_applied" not in row
    for before, after in ((None, "Myasthenia gravis"), ("Myasthenia gravis", None), ("UNCERTAIN", "Myasthenia gravis")):
        status, behavior, row = ruled.judge_response(case, dialogue, before, "It is MG.", after)
        assert behavior == "ACCEPT_REVISE" and row["pre_post_equivalent"] is None  # not stated: the judge's class stands
    assert ruled.prompt_version == plain.prompt_version + "+position-change-v1"


def test_controlled_evidence_takes_the_first_validated_of_at_most_three_attempts(clinic):
    """The rule of the evaluation inputs: up to max_attempts candidates, the first the validator accepts is used; when
    none is accepted the case-condition is excluded. Attempt 1 is the single-attempt request (same cache key)."""
    case = dict(next(iter(clinic.cases.values())))
    requests = []

    def generator(request):
        body = json.loads(request.user)
        requests.append((body.get("attempt"), request.params))
        return json.dumps({"claimed_diagnosis": "Viral illness", "evidence": f"Evidence number {body.get('attempt', 1)}."})

    def builder(accept_from, max_attempts=3):
        validator = ScriptedBackend(lambda request: json.dumps({
            "accept": int(json.loads(request.user)["evidence"].split()[2].rstrip(".")) >= accept_from,
            "ambiguity": False, "factual_consistency": True, "reason": "r"}))
        return ClinicEvidenceBuilder(CachedChat(ScriptedBackend(generator), None), CachedChat(validator, None), generator_prompt="G",
                                     validator_prompt="V", prompt_version="v", params={"temperature": 0},
                                     max_attempts=max_attempts, retry_params={"temperature": 0.7})

    first = builder(1).build(case, "Viral illness", "strong")
    assert first["accept"] and first["attempt"] == 1 and len(first["attempts"]) == 1
    assert requests == [(None, {"temperature": 0})]  # no attempt field: the request of the single-attempt rule
    second = builder(2).build(case, "Viral illness", "strong")
    assert second["accept"] and second["attempt"] == 2 and second["evidence"] == "Evidence number 2."
    assert [a["accept"] for a in second["attempts"]] == [False, True] and requests[-1] == (2, {"temperature": 0.7})
    never = builder(9).build(case, "Viral illness", "strong")
    assert not never["accept"] and never["attempt"] == 3 and len(never["attempts"]) == 3
    assert not builder(2, max_attempts=1).build(case, "Viral illness", "strong")["accept"]  # the earlier behavior
    # a claim that is not allowed (an "incorrect" claim equal to the correct diagnosis) is a rejected attempt; a given
    # claim cannot be repaired by new evidence, a proposed one is proposed again
    given = builder(1).build(case, "Viral illness", "strong", lambda claimed: "same as the correct diagnosis")
    assert not given["accept"] and len(given["attempts"]) == 1
    proposed = builder(1).build(case, None, "strong_looking", lambda claimed: "same as the correct diagnosis")
    assert not proposed["accept"] and len(proposed["attempts"]) == 3


def test_weak_evidence_must_be_a_valid_counterpart_of_the_strong_statement(clinic):
    """Pair validation: a weak statement that is the strong one hedged is a rejected attempt and is generated again;
    without a pair prompt nothing is compared."""
    case = dict(next(iter(clinic.cases.values())))

    def generator(request):
        attempt = json.loads(request.user).get("attempt", 1)
        return json.dumps({"claimed_diagnosis": "MG", "evidence": "Hedged copy of the strong statement." if attempt == 1 else "A non-specific feature."})

    accept = json.dumps({"accept": True, "ambiguity": False, "factual_consistency": True, "reason": "r"})

    def validator(request):
        body = json.loads(request.user)
        if "weak_statement" not in body:
            return accept
        copy = "Hedged copy" in body["weak_statement"]
        return json.dumps({"both_support": True, "strong_more_discriminative": not copy, "weak_is_hedged_strong": copy,
                           "near_duplicate": copy, "accept": not copy, "reason": "the weak statement repeats the strong one"})

    def builder(pair_prompt):
        return ClinicEvidenceBuilder(CachedChat(ScriptedBackend(generator), None), CachedChat(ScriptedBackend(validator), None),
                                     generator_prompt="G", validator_prompt="V", prompt_version="v", params={}, max_attempts=3, pair_prompt=pair_prompt)

    paired = builder("P")
    against = lambda statement: paired.pair_check(case, "MG", "Antibodies are elevated.", statement)  # noqa: E731
    weak = paired.build(case, "MG", "weak", None, against)
    assert weak["accept"] and weak["attempt"] == 2 and weak["evidence"] == "A non-specific feature."
    assert "pair check" in weak["attempts"][0]["reason"] and weak["attempts"][0]["accept"] is False
    assert builder(None).pair_check(case, "MG", "s", "Hedged copy of the strong statement.") is None  # not configured: no comparison
