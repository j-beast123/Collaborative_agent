"""Offline tests for PPT Step 2-1 (rerun validation), 2-3 (weighting) and 3 (reward)."""

import json
import math

import pytest

from scripts import build_rubrics
from src.collaboration.generator import RECONSTRUCTION_FIELDS
from src.collaboration.rerun import intervention_instruction, rerun_from_turn
from src.collaboration.runtime import CollaborationRuntime
from src.judges.llm import CachedChat, ScriptedBackend
from src.judges.reward import RewardSpec, TurnSignals, compute_reward, soft_min, turn_appropriateness
from src.judges.rubric_validation import load_validations, passed_records, validate_candidates
from test_judges import CONFIG, analysis, manager_responses, schema_rubric, trajectory

JUDGE_SYSTEM = "JUDGE-SYSTEM"


def rich_trajectory(tid, **kw):
    t = trajectory(tid, **kw)
    t.update(max_discussion_rounds=3, max_agent_turns=3, belief_enabled=True)
    return t


def fake_runtime(final_text="Final answer: option A - a", agent_text="I will verify the evidence first."):
    calls = {"agent_prompts": [], "reconstruct": 0, "judge": 0, "doctor": []}

    def agent(system, user, tokens):
        if system == JUDGE_SYSTEM:
            calls["judge"] += 1
            return json.dumps({"q1": {"answer": "A2", "reason": "r"}, "q2": {"answer": "Y", "reason": "r"}})
        calls["agent_prompts"].append(user)
        return agent_text

    def reconstruct(case, dialogue):
        calls["reconstruct"] += 1
        return dict({k: [] if k.endswith(("so_far", "evidence", "points", "details")) else "x"
                     for k in RECONSTRUCTION_FIELDS}, parse_status="valid")

    def doctor(system, user):
        calls["doctor"].append(user)
        return final_text if "This is the final decision" in user else "Let us keep discussing."

    runtime = CollaborationRuntime(doctor, agent, reconstruct, "DOC", "AGENT", JUDGE_SYSTEM, 64)
    return runtime, calls


def test_rerun_uses_generation_protocol_and_injects_only_at_failed_turn():
    runtime, calls = fake_runtime()
    instruction = intervention_instruction("Verify the claim against evidence before revising.")
    result = rerun_from_turn(rich_trajectory("t1"), 3, instruction, runtime)
    assert result["status"] == "completed" and result["correct"] is True and result["final_option"] == "A"
    # Failed turn is round 2 of 3: agent answers rounds 2 and 3, belief computed for both.
    assert [t["round"] for t in result["dialogue"] if t["speaker"] == "agent"] == [1, 2, 3]
    assert calls["reconstruct"] == 2 and calls["judge"] == 2
    assert instruction in calls["agent_prompts"][0] and instruction not in calls["agent_prompts"][1]
    assert "Private current Q1/Q2 judgment" in calls["agent_prompts"][0]  # belief prompt, not plain
    assert result["dialogue"][3]["intervention"] is True
    # History before the failed turn is kept verbatim.
    assert result["dialogue"][:3] == [{k: v for k, v in t.items() if k in ("speaker", "text", "round")}
                                      for t in rich_trajectory("t1")["turns"][:3]]
    with pytest.raises(ValueError):
        rerun_from_turn(rich_trajectory("t1"), 2, instruction, runtime)  # doctor turn


def test_rerun_wrong_final_answer_is_not_correct():
    runtime, _ = fake_runtime(final_text="Final answer: option B - b")
    assert rerun_from_turn(rich_trajectory("t1"), 3, "x", runtime)["correct"] is False


def valid_record(tid):
    from src.judges.failure_analysis import candidate_rubric

    parsed = json.loads(analysis(tid, f"rubric {tid}"))
    return {"rubric_id": f"cand-{tid}", "source_trajectory_id": tid, "source_case_id": f"case-{tid}",
            "analysis_status": "valid", "analysis": parsed, "candidate_rubric": candidate_rubric(parsed)}


def test_validate_candidates_pass_fail_error_and_resume(tmp_path):
    records = [valid_record("t1"), valid_record("t2"), valid_record("t3")]
    trajectories = {tid: rich_trajectory(tid) for tid in ("t1", "t2", "t3")}
    outcomes = {"t1": [True, True], "t2": [True, False]}

    def rerun(trajectory, turn, instruction, index):
        if trajectory["trajectory_id"] == "t3":
            raise RuntimeError("API down")
        return {"status": "completed", "correct": outcomes[trajectory["trajectory_id"]][index]}

    counts = validate_candidates(records, trajectories, rerun, num_reruns=2, min_successes=2, output_path=tmp_path / "v.jsonl")
    assert counts == {"pass": 1, "fail": 1, "error": 1, "skipped_done": 0}
    validations = load_validations(tmp_path / "v.jsonl")
    assert [r["rubric_id"] for r in passed_records(records, validations, required=True)] == ["cand-t1"]
    assert len(passed_records(records, validations, required=False)) == 3
    # Rerun: pass/fail are kept, the errored candidate is retried.
    counts = validate_candidates(records, trajectories, rerun, num_reruns=2, min_successes=2, output_path=tmp_path / "v.jsonl")
    assert counts["skipped_done"] == 2 and counts["error"] == 1
    with pytest.raises(ValueError):
        validate_candidates(records, trajectories, rerun, num_reruns=1, min_successes=2, output_path=tmp_path / "x")


def test_build_rubrics_consolidates_only_validated_candidates(tmp_path):
    rows = [rich_trajectory("t1"), rich_trajectory("t2")]
    (tmp_path / "traj.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    config = dict(CONFIG, validation={"enabled": True, "num_reruns": 1, "min_successes": 1})

    def factory(role, name):
        if name == "failure_analyzer":
            return CachedChat(ScriptedBackend(
                lambda r: analysis(json.loads(r.user.split("\n\n", 1)[1])["trajectory_id"], "Yield")), None)
        return CachedChat(ScriptedBackend(manager_responses), None)

    rerun = lambda t, turn, instruction, i: {"status": "completed", "correct": t["trajectory_id"] == "t1"}  # noqa: E731
    summary = build_rubrics.run(config, tmp_path / "traj.jsonl", tmp_path / "out", 0, chat_factory=factory, rerun_fn=rerun)
    assert summary["validation"]["pass"] == 1 and summary["validation"]["fail"] == 1
    assert summary["candidates_consolidated"] == 1
    rubric = json.loads((tmp_path / "out/rubric_set.json").read_text())["rubrics"][0]
    assert rubric["source_trajectories"] == ["t1"] and rubric["failure_count"] == 1



def test_reward_components_softmin_separately():
    # Appropriateness: 0/1/2 -> 0/0.5/1, averaged over applicable rubrics in a turn.
    assert turn_appropriateness({"a": 2, "b": 1, "c": "NA"}) == 0.75
    assert turn_appropriateness({"a": "NA"}) is None
    assert soft_min([1.0, 0.0], tau=0.001) == pytest.approx(0.0, abs=1e-2)  # ~ worst turn
    assert soft_min([1.0, 0.0], tau=1e6) == pytest.approx(0.5, abs=1e-3)  # ~ mean turn
    affine = lambda d: (d + 1) / 2  # noqa: E731
    spec = RewardSpec(weights={"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.3}, tau=1e6,
                      effectiveness_transform=affine)
    turns = [TurnSignals(3, {"a": 2}, delta_effectiveness=0.2), TurnSignals(5, {"a": 0}, delta_effectiveness=-0.2)]
    result = compute_reward(turns, outcome=1.0, spec=spec)
    # tau large -> SoftMin ~ mean: R_A ~ 0.5, R_E ~ mean(0.6, 0.4) = 0.5
    assert result["components"]["appropriateness"] == pytest.approx(0.5, abs=1e-3)
    assert result["components"]["effectiveness"] == pytest.approx(0.5, abs=1e-3)
    assert result["reward"] == pytest.approx(0.4 * 1.0 + 0.3 * 0.5 + 0.3 * 0.5, abs=1e-3)
    # Effectiveness and appropriateness are Soft-Min'ed separately (worst turn of each).
    low_tau = compute_reward(turns, 1.0, RewardSpec(spec.weights, 0.001, affine))
    assert low_tau["components"]["appropriateness"] == pytest.approx(0.0, abs=1e-2)
    assert low_tau["components"]["effectiveness"] == pytest.approx(0.4, abs=1e-2)
    # w_E = 0 ablation: no effectiveness needed at all.
    ablation = RewardSpec({"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.0}, 0.5)
    result = compute_reward([TurnSignals(3, {"a": 2})], 0.0, ablation)
    assert result["components"]["effectiveness"] is None and result["reward"] == pytest.approx(0.3)
    # w_E > 0 without a transform or a delta is an error.
    with pytest.raises(ValueError):
        RewardSpec(spec.weights, 0.5, None)
    with pytest.raises(ValueError):
        compute_reward([TurnSignals(3, {"a": 2})], 1.0, spec)
    # No applicable rubric in any turn: appropriateness unavailable, weights renormalized to the same total.
    result = compute_reward([TurnSignals(3, {"a": "NA"}, 0.0)], 1.0, spec)
    assert result["unavailable_components"] == ["appropriateness"]
    assert sum(result["weights_used"].values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        compute_reward([TurnSignals(3, {"a": "NA"}, 0.0)], 1.0,
                       RewardSpec(spec.weights, 0.5, affine, missing_component_policy="error"))
    assert not math.isnan(soft_min([0.0, 1.0], tau=1e-4))

def test_rerun_rejects_round_one_and_follows_final_decision_role():
    runtime, calls = fake_runtime()
    with pytest.raises(ValueError, match="round 1"):
        rerun_from_turn(rich_trajectory("t1"), 1, "x", runtime)
    t = rich_trajectory("t1")
    t["final_decision_role"] = "agent"
    result = rerun_from_turn(t, 3, "x", runtime)
    assert result["dialogue"][-1]["speaker"] == "agent"
    assert not any("This is the final decision" in u for u in calls["doctor"])


def test_reconciliation_and_registry_across_iterations(tmp_path):
    from src.judges.rubric_consolidation import (build_final_rubrics, reconcile_categories, update_registry)

    old = {"category_id": "cat-001", "name": "old", "definition": "d", "members": ["cand-old"],
           "created_iteration": 0, "status": "active", "frequency_history": {"0": 5}}
    new_a = {"category_id": "cat-002", "name": "a", "definition": "d", "members": ["cand-t1"], "created_iteration": 1,
             "status": "open", "frequency_history": {}}
    new_b = dict(new_a, category_id="cat-003", name="b", members=["cand-t2"])
    records = [valid_record("t1"), valid_record("t2")]
    replies = iter([json.dumps({"merges": [{"keep": "cat-002", "merge": ["cat-003"]},
                                           {"keep": "cat-002", "merge": ["cat-001"]}]})])
    chat = CachedChat(ScriptedBackend(lambda r: next(replies)), None)
    categories, report = reconcile_categories([old, new_a, new_b], records, chat, system_prompt="s",
                                              prompt_version="v", params={}, iteration=1)
    assert report["applied"] == [{"keep": "cat-002", "merged": ["cat-003"], "reason": None}]
    assert len(report["rejected"]) == 1  # older category cannot be merged away
    assert {c["category_id"]: c["status"] for c in categories}["cat-003"] == "merged"

    construct = CachedChat(ScriptedBackend(lambda r: json.dumps({"rubrics": [schema_rubric()]})), None)
    rubric_set = build_final_rubrics(categories, records, construct, system_prompt="s", prompt_version="v",
                                     params={}, iteration=1, max_active_categories=8)
    assert [r["category_id"] for r in rubric_set["rubrics"]] == ["cat-002"]
    assert rubric_set["rubrics"][0]["failure_count"] == 2 and rubric_set["rubrics"][0]["weight"] == 1.0
    registry = {c["category_id"]: c for c in update_registry(categories, rubric_set, records, 1)}
    # The old category is kept with its history even though it is not observed now.
    assert registry["cat-001"]["frequency_history"] == {"0": 5, "1": 0}
    assert registry["cat-001"]["status"] == "inactive"
    assert registry["cat-002"]["status"] == "active" and registry["cat-002"]["frequency_history"] == {"1": 2}
    assert registry["cat-003"]["merged_into"] == "cat-002"


def test_retry_delay_handles_http_date():
    from src.judges.llm import _retry_delay

    assert _retry_delay("7", 1) == 7.0
    assert _retry_delay("Wed, 21 Oct 2026 07:28:00 GMT", 3) == 8.0
    assert _retry_delay(None, 10) == 60.0
