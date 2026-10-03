"""SERL-GRPO integration: A1 prompts, environment rollouts, the SERL reward and the GRPOTrainer wiring."""

import json
from pathlib import Path

import pytest

from src.agentclinic.outcome import DiagnosisJudge
from src.agentclinic.protocol import PROVISIONAL, ClinicRuntime
from src.agentclinic.selection import split_eligible_cases
from src.common.config import IntegrationUnavailable
from src.data.schema import digest
from src.effectiveness.providers import EffectivenessProvider
from src.grpo.dataset import grpo_rows
from src.grpo.environment import ClinicEnvironment
from src.grpo.reward import SerlReward, completion_text
from src.judges.llm import CachedChat, ScriptedBackend
from src.judges.reward import RewardSpec
from src.judges.rubric_scoring import RubricScorer, ScoreResult
from src.training.reward_pipeline import load_reward_spec
from test_agentclinic import GOLD, osce
from test_judges import schema_rubric

ROOT = Path(__file__).resolve().parents[1]
from src.common.config import local_model_path  # noqa: E402

TOKENIZER = Path(local_model_path("${MODEL_ROOT}/Qwen3-8B"))
RUBRIC_SET = {"rubric_set_version": "rv1", "rubrics": [dict(schema_rubric(), rubric_id="r1")]}
REWARD = {"weights": {"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.3}, "softmin_temperature": 0.5,
          "missing_component_policy": "omit", "effectiveness": {"transform": "affine_unit"}}
GOOD = "Current assessment: Myasthenia gravis\nResponse: The ANTIBODY result is elevated, which supports MG."
BAD = "Current assessment: Viral illness\nResponse: I agree with you."
D1 = [{"speaker": "doctor", "stage": "D1", "text": f"I think this is viral.\n{PROVISIONAL} Viral illness",
       "provisional_diagnosis": "Viral illness"}]


def source_config(tmp_path, n=10) -> dict:
    """A v5 run config on a small synthetic AgentClinic file under tmp_path."""
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(osce(i)) for i in range(n)) + "\n")
    return {"dataset": {"path": "cases.jsonl", "file": "cases.jsonl", "min_test_findings": 2},
            "split": {"seed": 0, "fractions": {"train": 0.6, "dev": 0.2, "test": 0.2}},
            "models": {"doctor": {"model_id": "d"}, "judge": {"model_id": "j"}},
            "decoding": {"agent": {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 6},
                         "doctor": {"temperature": 0.4, "max_tokens": 50}},
            "prompts": {name: str(ROOT / f"prompts/agentclinic/{name}.txt")
                        for name in ("doctor", "consultant", "diagnosis_equivalence")}}


STATE = {"agent_current_assessment": "Myasthenia gravis", "doctor_current_assessment": "Viral illness",
         "better_supported_position": "agent", "evidence_sufficient": True,
         "supporting_evidence_summary": "Antibodies are elevated."}


def is_state_call(user: str) -> bool:
    return "return the JSON object" in user


def doctor(system, user):
    """The Doctor reaches the gold diagnosis only when the Agent communicated the antibody finding."""
    if "Give your final diagnosis" in user:
        if "UNPARSABLE" in user:
            return "no json here"
        return json.dumps({"final_diagnosis": GOLD if "ANTIBODY result" in user else "Viral illness", "brief_reason": "r"})
    return f"Noted.\n{PROVISIONAL} Viral illness"


def judge_chat(tmp_path):
    return CachedChat(ScriptedBackend(lambda request: json.dumps({"equivalent": False, "reason": "different"})),
                      tmp_path / "judge.jsonl")


class Scorer(RubricScorer):
    """r1 = 2 when the target turn reports the antibody finding, 0 on A1 otherwise, NA on later turns."""
    name = "fake"

    def __init__(self):
        self.items = []

    def score(self, item):
        self.items.append(item)
        text = item.context_turns[-1]["text"]
        label = 2 if "ANTIBODY" in text else 0 if item.item_id.endswith(":A1") else "NA"
        return ScoreResult(item.item_id, self.name, "scored", {"r1": label})


class Provider(EffectivenessProvider):
    name = "fake"

    def __init__(self):
        self.before_calls, self.after_calls = [], []

    def predict_before(self, history):
        self.before_calls.append(history)
        return 0.2

    def predict_after(self, history, response):
        self.after_calls.append((history, response))
        return 0.8 if "ANTIBODY" in response else 0.1


class VaryingProvider(Provider):
    """A different P(success | h, r) per response, so a GRPO group has reward variance."""

    def predict_after(self, history, response):
        return int(digest(response)[:4], 16) / 0xffff


@pytest.fixture
def environment(tmp_path):
    cases = split_eligible_cases(source_config(tmp_path), tmp_path)
    histories = [{"history_id": f"{cid}:A1", "case_id": cid, "history": D1}
                 for cid, case in cases.items() if case["split"] != "test"]
    runtime = ClinicRuntime(doctor=doctor, doctor_system="d", agent_system="consultant system", state_system="state",
                            agent=lambda system, user: json.dumps(dict(STATE, better_supported_position="unclear"))
                            if is_state_call(user) else "Current assessment: X\nResponse: Still my view.")
    judge = DiagnosisJudge(judge_chat(tmp_path), "judge")
    return ClinicEnvironment(cases, histories, runtime, judge, max_attempts=2,
                             a1_states={row["history_id"]: dict(STATE, state_status="elicited") for row in histories})


def reward_for(environment, tmp_path, reward=REWARD, scorer=None, provider=None, **kwargs):
    spec = load_reward_spec({"reward": reward})
    return SerlReward(environment, spec, rubric_set=RUBRIC_SET, scorer=scorer or Scorer(),
                      provider=provider if provider is not None or spec.weights["effectiveness"] == 0 else Provider(),
                      log_path=tmp_path / "reward_log.jsonl", iteration=3, workers=2, **kwargs)


def call(reward, sample_ids, texts, **kwargs):
    return reward(prompts=[None] * len(texts), completions=[[{"role": "assistant", "content": t}] for t in texts],
                  completion_ids=[[]] * len(texts), sample_id=sample_ids, **kwargs)


def test_rows_are_the_train_a1_prompts(environment):
    rows = grpo_rows(environment.cases, list(environment.histories.values()), "consultant system")
    assert {environment.cases[environment.histories[r["sample_id"]]["case_id"]]["split"] for r in rows} == {"train"}
    assert len(rows) == sum(c["split"] == "train" for c in environment.cases.values()) == 6
    system, user = rows[0]["prompt"]
    assert system == {"role": "system", "content": "consultant system"} and user["role"] == "user"
    assert "ANTIBODY-RESULT" in user["content"] and "I think this is viral." in user["content"]
    assert GOLD not in user["content"] and "SECRET-MANAGEMENT-PLAN" not in user["content"]
    with pytest.raises(ValueError):
        grpo_rows(environment.cases, [], "s")


def test_reward_combines_outcome_appropriateness_and_effectiveness(environment, tmp_path):
    """credit_assignment from_turn_on (not the main pipeline): the continuation's A2 is scored as well."""
    sample = sorted(environment.histories)[0]
    scorer, provider, metrics = Scorer(), Provider(), {}
    reward = reward_for(environment, tmp_path, scorer=scorer, provider=provider, credit_assignment="from_turn_on")
    rewards = call(reward, [sample, sample], [GOOD, BAD], log_metric=metrics.__setitem__)
    # GOOD: outcome 1, rubric 2 -> 1.0, delta 0.6 -> 0.8; BAD: outcome 0, rubric 0 -> 0.0, delta -0.1 -> 0.45
    assert rewards == pytest.approx([0.4 + 0.3 * 1.0 + 0.3 * 0.8, 0.3 * 0.45])
    assert len(provider.before_calls) == 1 and len(provider.after_calls) == 2  # P(success | h) is shared
    history, response = provider.after_calls[0]
    assert response.startswith("The ANTIBODY result") and [t["speaker"] for t in history["turns"]] == ["doctor"]
    assert "ANTIBODY-RESULT" in history["scenario"] and GOLD not in json.dumps(history)

    good, bad = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    assert (good["r_outcome"], good["r_appropriateness"], good["r_effectiveness"]) == pytest.approx((1.0, 1.0, 0.8))
    assert good["reward"] == pytest.approx(rewards[0]) and good["reward_source"] == "computed"
    assert good["active_rubric_ids"] == ["r1"] and good["rubric_set_version"] == "rv1" and good["iteration"] == 3
    assert good["final_diagnosis"] == GOLD and good["outcome"] == 1.0 and good["completion"] == GOOD
    assert good["rubric_scores"]["A1"]["labels"] == {"r1": 2} and good["rubric_scores"]["A2"]["labels"] == {"r1": "NA"}
    a1_effect = good["effectiveness"]["A1"]
    assert a1_effect["available"] is True and (a1_effect["before"], a1_effect["after"], a1_effect["delta"]) == pytest.approx((0.2, 0.8, 0.6))
    # A2 has no predictor: it is recorded as unavailable, and R_E is A1's value alone (not a soft-min with 0)
    assert good["effectiveness"]["A2"] == {"available": False, "reason": "no Effectiveness predictor for A2"}
    assert good["turn_effectiveness"] == [pytest.approx(0.8), None] and good["r_effectiveness"] == pytest.approx(0.8)
    parts = good["components"]
    assert set(parts) == {"outcome", "appropriateness", "effectiveness"}
    assert all(set(part) == {"value", "available", "source", "model_version", "metadata"} for part in parts.values())
    assert parts["effectiveness"]["metadata"]["per_turn"]["A2"]["available"] is False
    assert parts["appropriateness"]["metadata"]["per_turn"]["A2"] == {
        "value": None, "available": False, "no_applicable_rubric": True, "applicable_rubrics": {}}
    assert history["intervention_turn"] == "A1" and set(good["agent_states"]) == {"A1", "A2"}
    a1, a2 = good["turn_appropriateness_details"]
    assert a1 == {"applicable_rubrics": {"r1": {"score": 1.0, "weight": 1.0, "alpha": 1.0}}, "no_match_rubrics": [],
                  "aggregation": "weighted_mean", "temperature": None, "appropriateness": 1.0, "no_applicable_rubric": False}
    assert a2 == {"applicable_rubrics": {}, "no_match_rubrics": ["r1"], "aggregation": "weighted_mean", "temperature": None,
                  "appropriateness": None, "no_applicable_rubric": True}
    assert "A1 only" in reward.identity()["effectiveness_coverage"]  # A2 has no R_E and none is made up
    assert [m["stage"] for m in good["dialogue"]] == ["D1", "A1", "D2", "A2", "FINAL"]
    assert (bad["generation"], bad["outcome"], bad["final_diagnosis"]) == (1, 0.0, "Viral illness")
    assert metrics["serl/r_outcome"] == 0.5 and metrics["serl/completed_share"] == 1

    # the scorer sees the dialogue up to its turn and the Agent's view, never the outcome
    assert len(scorer.items) == 4
    for item in scorer.items:
        payload = json.dumps(item.payload())
        assert item.context_turns[-1]["speaker"] == "agent" and "ANTIBODY-RESULT" in payload
        assert "final_diagnosis" not in payload and "FINAL" not in payload
        state = item.pre_response_state
        assert state["state_status"] == "elicited" and "ANTIBODY-RESULT" in state["clinical_information_available_to_agent"]
        # before A1: the iteration's stored state of the history; before A2: elicited on the continued dialogue
        assert state["better_supported_position"] == ("agent" if item.item_id.endswith(":A1") else "unclear")


def test_soft_min_over_the_scored_agent_turns(environment, tmp_path):
    sample = sorted(environment.histories)[0]
    environment.runtime.agent = lambda system, user: "Current assessment: X\nResponse: The ANTIBODY result again."
    both = call(reward_for(environment, tmp_path, dict(REWARD, weights={"outcome": 0, "appropriateness": 1, "effectiveness": 0}),
                           credit_assignment="from_turn_on"), [sample], [BAD])
    first = call(reward_for(environment, tmp_path, dict(REWARD, weights={"outcome": 0, "appropriateness": 1, "effectiveness": 0}),
                            scored_stages=("A1",), credit_assignment="from_turn_on"), [sample], [BAD])
    assert first == [0.0]  # A1 alone scores 0
    assert 0.0 < both[0] < 0.5  # soft-min of A1 = 0.0 and A2 = 1.0 stays near the worse turn


def test_weights_and_temperature_come_from_the_config(environment, tmp_path):
    sample = sorted(environment.histories)[0]
    provider = Provider()
    outcome_only = dict(REWARD, weights={"outcome": 1, "appropriateness": 0, "effectiveness": 0})
    assert call(reward_for(environment, tmp_path, outcome_only, provider=provider), [sample, sample], [GOOD, BAD]) == [1.0, 0.0]
    assert provider.after_calls == []  # w_E = 0: the effectiveness provider is not called
    spec = load_reward_spec({"reward": REWARD})
    with pytest.raises(IntegrationUnavailable):
        SerlReward(environment, spec, rubric_set=RUBRIC_SET, scorer=Scorer(), provider=None,
                   log_path=tmp_path / "x.jsonl", iteration=0)
    with pytest.raises(IntegrationUnavailable):
        SerlReward(environment, spec, rubric_set=RUBRIC_SET, scorer=None, provider=Provider(),
                   log_path=tmp_path / "x.jsonl", iteration=0)
    with pytest.raises(ValueError):
        RewardSpec(weights=REWARD["weights"], tau=0, effectiveness_transform=lambda d: d)


def trajectory_of(environment, history_id: str, a1: str = "The ANTIBODY result is elevated.") -> dict:
    """theta_i's collaboration on a history, as the A2 samples read it (D1 -> A1 -> D2 with the stored states)."""
    row = environment.histories[history_id]
    state = dict(STATE, state_status="elicited", clinical_information_available_to_agent="rendered view")
    return {"history_id": history_id, "case_id": row["case_id"], "collaboration_state": "Challenge", "turns": [
        {"speaker": "doctor", "stage": "D1", "text": D1[0]["text"], "provisional_diagnosis": "Viral illness"},
        {"speaker": "agent", "stage": "A1", "text": a1, "assessment": "Myasthenia gravis", "pre_response_state": state},
        {"speaker": "doctor", "stage": "D2", "text": f"I am not convinced.\n{PROVISIONAL} Viral illness", "provisional_diagnosis": "Viral illness"},
        {"speaker": "agent", "stage": "A2", "text": "Still my view.", "assessment": "Myasthenia gravis",
         "pre_response_state": dict(state, better_supported_position="doctor")},
        {"speaker": "doctor", "stage": "FINAL", "text": "{}", "final": True}]}


def with_a2(environment, n: int = 2, a1: str = "I see your point.") -> tuple[ClinicEnvironment, list[dict]]:
    from src.grpo.environment import a2_samples

    histories = [h for h in environment.histories.values()]
    second = a2_samples([trajectory_of(environment, h["history_id"], a1) for h in histories[:n]], environment.cases,
                        environment.cases[histories[0]["case_id"]]["split"])
    return ClinicEnvironment(environment.cases, histories, environment.runtime, environment.judge, a2=second, max_attempts=2,
                             a1_states=environment.a1_states), second


def test_a2_training_samples_are_the_prompt_d1_a1_d2_and_the_reward_follows_from_final(environment, tmp_path):
    both, second = with_a2(environment)
    sample = second[0]
    assert sample["history_id"].endswith("/A2") and [m["stage"] for m in sample["history"]] == ["D1", "A1", "D2"]
    assert set(sample["states"]) == {"A1", "A2"} and "clinical_information_available_to_agent" not in sample["states"]["A2"]
    rows = grpo_rows(both.cases, [h for h in both.histories.values() if h.get("stage") != "A2"], "consultant system",
                     split=both.cases[sample["case_id"]]["split"], a2=second, stages=("A1", "A2"))
    a2_row = next(r for r in rows if r["sample_id"] == sample["history_id"])
    prompt = a2_row["prompt"][1]["content"]
    assert a2_row["stage"] == "A2" and "I see your point." in prompt and "I am not convinced." in prompt
    assert "Respond to the Doctor's latest message" in prompt and "Still my view." not in prompt and GOLD not in prompt
    assert {r["stage"] for r in rows} == {"A1", "A2"}
    assert {r["stage"] for r in grpo_rows(both.cases, list(environment.histories.values()), "s", a2=second, stages=("A1",),
                                          split=both.cases[sample["case_id"]]["split"])} == {"A1"}

    scorer, provider = Scorer(), Provider()
    reward = reward_for(both, tmp_path, scorer=scorer, provider=provider)
    good, bad = call(reward, [sample["history_id"]] * 2, [GOOD, BAD])
    # the Doctor's FINAL follows the trained A2: GOOD reports the antibody finding and turns the outcome
    rows = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["A2", "A2"] and [r["outcome"] for r in rows] == [1.0, 0.0]
    assert [m["stage"] for m in rows[0]["dialogue"]] == ["D1", "A1", "D2", "A2", "FINAL"]
    assert rows[0]["dialogue"][1]["text"] == "I see your point." and rows[0]["dialogue"][3]["text"].startswith("The ANTIBODY")
    # only A2 is scored (A1 belongs to the prompt), with the stored state before A2
    assert all(item.item_id.endswith(":A2") and item.turn_id == 3 for item in scorer.items) and len(scorer.items) == 2
    assert scorer.items[0].pre_response_state["better_supported_position"] == "doctor"
    assert rows[0]["scored_stages"] == ["A2"] and set(rows[0]["rubric_scores"]) == {"A2"}
    # A2 has no Effectiveness predictor: nothing is predicted and no value stands in for it
    assert provider.before_calls == [] and provider.after_calls == []
    for row in rows:
        assert row["effectiveness"] == {"A2": {"available": False, "reason": "no Effectiveness predictor for A2"}}
        assert row["r_effectiveness"] is None and row["components"]["effectiveness"]["available"] is False
        assert "effectiveness" in row["unavailable_components"] and "effectiveness" not in row["weights_used"]
        assert row["weights_used"] == {"outcome": 0.4, "appropriateness": 0.3}  # as configured: no 0.4/0.7, 0.3/0.7
    # R = 0.4 R_O + 0.3 R_A. GOOD: outcome 1, rubric 2; BAD: outcome 0, no rubric applies to it -> the group's mean R_A (1.0)
    assert good == pytest.approx(0.4 * 1.0 + 0.3 * 1.0) and bad == pytest.approx(0.3 * 1.0)
    assert rows[0]["raw_composite"] == rows[0]["reward_passed_to_trl"] == pytest.approx(0.7)
    assert rows[1]["no_applicable"] is True and rows[1]["imputed_R_A"] == 1.0 and rows[1]["raw_R_A"] is None
    assert reward.identity()["trained_stages"] == ["A1", "A2"]


def test_response_without_an_applicable_rubric_gets_its_groups_mean(environment, tmp_path):
    sample, other = sorted(environment.histories)[:2]

    class Partial(Scorer):
        """r1 applies to an A1 that argues from the finding (2) or agrees (0); to nothing else."""
        def score(self, item):
            text = item.context_turns[-1]["text"]
            applies = item.item_id.endswith(":A1") and ("supports MG" in text or "agree" in text)
            return ScoreResult(item.item_id, self.name, "scored", {"r1": (2 if "supports MG" in text else 0) if applies else "NA"})

    outside = "Current assessment: X\nResponse: Let us talk about something else."
    weights = dict(REWARD, weights={"outcome": 0, "appropriateness": 1, "effectiveness": 0})
    reward = reward_for(environment, tmp_path, weights, scorer=Partial())
    rewards = call(reward, [sample] * 4 + [other] * 2, [GOOD, BAD, outside, GOOD, outside, outside])
    rows = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    # group 1: [1, 0, missing, 1] -> the missing one gets 2/3; it is neither the best nor the worst of its group
    assert rewards[:4] == pytest.approx([1.0, 0.0, 2 / 3, 1.0])
    assert [r["raw_R_A"] for r in rows[:4]] == [1.0, 0.0, None, 1.0] and [r["no_applicable"] for r in rows[:4]] == [False, False, True, False]
    assert rows[2]["imputed_R_A"] == pytest.approx(2 / 3) and rows[2]["r_appropriateness"] == pytest.approx(2 / 3)
    assert rows[2]["reward_source"] == "computed" and rows[2]["components"]["appropriateness"]["source"] == "group_mean_imputation"
    assert rows[2]["raw_R_A_by_turn"] == {"A1": None} and rows[2]["group_turn_missing"] == {"A1": False}
    assert rows[2]["components"]["appropriateness"]["metadata"]["raw_value"] is None
    assert all(r["group_no_applicable_rate"] == 0.25 for r in rows[:4]) and rows[0]["imputed_R_A"] is None
    assert rows[0]["components"]["appropriateness"]["source"] == "rubric_scorer"
    # group 2: no response has a value -> R_A is unavailable for the whole group, the same for both (no 0, no imputation)
    assert [r["raw_R_A"] for r in rows[4:]] == [None, None] and [r["imputed_R_A"] for r in rows[4:]] == [None, None]
    assert all(r["group_no_applicable_rate"] == 1.0 and r["unavailable_components"] == ["appropriateness"] for r in rows[4:])
    assert rewards[4:] == [0.0, 0.0] and all(r["reward_source"] == "group_mean" for r in rows[4:])  # nothing to compute from

    # with the outcome in the reward, the whole group is renormalized alike, so its ranking is the outcome's
    with_outcome = dict(REWARD, weights={"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0})
    full = reward_for(environment, tmp_path, with_outcome, scorer=Partial(), provider=None)
    assert call(full, [other] * 2, [outside, outside]) == [0.0, 0.0]
    last = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()][-2:]
    assert all(r["weights_used"] == {"outcome": 0.4} and r["reward_source"] == "computed" for r in last)

    # the loophole this closes: two responses that both reach the right diagnosis, one scored 0 by the rubric,
    # one outside every rubric. Leaving the coverage must not pay.
    covered_low = "Current assessment: X\nResponse: I agree, though the ANTIBODY result is there."
    uncovered = "Current assessment: X\nResponse: See the ANTIBODY result."
    assert call(full, [sample] * 2, [covered_low, uncovered]) == pytest.approx([0.4, 0.4])
    earlier = reward_for(environment, tmp_path, dict(with_outcome, missing_component_policy="renormalize"), scorer=Partial(),
                         no_applicable_rubric_policy="none")
    assert call(earlier, [sample] * 2, [covered_low, uncovered]) == pytest.approx([0.4, 0.7])  # rescaled per response: it paid
    with pytest.raises(ValueError):
        reward_for(environment, tmp_path, no_applicable_rubric_policy="zero")


def test_scorer_receives_exactly_the_active_rubric_set(environment, tmp_path):
    sample = sorted(environment.histories)[0]
    active = {"rubric_set_version": "rv", "rubrics": [dict(schema_rubric(f"n{i}"), rubric_id=f"r{i}", weight=w, category_id=f"cat-{i}")
                                                       for i, w in enumerate((0.75, 0.25))]}

    class Recording(Scorer):
        def score(self, item):
            self.items.append(item)
            return ScoreResult(item.item_id, self.name, "scored", {"r0": 2, "r1": 0})

    scorer = Recording()
    reward = SerlReward(environment, load_reward_spec({"reward": dict(REWARD, weights={"outcome": 0, "appropriateness": 1, "effectiveness": 0})}),
                        rubric_set=active, scorer=scorer, provider=None, log_path=tmp_path / "reward_log.jsonl", iteration=0)
    assert call(reward, [sample], [GOOD]) == pytest.approx([0.75 * 1.0 + 0.25 * 0.0])  # failure-frequency weights
    assert all([r["rubric_id"] for r in item.rubrics] == ["r0", "r1"] for item in scorer.items)
    assert reward.identity()["active_rubric_ids"] == ["r0", "r1"]


def test_partial_turn_no_match_is_imputed_per_turn_before_the_soft_min(environment, tmp_path, monkeypatch):
    """credit_assignment from_turn_on (not the main pipeline): with several scored turns the imputation is per turn."""
    from src.judges.reward import soft_min

    plain = SerlReward.__init__
    monkeypatch.setattr(SerlReward, "__init__", lambda self, *a, **k: plain(self, *a, **dict(k, credit_assignment="from_turn_on")))

    sample = sorted(environment.histories)[0]
    environment.runtime.agent = lambda system, user: json.dumps(STATE) if is_state_call(user) \
        else "Current assessment: X\nResponse: Still my view."  # the continuation's A2, the same for every candidate
    partly = "Current assessment: X\nResponse: I partly agree."
    outside = "Current assessment: X\nResponse: Let us talk about something else."

    class ByTurn(Scorer):
        """A1: argues from the finding 2, partly agrees 1, agrees 0, otherwise no rubric applies. A2: a2_label."""
        a2_label = 2

        def score(self, item):
            text = item.context_turns[-1]["text"]
            if item.item_id.endswith(":A2"):
                label = self.a2_label
            else:
                label = 2 if "supports MG" in text else 1 if "partly" in text else 0 if "agree" in text else "NA"
            return ScoreResult(item.item_id, self.name, "scored", {"r1": label})

    only_a = dict(REWARD, weights={"outcome": 0, "appropriateness": 1, "effectiveness": 0})
    rewards = call(reward_for(environment, tmp_path, only_a, scorer=ByTurn()), [sample] * 4, [BAD, GOOD, outside, partly])
    rows = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    assert [r["raw_R_A_by_turn"] for r in rows] == [{"A1": 0.0, "A2": 1.0}, {"A1": 1.0, "A2": 1.0}, {"A1": None, "A2": 1.0}, {"A1": 0.5, "A2": 1.0}]
    escaped = rows[2]  # no rubric at A1 only: on its own turns it would be the best of the group (R_A = 1.0)
    assert escaped["raw_R_A"] == 1.0 and escaped["turn_applicability"] == {"A1": False, "A2": True}
    assert escaped["imputed_R_A_by_turn"] == {"A1": pytest.approx(0.5), "A2": 1.0}  # mean of 0, 1, 0.5 at A1
    assert escaped["group_turn_missing"] == {"A1": False, "A2": False} and escaped["no_applicable"] is False
    assert escaped["final_R_A"] == pytest.approx(soft_min([0.5, 1.0], 0.5)) == pytest.approx(escaped["imputed_R_A"])
    assert rewards[2] == pytest.approx(rewards[3]) and rewards[0] < rewards[2] < rewards[1]  # neither best nor worst
    assert rows[0]["imputed_R_A"] is None and rows[0]["final_R_A"] == pytest.approx(soft_min([0.0, 1.0], 0.5))
    assert escaped["components"]["appropriateness"]["metadata"]["imputed_turns"] == ["A1"]
    assert escaped["raw_composite"] == escaped["reward_passed_to_trl"] == pytest.approx(rewards[2])

    # a turn nobody in the group has a rubric for is left out for all of them: R_A is A1 alone, after imputation
    scorer = ByTurn()
    scorer.a2_label = "NA"
    (tmp_path / "reward_log.jsonl").unlink()
    rewards = call(reward_for(environment, tmp_path, only_a, scorer=scorer), [sample] * 4, [BAD, GOOD, outside, partly])
    rows = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    assert all(r["group_turn_missing"] == {"A1": False, "A2": True} and r["imputed_R_A_by_turn"]["A2"] is None for r in rows)
    assert rewards == pytest.approx([0.0, 1.0, 0.5, 0.5]) and rows[2]["raw_R_A"] is None and rows[2]["no_applicable"] is True

    # without the imputation (ablation) the response keeps the turn it escaped: it ties with the best candidate
    scorer = ByTurn()
    plain = call(reward_for(environment, tmp_path, only_a, scorer=scorer, no_applicable_rubric_policy="none"),
                 [sample] * 4, [BAD, GOOD, outside, partly])
    assert plain[2] == pytest.approx(plain[1]) == pytest.approx(1.0)


def softmin_reward(environment, tmp_path, rubric_set, scorer, weights=None, **kwargs):
    """The reward as the main pipeline configures it: weighted soft-min over rubrics (tau 0.5), turn-local credit."""
    config = {"reward": dict(REWARD, weights=weights or {"outcome": 0, "appropriateness": 1, "effectiveness": 0}),
              "rubric_aggregation": {"method": "weighted_softmin", "temperature": 0.5}}
    spec = load_reward_spec(config)
    return SerlReward(environment, spec, rubric_set=rubric_set, scorer=scorer,
                      provider=kwargs.pop("provider", None) or (Provider() if spec.weights["effectiveness"] > 0 else None),
                      log_path=tmp_path / "reward_log.jsonl", iteration=0, workers=2, **kwargs)


def test_weighted_softmin_over_the_rubrics_of_one_response():
    import math

    from src.judges.reward import turn_appropriateness, weighted_soft_min

    tau = 0.5
    softmin = lambda pairs: -tau * math.log(sum(a * math.exp(-s / tau) for s, a in pairs))  # noqa: E731
    weights = {"r1": 0.6, "r2": 0.3, "r3": 0.1}
    # the formula: alpha_k = w_k / sum(w), R_A = -tau * log(sum_k alpha_k * exp(-s_k / tau))
    value = turn_appropriateness({"r1": 2, "r2": 0, "r3": 1}, weights, "weighted_softmin", tau)
    assert value == pytest.approx(softmin([(1.0, 0.6), (0.0, 0.3), (0.5, 0.1)]))
    assert 0.0 < value < 0.6 * 1.0 + 0.1 * 0.5  # below the weighted mean: it leans toward the lowest score
    # one applicable rubric: exactly its score, whatever its weight
    for label, score in ((0, 0.0), (1, 0.5), (2, 1.0)):
        assert turn_appropriateness({"r1": "NA", "r2": label, "r3": "NA"}, weights, "weighted_softmin", tau) == pytest.approx(score, abs=1e-12)
    # a rubric that does not apply is left out before the weights are normalized: alpha is 0.6/0.7 and 0.1/0.7, not 0.6 and 0.1
    assert turn_appropriateness({"r1": 2, "r2": "NA", "r3": 1}, weights, "weighted_softmin", tau) == \
        pytest.approx(softmin([(1.0, 0.6 / 0.7), (0.5, 0.1 / 0.7)]))
    assert turn_appropriateness({"r1": "NA", "r2": "NA", "r3": "NA"}, weights, "weighted_softmin", tau) is None
    # equal scores give that score; a rubric with more weight pulls the value toward its score
    assert weighted_soft_min([0.5, 0.5], [0.9, 0.1], tau) == pytest.approx(0.5)
    heavy_low = turn_appropriateness({"r1": 0, "r3": 2}, weights, "weighted_softmin", tau)  # the 0 has weight 0.6
    light_low = turn_appropriateness({"r1": 2, "r3": 0}, weights, "weighted_softmin", tau)  # the 0 has weight 0.1
    assert heavy_low < light_low and heavy_low == pytest.approx(softmin([(0.0, 6 / 7), (1.0, 1 / 7)]))
    # uniform weights (no weights given) and the plain weighted mean of the MedMCQA protocol
    assert turn_appropriateness({"r1": 2, "r2": 0}, None, "weighted_softmin", tau) == pytest.approx(softmin([(1.0, 0.5), (0.0, 0.5)]))
    assert turn_appropriateness({"r1": 2, "r2": 0, "r3": 1}, weights) == pytest.approx(0.6 * 1.0 + 0.1 * 0.5)
    with pytest.raises(ValueError):
        RewardSpec(weights=REWARD["weights"], tau=0.5, effectiveness_transform=lambda d: d, rubric_aggregation="weighted_softmin")


def test_turn_local_reward_weight_strategies_and_no_cross_turn_soft_min(environment, tmp_path):
    import math

    sample = sorted(environment.histories)[0]
    rubric_set = {"rubric_set_version": "rv", "rubrics": [
        dict(schema_rubric(f"n{i}"), rubric_id=f"r{i}", weight=w, category_id=f"cat-{i}") for i, w in enumerate((0.75, 0.25))]}

    class Recording(Scorer):
        """A1: r0 = 2, r1 = 0. A2 would be scored 0 on both, if it were scored at all."""
        def score(self, item):
            self.items.append(item)
            return ScoreResult(item.item_id, self.name, "scored", {"r0": 2, "r1": 0} if item.item_id.endswith(":A1") else {"r0": 0, "r1": 0})

    softmin = lambda pairs: -0.5 * math.log(sum(a * math.exp(-s / 0.5) for s, a in pairs))  # noqa: E731
    scorer = Recording()
    frequency = softmin_reward(environment, tmp_path, rubric_set, scorer)
    assert call(frequency, [sample], [GOOD]) == pytest.approx([softmin([(1.0, 0.75), (0.0, 0.25)])])
    # turn-local credit: the A1 response is rewarded for A1; the A2 the continuation wrote is neither scored nor soft-min'ed in
    assert [item.item_id.split(":")[-1] for item in scorer.items] == ["A1"]
    row = json.loads((tmp_path / "reward_log.jsonl").read_text().splitlines()[-1])
    assert row["scored_stages"] == ["A1"] and set(row["rubric_scores"]) == {"A1"} and [m["stage"] for m in row["dialogue"]][3] == "A2"
    detail, = row["turn_appropriateness_details"]
    assert detail["aggregation"] == "weighted_softmin" and detail["temperature"] == 0.5
    assert detail["applicable_rubrics"] == {"r0": {"score": 1.0, "weight": 0.75, "alpha": 0.75}, "r1": {"score": 0.0, "weight": 0.25, "alpha": 0.25}}
    identity = frequency.identity()
    assert (identity["rubric_weight_strategy"], identity["rubric_aggregation"], identity["credit_assignment"]) == \
        ("failure_frequency", "weighted_softmin", "turn_local") and identity["rubric_weights"] == {"r0": 0.75, "r1": 0.25}
    # the weight source is a setting: uniform gives every active rubric the same weight
    uniform = softmin_reward(environment, tmp_path, rubric_set, Recording(), rubric_weight_strategy="uniform")
    assert call(uniform, [sample], [GOOD]) == pytest.approx([softmin([(1.0, 0.5), (0.0, 0.5)])])
    assert uniform.identity()["rubric_weights"] == {"r0": 0.5, "r1": 0.5}
    # effectiveness_gap (the main method): the weights written in the rubric set, as they are
    gap = softmin_reward(environment, tmp_path, rubric_set, Recording(), rubric_weight_strategy="effectiveness_gap")
    assert gap.identity()["rubric_weights"] == {"r0": 0.75, "r1": 0.25}
    with pytest.raises(ValueError):
        softmin_reward(environment, tmp_path, rubric_set, Recording(), rubric_weight_strategy="task_success_gap")

    # A1: R = 0.4 R_O + 0.3 R_A(A1) + 0.3 R_E(A1); R_E is A1's delta alone (identity transform), no soft-min with A2
    config = {"reward": dict(REWARD, effectiveness={"transform": "identity"}), "rubric_aggregation": {"method": "weighted_softmin", "temperature": 0.5}}
    full = SerlReward(environment, load_reward_spec(config), rubric_set=rubric_set, scorer=Recording(), provider=Provider(),
                      log_path=tmp_path / "reward_log.jsonl", iteration=0, workers=2)
    reward, = call(full, [sample], [GOOD])
    assert reward == pytest.approx(0.4 * 1.0 + 0.3 * softmin([(1.0, 0.75), (0.0, 0.25)]) + 0.3 * 0.6)
    row = json.loads((tmp_path / "reward_log.jsonl").read_text().splitlines()[-1])
    assert set(row["effectiveness"]) == {"A1"} and row["turn_effectiveness"] == [pytest.approx(0.6)]


def test_outcome_only_reward_queries_neither_scorer_nor_predictor_and_reports_the_group_signal(environment, tmp_path):
    from scripts.reward_report import report

    first, second = sorted(environment.histories)[:2]
    scorer, provider, metrics = Scorer(), Provider(), {}
    outcome_only = dict(REWARD, weights={"outcome": 1.0, "appropriateness": 0.0, "effectiveness": 0.0})
    reward = reward_for(environment, tmp_path, outcome_only, scorer=scorer, provider=provider)
    # group 1: one response reaches the right diagnosis, one does not; group 2: both fail
    rewards = call(reward, [first, first, second, second], [GOOD, BAD, BAD, BAD], log_metric=metrics.__setitem__)
    assert rewards == [1.0, 0.0, 0.0, 0.0]  # R = R_Outcome
    assert scorer.items == [] and provider.before_calls == [] and provider.after_calls == []  # R_A and R_E are never asked for
    rows = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    assert all(r["weights_used"] == {"outcome": 1.0} and r["rubric_scores"] == {} and r["r_appropriateness"] is None
               and r["r_effectiveness"] is None for r in rows)
    assert metrics["serl/groups_with_reward_variance"] == 0.5
    signal = report(rows, outcome_only["weights"])["group_signal"]
    diversity = signal.pop("response_diversity")  # whether the responses of a group differ at all (saved with the signal)
    assert set(diversity) == {"groups_with_all_responses_distinct_share", "mean_pairwise_word_overlap", "distinct_stated_assessments_per_group",
                              "distinct_final_diagnoses_per_group", "groups_without_signal"} and diversity["groups_without_signal"]["groups"] == 1
    assert signal == {"groups": 2, "outcome_all_0_share": 0.5, "outcome_all_1_share": 0.0, "outcome_varies_share": 0.5,
                      "mean_outcome_variance_within_group": pytest.approx(0.125), "groups_with_nonzero_reward_variance_share": 0.5,
                      "groups_without_learning_signal_share": 0.5}


def test_no_active_rubrics_adds_nothing_and_leaves_the_other_weights_as_configured(environment, tmp_path):
    sample = sorted(environment.histories)[0]
    spec = load_reward_spec({"reward": REWARD})
    reward = SerlReward(environment, spec, rubric_set={"rubrics": []}, scorer=None, provider=Provider(),
                        log_path=tmp_path / "reward_log.jsonl", iteration=0)
    assert call(reward, [sample], [GOOD]) == pytest.approx([0.4 * 1.0 + 0.3 * 0.8])
    row = json.loads((tmp_path / "reward_log.jsonl").read_text())
    assert row["unavailable_components"] == ["appropriateness"] and row["r_appropriateness"] is None
    assert row["weights_used"] == {"outcome": 0.4, "effectiveness": 0.3} and row["components"]["appropriateness"]["available"] is False
    # the MedMCQA protocol's rule (renormalize) scales the remaining weights up instead
    legacy = SerlReward(environment, load_reward_spec({"reward": dict(REWARD, missing_component_policy="renormalize")}),
                        rubric_set={"rubrics": []}, scorer=None, provider=Provider(), log_path=tmp_path / "legacy.jsonl", iteration=0)
    assert call(legacy, [sample], [GOOD]) == pytest.approx([(0.4 * 1.0 + 0.3 * 0.8) / 0.7])


def test_invalid_response_scores_zero_and_environment_failure_gets_the_group_mean(environment, tmp_path):
    first, second = sorted(environment.histories)[:2]
    broken = "Current assessment: X\nResponse: UNPARSABLE"
    rewards = call(reward_for(environment, tmp_path), [first, first, first, second], [GOOD, "", broken, broken])
    assert rewards[1] == 0.0  # no message: the policy's failure
    assert rewards[2] == pytest.approx((rewards[0] + 0.0) / 2)  # FINAL never parsed: neutral within its group
    assert rewards[3] == 0.0  # no other response in its group
    rows = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["completed", "invalid_response", "rollout_failed", "rollout_failed"]
    assert [r["reward_source"] for r in rows] == ["computed", "computed", "group_mean", "group_mean"]
    assert completion_text("plain") == "plain" and completion_text([{"role": "assistant", "content": "a"}]) == "a"


def test_test_split_histories_are_refused(tmp_path):
    from src.agentclinic.runtime import frozen_histories

    cases = split_eligible_cases(source_config(tmp_path), tmp_path)
    rows = [{"history_id": f"{cid}:A1", "case_id": cid, "history": D1} for cid in cases]
    (tmp_path / "histories.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="TEST"):
        frozen_histories({"environment": {"histories": "histories.jsonl"}}, tmp_path, cases)


@pytest.mark.skipif(not TOKENIZER.is_dir(), reason="needs the local Qwen3 tokenizer")
def test_grpo_trainer_runs_an_iteration_and_the_next_one_continues(tmp_path):
    """The real TRL GRPOTrainer on CPU with a tiny random Qwen3: reward function, logs, adapter, hand-off."""
    pytest.importorskip("trl")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from src.grpo.trainer import checkpoint_trajectories, train_iteration

    source = source_config(tmp_path)
    (tmp_path / "source.json").write_text(json.dumps(source))
    cases = split_eligible_cases(source, tmp_path)
    rows = [{"history_id": f"{cid}:A1", "case_id": cid, "history": D1} for cid, c in cases.items() if c["split"] != "test"]
    (tmp_path / "histories.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "rubrics").mkdir()
    (tmp_path / "rubrics/set0.json").write_text(json.dumps(RUBRIC_SET))
    config = {
        "run_id": "smoke", "output_dir": "out",
        "environment": {"config": "source.json", "histories": "histories.jsonl", "train_split": "train",
                        "state_prompt": str(ROOT / "prompts/agentclinic/state_elicitation.txt")},
        "policy": {"model_id": "tiny-qwen3", "revision": "none"},
        "grpo": {"num_generations": 2, "per_device_train_batch_size": 2, "max_completion_length": 6, "max_steps": 2,
                 "learning_rate": 1e-3, "logging_steps": 1, "save_strategy": "no", "use_cpu": True, "bf16": False},
        "rubrics": {"rubric_set": "rubrics/set{iteration}.json"},
        "reward": dict(REWARD, scored_stages=["A1", "A2"], effectiveness=dict(REWARD["effectiveness"], stages=["A1"])),
        "rollout": {"workers": 2, "agent_batch_size": 2, "max_attempts": 1}, "trajectories": {"split": "train"},
        "training_samples": {"stages": ["A1"]}}

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tiny = AutoConfig.from_pretrained(TOKENIZER)
    tiny.update({"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
                 "num_key_value_heads": 2, "head_dim": 16})
    model = get_peft_model(AutoModelForCausalLM.from_config(tiny), LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    fakes = {"policy": (model, tokenizer), "doctor": doctor, "judge_chat": judge_chat(tmp_path)}

    # every generate call: who called it (GRPOTrainer samples completions in the main thread, the continuation of a
    # rollout runs in the batching thread of the reward's environment), the model's mode, whether the output carries a graph
    import threading
    generated, generate = [], model.generate

    def spy(*args, **kwargs):
        output = generate(*args, **kwargs)
        generated.append({"rollout": threading.current_thread() is not threading.main_thread(), "training": model.training,
                          "requires_grad": bool(getattr(output, "requires_grad", False))})
        return output

    model.generate = spy
    lora_b = next(p for n, p in model.named_parameters() if "lora_B" in n)
    assert not lora_b.any()
    summary = train_iteration(config, tmp_path, 0, scorer=Scorer(), provider=VaryingProvider(), **fakes)
    assert lora_b.any()  # A1 samples: the group-relative loss on the A1 completions reached the adapter
    assert summary["train_prompts_by_stage"] == {"A1": 6}
    # the A2 written inside an A1 sample's continuation is generated in eval mode and carries no gradient
    inside = [g for g in generated if g["rollout"]]
    assert inside and not any(g["training"] or g["requires_grad"] for g in inside)
    out = tmp_path / "out/iter0"
    assert (out / "adapter/adapter_config.json").is_file() and summary["global_step"] == 2
    assert summary["checkpoint_id"] == "smoke-iter0" and summary["rubric_version"] == "rv1"
    assert summary["initial_adapter"] is None and summary["train_prompts"] == 6
    assert json.loads((out / "checkpoint.json").read_text())["reward"]["weights"] == REWARD["weights"]
    log = [json.loads(line) for line in (out / "reward_log.jsonl").read_text().splitlines()]
    assert len(log) == 4 and {r["global_step"] for r in log} == {0, 1} and {r["stage"] for r in log} == {"A1"}
    assert all(r["status"] in ("completed", "invalid_response") and 0.0 <= r["reward"] <= 1.0 for r in log)
    steps = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert "loss" in steps[0] and "reward" in steps[0] and "serl/completed_share" in steps[0]
    assert steps[0]["reward_std"] > 0 and steps[0]["grad_norm"] > 0 and steps[0]["iteration"] == 0

    trajectories = checkpoint_trajectories(config, tmp_path, 0, **fakes)
    assert trajectories["trajectories"] == 6 and trajectories["checkpoint_id"] == "smoke-iter0"
    row = json.loads((out / "trajectories.jsonl").read_text().splitlines()[0])
    assert [t["stage"] for t in row["turns"]] == ["D1", "A1", "D2", "A2", "FINAL"] and row["split"] == "train"
    assert row["trajectory_status"] == "valid" and row["is_correct"] is False and row["turns"][-1]["final"] is True
    assert row["iteration"] == 1 and row["checkpoint_id"] == "smoke-iter0" and row["protocol"] == "agentclinic_two_turn"
    assert checkpoint_trajectories(config, tmp_path, 0, **fakes)["trajectories"] == 6  # resumable: nothing is redone
    dev = checkpoint_trajectories(config, tmp_path, None, split="dev", output=tmp_path / "eval/base_dev.jsonl", **fakes)
    assert dev["checkpoint_id"] == "base" and dev["split"] == "dev" and dev["trajectories"] == 2

    # iteration 1 trains A2 samples only: prompts D1 -> A1 -> D2 of theta_1's collaborations, completions A2
    (tmp_path / "rubrics/set1.json").write_text(json.dumps(dict(RUBRIC_SET, rubric_set_version="rv2")))
    config["training_samples"]["stages"] = ["A2"]

    class A2Scorer(Scorer):
        """A different score per A2 text, so that the A2 groups have reward variance."""
        def score(self, item):
            self.items.append(item)
            return ScoreResult(item.item_id, self.name, "scored", {"r1": int(digest(item.context_turns[-1]["text"])[:2], 16) % 3})

    before, scorer = lora_b.detach().clone(), A2Scorer()
    following = train_iteration(config, tmp_path, 1, scorer=scorer, provider=Provider(), **fakes)
    assert following["initial_adapter"] == "out/iter0/adapter" and following["rubric_version"] == "rv2"
    assert set(following["train_prompts_by_stage"]) == {"A2"} and following["train_prompts"] >= 1
    second = [json.loads(line) for line in (tmp_path / "out/iter1/reward_log.jsonl").read_text().splitlines()]
    assert second and {r["stage"] for r in second} == {"A2"} and all(r["sample_id"].endswith("/A2") for r in second)
    assert all(r["r_effectiveness"] is None for r in second if r["status"] == "completed")  # no A2 predictor
    assert all(item.item_id.endswith(":A2") for item in scorer.items)
    steps = [json.loads(line) for line in (tmp_path / "out/iter1/train_log.jsonl").read_text().splitlines()]
    assert any(s.get("grad_norm", 0) > 0 for s in steps)
    assert not torch.equal(before, lora_b.detach())  # the loss on the A2 completions reached the same adapter
    config["training_samples"]["stages"] = ["A1", "A2"]
    with pytest.raises(IntegrationUnavailable):
        train_iteration(config, tmp_path, 3, scorer=Scorer(), provider=Provider(), **fakes)

    # an interrupted iteration continues from its last trainer checkpoint: no finished step is trained again, the steps
    # the killed process had logged after that checkpoint are replayed and their rows moved aside, the resume is recorded
    import shutil

    config["training_samples"]["stages"] = ["A1"]
    (tmp_path / "rubrics/set2.json").write_text(json.dumps(RUBRIC_SET))
    config["grpo"] = dict(config["grpo"], max_steps=3, save_strategy="steps", save_steps=1, save_total_limit=5)
    (tmp_path / "out/iter2").mkdir()
    shutil.copyfile(tmp_path / "out/iter0/trajectories.jsonl", tmp_path / "out/iter1/trajectories.jsonl") if not (
        tmp_path / "out/iter1/trajectories.jsonl").exists() else None
    train_iteration(config, tmp_path, 2, scorer=Scorer(), provider=VaryingProvider(), **fakes)
    out2 = tmp_path / "out/iter2"
    assert sorted(p.name for p in (out2 / "trainer").glob("checkpoint-*")) == ["checkpoint-1", "checkpoint-2", "checkpoint-3"]
    for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth", "adapter_model.safetensors"):
        assert (out2 / "trainer/checkpoint-2" / name).is_file(), name  # what a resume restores
    # the state of a process killed during step 3: checkpoint-2 is the last one, step 3's rewards are already logged
    shutil.rmtree(out2 / "trainer/checkpoint-3")
    shutil.rmtree(out2 / "adapter")
    for name in ("checkpoint.json", "run_config.json"):
        (out2 / name).unlink()
    first_two = [json.loads(line) for line in (out2 / "reward_log.jsonl").read_text().splitlines() if json.loads(line)["global_step"] < 2]
    resumed = train_iteration(config, tmp_path, 2, resume=True, scorer=Scorer(), provider=VaryingProvider(), **fakes)
    assert resumed["global_step"] == 3
    log = [json.loads(line) for line in (out2 / "reward_log.jsonl").read_text().splitlines()]
    assert sorted(r["global_step"] for r in log) == [0, 0, 1, 1, 2, 2]  # every step once; steps 0 and 1 were not computed again
    assert [r for r in log if r["global_step"] < 2] == first_two
    steps = [json.loads(line)["global_step"] for line in (out2 / "train_log.jsonl").read_text().splitlines() if "loss" in json.loads(line)]
    assert steps == [1, 2, 3]
    record, = [json.loads(line) for line in (out2 / "resume_log.jsonl").read_text().splitlines()]
    assert record["resumed_from"] == "checkpoint-2" and record["global_step"] == 2 and record["host"]
    assert record["rows_moved_to_replayed"]["reward_log.jsonl"] == 2
    replayed, = out2.glob("replayed_*")
    assert len((replayed / "reward_log.jsonl").read_text().splitlines()) == 2


def test_config_values_can_be_overridden_from_the_command_line():
    from src.common.main_config import override

    config = override({"reward": {"weights": {"effectiveness": 0.3}}, "grpo": {"max_steps": None}},
                      ["reward.weights.effectiveness=0", "grpo.max_steps=3"])
    assert config == {"reward": {"weights": {"effectiveness": 0}}, "grpo": {"max_steps": 3}}
    with pytest.raises(SystemExit):
        override(config, ["reward.weights.efectiveness=0"])


def test_effectiveness_checkpoint_is_fixed_unless_an_iteration_names_another():
    from src.grpo.trainer import effectiveness_settings

    config = {"reward": {"effectiveness": {"provider": "learned_predictor", "checkpoint": "iter0_predictor"}}}
    assert [effectiveness_settings(config, i)["checkpoint"] for i in (0, 1, 2)] == ["iter0_predictor"] * 3
    config["reward"]["effectiveness"]["checkpoint_by_iteration"] = {"2": "refit_for_theta2"}
    assert [effectiveness_settings(config, i)["checkpoint"] for i in (0, 1, 2)] == ["iter0_predictor", "iter0_predictor", "refit_for_theta2"]
    assert config["reward"]["effectiveness"]["checkpoint"] == "iter0_predictor"
