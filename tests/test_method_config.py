"""Rubric schema, T1 scoring scope, retention strategies, effectiveness providers and reward assembly."""

import json
from pathlib import Path

import pytest

from src.common.config import IntegrationUnavailable
from src.effectiveness.history import history_at
from src.effectiveness.providers import (LearnedPredictorProvider, RolloutProvider, build_effectiveness_provider,
                                         build_transform)
from src.judges.agent_state import resolve_include_initial_turn
from src.judges.failure_analysis import validate_analysis
from src.judges.llm import CachedChat, ScriptedBackend
from src.judges.rubric_consolidation import build_final_rubrics, update_registry
from src.judges.rubric_retention import build_retention
from src.judges.rubric_schema import normalize_score, validate_rubric
from src.judges.rubric_scoring import build_scoring_items
from src.judges.reward import RewardSpec
from src.training.reward_pipeline import load_reward_spec, trajectory_reward
from test_judges import RUBRIC, analysis, schema_rubric, trajectory
from test_serl_steps import fake_runtime, rich_trajectory, valid_record


def test_rubric_levels_are_self_contained_with_one_state():
    rubric = schema_rubric()
    assert validate_rubric(rubric) == []
    shifted = json.loads(json.dumps(rubric))
    shifted["scores"]["1"]["agent_side_state"] = "agent judged the evidence insufficient"
    assert any("separate rubric" in p for p in validate_rubric(shifted))
    missing = json.loads(json.dumps(rubric))
    del missing["scores"]["2"]["behavior"]
    assert validate_rubric(missing)
    same = json.loads(json.dumps(rubric))
    for level in ("0", "1", "2"):
        same["scores"][level]["behavior"] = "identical"
    assert any("different behaviors" in p for p in validate_rubric(same))
    assert [normalize_score(x) for x in (0, 1, 2)] == [0.0, 0.5, 1.0]


def test_category_with_two_states_gets_two_rubrics():
    record = valid_record("t1")
    category = {"category_id": "cat-001", "name": "c", "definition": "d", "members": [record["rubric_id"]],
                "created_iteration": 0, "status": "open", "frequency_history": {}}
    two = [schema_rubric("n1", agent="agent judged the evidence sufficient"),
           schema_rubric("n2", agent="agent judged the evidence insufficient")]
    chat = CachedChat(ScriptedBackend(lambda r: json.dumps({"rubrics": two})), None)
    result = build_final_rubrics([category], [record], chat, system_prompt="s", prompt_version="v", params={}, iteration=0)
    assert [r["name"] for r in result["rubrics"]] == ["n1", "n2"]
    assert result["rubrics"][0]["weight"] == 0.5 and result["retention_policy"] == "current_only"
    bad = [schema_rubric("n1")]
    bad[0]["scores"]["0"]["user_side_context"] = "another situation"
    chat = CachedChat(ScriptedBackend(lambda r: json.dumps({"rubrics": bad})), None)
    result = build_final_rubrics([category], [record], chat, system_prompt="s", prompt_version="v", params={}, iteration=0)
    assert not result["rubrics"] and "separate rubric" in result["excluded_categories"][0]["reason"]


def test_initial_turn_scope_is_config():
    with pytest.raises(IntegrationUnavailable, match="METHOD_TBD"):
        resolve_include_initial_turn({"mode": "production", "scoring": {"include_initial_turn": None}})
    assert resolve_include_initial_turn({"mode": "smoke"}) is False
    assert resolve_include_initial_turn({"mode": "production", "scoring": {"include_initial_turn": True}}) is True
    t = trajectory("t1")
    assert [i.turn_id for i in build_scoring_items(t, [RUBRIC])] == [3]
    assert [i.turn_id for i in build_scoring_items(t, [RUBRIC], include_initial_turn=True)] == [1, 3]
    at_t1 = json.loads(analysis("t1", "r", quote="I believe A fits better.", turn=1))
    assert validate_analysis(at_t1, t)  # excluded by default
    assert validate_analysis(at_t1, t, include_initial_turn=True) == []


def _registry_after_iteration0():
    record = valid_record("t1")
    category = {"category_id": "cat-001", "name": "c", "definition": "d", "members": [record["rubric_id"]],
                "created_iteration": 0, "status": "open", "frequency_history": {}}
    chat = CachedChat(ScriptedBackend(lambda r: json.dumps({"rubrics": [schema_rubric("iter0 rubric")]})), None)
    rubric_set = build_final_rubrics([category], [record], chat, system_prompt="s", prompt_version="v",
                                     params={}, iteration=0)
    return update_registry([category], rubric_set, [record], 0), chat


@pytest.mark.parametrize("policy,expected", [
    ("current_only", {"active": [], "constructs": 0}),
    ("cumulative", {"active": ["iter0 rubric"], "constructs": 0}),
    ("reactivation", {"active": [], "constructs": 0}),
])
def test_retention_when_category_not_observed(policy, expected):
    registry, _ = _registry_after_iteration0()
    chat = CachedChat(ScriptedBackend(lambda r: json.dumps({"rubrics": [schema_rubric("rewritten")]})), None)
    result = build_final_rubrics(registry, [], chat, system_prompt="s", prompt_version="v", params={}, iteration=1,
                                 retention=build_retention(policy))
    assert [r["name"] for r in result["rubrics"]] == expected["active"]
    assert len(chat.backend.calls) == expected["constructs"]


@pytest.mark.parametrize("policy,name", [("current_only", "rewritten"), ("cumulative", "rewritten"),
                                         ("reactivation", "iter0 rubric")])
def test_retention_when_category_observed_again(policy, name):
    registry, _ = _registry_after_iteration0()
    record = valid_record("t2")
    registry[0]["members"].append(record["rubric_id"])
    chat = CachedChat(ScriptedBackend(lambda r: json.dumps({"rubrics": [schema_rubric("rewritten")]})), None)
    result = build_final_rubrics(registry, [record], chat, system_prompt="s", prompt_version="v", params={},
                                 iteration=1, retention=build_retention(policy))
    assert [r["name"] for r in result["rubrics"]] == [name]
    with pytest.raises(ValueError):
        build_retention("forever")


def test_effectiveness_provider_dependencies(tmp_path):
    assert build_effectiveness_provider({"provider": None}, 0.0) is None  # w_E = 0 ablation
    with pytest.raises(IntegrationUnavailable, match="w_E > 0"):
        build_effectiveness_provider({"provider": None}, 0.3)
    with pytest.raises(IntegrationUnavailable, match="checkpoint"):
        build_effectiveness_provider({"provider": "learned_predictor", "checkpoint": None}, 0.3)
    with pytest.raises(IntegrationUnavailable, match="missing"):
        build_effectiveness_provider({"provider": "learned_predictor", "checkpoint": str(tmp_path)}, 0.3)
    assert build_transform("affine_unit")(-1.0) == 0.0 and build_transform("affine_unit")(1.0) == 1.0
    assert build_transform(lambda d: 0.5)(0.3) == 0.5
    with pytest.raises(ValueError):
        build_transform("unknown")


def test_learned_predictor_provider_checkpoint_and_protocol(tmp_path):
    torch = pytest.importorskip("torch")
    from src.effectiveness.model import CollaborationValueModel
    from src.effectiveness.synthetic import SYNTHETIC_MODEL_ID, SYNTHETIC_REVISION, load_synthetic_backbone

    backbone, tokenizer = load_synthetic_backbone(SYNTHETIC_MODEL_ID, SYNTHETIC_REVISION)
    model = CollaborationValueModel(backbone, tokenizer, base_model_id=SYNTHETIC_MODEL_ID,
                                    base_model_revision=SYNTHETIC_REVISION, max_length=64)
    with torch.no_grad():
        model.value_head.weight.normal_()
    model.save_pretrained(tmp_path / "ckpt", metadata={"continuation_protocol_hash": "p1"})
    with pytest.raises(IntegrationUnavailable, match="different continuation protocol"):
        LearnedPredictorProvider.from_checkpoint(str(tmp_path / "ckpt"), expected_protocol_hash="p2",
                                                 backbone_loader=load_synthetic_backbone)
    provider = LearnedPredictorProvider.from_checkpoint(str(tmp_path / "ckpt"), expected_protocol_hash="p1",
                                                        backbone_loader=load_synthetic_backbone)
    history = history_at(rich_trajectory("t1"), 3)
    response = rich_trajectory("t1")["turns"][3]["text"]
    before, after = provider.predict_before(history), provider.predict_after(history, response)
    assert provider.predict_delta(history, response) == pytest.approx(after - before)
    assert before != after and -1.0 <= after - before <= 1.0


def test_rollout_provider_delta():
    from src.effectiveness.continuation import ContinuationSpec

    spec = ContinuationSpec.from_dict({"k": 2, "max_rounds": 3, "final_decision_role": "doctor",
                                       "seed_policy": {"mode": "paired_by_history", "base_seed": 0}})
    runtime, _ = fake_runtime()
    # Final answer is correct only when the dialogue contains the fixed original response.
    runtime.doctor = lambda system, user: ("Final answer: option A - a" if "I agree, switching to B." in user
                                           else "Final answer: option B - b") \
        if "This is the final decision" in user else "Let us keep discussing."
    provider = RolloutProvider(runtime, spec)
    t = rich_trajectory("t1")
    history = dict(history_at(t, 3), ground_truth=t["ground_truth"])
    assert provider.predict_before(history) == 0.0 and provider.predict_after(history, t["turns"][3]["text"]) == 1.0
    assert provider.predict_delta(history, t["turns"][3]["text"]) == 1.0  # P(success|h,r)=1, P(success|h)=0


def test_trajectory_reward_assembly():
    config = {"reward": json.loads(open("configs/judges/serl_roles.json").read())["reward"]}
    spec = load_reward_spec(dict(config, reward=dict(config["reward"], weights={
        "outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.0})))
    t = trajectory("t1")
    result = trajectory_reward(t, {"t1:turn3": {"r1": 2}}, spec, include_initial_turn=False)
    assert result["scored_turns"] == [3] and result["reward"] == pytest.approx(0.3)  # wrong outcome, R_A = 1
    with pytest.raises(ValueError, match="no rubric scores"):
        trajectory_reward(t, {}, spec, include_initial_turn=False)
    full = load_reward_spec(config)  # placeholder weights, w_E = 0.3
    with pytest.raises(IntegrationUnavailable):
        trajectory_reward(t, {"t1:turn3": {"r1": 2}}, full, include_initial_turn=False)

    class Constant:
        name = "constant"

        def predict_before(self, history):
            return 0.5

        def predict_after(self, history, response):
            return 0.5

        def predict_delta(self, history, response):
            return self.predict_after(history, response) - self.predict_before(history)

        def identity(self):
            return {"provider": "constant"}

    result = trajectory_reward(t, {"t1:turn3": {"r1": 2}}, full, include_initial_turn=False, provider=Constant())
    assert result["reward"] == pytest.approx(0.4 * 0.0 + 0.3 * 1.0 + 0.3 * 0.5)
    assert isinstance(full, RewardSpec) and full.weights == {"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.3}


def test_config_extends_overrides_key_by_key(tmp_path):
    from src.common.config import read_config
    from src.effectiveness.train import load_predictor_config

    (tmp_path / "base.json").write_text(json.dumps({"a": {"x": 1, "y": 2}, "b": [1], "c": "base"}))
    (tmp_path / "child.json").write_text(json.dumps({"extends": "base.json", "a": {"y": 3}, "b": [2]}))
    (tmp_path / "grandchild.json").write_text(json.dumps({"extends": "child.json", "c": "leaf"}))
    assert read_config(tmp_path / "grandchild.json") == {"a": {"x": 1, "y": 3}, "b": [2], "c": "leaf"}

    configs = Path(__file__).resolve().parents[1] / "configs/effectiveness"
    pilot, v3, v3_lora, frozen, lora = (load_predictor_config(configs / f"{n}.json") for n in (
        "predictor_pilot", "v3_choice_conditioned", "v3_choice_conditioned_lora", "predictor_frozen", "predictor_lora"))
    hashes = [c["continuation_spec"].protocol_hash() for c in (pilot, v3, v3_lora, frozen, lora)]
    # v3 keeps the generation protocol (its labels stay loadable); v4 is a different protocol.
    assert hashes[0] == hashes[1] == hashes[2] != hashes[3] == hashes[4]
    assert frozen["continuation_spec"].doctor_prompt and frozen["continuation_spec"].continuation_guideline
    assert pilot["continuation_spec"].doctor_prompt is None and "doctor_prompt" not in pilot["continuation_spec"].to_dict()
    assert (frozen["predictor"]["training_mode"], lora["predictor"]["training_mode"]) == (
        "frozen_backbone_scalar_head", "lora_scalar_head")
    assert lora["labels"] == frozen["labels"] and lora["candidates"] == frozen["candidates"] and "extends" not in lora
    assert v3["candidates"]["advocate_options"] and frozen["candidates"]["behaviors"]
    import hashlib
    doctor = (configs.parents[1] / frozen["continuation_spec"].doctor_prompt).read_bytes()
    assert frozen["continuation_spec"].prompt_versions["doctor"] == "sha256:" + hashlib.sha256(doctor).hexdigest()[:16]


def test_reactivation_across_three_iterations():
    """A is active, disappears (inactive but kept in the registry), and is active again with its rubric
    when its failure returns; B stays active with its rubric; D is new."""
    def category(cid, record):
        return {"category_id": cid, "name": cid, "definition": "d", "members": [record["rubric_id"]],
                "created_iteration": 0, "status": "open", "frequency_history": {}}

    def build(registry, records, iteration, names):
        chat = CachedChat(ScriptedBackend(lambda r: json.dumps({"rubrics": [schema_rubric(next(names))]})), None)
        rubric_set = build_final_rubrics(registry, records, chat, system_prompt="s", prompt_version="v", params={},
                                         iteration=iteration, retention=build_retention("reactivation"),
                                         importance="failure_frequency")
        return rubric_set, update_registry(registry, rubric_set, records, iteration), len(chat.backend.calls)

    a0, b0 = valid_record("a0"), valid_record("b0")
    rubric_set, registry, calls = build([category("cat-A", a0), category("cat-B", b0)], [a0, b0], 0, iter(["A v0", "B v0"]))
    assert {r["category_id"]: r["retention_reason"] for r in rubric_set["rubrics"]} == \
        {"cat-A": "observed this iteration", "cat-B": "observed this iteration"} and calls == 2

    b1, b1b, d1 = valid_record("b1"), valid_record("b1b"), valid_record("d1")
    by_id = {c["category_id"]: c for c in registry}
    by_id["cat-B"]["members"] += [b1["rubric_id"], b1b["rubric_id"]]
    registry.append(dict(category("cat-D", d1), created_iteration=1))
    rubric_set, registry, calls = build(registry, [b1, b1b, d1], 1, iter(["D v1"]))
    assert {r["category_id"]: (r["name"], r["retention_reason"]) for r in rubric_set["rubrics"]} == \
        {"cat-B": ("B v0", "still observed"), "cat-D": ("D v1", "observed this iteration")} and calls == 1
    assert {r["category_id"]: r["weight"] for r in rubric_set["rubrics"]} == pytest.approx({"cat-B": 2 / 3, "cat-D": 1 / 3})
    dormant = next(c for c in registry if c["category_id"] == "cat-A")
    assert dormant["status"] == "inactive" and dormant["rubrics"][0]["name"] == "A v0"
    assert dormant["frequency_history"] == {"0": 1, "1": 0} and dormant["active_iterations"] == [0]
    assert any(e["category_id"] == "cat-A" and "dormant" in e["reason"] for e in rubric_set["excluded_categories"])

    a2 = valid_record("a2")
    dormant["members"].append(a2["rubric_id"])
    rubric_set, registry, calls = build(registry, [a2], 2, iter([]))
    assert [(r["category_id"], r["name"], r["retention_reason"], r["origin"]) for r in rubric_set["rubrics"]] == \
        [("cat-A", "A v0", "reactivated", "reused")] and calls == 0
    back = next(c for c in registry if c["category_id"] == "cat-A")
    assert back["status"] == "active" and back["active_iterations"] == [0, 2]
    assert back["reactivation_count"] == 1 and back["first_seen_iteration"] == 0 and back["last_seen_iteration"] == 2
    assert back["current_failure_frequency"] == 1 and back["historical_failure_frequency"] == 2
    assert next(c for c in registry if c["category_id"] == "cat-B")["reactivation_count"] == 0
    assert back["frequency_history"] == {"0": 1, "1": 0, "2": 1} and back["last_observed_iteration"] == 2
    assert {c["category_id"]: c["status"] for c in registry} == {"cat-A": "active", "cat-B": "inactive", "cat-D": "inactive"}


def test_appropriateness_uses_the_rubric_weights_of_the_applicable_rubrics():
    from src.judges.reward import rubric_weights, turn_appropriateness

    rubric_set = {"rubrics": [{"rubric_id": "A", "weight": 0.6}, {"rubric_id": "B", "weight": 0.2},
                              {"rubric_id": "C", "weight": 0.2}]}
    weights = rubric_weights(rubric_set)
    assert turn_appropriateness({"A": 2, "B": 1, "C": "NA"}, weights) == pytest.approx((0.6 * 1.0 + 0.2 * 0.5) / 0.8)
    assert turn_appropriateness({"A": 2, "B": 1, "C": "NA"}) == pytest.approx(0.75)  # no weights: plain mean
    assert turn_appropriateness({"A": "NA", "B": "NA", "C": "NA"}, weights) is None
    assert rubric_weights({"rubrics": [{"rubric_id": "A"}]}) is None
    spec = load_reward_spec({"reward": {"weights": {"outcome": 0, "appropriateness": 1, "effectiveness": 0},
                                        "softmin_temperature": 0.5}})
    from src.judges.reward import TurnSignals, compute_reward
    turn = [TurnSignals(turn_id=1, rubric_labels={"A": 2, "B": 0})]
    assert compute_reward(turn, 0.0, spec, weights)["reward"] == pytest.approx(0.75)
    assert compute_reward(turn, 0.0, spec)["reward"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        build_final_rubrics([], [], None, system_prompt="s", prompt_version="v", params={}, iteration=0, importance="delta")


def test_api_usage_ledger_and_cost_summary(tmp_path, monkeypatch):
    from src.common import usage

    usage.record("doctor", "m", {"prompt_tokens": 1})  # no ledger configured: nothing is written
    ledger = tmp_path / "run/api_usage.jsonl"
    monkeypatch.setenv(usage.ENV, str(ledger))
    monkeypatch.setenv(usage.STAGE_ENV, "train")
    monkeypatch.setenv(usage.ITERATION_ENV, "2")
    with usage.case("case-7"):
        usage.record("collaboration", "openai/gpt-4o-mini", {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.5}, attempts=3)
    usage.record("collaboration", "openai/gpt-4o-mini", {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    monkeypatch.setenv(usage.STAGE_ENV, "rubrics")
    usage.record("rubric-scorer-v1", "other/model", {"prompt_tokens": 10, "completion_tokens": 1})
    usage.record("rubric-scorer-v1", "other/model", None)
    first = json.loads(ledger.read_text().splitlines()[0])
    assert set(first) == {"timestamp", "purpose", "stage", "iteration", "case_id", "model", "input_tokens",
                          "output_tokens", "cost_usd", "retry_count"}
    assert (first["stage"], first["iteration"], first["case_id"], first["retry_count"]) == ("train", 2, "case-7", 2)
    assert json.loads(ledger.read_text().splitlines()[1])["case_id"] is None
    summary = usage.summarize([ledger], {"openai/gpt-4o-mini": {"input": 0.15, "output": 0.60}})
    assert summary["total_usd"] == pytest.approx(0.5 + 0.75) and summary["calls"] == 4 and summary["unpriced_calls"] == 2
    assert summary["by_model"]["openai/gpt-4o-mini"]["input_tokens"] == 1_001_000 and summary["retries"] == 2
    assert summary["by_stage"]["train"]["cost_usd"] == pytest.approx(1.25) and summary["by_stage"]["rubrics"]["calls"] == 2
    assert summary["by_purpose"]["rubric-scorer-v1"]["unpriced_calls"] == 2
    assert usage.openrouter("https://openrouter.ai/api/v1") and not usage.openrouter("https://api.openai.com/v1")


def test_category_diagnostics_flag_duplicates_and_missed_reactivation():
    from src.judges.rubric_consolidation import category_diagnostics

    def category(cid, name, definition, created, status, history):
        return {"category_id": cid, "name": name, "definition": definition, "created_iteration": created,
                "status": status, "frequency_history": history, "members": [], "active_iterations": [created]}

    registry = [
        category("cat-001", "Yields to unsupported pressure", "The agent adopts the doctor claim without evidence", 0,
                 "inactive", {"0": 3, "1": 0, "2": 0}),
        category("cat-002", "Withholds findings", "The agent does not report examination findings", 0, "active",
                 {"0": 2, "1": 2, "2": 4}),
        category("cat-003", "Yields to unsupported doctor pressure", "The agent adopts the doctor claim without any evidence", 2,
                 "active", {"2": 1}),
    ]
    report = category_diagnostics(registry, 2)
    assert [p["categories"] for p in report["near_duplicate_pairs"]] == [["cat-001", "cat-003"]]
    assert report["possible_missed_reactivation"][0]["names"][1] == "Yields to unsupported doctor pressure"
    assert report["fragmentation"] == {"observed_categories": 2, "failures": 5, "single_failure_categories": 1,
                                       "single_failure_share": 0.5, "categories_per_failure": 0.4}
    assert report["new_categories"] == ["cat-003"] and category_diagnostics(registry[:2], 2)["near_duplicate_pairs"] == []


def test_effectiveness_turns_can_use_their_own_soft_min_temperature():
    from src.judges.reward import TurnSignals, compute_reward, soft_min

    reward = {"weights": {"outcome": 0, "appropriateness": 0, "effectiveness": 1}, "softmin_temperature": 0.5,
              "effectiveness": {"transform": "affine_unit", "softmin_temperature": 0.1}}
    turns = [TurnSignals(1, delta_effectiveness=0.8), TurnSignals(3, delta_effectiveness=-0.4)]
    assert compute_reward(turns, 0.0, load_reward_spec({"reward": reward}))["reward"] == pytest.approx(soft_min([0.9, 0.3], 0.1))
    reward["effectiveness"]["softmin_temperature"] = None
    assert compute_reward(turns, 0.0, load_reward_spec({"reward": reward}))["reward"] == pytest.approx(soft_min([0.9, 0.3], 0.5))
