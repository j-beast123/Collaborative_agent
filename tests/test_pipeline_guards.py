"""Guards of the main pipeline: information boundaries, split isolation, the Effectiveness service
(local = gRPC, calibration, version checks, missing turns), output preservation, config validation."""

from conftest import AGENTCLINIC, require_env_histories, require_local  # noqa: F401
import json
import math
from pathlib import Path

import pytest

from src.agentclinic.adapter import agent_view, doctor_view, render, render_private
from src.agentclinic.outcome import DiagnosisJudge
from src.agentclinic.protocol import ClinicRuntime
from src.agentclinic.selection import split_eligible_cases
from src.agentclinic.trajectory import collaborate
from src.common.config import IntegrationUnavailable, read_config
from src.common.main_config import load_main_config, problems_of
from src.common.run_dir import config_hash, create_run, write_new
from src.effectiveness import calibration
from src.effectiveness.service import (EffectivenessUnavailable, GrpcEffectivenessClient, LocalEffectiveness, check_identity,
                                       start_server)
from src.effectiveness.synthetic import load_synthetic_backbone
from src.grpo.dataset import grpo_rows
from src.judges.failure_analysis import analyzer_input
from src.judges.rubric_scoring import build_scoring_items
from test_agentclinic import GOLD
from test_effectiveness_predictor import tiny_model
from test_grpo import D1, STATE, is_state_call, judge_chat, source_config
from test_judges import RUBRIC

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/serl_agentclinic.json"
PRIVATE_MARKERS = ("PTOSIS-FINDING", "ANTIBODY-RESULT", "Decrement on repetitive stimulation")
HISTORY = {"question": "q", "options": {}, "scenario": "the Agent's information", "turns": [{"speaker": "doctor", "text": "d1"}]}


def recording_runtime():
    """Doctor and Agent that never name the gold diagnosis or a private finding; every prompt is recorded."""
    prompts = {"doctor": [], "agent": [], "state": []}

    def doctor(system, user):
        prompts["doctor"].append(user)
        if "Give your final diagnosis" in user:
            return json.dumps({"final_diagnosis": "Viral illness", "brief_reason": "r"})
        return "I still think this is viral.\nProvisional diagnosis: Viral illness"

    def agent(system, user):
        if is_state_call(user):
            prompts["state"].append(user)
            return json.dumps(STATE | {"agent_current_assessment": "Something else"})
        prompts["agent"].append(user)
        if "No one has commented" in user:
            return "Current assessment: Something else"
        return "Current assessment: Something else\nResponse: I am not sure this is viral."

    return ClinicRuntime(doctor=doctor, agent=agent, doctor_system="d", agent_system="a", state_system="s"), prompts


def test_information_boundaries_in_every_prompt_of_a_collaboration(tmp_path):
    cases = split_eligible_cases(source_config(tmp_path), tmp_path)
    runtime, prompts = recording_runtime()
    judge = DiagnosisJudge(judge_chat(tmp_path), "judge")
    for case in list(cases.values())[:3]:
        trajectory = collaborate(case, D1, runtime, judge, trajectory_id=f"{case['case_id']}:t")
        # 1. the Doctor never receives the Agent's private evidence
        assert len(prompts["doctor"]) >= 2
        for prompt in prompts["doctor"]:
            assert not any(marker in prompt for marker in PRIVATE_MARKERS)
        # 2. nobody is given the correct diagnosis or the management plan
        for kind in ("doctor", "agent", "state"):
            for prompt in prompts[kind]:
                assert GOLD not in prompt and "SECRET-MANAGEMENT-PLAN" not in prompt and "Correct_Diagnosis" not in prompt
        # the Agent does see its private evidence, in its response prompts and in the state elicitation
        assert all(any(marker in prompt for marker in PRIVATE_MARKERS) for prompt in prompts["agent"] + prompts["state"])
        # 3. rubric generation, rubric scoring and the Effectiveness input do not carry the correct diagnosis
        view = json.dumps(analyzer_input(trajectory))
        assert GOLD not in view and "ground_truth" not in view and "correct_diagnosis" not in view
        for item in build_scoring_items(trajectory, [RUBRIC], include_initial_turn=True):
            payload = json.dumps(item.payload())
            assert GOLD not in payload and "final_diagnosis" not in payload and "is_correct" not in payload
        for kind in prompts:
            prompts[kind].clear()
    assert GOLD not in render(doctor_view(case)) and GOLD not in render(agent_view(case)) and GOLD not in render_private(case)
    rows = grpo_rows(cases, [{"history_id": f"{c}:A1", "case_id": c, "history": D1} for c in cases], "consultant system")
    assert all(GOLD not in json.dumps(row) for row in rows)  # the GRPO policy input


def test_legacy_analyzer_input_still_shows_the_correct_option():
    from test_judges import trajectory

    assert analyzer_input(trajectory("t1"))["ground_truth"] == {"correct_option": "A"}  # MedMCQA protocol, unchanged


def test_real_split_is_disjoint_and_test_cases_are_never_in_the_frozen_histories():
    source = read_config(ROOT / "configs/effectiveness/v5_agentclinic_full.json")
    if not (ROOT / source["dataset"]["path"]).is_file():
        pytest.skip("AgentClinic data file not present")
    cases = split_eligible_cases(source, ROOT)
    by_split = {name: {cid for cid, case in cases.items() if case["split"] == name} for name in ("train", "dev", "test")}
    assert {name: len(ids) for name, ids in by_split.items()} == {"train": 103, "dev": 34, "test": 35}
    assert not (by_split["train"] & by_split["dev"]) and not (by_split["train"] & by_split["test"]) and not (by_split["dev"] & by_split["test"])
    assert split_eligible_cases(source, ROOT).keys() == cases.keys()  # 12. deterministic
    assert {cid: c["split"] for cid, c in split_eligible_cases(source, ROOT).items()} == {cid: c["split"] for cid, c in cases.items()}
    histories = ROOT / "outputs/effectiveness/v5_agentclinic_full/histories.jsonl"
    if histories.is_file():
        used = {json.loads(line)["case_id"] for line in histories.read_text().splitlines() if line.strip()}
        assert used <= by_split["train"] | by_split["dev"] and not (used & by_split["test"])


@pytest.fixture
def checkpoint(tmp_path):
    import torch

    torch.manual_seed(0)
    model = tiny_model()
    with torch.no_grad():
        model.value_head.weight.normal_(0, 0.5)
    model.save_pretrained(tmp_path / "ckpt", metadata={"continuation_protocol_hash": "hash-1", "k": 9})
    calibration.save({"method": "temperature", "a": 1 / 2.97, "b": 0.0, "temperature": 2.97}, tmp_path / "calibration.json")
    return tmp_path


def local(checkpoint, **kwargs):
    settings = dict(model_version="v1-test", calibration=str(checkpoint / "calibration.json"), device="cpu", allow_cpu=True,
                    supported_turns=("A1",), expected_protocol_hash="hash-1", expected_temperature=2.97,
                    backbone_loader=load_synthetic_backbone)
    return LocalEffectiveness.from_checkpoint(str(checkpoint / "ckpt"), **dict(settings, **kwargs))


def test_effectiveness_service_parity_calibration_and_missing_turns(checkpoint):
    pytest.importorskip("grpc")
    predictor = local(checkpoint)
    server, port = start_server(predictor, port=0, log_path=checkpoint / "grpc_logs/requests.jsonl")
    try:
        client = GrpcEffectivenessClient(f"127.0.0.1:{port}", expected={
            "model_version": "v1-test", "calibration_temperature": 2.97, "continuation_protocol_hash": "hash-1",
            "supported_turns": ["A1"]}, timeout_s=30, retries=0)
        direct = predictor.predict(HISTORY, "The antibody result is elevated.", "A1", request_id="r1", trajectory_id="t1")
        served = client.predict(HISTORY, "The antibody result is elevated.", "A1", request_id="r1", trajectory_id="t1")
        # 5. the served prediction is the local one, field by field
        for name in ("p_before", "p_after", "delta", "logit_before", "logit_after", "model_version", "checkpoint_id",
                     "calibration_temperature", "intervention_turn", "request_id", "trajectory_id", "schema_version"):
            assert getattr(served, name) == getattr(direct, name), name
        assert (served.source, direct.source) == ("grpc", "local")
        assert set(served.to_dict()) == set(direct.to_dict())
        # 6. the temperature is really applied: p = sigmoid(logit / T), and it differs from the raw sigmoid
        sigmoid = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
        assert served.p_before == pytest.approx(sigmoid(served.logit_before / 2.97), abs=1e-12)
        assert served.p_after == pytest.approx(sigmoid(served.logit_after / 2.97), abs=1e-12)
        assert served.delta == pytest.approx(served.p_after - served.p_before, abs=1e-12)
        assert abs(served.p_after - sigmoid(served.logit_after)) > 1e-4 and served.calibration_temperature == 2.97
        raw = local(checkpoint, calibration=None, expected_temperature=None).predict(HISTORY, "The antibody result is elevated.")
        assert raw.p_after == pytest.approx(sigmoid(raw.logit_after)) and raw.logit_after == served.logit_after
        # P(success | h) alone, and its cache: the same history gives the same BEFORE value
        before_only = client.predict(HISTORY, None)
        assert (before_only.p_after, before_only.delta) == (None, None) and before_only.p_before == served.p_before
        # 7. a turn without a predictor is refused, locally and over gRPC; nothing is substituted
        for target in (predictor, client):
            with pytest.raises(EffectivenessUnavailable, match="A2"):
                target.predict(HISTORY, "x", "A2")
        with pytest.raises(ValueError):  # malformed requests are errors, not defaults
            client.predict({"turns": "not a list"}, "x")
        with pytest.raises(ValueError):
            client.predict(HISTORY, "   ")
        health = client.health()
        assert health["status"] == "SERVING" and health["requests"] >= 2 and health["errors"] >= 2
        log = [json.loads(line) for line in (checkpoint / "grpc_logs/requests.jsonl").read_text().splitlines()]
        assert log[0]["request_id"] == "r1" and log[0]["trajectory_id"] == "t1" and log[0]["status"] == "OK"
        assert {row["status"] for row in log} >= {"OK", "INVALID_ARGUMENT"}
        # 13. a client that expects another model, temperature or protocol does not connect
        for wrong in ({"model_version": "v2"}, {"calibration_temperature": 1.0}, {"continuation_protocol_hash": "other"},
                      {"supported_turns": ["A1", "A2"]}):
            with pytest.raises(IntegrationUnavailable, match="mismatch"):
                GrpcEffectivenessClient(f"127.0.0.1:{port}", expected=wrong, timeout_s=30, retries=0)
        client.close()
    finally:
        server.stop(grace=1).wait()
    with pytest.raises(IntegrationUnavailable, match="failed"):  # no service: an error after the retries, no fallback
        GrpcEffectivenessClient(f"127.0.0.1:{port}", timeout_s=1, retries=1, retry_wait_s=0.1)


def test_predictor_start_up_checks(checkpoint):
    with pytest.raises(IntegrationUnavailable, match="temperature"):
        local(checkpoint, expected_temperature=1.5)
    with pytest.raises(IntegrationUnavailable, match="different continuation protocol"):
        local(checkpoint, expected_protocol_hash="other")
    with pytest.raises(IntegrationUnavailable, match="CPU"):
        local(checkpoint, allow_cpu=False)
    with pytest.raises(IntegrationUnavailable, match="explicit device"):
        local(checkpoint, device=None)
    with pytest.raises(IntegrationUnavailable, match="missing"):
        LocalEffectiveness.from_checkpoint(str(checkpoint / "nowhere"), model_version="v", calibration=None, device="cpu", allow_cpu=True)
    calibration.save({"method": "temperature", "a": -1.0, "b": 0.0}, checkpoint / "bad.json")
    with pytest.raises(ValueError, match="monotonic"):
        local(checkpoint, calibration=str(checkpoint / "bad.json"), expected_temperature=None)
    check_identity({"model_version": "v1", "calibration_temperature": 2.9657}, {"model_version": "v1", "calibration_temperature": 2.97})
    with pytest.raises(IntegrationUnavailable):
        check_identity({"model_version": "v1", "calibration_temperature": None}, {"calibration_temperature": 2.97})


def test_runs_are_versioned_and_never_overwritten(tmp_path):
    config = {"seed": 0, "reward": {"weights": {"outcome": 1}}}
    run = create_run(tmp_path / "serl", "e2e", config, root=ROOT, seed=0, purpose="test", run_id="e2e_fixed")
    assert sorted(p.name for p in run.iterdir()) == ["environment.json", "manifest.json", "resolved_config.json"]
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["config_hash"] == config_hash(config) and manifest["seed"] == 0 and "git" in manifest
    with pytest.raises(IntegrationUnavailable, match="never overwritten"):  # 11. the same id again
        create_run(tmp_path / "serl", "e2e", config, root=ROOT, run_id="e2e_fixed")
    assert create_run(tmp_path / "serl", "e2e", config, root=ROOT, run_id="e2e_fixed", resume=True) == run
    with pytest.raises(IntegrationUnavailable, match="another configuration"):
        create_run(tmp_path / "serl", "e2e", dict(config, seed=1), root=ROOT, run_id="e2e_fixed", resume=True)
    fresh = create_run(tmp_path / "serl", "e2e", config, root=ROOT)
    assert fresh != run and fresh.name.startswith("e2e_")
    with pytest.raises(FileExistsError):
        write_new(run / "manifest.json", "{}")
    assert config_hash(config) == config_hash(json.loads(json.dumps(config)))  # 12. deterministic


def test_main_config_is_the_current_design_and_legacy_configs_are_refused():
    require_env_histories("configs/serl_agentclinic.json")
    config = load_main_config(MAIN, ROOT) if (ROOT / read_config(MAIN)["environment"]["histories"]).is_file() else read_config(MAIN)
    assert config["collaboration"]["protocol"] == "agentclinic_two_turn" and config["collaboration"]["max_rounds"] == 2
    effectiveness = config["reward"]["effectiveness"]
    assert effectiveness["available"] == {"A1": True, "A2": False} and effectiveness["stages"] == ["A1"]
    assert effectiveness["expected_temperature"] == 2.97 and effectiveness["provider"] == "grpc"
    assert config["roles"]["rubric_manager"]["max_active_rubrics"] is None
    # 10. no fixed expected action for Recover, in any evaluation config
    for name in ("eval.json", "agentclinic.json"):
        assert read_config(ROOT / "configs/eval" / name)["expected_actions"]["states"]["Recover"] is None

    def broken(**changes):
        other = json.loads(json.dumps(config))
        for path, value in changes.items():
            *parents, leaf = path.split("__")
            section = other
            for parent in parents:
                section = section[parent]
            section[leaf] = value
        return " | ".join(problems_of(other, ROOT))

    assert "max_rounds must be 2" in broken(collaboration__max_rounds=7)
    assert "stages must be exactly the turns marked available" in broken(reward__effectiveness__stages=["A1", "A2"])
    assert "calibration is required" in broken(reward__effectiveness__calibration=None)
    assert broken(rubric_importance__strategy="uniform") == ""  # the weight source is a setting, not fixed
    assert "failure_frequency or uniform" in broken(rubric_importance__strategy="effectiveness_gap")
    assert config["rubric_importance"]["status"] == "provisional"
    assert config["rubric_aggregation"]["method"] == "weighted_softmin" and config["rubric_aggregation"]["temperature"] == 0.5
    assert "weighted_softmin" in broken(rubric_aggregation__method="weighted_mean")
    assert config["reward"]["credit_assignment"] == "turn_local" and "turn_local" in broken(reward__credit_assignment="from_turn_on")
    assert config["rubric_registry"] == {**config["rubric_registry"], "max_active_categories": 8, "schedule": "evolving"}
    assert config["training_samples"]["stages"] == ["A1", "A2"] and config["reward"]["no_applicable_rubric_policy"] == "group_mean"
    assert "rubric_registry.max_active_categories must be a positive integer" in broken(rubric_registry__max_active_categories=None)
    assert "rubric_registry.max_active_categories" in broken(roles__rubric_manager__max_active_rubrics=8)
    assert "missing_component_policy must be omit" in broken(reward__missing_component_policy="renormalize")
    assert "fixed_rubric_set" in broken(rubric_registry__schedule="fixed")
    assert "training_samples.stages" in broken(training_samples__stages=[])
    assert "group_mean" in broken(reward__no_applicable_rubric_policy="renormalize")
    assert "devices.policy" in broken(devices__policy="cpu")
    legacy = " | ".join(problems_of(read_config(ROOT / "configs/judges/serl_roles.json"), ROOT))
    assert "pipeline must be" in legacy and "collaboration.protocol" in legacy
    with pytest.raises(IntegrationUnavailable, match="not valid"):
        load_main_config(ROOT / "configs/judges/serl_roles.json", ROOT)


# ----------------------------------------------------------------------------- added by the integration audit

def test_scorer_failure_is_explicit_and_never_read_as_no_match(tmp_path):
    """A response whose rubric scoring failed has no computed reward (status scoring_failed, group mean);
    its labels are not treated as No Match or 0, and a batch that is mostly failures is an error."""
    from src.judges.rubric_scoring import RubricScorer, ScoreResult
    from test_grpo import GOOD, REWARD, call, environment as environment_fixture, reward_for

    environment = environment_fixture.__wrapped__(tmp_path)
    sample = sorted(environment.histories)[0]

    class Flaky(RubricScorer):
        name = "flaky"

        def score(self, item):
            text = item.context_turns[1]["text"]  # the A1 of the response being scored
            if "second" in text:
                return ScoreResult(item.item_id, self.name, "invalid", problems=["not json"])
            if "third" in text:
                return ScoreResult(item.item_id, self.name, "scored", {})  # scored, but a rubric label is missing
            return ScoreResult(item.item_id, self.name, "scored", {"r1": 2})

    texts = [GOOD, GOOD.replace("supports MG.", "supports MG, second."), GOOD.replace("supports MG.", "supports MG, third.")]
    reward = reward_for(environment, tmp_path, scorer=Flaky(), max_failed_share=0.7)
    rewards = call(reward, [sample] * 3, texts)
    rows = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["completed", "scoring_failed", "scoring_failed"]
    assert [r["reward_source"] for r in rows] == ["computed", "group_mean", "group_mean"]
    assert rewards[1] == rewards[2] == rewards[0]  # neutral within the group, not a low score
    assert rows[1]["components"] is None and rows[1]["r_appropriateness"] is None
    assert rows[1]["rubric_scores"]["A1"]["status"] == "invalid" and rows[2]["rubric_scores"]["A1"]["status"] == "incomplete"
    assert rows[1]["rubric_scores"]["A1"]["labels"] == {} == rows[2]["rubric_scores"]["A1"]["labels"]
    strict = reward_for(environment, tmp_path, scorer=Flaky(), max_failed_share=0.5)
    with pytest.raises(IntegrationUnavailable, match="no computed reward"):
        call(strict, [sample] * 3, texts)


def test_no_match_differs_from_zero_and_weights_renormalize_over_applicable_rubrics():
    """8 and 9: a No Match rubric leaves the mean untouched (a 0 would pull it down), and the weights of the
    applicable rubrics are renormalized by their own sum."""
    from src.judges.reward import appropriateness_detail, soft_min, turn_appropriateness

    weights = {"A": 0.6, "B": 0.3, "C": 0.1}
    assert turn_appropriateness({"A": 2, "B": 1, "C": "NA"}, weights) == pytest.approx((0.6 * 1.0 + 0.3 * 0.5) / 0.9)
    assert turn_appropriateness({"A": 2, "B": 1, "C": 0}, weights) == pytest.approx((0.6 * 1.0 + 0.3 * 0.5 + 0.1 * 0.0) / 1.0)
    assert turn_appropriateness({"A": 2, "B": 1, "C": "NA"}, weights) > turn_appropriateness({"A": 2, "B": 1, "C": 0}, weights)
    assert turn_appropriateness({"A": "NA", "B": 2, "C": "NA"}, weights) == 1.0  # one applicable rubric: its own score
    nothing = appropriateness_detail({"A": "NA", "B": "NA", "C": "NA"}, weights)
    assert nothing["appropriateness"] is None and nothing["no_applicable_rubric"] is True  # not 0.0
    zero = appropriateness_detail({"A": 0, "B": "NA", "C": "NA"}, weights)
    assert zero["appropriateness"] == 0.0 and zero["no_applicable_rubric"] is False
    assert soft_min([1.0, 0.0], 0.5) < 0.5 and soft_min([0.75], 0.5) == pytest.approx(0.75) and soft_min([], 0.5) is None


def test_results_of_a_run_are_written_once(tmp_path):
    """11: base trajectories and a trained iteration are never rewritten inside a run."""
    from src.grpo.trainer import base_trajectories, train_iteration, trajectories_path

    config = {"output_dir": "out", "reward": {"weights": {"outcome": 1, "appropriateness": 0, "effectiveness": 0}}}
    path = trajectories_path(config, tmp_path, 0)
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    with pytest.raises(IntegrationUnavailable, match="written once"):
        base_trajectories(config, tmp_path)
    assert path.read_text() == "{}\n"
    (tmp_path / "out/iter0").mkdir()
    (tmp_path / "out/iter0/checkpoint.json").write_text("{}")
    with pytest.raises(IntegrationUnavailable, match="already trained"):
        train_iteration(config, tmp_path, 0)
    assert (tmp_path / "out/iter0/checkpoint.json").read_text() == "{}"


def test_device_placement_is_explicit():
    config = load_main_config(MAIN, ROOT) if (ROOT / read_config(MAIN)["environment"]["histories"]).is_file() else read_config(MAIN)
    assert config["devices"]["placement"] == "concurrent"

    def problems(**devices):
        other = json.loads(json.dumps(config))
        other["devices"].update(devices)
        return " | ".join(problems_of(other, ROOT))

    assert "placement must be" in problems(placement="auto")
    assert "effectiveness_service must name a GPU" in problems(effectiveness_service="cpu")
    assert "physical_gpus must name" in problems(physical_gpus={})
    pilot = read_config(ROOT / "configs/serl_audit_pilot.json")
    assert pilot["devices"]["placement"] == "sequential" and pilot["run_label"] == "e2e"

    # GRPO with the Effectiveness reward needs both models at once, on two different GPUs; checked before any model loads
    from src.common.main_config import training_device_problems
    from src.grpo.trainer import train_iteration
    weights = {"weights": {"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.3}}

    def training(**devices):
        return " | ".join(training_device_problems({"reward": weights, "devices": dict(config["devices"], **devices)}))

    assert "two different GPUs" in training()  # the main config: no second GPU has been assigned yet
    assert "two different GPUs" in training(physical_gpus={"policy": 7, "effectiveness_service": 7})
    assert training(physical_gpus={"policy": 6, "effectiveness_service": 7}) == ""
    assert "placement must be concurrent" in training(placement="sequential", physical_gpus={"policy": 6, "effectiveness_service": 7})
    assert training_device_problems({"reward": {"weights": {"outcome": 1, "appropriateness": 0, "effectiveness": 0}}, "devices": {}}) == []
    with pytest.raises(IntegrationUnavailable, match="cannot start"):
        train_iteration({"output_dir": "out", "devices": pilot["devices"], "reward": weights}, ROOT / "nowhere", 0)


def test_deterministic_components_repeat_exactly(tmp_path):
    """12: with the same inputs the deterministic parts give the same result: split, GRPO prompts, the
    Effectiveness input text, the reward arithmetic and the configuration hash."""
    from src.effectiveness.history import format_input
    from src.judges.reward import TurnSignals, compute_reward
    from src.training.reward_pipeline import load_reward_spec

    source = source_config(tmp_path)
    first, second = split_eligible_cases(source, tmp_path), split_eligible_cases(source, tmp_path)
    assert {c: v["split"] for c, v in first.items()} == {c: v["split"] for c, v in second.items()}
    histories = [{"history_id": f"{c}:A1", "case_id": c, "history": D1} for c in first]
    assert grpo_rows(first, histories, "s") == grpo_rows(second, list(reversed(histories)), "s")  # order does not matter
    assert format_input(HISTORY, "r") == format_input(json.loads(json.dumps(HISTORY)), "r")
    spec = load_reward_spec({"reward": {"weights": {"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.3},
                                        "softmin_temperature": 0.5, "effectiveness": {"transform": "identity"}}})
    turns = [TurnSignals(1, {"A": 2, "B": 1}, 0.25, True), TurnSignals(3, {"A": "NA", "B": 0}, None, False)]
    one, two = compute_reward(turns, 1.0, spec, {"A": 0.7, "B": 0.3}), compute_reward(turns, 1.0, spec, {"A": 0.7, "B": 0.3})
    assert one == two and one["components"]["effectiveness"] == 0.25  # A2 has no value: R_E is A1's delta alone
    assert one["turn_effectiveness"] == [0.25, None]


def test_audit_helpers(tmp_path):
    from scripts.audit_pipeline import contains, sha256, verify

    assert contains("Likely myasthenia gravis, given the antibodies.", "Myasthenia Gravis")
    assert not contains("The patient has hypertensive crisis.", "hypertension") and not contains("anything", "")
    kept, changed = tmp_path / "kept.txt", tmp_path / "changed.txt"
    kept.write_text("a"), changed.write_text("b")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"created": "t", "files": [
        {"path": str(kept), "sha256": sha256(kept)}, {"path": str(changed), "sha256": sha256(changed)},
        {"path": str(tmp_path / "gone.txt"), "sha256": "0"}, {"path": str(tmp_path / "running/x.pt"), "sha256": "0"}]}))
    changed.write_text("c")
    result = verify(manifest, running=("running/",))
    assert (result["unchanged"], len(result["changed"]), len(result["missing"]), len(result["of_a_run_in_progress"])) == (1, 1, 1, 1)
    assert result["preserved"] is False
    changed.write_text("b"), (tmp_path / "gone.txt").write_text("x")
    manifest.write_text(json.dumps({"created": "t", "files": [{"path": str(kept), "sha256": sha256(kept)}]}))
    assert verify(manifest)["preserved"] is True


def test_an_unreachable_api_gives_no_computed_reward_but_a_predictor_failure_stops_the_run(tmp_path):
    from test_grpo import GOOD, call, environment as environment_fixture, reward_for

    environment = environment_fixture.__wrapped__(tmp_path)
    sample = sorted(environment.histories)[0]
    doctor = environment.runtime.doctor

    def flaky_doctor(system, user):
        if "second" in user:
            raise ConnectionError("OpenRouter unreachable")
        return doctor(system, user)

    environment.runtime.doctor = flaky_doctor
    rewards = call(reward_for(environment, tmp_path), [sample, sample], [GOOD, GOOD.replace("supports MG.", "supports MG, second.")])
    rows = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["completed", "api_failed"] and "unreachable" in rows[1]["error"]
    assert rows[1]["reward_source"] == "group_mean" and rewards[1] == rewards[0]

    class BrokenPredictor:
        supported_turns, source = ("A1",), "grpc"

        def identity(self):
            return {"source": "grpc"}

        def predict(self, *args, **kwargs):
            raise IntegrationUnavailable("Effectiveness service failed (UNAVAILABLE)")

    environment.runtime.doctor = doctor
    with pytest.raises(IntegrationUnavailable, match="Effectiveness service failed"):  # never a default value
        call(reward_for(environment, tmp_path, provider=BrokenPredictor()), [sample], [GOOD])


def test_a_continued_run_keeps_the_configuration_it_was_created_with(tmp_path, capsys):
    """Config files are edited while a run is in progress (a role connected, a predictor added). The run continues with
    its own resolved configuration; only a changed prompt file stops it."""
    from src.common.main_config import open_run

    config = {"output_root": "serl", "run_label": "cond", "seed": 0, "reward": {"weights": {"outcome": 0.4}},
              "resolved_environment": {"models": {"doctor": "d"}}, "input_hashes": {"prompts/a.txt": "h1"}}
    created = open_run(config, tmp_path, purpose="test")
    edited = dict(config, resolved_environment={"models": {"doctor": "d"}, "label_roles": {"compliance_judge": "jev"}},
                  reward={"weights": {"outcome": 0.4}, "effectiveness": {"checkpoint": "new"}})
    continued = open_run(edited, tmp_path, run_id=created["run_id"], resume=True)
    assert continued["resolved_environment"] == config["resolved_environment"] and continued["reward"] == config["reward"]
    assert continued["run_id"] == created["run_id"] and "continues with its own resolved_config.json" in capsys.readouterr().err
    assert open_run(config, tmp_path, run_id=created["run_id"], resume=True)["reward"] == config["reward"]  # unchanged files: no notice
    assert capsys.readouterr().err == ""
    with pytest.raises(IntegrationUnavailable, match="prompt files changed"):
        open_run(dict(config, input_hashes={"prompts/a.txt": "h2"}), tmp_path, run_id=created["run_id"], resume=True)


def test_model_directories_can_be_mapped_on_another_machine(monkeypatch):
    from src.common.config import local_model_path

    monkeypatch.delenv("MODEL_PATH_MAP", raising=False)
    assert local_model_path("/mnt/shared/models/Qwen3-4B") == "/mnt/shared/models/Qwen3-4B"
    monkeypatch.setenv("MODEL_PATH_MAP", "/mnt/shared/models=/data/models")
    assert local_model_path("/mnt/shared/models/Qwen3-4B") == "/data/models/Qwen3-4B"
    assert local_model_path("/mnt/shared/models") == "/data/models"
    assert local_model_path("/mnt/shared/models_other/x") == "/mnt/shared/models_other/x"  # a prefix, not a substring
    assert local_model_path("Qwen/Qwen3-4B") == "Qwen/Qwen3-4B" and local_model_path(None) is None
