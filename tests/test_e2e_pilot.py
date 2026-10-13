"""The end-to-end pipeline check itself (scripts/run_e2e_pilot.py), run with fake models on the real cases."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentclinic.outcome import DiagnosisJudge
from src.agentclinic.protocol import PROVISIONAL, ClinicRuntime
from src.agentclinic.runtime import Clinic
from src.agentclinic.selection import split_eligible_cases
from src.common.config import IntegrationUnavailable, read_config
from src.common.io import read_jsonl
from src.effectiveness import calibration
from src.effectiveness.history import format_input
from src.effectiveness.service import LocalEffectiveness, start_server
from src.effectiveness.synthetic import load_synthetic_backbone
from src.judges.llm import CachedChat, ScriptedBackend
from src.judges.rubric_scoring import RubricScorer, ScoreResult
from test_effectiveness_predictor import tiny_model
from test_grpo import STATE, is_state_call

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "outputs/effectiveness/v5_agentclinic_full"
pytestmark = pytest.mark.skipif(not (LABELS / "histories.jsonl").is_file(), reason="needs the label run's files")


class FixtureScorer(RubricScorer):
    """The first rubric applies at A1 only (score 2); the second never applies. A2 therefore has no applicable rubric."""
    name = "fake"

    def score(self, item):
        first, second = (r["rubric_id"] for r in item.rubrics)
        return ScoreResult(item.item_id, self.name, "scored", {first: 2 if item.turn_id == 1 else "NA", second: "NA"})


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A repository root in tmp: configs, prompts and data are the real ones, the label run's files are linked
    read-only, the Effectiveness checkpoint is a tiny synthetic one, and outputs/serl is new."""
    pytest.importorskip("grpc")
    import torch

    from scripts import run_e2e_pilot as e2e

    root = tmp_path / "repo"
    labels = root / "outputs/effectiveness/v5_agentclinic_full"
    (labels / "model_lora").mkdir(parents=True)
    for name in ("configs", "prompts", "data"):
        (root / name).symlink_to(ROOT / name)
    for name in ("histories.jsonl", "prepare_diagnostics.jsonl", "labels_dev.jsonl", "split_manifest.json"):
        (labels / name).symlink_to(LABELS / name)
    main = read_config(ROOT / "configs/serl_audit_pilot.json")
    torch.manual_seed(0)
    model = tiny_model()
    with torch.no_grad():
        model.value_head.weight.normal_(0, 0.5)
    model.save_pretrained(root / "ckpt", metadata={"epoch": 3, "continuation_protocol_hash":
                                                   main["reward"]["effectiveness"]["expected_continuation_protocol_hash"]})
    calibration.save({"method": "temperature", "a": 1 / 2.97, "b": 0.0, "temperature": 2.97}, root / "calibration.json")
    # the "offline evaluation" of the synthetic checkpoint on DEV: logits computed in batches, as evaluate_predictor_epochs does
    examples, seen = [], set()
    for row in read_jsonl(LABELS / "labels_dev.jsonl")[:40]:
        if row["history_id"] not in seen:
            seen.add(row["history_id"])
            examples.append(SimpleNamespace(kind="before", history_id=row["history_id"], response_id=None, text=format_input(row["history"])))
        examples.append(SimpleNamespace(kind="after", history_id=row["history_id"], response_id=row["response_id"],
                                        text=format_input(row["history"], row["response"])))
    (labels / "model_lora/dev_logits.json").write_text(json.dumps({
        "examples": [{"kind": e.kind, "history_id": e.history_id, "response_id": e.response_id} for e in examples],
        "checkpoints": {"epoch3": {"logits": model.example_logits(examples, 8)}}}))

    predictor = LocalEffectiveness.from_checkpoint(
        str(root / "ckpt"), model_version=main["reward"]["effectiveness"]["model_version"], calibration=str(root / "calibration.json"),
        device="cpu", allow_cpu=True, supported_turns=("A1",), expected_temperature=2.97, backbone_loader=load_synthetic_backbone)
    server, port = start_server(predictor, port=0, log_path=root / "service_requests.jsonl")
    (root / "check.json").write_text(json.dumps({
        "extends": "configs/serl_audit_pilot.json",
        "reward": {"effectiveness": {"checkpoint": "ckpt", "calibration": "calibration.json", "endpoint": f"127.0.0.1:{port}",
                                     "parity_tolerance": 1e-4, "parity_logit_tolerance": 1e-3}},
        "pilot": {"parity_histories": 2, "service_request_log": str(root / "service_requests.jsonl")}}))

    cases = split_eligible_cases(read_config(ROOT / "configs/effectiveness/v5_agentclinic_full.json"), ROOT)
    calls = []

    def doctor(system, user):
        if "Give your final diagnosis" in user:
            return json.dumps({"final_diagnosis": "Sarcoidosis", "brief_reason": "r"})
        return f"I will keep my view for now.\n{PROVISIONAL} Sarcoidosis"

    def agent(system, user):
        calls.append(user)
        if is_state_call(user):
            return json.dumps(STATE)
        return "Current assessment: Sarcoidosis\nResponse: The findings fit your diagnosis; I would confirm it."

    def clinic(config, root_, model_, tokenizer_, **_):
        runtime = ClinicRuntime(doctor=doctor, agent=agent, agent_greedy=agent, doctor_system="doctor system",
                                agent_system=(ROOT / config["resolved_environment"]["prompts"]["consultant"]).read_text(),
                                state_system="state system")
        judge = DiagnosisJudge(CachedChat(ScriptedBackend(lambda r: json.dumps({"equivalent": False, "reason": "x"})), None), "j")
        return Clinic(cases, runtime, judge, nullcontext, config["resolved_environment"], lambda: None)

    monkeypatch.setattr(e2e, "ROOT", root)
    monkeypatch.setattr(e2e, "load_policy", lambda *a, **k: (None, None))
    monkeypatch.setattr(e2e, "build_clinic", clinic)
    monkeypatch.setattr(e2e, "build_scorer", lambda config, root_: FixtureScorer())
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")
    yield e2e, root, calls
    server.stop(grace=1).wait()


def weights_of(root) -> dict:
    return read_config(root / "check.json")["reward"]["weights"]


def stage(e2e, root, monkeypatch, *arguments):
    monkeypatch.setattr("sys.argv", ["run_e2e_pilot", *arguments, "--config", str(root / "check.json")])
    try:
        e2e.main()
    except SystemExit as stop:
        return stop.code
    except IntegrationUnavailable as error:  # a refusal (e.g. a run that already has results)
        return f"refused: {error}"
    return 0


def test_one_case_then_two_cases_pass_every_check(workspace, monkeypatch, capsys):
    e2e, root, _ = workspace
    assert stage(e2e, root, monkeypatch, "rollout", "--cases", "1", "--write-run-id", str(root / "id")) == 0
    run_id = (root / "id").read_text()
    run = root / "outputs/serl" / run_id
    assert run_id.startswith("e2e_") and not (run / "rewards.jsonl").exists()  # the rollout stage computes no reward
    assert stage(e2e, root, monkeypatch, "parity", "--run-id", run_id) == 0
    code = stage(e2e, root, monkeypatch, "reward", "--run-id", run_id)
    checks = json.loads((run / "checks.json").read_text())
    assert checks["passed"] and code == 0, checks["failed"]
    (sample, per_case), (second_sample, second_checks) = checks["per_case"].items()  # the A1 sample and the A2 sample of the case
    assert len(per_case) >= 30 and all(per_case.values()) and all(checks["run"].values())
    assert second_sample == f"{sample}/A2" and len(second_checks) >= 20 and all(second_checks.values())

    row, trained_a2 = read_jsonl(run / "rewards.jsonl")
    assert (row["stage"], trained_a2["stage"]) == ("A1", "A2")
    # A2 sample: the prompt is D1 -> A1 -> D2 of the base collaboration, only A2 is scored, and it has no R_E at all
    frozen = next(h for h in read_jsonl(LABELS / "histories.jsonl") if h["history_id"] == sample)
    assert [m["text"] for m in trained_a2["dialogue"][:3]] == [m["text"] for m in frozen["natural"]["dialogue"][:3]]
    assert set(trained_a2["rubric_scores"]) == {"A2"} and set(trained_a2["effectiveness"]) == {"A2"}
    assert trained_a2["components"]["effectiveness"]["available"] is False and trained_a2["r_effectiveness"] is None
    # no rubric applies to it and it is alone in its group: R_A is unavailable (not 0), the outcome carries the reward
    assert trained_a2["no_applicable"] is True and trained_a2["raw_R_A"] is None and trained_a2["imputed_R_A"] is None
    assert trained_a2["group_no_applicable_rate"] == 1.0 and trained_a2["weights_used"] == {"outcome": 0.4}  # not rescaled
    assert sorted(trained_a2["unavailable_components"]) == ["appropriateness", "effectiveness"] and trained_a2["reward"] == 0.0
    assert trained_a2["raw_R_A_by_turn"] == {"A2": None} and trained_a2["group_turn_missing"] == {"A2": True}
    assert trained_a2["raw_composite"] == trained_a2["reward_passed_to_trl"] == 0.0
    # A1 sample, turn-local credit: only A1 is scored and predicted; the continuation's A2 is not part of its reward
    assert row["raw_R_A_by_turn"] == {"A1": 1.0} and row["group_turn_missing"] == {"A1": False} and row["scored_stages"] == ["A1"]
    assert set(row["rubric_scores"]) == {"A1"} == set(row["effectiveness"]) and [m["stage"] for m in row["dialogue"]][3] == "A2"
    assert row["final_R_A"] == 1.0 and row["weights_used"] == weights_of(root)
    meta = row["components"]["appropriateness"]["metadata"]
    assert meta["aggregation"] == "weighted_softmin" and meta["rubric_temperature"] == 0.5 and meta["credit_assignment"] == "turn_local"
    parts = row["components"]
    a1 = row["effectiveness"]["A1"]
    assert a1["available"] is True and a1["source"] == "grpc" and a1["calibration_temperature"] == 2.97
    assert trained_a2["effectiveness"]["A2"] == {"available": False, "reason": "no Effectiveness predictor for A2"}
    assert parts["effectiveness"]["value"] == pytest.approx(a1["delta"]) and row["turn_effectiveness"] == [pytest.approx(a1["delta"])]
    per_turn = parts["appropriateness"]["metadata"]["per_turn"]
    assert set(per_turn) == {"A1"} and per_turn["A1"]["value"] == 1.0
    assert parts["appropriateness"]["value"] == 1.0 and parts["outcome"]["value"] == 0.0
    weights = read_config(root / "check.json")["reward"]["weights"]
    assert row["reward"] == pytest.approx(weights["appropriateness"] * 1.0 + weights["effectiveness"] * a1["delta"])
    parity = json.loads((run / "evaluation/effectiveness_parity.json").read_text())
    assert parity["passed"] and parity["max_abs_diff"]["delta"] < 1e-4 and parity["pairs"] >= 2
    for name in ("resolved_config.json", "manifest.json", "environment.json", "prompts.jsonl", "states.jsonl", "trajectories.jsonl",
                 "rubric_scores.jsonl", "effectiveness.jsonl", "run_summary.md", "api_cost.json", "grpc_logs/effectiveness_requests.jsonl"):
        assert (run / name).is_file(), name
    assert json.loads((run / "cases.json").read_text())[0]["split"] == "train"

    # a finished run is never written again, and the 2-case pilot needs a passed 1-case run
    assert stage(e2e, root, monkeypatch, "reward", "--run-id", run_id) != 0
    assert stage(e2e, root, monkeypatch, "rollout", "--cases", "2") != 0
    assert stage(e2e, root, monkeypatch, "rollout", "--cases", "2", "--prerequisite-run", run_id, "--write-run-id", str(root / "id2")) == 0
    second = (root / "id2").read_text()
    assert second != run_id
    assert stage(e2e, root, monkeypatch, "parity", "--run-id", second) == 0 and stage(e2e, root, monkeypatch, "reward", "--run-id", second) == 0
    pilot = json.loads((root / "outputs/serl" / second / "checks.json").read_text())
    assert pilot["passed"], pilot["failed"]
    states = {c["collaboration_state_of_the_base_agent"] for c in json.loads((root / "outputs/serl" / second / "cases.json").read_text())}
    assert states == {"Challenge", "Preserve"} and len(pilot["per_case"]) == 4  # an A1 and an A2 sample per case


def test_the_check_fails_when_a_property_is_broken(workspace, monkeypatch):
    """The verification is not a formality: a state that reaches the response prompt, or a leaked diagnosis, fails it."""
    e2e, root, calls = workspace
    assert stage(e2e, root, monkeypatch, "rollout", "--cases", "1", "--write-run-id", str(root / "id")) == 0
    run = root / "outputs/serl" / (root / "id").read_text()
    assert stage(e2e, root, monkeypatch, "parity", "--run-id", run.name) == 0
    prompts = read_jsonl(run / "prompts.jsonl")
    case = split_eligible_cases(read_config(ROOT / "configs/effectiveness/v5_agentclinic_full.json"), ROOT)[prompts[0]["case_id"]]
    for p in prompts:  # tamper with the record: the A2 prompt now carries the elicited state, a Doctor prompt the diagnosis
        if p["role"] == "agent":
            p["user"] += "\n" + json.dumps(STATE)
        if p["role"] == "doctor":
            p["user"] = p["user"].replace("Conversation so far:", f"Confirmed: {case['gold']}\n\nConversation so far:", 1)
    (run / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
    assert stage(e2e, root, monkeypatch, "reward", "--run-id", run.name) == 1
    failed = " ".join(json.loads((run / "checks.json").read_text())["failed"])
    assert "state_output_absent_from_response_and_doctor_prompts" in failed and "state_fields_absent_from_response_prompts" in failed
    assert "correct_diagnosis_in_no_information_section" in failed and "a2_prompt_is_the_label_pipeline_prompt" in failed
    assert "/A2: correct_diagnosis_in_no_information_section" in failed  # the A2 sample's Doctor prompt was tampered with too
