"""A2 Effectiveness labels and the turn-conditioned (A1 + A2) Effectiveness Model."""

from conftest import AGENTCLINIC, require_env_histories, require_local  # noqa: F401
import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_agentclinic_a2 as a2
from src.agentclinic.pipeline import a2_histories, build_labels, generate_candidates, prepare
from src.agentclinic.protocol import agent_prompt
from src.common.io import read_jsonl
from test_agentclinic import GOLD, cases, compliance_judge, diagnosis_judge, fake_runtime  # noqa: F401 (cases is a fixture)

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("evidence_grounded_maintain", "unsupported_maintain", "accept_revise", "request_evidence", "evidence_only_report")


def digests(directory: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}


@pytest.fixture
def labelled(cases, tmp_path):
    """A1 histories of four cases (two TRAIN, two DEV) in one directory, A2 labels built from them in another."""
    chosen = cases[:4]
    for index, case in enumerate(chosen):
        case["split"] = "train" if index < 2 else "dev"
    by_id = {c["case_id"]: c for c in chosen}
    # the Doctor reaches the right diagnosis only when a message told it the antibody result
    runtime, calls = fake_runtime(doctor_final=lambda user: GOLD if "antibody result" in user else "Viral illness")
    judge, _ = diagnosis_judge(tmp_path / "judge_cache.jsonl")
    source, out = tmp_path / "v5", tmp_path / "v6"
    prepare(chosen, runtime, judge, source, workers=4)
    before = digests(source)
    histories, skipped = a2_histories(read_jsonl(source / "histories.jsonl"))
    del calls[:]
    out.mkdir()
    made = generate_candidates(by_id, histories, runtime, compliance_judge(), out, workers=4, max_attempts=2, stage="A2", names=NAMES)
    built = build_labels(by_id, histories, runtime, judge, out, k=9, workers=4, require_compliance=True, stage="A2",
                         metadata={"version": "t", "k": 9, "split": "mixed", "continuation_protocol_hash": "h", "intervention_turn": "A2"})
    return {"cases": by_id, "histories": histories, "skipped": skipped, "out": out, "source": source, "source_digests": before,
            "calls": calls, "made": made, "built": built}


def test_a2_labels_are_k9_continuations_of_d1_a1_d2_through_final_only(labelled):
    out, histories = labelled["out"], labelled["histories"]
    assert len(histories) == 4 and labelled["skipped"] == [] and all(h["history_id"].endswith(":A2") for h in histories)
    first = histories[0]
    assert [m["stage"] for m in first["history"]] == ["D1", "A1", "D2"] and first["intervention_turn"] == "A2"
    assert first["agent_assessment_before"] == "Viral illness"  # what the Agent declared with its natural A1
    # the A1 run's directory was only read
    assert digests(labelled["source"]) == labelled["source_digests"]

    assert labelled["made"]["candidates"] == 20 and labelled["built"]["labels"] == 20 and labelled["built"]["incomplete_pairs"] == 0
    before, after = read_jsonl(out / "before_rollouts.jsonl"), read_jsonl(out / "after_rollouts.jsonl")
    assert len(before) == 4 * 9 and len(after) == 4 * 5 * 9
    for row in before + after:
        h2 = next(h for h in histories if h["history_id"] == row["history_id"])["history"]
        assert [m["stage"] for m in row["dialogue"]] == ["D1", "A1", "D2", "A2", "FINAL"]
        assert [m["text"] for m in row["dialogue"][:3]] == [m["text"] for m in h2]  # the history is never regenerated
    assert all(r["dialogue"][3]["text"] == "NATURAL reply." and "fixed" not in r["dialogue"][3] for r in before)
    assert all(r["dialogue"][3].get("fixed") is True for r in after)
    # candidates are generated from the A2 prompt, and a continuation asks the Doctor for the FINAL only (no new D2)
    agent_calls = [user for who, user in labelled["calls"] if who == "agent"]
    assert all("Respond to the Doctor's latest message" in user and "NATURAL reply." in user for user in agent_calls)
    doctor_calls = [user for who, user in labelled["calls"] if who == "doctor"]
    assert len(doctor_calls) == 4 * 6 * 9 and all("Give your final diagnosis" in user for user in doctor_calls)

    labels = read_jsonl(out / "labels.jsonl")
    for row in labels:
        assert row["intervention_turn"] == "A2" == row["history"]["intervention_turn"] and row["metadata"]["intervention_turn"] == "A2"
        assert row["before"] == {"successes": 0, "trials": 9} and row["after"]["trials"] == 9
        assert [t["speaker"] for t in row["history"]["turns"]] == ["doctor", "agent", "doctor"]
        assert row["delta_effectiveness"] == row["after_rate"] - row["before_rate"]
    informative = {r["candidate_id"]: r["delta_effectiveness"] for r in labels if r["history_id"] == histories[0]["history_id"]}
    assert informative == {"evidence_grounded_maintain": 1.0, "evidence_only_report": 1.0, "unsupported_maintain": 0.0,
                           "accept_revise": 0.0, "request_evidence": 0.0}


def test_a2_labels_carry_no_correct_diagnosis_and_keep_the_case_split(labelled):
    out, cases_by_id, histories = labelled["out"], labelled["cases"], labelled["histories"]
    audit = a2.leakage_audit(out, cases_by_id, histories)
    assert audit["passed"], audit["problems"]
    assert audit["cases_by_split"] == {"train": 2, "dev": 2} and audit["labels"] == 20
    labels = read_jsonl(out / "labels.jsonl")
    assert {r["split"] for r in labels} == {"train", "dev"}
    assert not {r["case_id"] for r in labels if r["split"] == "train"} & {r["case_id"] for r in labels if r["split"] == "dev"}
    for row in labels:  # predictor input: the Agent's view and the dialogue; no outcome, no evaluator field
        text = json.dumps(row["history"])
        assert "Correct_Diagnosis" not in text and "final_diagnosis" not in text and GOLD not in row["history"]["scenario"]
    for history in histories:
        prompt = agent_prompt(cases_by_id[history["case_id"]], history["history"], "A2")
        assert GOLD not in prompt.split("Conversation so far:")[0] and "SECRET-MANAGEMENT-PLAN" not in prompt
    # a label of a case moved to another split, or a leaked diagnosis, fails the audit
    rows = read_jsonl(out / "labels.jsonl")
    rows[0]["split"] = "dev"
    rows[1]["history"]["scenario"] += f" Diagnosis: {GOLD}"
    (out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    broken = a2.leakage_audit(out, cases_by_id, histories)
    assert not broken["passed"] and len(broken["problems"]) >= 2

    report = a2.sanity(out, histories, 9, list(NAMES))
    assert report["labelled_pairs"] == 20 and report["histories_with_candidate_ordering"] == 4
    assert report["before_floor_share"] == 1.0 and report["nonzero_delta_pairs"] == 8
    assert report["histories_where_the_final_diagnosis_depends_on_the_candidate"] == 4
    assert report["candidate_compliance"]["request_evidence"] == {"compliant": 4, "generated": 4}


# ---------------------------------------------------------------- the turn-conditioned predictor

HISTORY_A1 = {"scenario": "Agent view of the case", "question": "What is the most likely diagnosis?", "options": {},
              "turns": [{"speaker": "doctor", "text": "I think this is viral."}]}
HISTORY_A2 = {"intervention_turn": "A2", "scenario": "Agent view of the case", "question": "What is the most likely diagnosis?",
              "options": {}, "turns": [{"speaker": "doctor", "text": "I think this is viral."},
                                       {"speaker": "agent", "text": "The antibody result is elevated."},
                                       {"speaker": "doctor", "text": "I am not convinced."}]}


def pair(history_id: str, case: str, history: dict, response_id: str, before: int, after: int, split: str = "train") -> dict:
    row = {"case_id": case, "original_case_id": case, "history_id": history_id, "response_id": response_id, "split": split,
           "history": history, "response": f"response {response_id}", "before": {"successes": before, "trials": 9},
           "after": {"successes": after, "trials": 9}, "metadata": {"continuation_protocol_hash": "h"}}
    return dict(row, intervention_turn="A2") if history.get("intervention_turn") == "A2" else row


def test_turn_conditioned_input_names_the_turn_and_the_v1_input_is_unchanged():
    from src.effectiveness.history import format_input
    from src.effectiveness.pair_data import to_examples

    legacy = format_input(HISTORY_A1, "r")
    assert legacy.splitlines()[:2] == ["[STATE] after_response", "[CASE]"] and "Intervention turn" not in legacy
    assert format_input(HISTORY_A1, "r", "v1") == legacy  # Effectiveness v1 reads exactly what it was trained on
    first, second = format_input(HISTORY_A1, "r", "turn_v1"), format_input(HISTORY_A2, "r", "turn_v1")
    assert first.splitlines()[1] == "Intervention turn: A1" and second.splitlines()[1] == "Intervention turn: A2"
    assert first.replace("Intervention turn: A1\n", "") == legacy  # the turn line is the only difference
    assert format_input(HISTORY_A1, None, "turn_v1", "A2").splitlines()[:2] == ["[STATE] before_response", "Intervention turn: A2"]
    with pytest.raises(ValueError):
        format_input(HISTORY_A1, "r", "turn_v1", "D2")
    pairs = [pair("c1:A1", "c1", HISTORY_A1, "x", 3, 6), pair("c1:A2", "c1", HISTORY_A2, "x", 9, 0)]
    examples = to_examples(pairs, "turn_v1")
    assert [(e.kind, e.intervention_turn) for e in examples] == [("before", "A1"), ("after", "A1"), ("before", "A2"), ("after", "A2")]
    assert all(f"Intervention turn: {e.intervention_turn}" in e.text for e in examples)
    assert [e.text for e in to_examples(pairs[:1])] == [format_input(HISTORY_A1), format_input(HISTORY_A1, "response x")]


def test_turn_balanced_batches_mix_a1_and_a2_evenly():
    import random

    from src.effectiveness.pair_data import to_examples
    from src.effectiveness.train import turn_balanced_batches

    pairs = [pair(f"c{i}:A1", f"c{i}", HISTORY_A1, r, 3, 6) for i in range(10) for r in "xy"] + \
        [pair(f"c{i}:A2", f"c{i}", HISTORY_A2, "x", 9, 0) for i in range(4)]
    examples = to_examples(pairs, "turn_v1")  # A1: 30 examples, A2: 8
    order = list(range(len(examples)))
    batches = turn_balanced_batches(examples, order, 8, random.Random(0))
    turns = [[examples[i].intervention_turn for i in batch] for batch in batches]
    assert all(batch.count("A1") == batch.count("A2") for batch in turns[:-1]) and all(len(b) == 8 for b in batches[:-1])
    seen = [i for batch in batches for i in batch]
    assert {i for i in seen if examples[i].intervention_turn == "A1"} == {i for i in order if examples[i].intervention_turn == "A1"}
    assert sum(examples[i].intervention_turn == "A1" for i in seen) == 30 == sum(examples[i].intervention_turn == "A2" for i in seen)
    assert {i for i in seen if examples[i].intervention_turn == "A2"} == {i for i in order if examples[i].intervention_turn == "A2"}
    only_a1 = [i for i in order if examples[i].intervention_turn == "A1"]
    assert turn_balanced_batches(examples, only_a1, 8, random.Random(0))[0] == only_a1[:8]  # one turn: plain batches


@pytest.fixture
def joint_checkpoint(tmp_path):
    import torch

    from src.effectiveness import calibration
    from test_effectiveness_predictor import tiny_model

    torch.manual_seed(0)
    model = tiny_model()
    with torch.no_grad():
        model.value_head.weight.normal_(0, 0.5)
    model.save_pretrained(tmp_path / "v1", metadata={"continuation_protocol_hash": "hash-1", "k": 9})
    model.input_format, model.supported_turns = "turn_v1", ["A1", "A2"]
    model.save_pretrained(tmp_path / "joint", metadata={"continuation_protocol_hash": "hash-1", "k": 9})
    calibration.save({"method": "temperature", "a": 1 / 2.97, "b": 0.0, "temperature": 2.97}, tmp_path / "v1_calibration.json")
    (tmp_path / "by_turn.json").write_text(json.dumps({"method": "temperature_by_turn", "by_turn": {
        "A1": {"method": "temperature", "a": 0.5, "b": 0.0, "temperature": 2.0},
        "A2": {"method": "temperature", "a": 0.25, "b": 0.0, "temperature": 4.0}}}))
    return tmp_path


def test_one_checkpoint_serves_both_turns_with_the_calibration_of_each_turn(joint_checkpoint):
    pytest.importorskip("grpc")
    import math

    from src.common.config import IntegrationUnavailable
    from src.effectiveness.history import format_input
    from src.effectiveness.service import EffectivenessUnavailable, GrpcEffectivenessClient, LocalEffectiveness, start_server
    from src.effectiveness.synthetic import load_synthetic_backbone
    from types import SimpleNamespace

    settings = dict(device="cpu", allow_cpu=True, expected_protocol_hash="hash-1", backbone_loader=load_synthetic_backbone)
    joint = LocalEffectiveness.from_checkpoint(str(joint_checkpoint / "joint"), model_version="joint-test",
                                               calibration=str(joint_checkpoint / "by_turn.json"), supported_turns=("A1", "A2"), **settings)
    identity = joint.identity()
    assert identity["supported_turns"] == ["A1", "A2"] and identity["input_format"] == "turn_v1"
    assert identity["calibration_temperature_by_turn"] == {"A1": 2.0, "A2": 4.0}
    sigmoid = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    server, port = start_server(joint, port=0)
    try:
        client = GrpcEffectivenessClient(f"127.0.0.1:{port}", expected={"model_version": "joint-test", "supported_turns": ["A1", "A2"]})
        for history, turn, temperature in ((HISTORY_A1, "A1", 2.0), (HISTORY_A2, "A2", 4.0)):
            direct, served = joint.predict(history, "It is elevated.", turn), client.predict(history, "It is elevated.", turn)
            # offline = served, for each turn: the same logits, the turn's own temperature, the turn named in the input
            for name in ("p_before", "p_after", "delta", "logit_before", "logit_after", "calibration_temperature", "intervention_turn"):
                assert getattr(direct, name) == getattr(served, name), name
            assert served.intervention_turn == turn and served.calibration_temperature == temperature
            assert served.p_after == pytest.approx(sigmoid(served.logit_after / temperature))
            offline = joint.model.example_logits([SimpleNamespace(kind="after", history_id="h", text=format_input(
                history, "It is elevated.", "turn_v1", turn))], 1)[0]
            assert served.logit_after == pytest.approx(offline, abs=1e-6)
        # the turn is part of the input: the same history asked as A1 and as A2 is two different predictions
        assert joint.predict(HISTORY_A1, "It is elevated.", "A1").logit_after != joint.predict(HISTORY_A1, "It is elevated.", "A2").logit_after
        client.close()
    finally:
        server.stop(grace=1).wait()

    # Effectiveness v1 stays A1-only: it reads the v1 input, serves A1 exactly as before, and refuses A2
    v1 = LocalEffectiveness.from_checkpoint(str(joint_checkpoint / "v1"), model_version="v1-test", expected_temperature=2.97,
                                            calibration=str(joint_checkpoint / "v1_calibration.json"), supported_turns=("A1",), **settings)
    assert v1.identity()["input_format"] == "v1" and v1.identity()["supported_turns"] == ["A1"]
    reference = v1.model.example_logits([SimpleNamespace(kind="after", history_id="h", text=format_input(HISTORY_A1, "It is elevated."))], 1)[0]
    prediction = v1.predict(HISTORY_A1, "It is elevated.")
    assert prediction.logit_after == pytest.approx(reference, abs=1e-6) and prediction.calibration_temperature == 2.97
    with pytest.raises(EffectivenessUnavailable):
        v1.predict(HISTORY_A2, "It is elevated.", "A2")
    with pytest.raises(IntegrationUnavailable, match="trained for turns"):
        LocalEffectiveness.from_checkpoint(str(joint_checkpoint / "v1"), model_version="v1-test",
                                           calibration=str(joint_checkpoint / "v1_calibration.json"), supported_turns=("A1", "A2"), **settings)
    assert "input_format" not in json.loads((joint_checkpoint / "v1/predictor_config.json").read_text())  # the v1 checkpoint form is unchanged


def test_reward_with_a_predictor_for_both_turns_and_the_main_config_stays_a1_only(tmp_path):
    require_env_histories("configs/serl_agentclinic_max7.json")
    """What the reward does once a predictor covers A2: an A1 sample still gets R_E of A1 alone (turn-local credit, no
    soft-min across the turns) and an A2 sample gets R_E of its A2.
    """
    from src.common.config import read_config
    from src.common.main_config import problems_of
    from src.judges.reward import soft_min

    import test_grpo as g

    environment = g.environment.__wrapped__(tmp_path)
    both, second = g.with_a2(environment)
    sample, trained_a2 = sorted(h for h in both.histories if not h.endswith("/A2"))[0], second[0]["history_id"]

    class Provider(g.Provider):
        """P(success | h) = 0.2; a response that names the antibody 0.8, any other 0.1."""

    weights = dict(g.REWARD, weights={"outcome": 0, "appropriateness": 0, "effectiveness": 1}, effectiveness={"transform": "identity"})
    reward = g.reward_for(both, tmp_path, weights, provider=Provider(), effectiveness_stages=("A1", "A2"))
    a1_reward, a2_reward = g.call(reward, [sample, trained_a2], [g.GOOD, g.GOOD])
    rows = [json.loads(line) for line in (tmp_path / "reward_log.jsonl").read_text().splitlines()]
    # A1 sample: R_E is A1's own delta; the continuation's A2 (which says nothing new, delta -0.1) is not predicted or mixed in
    assert rows[0]["turn_effectiveness"] == pytest.approx([0.6]) and a1_reward == pytest.approx(0.6)
    assert set(rows[0]["effectiveness"]) == {"A1"} and a1_reward != pytest.approx(soft_min([0.6, -0.1], 0.5))
    # A2 sample: R_E is the trained A2's own value, with the history D1 -> A1 -> D2; A1 of the prompt gets none
    assert set(rows[1]["effectiveness"]) == {"A2"} and a2_reward == pytest.approx(0.6)
    request = rows[1]["effectiveness"]["A2"]["request"]
    assert request["history"]["intervention_turn"] == "A2" and [t["speaker"] for t in request["history"]["turns"]] == ["doctor", "agent", "doctor"]
    assert reward.identity()["effectiveness_coverage"] == "all scored turns"

    main = read_config(ROOT / "configs/serl_agentclinic.json")
    effectiveness = main["reward"]["effectiveness"]
    assert effectiveness["stages"] == ["A1"] and effectiveness["available"] == {"A1": True, "A2": False}
    assert effectiveness["model_version"] == "effectiveness-v1-lora-epoch3-T2.97" and effectiveness["checkpoint"].endswith("model_lora/v1_epoch3")
    assert problems_of(main, ROOT) == []
    main["reward"]["effectiveness"].update(stages=["A1", "A2"], available={"A1": True, "A2": True})
    assert any("A2 must be unavailable" in p for p in problems_of(main, ROOT))


def test_joint_evaluation_handles_a_turn_without_any_informative_pair():
    """The metrics and the gate inputs are defined (or explicitly undefined) when a turn's labels never differ
    within a history, as in the A2 smoke labels."""
    from scripts.train_joint import bootstrap_ordering, by_turn_metrics, calibrated, fit_calibration
    from src.effectiveness.pair_data import to_examples

    pairs = [pair(f"c{i}:A1", f"c{i}", HISTORY_A1, r, 3, after) for i in range(6) for r, after in (("x", 6), ("y", 1))] + \
        [pair(f"c{i}:A2", f"c{i}", HISTORY_A2, r, 9 * (i % 2), 9 * (i % 2)) for i in range(6) for r in "xy"]  # A2: delta 0 everywhere
    examples = to_examples(pairs, "turn_v1")
    logits = [1.5 if e.empirical_value > 0.5 else -1.5 for e in examples]  # a predictor that knows the success rates
    metrics = by_turn_metrics([1 / (1 + 2.718281828 ** -z) for z in logits], examples)
    assert metrics["A1"]["ordering"] == 1.0 and metrics["A1"]["ordering_pairs"] == 6 and metrics["A1"]["nonzero_delta_pairs"] == 12
    assert metrics["A2"]["ordering"] is None and metrics["A2"]["ordering_pairs"] == 0 and metrics["A2"]["nonzero_delta_pairs"] == 0
    assert metrics["A2"]["after_nll_gain_from_reading_the_response"] == pytest.approx(0.0)  # nothing to gain from the response
    assert metrics["A1"]["after_nll_gain_from_reading_the_response"] > 0
    a2 = [e for e in examples if e.intervention_turn == "A2"]
    boot = bootstrap_ordering([0.5] * len(a2), a2, 50, 0)
    assert boot["estimate"] is None and boot["ci95"] is None and boot["histories_with_an_informative_pair"] == 0
    a1 = [e for e in examples if e.intervention_turn == "A1"]
    assert bootstrap_ordering([0.9 if e.empirical_value > 0.5 else 0.1 for e in a1], a1, 50, 0)["ci95"] == [1.0, 1.0]
    # one temperature for all turns, or one per turn with the same weights
    single, by_turn = fit_calibration(logits, examples, by_turn=False), fit_calibration(logits, examples, by_turn=True)
    assert single["method"] == "temperature" and set(by_turn["by_turn"]) == {"A1", "A2"}
    assert len(calibrated(by_turn, logits, examples)) == len(examples)


def test_joint_training_script_runs_cv_final_and_one_dev_evaluation(tmp_path, monkeypatch):
    """scripts/train_joint.py from start to end on a tiny seeded backbone and toy labels: mixed A1/A2 training with
    turn-balanced batches, epoch and calibration chosen on out-of-fold TRAIN logits, one checkpoint for both turns ...
    """
    import random

    from scripts import train_joint
    from src.effectiveness import calibration
    from src.effectiveness.model import CollaborationValueModel
    from src.effectiveness.synthetic import load_synthetic_backbone
    from src.effectiveness.train import load_predictor_config
    from test_effectiveness_predictor import tiny_model

    smoke = json.loads((ROOT / "configs/effectiveness/predictor_smoke.json").read_text())
    smoke["continuation"]["k"] = 9
    smoke["predictor"].update(input_format="turn_v1", supported_turns=["A1", "A2"])
    smoke["training"].update(epochs=2, learning_rate=0.05, cache_features=False, turn_balanced=True, init_from="v1",
                             selection={"folds": 3, "seed": 0, "nll_tolerance": 0.02, "calibration_gain_for_turn_specific": 0.005, "rule": "r"})
    smoke["data"] = {f"{turn}_{split}": f"{turn}_{split}.jsonl" for turn in ("a1", "a2") for split in ("train", "dev")}
    smoke["baseline"] = {"v1_checkpoint": "v1", "v1_calibration": "v1_calibration.json", "v1_dev_logits": "v1_dev_logits.json"}
    smoke["gate"] = dict(json.loads((ROOT / "configs/effectiveness/v6_predictor_joint.json").read_text())["gate"], bootstrap=50)
    (tmp_path / "config.json").write_text(json.dumps(smoke))
    config = load_predictor_config(tmp_path / "config.json")
    protocol = config["continuation_spec"].protocol_hash()
    rng = random.Random(0)

    def rows(split: str, turn: str, cases: range) -> list[dict]:
        out = []
        for i in cases:
            history = dict(HISTORY_A2 if turn == "A2" else HISTORY_A1, scenario=f"case {split}{i} view")
            before = rng.randrange(10)
            for name in ("x", "y", "z"):
                row = pair(f"{split}{i}:{turn}", f"{split}{i}", history, name, before, rng.randrange(10), split)
                out.append(dict(row, metadata={"continuation_protocol_hash": protocol}))
        return out

    for split, cases in (("train", range(9)), ("dev", range(4))):
        for turn in ("A1", "A2"):
            (tmp_path / f"{turn.lower()}_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows(split, turn, cases)))
    # "Effectiveness v1": a tiny A1-only checkpoint, its calibration and its stored DEV logits
    v1 = tiny_model()
    v1.save_pretrained(tmp_path / "v1", metadata={"continuation_protocol_hash": protocol})
    calibration.save({"method": "temperature", "a": 0.5, "b": 0.0, "temperature": 2.0}, tmp_path / "v1_calibration.json")
    from src.effectiveness.pair_data import load_pairs, to_examples
    a1_dev = to_examples(load_pairs(tmp_path / "a1_dev.jsonl"))  # v1 reads the v1 input format
    (tmp_path / "v1_dev_logits.json").write_text(json.dumps({
        "examples": [{"kind": e.kind, "history_id": e.history_id, "response_id": e.response_id} for e in a1_dev],
        "checkpoints": {"epoch3": {"logits": v1.example_logits(a1_dev, 8)}}}))
    v1_files = digests(tmp_path / "v1")
    monkeypatch.setattr(train_joint, "ROOT", tmp_path)
    out = tmp_path / "joint"
    out.mkdir()

    summary = train_joint.cross_validate(config, out)
    assert summary["dev_used"] is False and summary["chosen_epoch"] in (1, 2) and summary["train_examples"] == {"A1": 36, "A2": 36}
    assert len(summary["candidates"]) == 2 and (out / "calibration.json").is_file() and not (out / "cv/fold0").exists()
    parameters = calibration.load(out / "calibration.json")
    assert parameters["method"] in ("temperature", "temperature_by_turn") and "cross_fitted_nll" in parameters
    final = train_joint.final(config, out)
    checkpoint = json.loads((out / "joint_v2/predictor_config.json").read_text())
    assert checkpoint["input_format"] == "turn_v1" and checkpoint["supported_turns"] == ["A1", "A2"]  # one model for both turns
    assert final["epochs"] == summary["chosen_epoch"] and (out / "joint_v2/calibration.json").is_file()
    assert digests(tmp_path / "v1") == v1_files  # the warm start read v1 and did not touch it
    model = CollaborationValueModel.from_pretrained(out / "joint_v2", backbone_loader=load_synthetic_backbone)
    assert model.input_format == "turn_v1" and model.supported_turns == ["A1", "A2"]
    with pytest.raises(SystemExit):
        train_joint.final(config, out)  # the final checkpoint is written once

    report = train_joint.evaluate(config, out)
    assert set(report["joint"]) == {"A1", "A2", "pooled"} and report["v1_on_A1"]["histories"] == 4 == report["joint"]["A1"]["histories"]
    assert set(report["gate"]["checks"]) == {
        "a2_ordering_ci_lower_bound_above_0.5", "a2_after_nll_below_history_only",
        "a2_delta_sign_accuracy_above_0.5_on_enough_nonzero_pairs", "a1_ordering_within_margin_of_v1", "a1_after_nll_within_margin_of_v1"}
    assert report["gate"]["usable_for_reward"] == (report["gate"]["a2_response_sensitive"] and report["gate"]["a1_preserved"])
    for name in ("dev_logits.json", "dev_report.json", "dev_report.md", "gate.json", "cv_summary.md", "final_summary.json"):
        assert (out / name).is_file(), name
    json.loads((out / "dev_report.json").read_text())  # valid JSON: undefined metrics are null, not NaN
    with pytest.raises(SystemExit, match="read once"):
        train_joint.evaluate(config, out)  # DEV is read once
