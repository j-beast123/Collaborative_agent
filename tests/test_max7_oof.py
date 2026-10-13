"""Guards and real tiny-model CV for the new max7 workflow; no API/data/GPU access."""
import json
from pathlib import Path

import pytest

from scripts import train_max7 as workflow
from src.agentclinic.adapter import load_cases
from src.effectiveness import calibration
from src.effectiveness.pair_data import to_examples
from src.effectiveness.synthetic import synthetic_pair_rows


def test_case_filter_does_not_deserialize_excluded_record(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"OSCE_Examination": {"Correct_Diagnosis": "allowed"}}) + "\nTHIS EXCLUDED RECORD IS NOT JSON\n")
    rows = load_cases(path, case_ids={"agentclinic_medqa_extended_0000"})
    assert len(rows) == 1 and rows[0]["gold"] == "allowed"


def test_identity_calibration_roundtrip_and_reject_changed_coefficients(tmp_path):
    parameters = calibration.fit_logits([], [], [], "identity")
    calibration.save(parameters, tmp_path / "calibration.json")
    assert calibration.apply_logits(calibration.load(tmp_path / "calibration.json"), [-2, 0, 2]) == pytest.approx(
        [0.119202922, 0.5, 0.880797078])
    parameters["temperature"] = 2.9657
    calibration.save(parameters, tmp_path / "calibration.json")
    with pytest.raises(ValueError, match="identity"):
        calibration.load(tmp_path / "calibration.json")


def test_real_tiny_case_cv_final_and_frozen_label_guard(tmp_path, monkeypatch):
    c = json.loads((workflow.ROOT / "configs/effectiveness/predictor_smoke.json").read_text())
    c["predictor"].update(input_format="turn_v1", supported_turns=list(workflow.TURNS))
    c["training"].update(epochs=2, init_from=None, turn_balanced=False, cache_features=False,
                          selection={"folds": 3, "seed": 0})
    c["protocol_hash"] = "synthetic-max7"
    workflow.write(tmp_path / "training.frozen.json", c)
    pairs = []
    for turn in ("A1", "A7"):
        for row in synthetic_pair_rows(9, 9, "synthetic-max7", 0, "case"):
            row["history_id"] += ":" + turn
            row["intervention_turn"] = turn
            row["history"]["intervention_turn"] = turn
            pairs.append(row)
    rows = to_examples(pairs, "turn_v1")
    read_splits = []

    def only_train(config, split):
        read_splits.append(split)
        assert split == "train", "DEV was read during model selection"
        return rows

    monkeypatch.setattr(workflow, "examples", only_train)
    labels = tmp_path / "labels/labels_train.jsonl"
    labels.parent.mkdir()
    labels.write_text("immutable synthetic labels\n")
    workflow.write(tmp_path / "labels.freeze.json", {"files": {labels.name: {"sha256": workflow.sha(labels)}}})
    workflow.cv(tmp_path)
    summary = json.loads((tmp_path / "cv_summary.json").read_text())
    assert summary["dev_used"] is False and read_splits == ["train"]
    oof = json.loads((tmp_path / "oof_logits.json").read_text())
    by_case = {}
    for row in oof:
        by_case.setdefault(row["case_id"], set()).add(row["fold"])
    assert all(len(folds) == 1 for folds in by_case.values())
    assert len(oof) == len(rows) and {r["intervention_turn"] for r in oof} == {"A1", "A7"}
    parameters = calibration.load(tmp_path / "calibration.json")
    assert parameters["method"] in ("temperature", "identity")
    assert parameters["chosen_epoch"] == summary["chosen_epoch"]
    workflow.final(tmp_path)
    checkpoint = tmp_path / "joint_max7/trainable_state.pt"
    checksum = workflow.sha(checkpoint)
    workflow.final(tmp_path)
    assert workflow.sha(checkpoint) == checksum
    labels.write_text("changed labels\n")
    with pytest.raises(ValueError, match="frozen labels changed"):
        workflow.cv(tmp_path)


def test_cv_folds_fitted_one_by_one_give_the_selection_of_the_single_cv_run(tmp_path, monkeypatch):
    """--fold: every fold is its own run (own directory, own logits); the selection happens only when all are there."""
    c = json.loads((workflow.ROOT / "configs/effectiveness/predictor_smoke.json").read_text())
    c["predictor"].update(input_format="turn_v1", supported_turns=list(workflow.TURNS))
    c["training"].update(epochs=2, init_from=None, turn_balanced=False, cache_features=False, selection={"folds": 3, "seed": 0})
    c["protocol_hash"] = "synthetic-max7"
    pairs = []
    for row in synthetic_pair_rows(9, 9, "synthetic-max7", 0, "case"):
        row["history_id"] += ":A1"
        row["intervention_turn"] = row["history"]["intervention_turn"] = "A1"
        pairs.append(row)
    rows = to_examples(pairs, "turn_v1")
    monkeypatch.setattr(workflow, "examples", lambda config, split: rows)
    results = {}
    for name in ("single", "by_fold"):
        out = tmp_path / name
        workflow.write(out / "training.frozen.json", c)
        (out / "labels").mkdir()
        (out / "labels/labels_train.jsonl").write_text("immutable synthetic labels\n")
        workflow.write(out / "labels.freeze.json", {"files": {"labels_train.jsonl": {"sha256": workflow.sha(out / "labels/labels_train.jsonl")}}})
        if name == "by_fold":
            for fold in (2, 0, 1):
                workflow.cv(out, only_fold=fold)
                assert (out / "cv" / f"fold{fold}_logits.json").is_file()
                assert not (out / "cv_summary.json").exists() and not (out / "calibration.json").exists()  # nothing is selected yet
            fitted = {fold: (out / "cv" / f"fold{fold}_logits.json").read_text() for fold in range(3)}
        workflow.cv(out)
        if name == "by_fold":
            assert fitted == {fold: (out / "cv" / f"fold{fold}_logits.json").read_text() for fold in range(3)}  # no fold was fitted again
        results[name] = (json.loads((out / "oof_logits.json").read_text()), json.loads((out / "cv_summary.json").read_text())["chosen_epoch"])
    single, by_fold = results["single"], results["by_fold"]
    assert [(r["history_id"], r["response_id"], r["kind"], r["fold"]) for r in single[0]] == [(r["history_id"], r["response_id"], r["kind"], r["fold"]) for r in by_fold[0]]
    assert all(abs(a["raw_logit"] - b["raw_logit"]) < 1e-4 for a, b in zip(single[0], by_fold[0])) and single[1] == by_fold[1]
