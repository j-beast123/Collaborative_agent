"""Offline guards for the reusable full-trajectory GRPO smoke path."""

import json

import pytest

from scripts import run_medmcqa_full_grpo_smoke as grpo
from scripts.run_medmcqa_full_grpo_smoke import load_saved_records
from src.judges.rubric_scoring import TWO_STAGE_JEV_PROTOCOL


def saved_record(**updates):
    row = {
        "trajectory_id": "case-1:rollout_0",
        "case_id": "case-1",
        "rollout_index": 0,
        "ground_truth": {"correct_option": "A"},
        "final_option": "A",
        "rubric_set_version": "version-1",
        "rubric_set_content_sha256": "content-1",
        "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        "turn_scores": [{
            "turn_id": 1,
            "labels": {"r1": 2},
            "normalized_labels": {"r1": 2},
            "raw_rubric_adherence": 1.0,
        }],
        "agent_traces": [{
            "stage": "A1",
            "prompt_input_ids": [1, 2],
            "completion_ids": [3],
            "completion_logprobs": [-0.5],
            "sampling_temperature": 0.7,
            "sampling_top_p": 0.9,
            "sampling_top_k": 20,
        }],
    }
    row.update(updates)
    return row


def load(path, **overrides):
    kwargs = {
        "rubric_weight": 0.3,
        "outcome_weight": 0.4,
        "rubric_temperature": 0.5,
        "expected_rollouts": 1,
        "sampling_temperature": 0.7,
        "sampling_top_p": 0.9,
        "sampling_top_k": 20,
        "rubric_ids": {"r1"},
        "expected_rubric_set_version": "version-1",
        "expected_rubric_content_sha256": "content-1",
        "expected_parent_rubric_set_version": None,
        "expected_parent_rubric_content_sha256": None,
        "allow_unversioned_rubric_scores": False,
        "allow_legacy_sampling_metadata": False,
        "expected_scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        "allow_legacy_rubric_scoring_protocol": False,
    }
    kwargs.update(overrides)
    return load_saved_records(path, **kwargs)


def write_row(tmp_path, row):
    path = tmp_path / "trajectories.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_saved_grpo_record_requires_matching_rubric_and_protocol(tmp_path):
    path = write_row(tmp_path, saved_record())
    record, = load(path)
    assert record["reward"] == pytest.approx(0.7)
    assert record["conversation_rubric_adherence"] == pytest.approx(1.0)

    path = write_row(tmp_path, saved_record(rubric_set_content_sha256="other"))
    with pytest.raises(ValueError, match="content hash mismatch"):
        load(path)

    row = saved_record()
    row.pop("rubric_set_content_sha256")
    path = write_row(tmp_path, row)
    with pytest.raises(ValueError, match="no rubric content hash"):
        load(path)

    path = write_row(tmp_path, saved_record(scoring_protocol="single_call_legacy"))
    with pytest.raises(ValueError, match="scoring protocol mismatch"):
        load(path)

    path = write_row(tmp_path, saved_record(
        rubric_set_version="parent-version",
        rubric_set_content_sha256="parent-content",
    ))
    record, = load(
        path,
        expected_rubric_set_version="retained-version",
        expected_rubric_content_sha256="retained-content",
        expected_parent_rubric_set_version="parent-version",
        expected_parent_rubric_content_sha256="parent-content",
    )
    assert record["rubric_ids_used"] == ["r1"]


def test_saved_grpo_record_requires_exact_sampling_metadata(tmp_path):
    row = saved_record()
    row["agent_traces"][0].pop("sampling_top_k")
    path = write_row(tmp_path, row)
    with pytest.raises(ValueError, match="exact sampling metadata"):
        load(path)

    record, = load(path, allow_legacy_sampling_metadata=True)
    assert record["agent_traces"][0]["sampling_top_k"] == 20


def test_grpo_objective_weights_trajectories_not_number_of_turns(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1)

    def sequence(model, prompt_ids, completion_ids, **kwargs):
        return torch.zeros(len(completion_ids), requires_grad=kwargs.get("grad", False))

    monkeypatch.setattr(grpo, "sequence_logprobs", sequence)
    monkeypatch.setattr(
        grpo, "reference_logprobs",
        lambda model, prompt_ids, completion_ids, **kwargs: torch.zeros(len(completion_ids)),
    )

    def trace(stage):
        return {
            "stage": stage,
            "prompt_input_ids": [1],
            "completion_ids": [2],
            "completion_logprobs": [0.0],
            "sampling_temperature": 1.0,
            "sampling_top_p": 1.0,
            "sampling_top_k": 0,
        }

    records = [
        {"case_id": "c", "trajectory_id": "low", "reward": 0.0, "agent_traces": [trace("A1")]},
        {"case_id": "c", "trajectory_id": "high", "reward": 1.0,
         "agent_traces": [trace("A1"), trace("A2"), trace("A3")]},
    ]
    total, policy, kl, used, stats = grpo.compute_grpo_loss(
        model, records, beta=0.04, clip_epsilon=0.2,
        temperature=1.0, top_p=1.0, top_k=0, backward=False,
    )
    assert float(policy) == pytest.approx(0.0, abs=1e-7)
    assert float(kl) == pytest.approx(0.0, abs=1e-7)
    assert float(total) == pytest.approx(0.0, abs=1e-7)
    assert used == 4 and stats["eligible_trajectories"] == 2
    assert stats["objective_weighting"] == "mean_agent_tokens_within_trajectory_then_mean_trajectories"

    records[1]["reward"] = None
    with pytest.raises(RuntimeError, match="at least two trajectories"):
        grpo.compute_grpo_loss(
            model, records, beta=0.04, clip_epsilon=0.2,
            temperature=1.0, top_p=1.0, top_k=0, backward=False,
        )
