import copy

import pytest

from src.online_rl.integrated import grpo


def test_alignment_preserves_recorded_sampling_evidence(monkeypatch):
    trace = {'behavior_logprobs': [-1.0, -2.0], 'old_policy_logprobs': [-1.0, -2.0],
             'sampling_logprobs': [-1.0, -2.0], 'completion_ids': [1, 2]}
    records = [{'trajectory_id': 'one', 'agent_traces': [trace]}]
    original = copy.deepcopy(records)
    monkeypatch.setattr(grpo, 'behavior_logprobs', lambda *args: [[-0.9, -1.8]])
    aligned, audit = grpo.teacher_forced_behavior_records(None, records, 16)
    assert records == original
    assert aligned[0]['agent_traces'][0]['behavior_logprobs'] == [-0.9, -1.8]
    assert aligned[0]['agent_traces'][0]['old_policy_logprobs'] == [-1.0, -2.0]
    assert audit['sampling_evidence_preserved']
    assert audit['tokens'] == 2
    assert audit['max_abs_logprob_gap'] == pytest.approx(0.2)
    monkeypatch.setattr(grpo, 'behavior_logprobs', lambda *args: [[-0.9]])
    with pytest.raises(RuntimeError, match='alignment changed'):
        grpo.teacher_forced_behavior_records(None, records, 16)
