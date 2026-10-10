"""Regression checks for active reward dependencies and auditable sibling credit."""
from types import SimpleNamespace

import pytest

from src.online_rl.integrated.metrics import advantages, reward
from src.online_rl.tree.team import compose_reward, effectiveness_weighted_r_a


@pytest.mark.parametrize('ra', [0.0, 0.5, 1.0])
def test_rubric_only_does_not_require_doctor_outcome(ra):
    assert reward(None, ra, 0.0, 1.0, True) == ra
    assert compose_reward(None, ra, 0.0, 1.0) == ra
    assert reward(None, None, 0.0, 1.0, False) is None
    assert compose_reward(None, None, 0.0, 1.0) is None


@pytest.mark.parametrize('valid', [True, False])
def test_outcome_only_does_not_require_rubric(valid):
    assert reward(0.75, None, 1.0, 0.0, valid) == 0.75
    assert compose_reward(0.75, None, 1.0, 0.0) == 0.75
    assert reward(None, 1.0, 1.0, 0.0, valid) is None


def test_mixed_reward_keeps_na_fallback_and_zero_separate():
    assert compose_reward(0.75, None, 0.5, 0.5) == 0.375
    assert compose_reward(0.75, 0.0, 0.5, 0.5) == 0.375
    assert compose_reward(None, 1.0, 0.5, 0.5) is None
    assert reward(1.0, 1.0, 0.0, 1.0, False) is None
    with pytest.raises(RuntimeError, match='unscored'):
        reward(1.0, None, 0.5, 0.5, True)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_rewards_fail_before_optimizer(bad):
    with pytest.raises(ValueError, match='finite'):
        advantages([None, bad], 'mean_center')


def assessment(score, status='scored', match=True):
    return SimpleNamespace(response_node_id='n', rubric_id='r', score=score,
                           scoring_status=status, match=match)


def test_explicit_unscorable_is_omitted_but_missing_scored_value_is_error():
    assert effectiveness_weighted_r_a([assessment(None, 'group_unscorable')], {'r': 1.0}) is None
    assert effectiveness_weighted_r_a([assessment(None, 'rubric_unscorable')], {'r': 1.0}) is None
    assert effectiveness_weighted_r_a([assessment(None, 'not_applicable', False)], {'r': 1.0}) is None
    assert effectiveness_weighted_r_a([assessment(0.0)], {'r': 1.0}) == 0.0
    for bad in (None, float('nan'), float('inf'), -0.1, 1.1):
        with pytest.raises(RuntimeError, match='scored rubric value'):
            effectiveness_weighted_r_a([assessment(bad)], {'r': 1.0})


def test_real_update_keeps_advantages_aligned_and_explains_skipped_groups():
    from test_integrated_v1 import _tiny_updater, _group
    from src.online_rl.integrated.continuation import adapter_sha256

    _, model, updater = _tiny_updater()
    before = adapter_sha256(model)
    used = _group(updater, [None, 0.0, 1.0, None])
    flat = _group(updater, [0.5, 0.5, 0.5, 0.5], offset=5)
    missing = _group(updater, [None, None, 0.0, None], offset=9)
    result = updater.update_groups([used, flat, missing])
    details = result['advantage_diagnostics']
    assert result['optimizer_step'] and before != adapter_sha256(model)
    assert result['used_agent_completions'] == 2 and result['groups_used'] == 1
    assert result['skip_counts'] == {'fewer_than_two_rewarded_responses': 1, 'zero_advantage_group': 1}
    assert details['groups'][0]['advantages'] == [None, -0.5, 0.5, None]
    assert details['groups'][1]['advantages'] == [0.0] * 4
    assert details['groups'][2]['advantages'] == [None] * 4
    assert details['used']['positive'] == details['used']['negative'] == 1
    assert result['policy_loss'] == pytest.approx(0.0)
    assert result['grad_norm'] > 0
    assert result['ratio']['measured_at'] == 'before_optimizer_step'
    assert result['kl_scope'] == 'used_nonzero_advantage_groups_only'


def test_loss_rejects_misaligned_advantages_before_model_forward():
    from src.online_rl.integrated.grpo import grpo_loss
    with pytest.raises(ValueError, match='aligned'):
        grpo_loss(None, [{}], [], 0.04, 0.2, 16)


def test_failed_loss_clears_partial_gradients_and_restores_cache(monkeypatch):
    from test_integrated_v1 import _tiny_updater, _group
    from src.online_rl.integrated import grpo
    from src.online_rl.integrated.continuation import adapter_sha256

    torch, model, updater = _tiny_updater()
    records = _group(updater, [0.0, 1.0])
    before = adapter_sha256(model)
    model.config.use_cache = True

    def broken_loss(*args, **kwargs):
        for parameter in updater.trainable:
            parameter.grad = torch.ones_like(parameter)
        raise RuntimeError('bad later response')

    monkeypatch.setattr(grpo, 'grpo_loss', broken_loss)
    with pytest.raises(RuntimeError, match='bad later response'):
        updater.update(records)
    assert model.config.use_cache is True
    assert all(p.grad is None for p in updater.trainable)
    assert updater.steps == 0 and adapter_sha256(model) == before


@pytest.mark.parametrize('field,value', [('temperature', 0.7), ('top_p', 0.9),
                                       ('top_k', 20), ('top_k', None)])
def test_strict_raw_updater_rejects_sampling_warpers_before_loading_optimizer(field, value):
    from src.online_rl.integrated.grpo import IntegratedGRPOUpdater
    rollout = dict(temperature=1.0, top_p=1.0, top_k=0)
    rollout[field] = value
    with pytest.raises(ValueError, match='raw on-policy sampling'):
        IntegratedGRPOUpdater(None, {'optimizer': {'require_raw_on_policy_sampling': True}, 'rollout': rollout})


def strict_updater():
    from test_integrated_v1 import _tiny_updater
    from src.online_rl.integrated.grpo import IntegratedGRPOUpdater
    torch, model, old = _tiny_updater()
    cfg = {'optimizer': {**old.o, 'require_raw_on_policy_sampling': True},
           'rollout': {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0},
           'advantage_normalization': {'mode': 'mean_center'}}
    return torch, model, IntegratedGRPOUpdater(old.policy, cfg)


def test_strict_raw_real_sampling_matches_loss_and_updates_lora():
    from src.online_rl.agent_batch import generate_batch
    from src.online_rl.integrated.continuation import adapter_sha256
    from test_agent_batch import Tokenizer
    _, model, updater = strict_updater()
    model.eval()
    traces = generate_batch(model, Tokenizer(), ['3 4'] * 4, [10, 20, 30, 40], 4, 1.0, 1.0, 0)
    for trace, old in zip(traces, updater.behavior_logprobs(traces)):
        trace['behavior_logprobs'] = old
        assert trace['sampling_logprobs'] == pytest.approx(old, abs=1e-5)
    before = adapter_sha256(model)
    records = [{'trajectory_id': str(i), 'case_id': 'same-state', 'reward': float(i % 2),
                'rollout_policy_sha256': before, 'agent_traces': [trace]} for i, trace in enumerate(traces)]
    result = updater.update(records)
    assert result['optimizer_step'] and result['sampling_policy_alignment'] == 'raw_full_support_verified'
    assert result['ratio']['max_abs_deviation'] < 1e-6
    assert result['grad_norm'] > 0 and adapter_sha256(model) != before


def test_strict_raw_rejects_legacy_trace_and_checks_each_segment():
    from test_integrated_v1 import _group
    _, _, updater = strict_updater()
    group = _group(updater, [0.0, 1.0])
    with pytest.raises(ValueError, match='raw on-policy sampling'):
        updater.update(group)
    for record in group:
        trace = record['agent_traces'][0]
        trace.update(sampling_temperature=1.0, sampling_top_p=1.0, sampling_top_k=0,
                     logprob_semantics='raw_policy_before_sampling_warpers_v1')
        trace['segments'] = [dict(trace)]
    group[0]['agent_traces'][0]['segments'][0]['logprob_semantics'] = 'unknown'
    with pytest.raises(ValueError, match='logprob semantics'):
        updater.update(group)
    assert updater.steps == 0


def test_strict_raw_rejects_unreported_sampling_transform():
    from src.online_rl.integrated.grpo import validate_raw_sampling
    trace = dict(sampling_temperature=1.0, sampling_top_p=1.0, sampling_top_k=0,
                 logprob_semantics='raw_policy_before_sampling_warpers_v1',
                 completion_ids=[1], old_policy_logprobs=[-2.0], sampling_logprobs=[-1.5])
    with pytest.raises(ValueError, match='actual sampling logprobs differ'):
        validate_raw_sampling(trace, recorded=True, label='trace')
