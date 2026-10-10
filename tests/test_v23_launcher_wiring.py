"""Offline admission and cache identity for the isolated v2.3 experiment."""
import copy
from types import SimpleNamespace

import pytest

from scripts.online_rl import run_tree as RT
from src.online_rl.integrated.continuation import protocol_sha256
from src.online_rl.tree.experiment import load_config

CONFIG = 'configs/experiments/train120_max2_fixed_p0_v23.json'
PREVIOUS = 'configs/experiments/train120_max2_fixed_p0.json'


def test_v23_changes_audit_identity_without_changing_training_or_witness_design():
    cfg, previous = load_config(CONFIG), load_config(PREVIOUS)
    assert RT.blockers(cfg) == []
    for section in ('policy', 'data', 'rollout', 'schedule', 'optimizer', 'verification',
                    'agent_reward', 'five_gpu', 'intervention', 'prompt_update'):
        assert cfg[section] == previous[section]
    assert cfg['output_dir'] != previous['output_dir']
    assert cfg['rollout_protocol_sha256_override'] is None
    assert cfg['rubric']['generation']['instruction_alignment_protocol'] == RT.INSTRUCTION_ALIGNMENT_PROTOCOL
    assert cfg['judges']['intervention_audit_model'] == cfg['judges']['gate_model']
    assert RT.intervention_audit_identity(previous) is None
    assert protocol_sha256(cfg, RT.ROOT) != protocol_sha256(previous, RT.ROOT)


@pytest.mark.parametrize('field,value,message', [
    ('intervention_audit_protocol', 'unknown', 'Unknown judges.intervention_audit_protocol'),
    ('intervention_audit_structured', False, 'structured = true'),
    ('intervention_audit_model', 'typesafe/jev', 'structured Luna model'),
    ('intervention_audit_max_tokens', True, 'max_tokens >= 2048'),
    ('intervention_audit_max_tokens', 1000, 'max_tokens >= 2048'),
    ('intervention_audit_timeout_s', 0, 'positive finite timeout'),
    ('intervention_audit_timeout_s', float('inf'), 'positive finite timeout'),
    ('intervention_audit_prompt_sha256', '0' * 64, 'intervention_audit_prompt_file differs'),
    ('intervention_audit_prompt_sha256', 'TBD', 'pinned instruction audit prompt'),
    ('intervention_audit_prompt_file', 'missing-audit-prompt.txt', 'Missing'),
])
def test_bad_audit_configuration_fails_before_clients_or_credentials(field, value, message, monkeypatch):
    monkeypatch.setattr(RT, 'load_key', lambda: pytest.fail('preflight read credentials'))
    cfg = load_config(CONFIG)
    cfg['judges'][field] = value
    assert any(message in reason for reason in RT.blockers(cfg))


def test_alignment_contract_cannot_silently_use_old_schema_or_skip_audit():
    cfg = load_config(CONFIG)
    previous = load_config(PREVIOUS)
    cfg['rubric']['generation']['local_schema_file'] = previous['rubric']['generation']['local_schema_file']
    assert any('valid pinned alignment schema' in reason for reason in RT.blockers(cfg))
    cfg = load_config(CONFIG)
    cfg['judges'].pop('intervention_audit_protocol')
    assert any('requires intervention_instruction_audit_v1' in reason for reason in RT.blockers(cfg))


def test_identity_uses_exact_runtime_settings_and_invalidates_for_audit_changes(monkeypatch):
    from src.online_rl.tree import instruction_audit
    from src.online_rl.tree.luna import LunaDecisionClient
    monkeypatch.setattr(LunaDecisionClient, '_chat', lambda *args, **kwargs: pytest.fail('offline identity called an API'))
    cfg = load_config(CONFIG)
    callback = RT.make_intervention_audit(cfg, 'unused-test-key')
    before = RT.intervention_audit_identity(cfg)
    assert callback.settings == before
    assert before['prompt_sha256'] == cfg['judges']['intervention_audit_prompt_sha256']
    before_protocol = protocol_sha256(cfg, RT.ROOT)
    changed = copy.deepcopy(cfg)
    changed['judges']['intervention_audit_timeout_s'] += 1
    assert RT.intervention_audit_identity(changed) != before
    assert protocol_sha256(changed, RT.ROOT) != before_protocol
    monkeypatch.setitem(instruction_audit.ITEM, 'task', instruction_audit.ITEM['task'] + ' Changed contract.')
    assert RT.intervention_audit_identity(cfg)['contract_sha256'] != before['contract_sha256']
    assert protocol_sha256(cfg, RT.ROOT) != before_protocol


def test_build_binds_audit_to_actual_tree_environment_for_both_entrypoints(monkeypatch):
    from src.online_rl.tree import trainer
    from src.online_rl.tree.luna import LunaDecisionClient
    cfg = load_config(CONFIG)
    doctor = SimpleNamespace(model=cfg['feedback']['model'], protocol='fixture-doctor',
                             verify=lambda: {'problems': []})
    policy = SimpleNamespace(args=SimpleNamespace(rounds=cfg['rollout']['rounds'],
                             doctor_model=cfg['rollout']['doctor_model']), doctor=doctor)
    monkeypatch.setattr(LunaDecisionClient, '_chat', lambda *args, **kwargs: pytest.fail('build called an API'))
    monkeypatch.setattr(RT, 'load_key', lambda: 'unused-test-key')
    monkeypatch.setattr(RT, 'make_feedback', lambda *args: None)
    monkeypatch.setattr(RT, 'opening_table', lambda *args: None)
    monkeypatch.setattr(RT, 'runtime_execution', lambda *args: {})
    monkeypatch.setattr(RT, 'replica_barrier', lambda *args: None)
    monkeypatch.setattr(RT, 'milestone_evaluator', lambda *args: None)
    # Stop before persistence or model initialization; exercise the real client,
    # callback factory, TreeEnv construction and shared worker build path.
    monkeypatch.setattr(trainer, 'TreeTrainer', lambda config, **kwargs: SimpleNamespace(**kwargs))
    built = RT.build(cfg, policy=policy, cases=[{'case_id': 'fixture'}], defer_update=True)
    assert isinstance(built.env, trainer.TreeEnv)
    assert callable(built.env.intervention_audit_fn)
    assert built.env.intervention_audit_fn.settings == RT.intervention_audit_identity(cfg)
    assert built.env.intervention_audit_fn.model == cfg['judges']['gate_model']


@pytest.mark.parametrize('section,key,value,message', [
    ('rubric', 'enabled', False, 'requires rubric.enabled'),
    ('verification', 'mode', 'exhaustive_members_v1', 'requires bounded witness verification'),
    ('judges', 'scorer_protocol', 'legacy_v1', 'structured scorer with Stage-1 applicability'),
])
def test_audit_runtime_requirements_fail_during_preflight(section, key, value, message, monkeypatch):
    cfg = load_config(CONFIG)
    cfg[section][key] = value
    monkeypatch.setattr(RT, 'load_key', lambda: pytest.fail('preflight read credentials'))
    monkeypatch.setattr(RT, 'make_policy', lambda *args: pytest.fail('preflight loaded a model'))
    assert any(message in reason for reason in RT.blockers(cfg))
