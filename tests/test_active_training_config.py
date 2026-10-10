"""The published max-two-turn configuration binds the active training protocols."""
import copy
import json

import pytest

from scripts.online_rl import run_tree as RT
from src.online_rl.integrated.continuation import protocol_sha256
from src.online_rl.tree.backends import generation_settings
from src.online_rl.tree.experiment import load_config


CONFIG = 'configs/experiments/train120_max2_fixed_p0.json'


def test_standalone_config_keeps_fixed_prompt_noeval_and_fresh_two_turn_identity():
    raw = json.loads((RT.ROOT / CONFIG).read_text())
    assert 'extends' not in raw
    cfg = load_config(CONFIG)
    assert RT.blockers(cfg) == []
    assert cfg['schedule']['cases'] == 120
    assert cfg['rollout']['rounds'] == cfg['rollout']['halting']['min_agent_turns'] == 2
    assert cfg['rollout']['branching'] == 4
    assert cfg['rollout_protocol_sha256_override'] is None
    assert cfg['policy']['init_adapter'] is None
    assert cfg['prompt_update']['enabled'] is False
    assert cfg['five_gpu']['checkpoint_evaluation_enabled'] is False
    assert cfg['five_gpu']['base_evaluation']['enabled'] is False
    assert cfg['five_gpu']['batch_cases'] == 5
    assert cfg['five_gpu']['checkpoint_every_cases'] == 30
    longer = copy.deepcopy(cfg)
    longer['rollout']['rounds'] = 4
    assert protocol_sha256(longer, RT.ROOT) != protocol_sha256(cfg, RT.ROOT)


def test_quote_config_reaches_actual_gate_and_changes_its_identity():
    cfg = load_config(CONFIG)
    expected = RT.evidence_gate_identity(cfg)
    client = RT.judge_client(cfg['judges']['gate_model'], 'unused',
        (RT.ROOT / cfg['judges']['gate_prompt_file']).read_text(),
        max_tokens=cfg['judges']['gate_max_tokens'])
    gate = RT.make_gate(cfg, client, [{'case_id': 'fixture'}])
    assert gate.settings == expected
    assert expected['rubric_quote_protocol'] == 'visible_field_enum_v1'
    previous = copy.deepcopy(cfg)
    previous['judges'].pop('gate_quote_protocol')
    assert RT.evidence_gate_identity(previous) != expected


@pytest.mark.parametrize('field,value,expected', [
    ('gate_protocol', 'unknown', 'gate_quote_protocol requires evidence_gate_v2'),
    ('gate_max_tokens', 2000, 'gate_max_tokens >= 2048'),
    ('gate_max_tokens', True, 'gate_max_tokens >= 2048'),
    ('gate_prompt_sha256', 'TBD', 'pinned gate prompt'),
    ('gate_prompt_sha256', '0' * 64, 'gate_prompt_file differs'),
    ('gate_model', 'typesafe/jev', 'structured Luna client'),
    ('gate_quote_protocol', 'misspelled', 'Unknown judges.gate_quote_protocol'),
])
def test_evidence_gate_rejects_incompatible_settings(field, value, expected):
    cfg = load_config(CONFIG)
    cfg['judges'][field] = value
    assert any(expected in reason for reason in RT.blockers(cfg))


def test_run_identity_binds_gate_contract_without_an_api_call(monkeypatch):
    from src.online_rl.tree import quality_gate
    from src.online_rl.tree.luna import LunaDecisionClient

    def forbidden(*args, **kwargs):
        raise AssertionError('Computing a protocol identity must not call an API')

    monkeypatch.setattr(LunaDecisionClient, '_chat', forbidden)
    cfg = load_config(CONFIG)
    before = RT.evidence_gate_identity(cfg)
    monkeypatch.setitem(quality_gate.ITEM, 'task', quality_gate.ITEM['task'] + ' Changed contract.')
    after = RT.evidence_gate_identity(cfg)
    assert before['contract_sha256'] != after['contract_sha256']
    assert before['prompt_sha256'] == after['prompt_sha256']


def test_reflection_audit_requires_complete_schema_and_actual_doctor(tmp_path):
    cfg = load_config(CONFIG)
    assert generation_settings(cfg['rubric'])['feedback_check_protocol'] == 'all_reflection_claims_v1'
    schema = json.loads((RT.ROOT / cfg['rubric']['generation']['local_schema_file']).read_text())
    check = schema['properties']['analysis']['properties']['feedback_check']
    check['required'].remove('claim_reviews')
    schema_file = tmp_path / 'incomplete_schema.json'
    schema_file.write_text(json.dumps(schema))
    bad = copy.deepcopy(cfg)
    bad['rubric']['generation']['local_schema_file'] = str(schema_file)
    assert any('mandatory claim_reviews' in reason for reason in RT.joint_protocol_problems(bad))
    bad = copy.deepcopy(cfg)
    bad['feedback']['protocol'] = 'observer_v1'
    assert any('participating Doctor' in reason for reason in RT.joint_protocol_problems(bad))
    bad = copy.deepcopy(cfg)
    bad['rubric']['generation']['feedback_check_protocol'] = 'misspelled'
    assert any('Unknown rubric.generation.feedback_check_protocol' in reason
               for reason in RT.joint_protocol_problems(bad))
