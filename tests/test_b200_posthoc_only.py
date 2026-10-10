"""Public Doctor dialogue + terminal reflection; no model/API execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.online_rl.run_tree import make_feedback
from src.online_rl.tree.b200_runner import load_config, preflight, validate_resume_identity, ROOT
from src.online_rl.doctor import make_doctor
from src.online_rl.integrated.records import sha256_json
from test_b200_partner import CASE
from test_hf_doctor import local_model


def test_posthoc_config_preserves_science_and_pins_new_prompts(monkeypatch):
    monkeypatch.setattr('src.online_rl.tree.b200_runner.physical_devices', lambda: ['0', '1'])
    cfg = load_config()
    assert not cfg['doctor_need']['enabled']
    assert 'private_need' not in cfg['five_gpu']['opening_identity']
    assert cfg['feedback']['protocol'] == 'physician_post_final_reflection_v1'
    assert cfg['feedback']['max_tokens'] == 512
    assert cfg['rollout']['agent_protocol'] == 'precommit_v1'
    assert cfg['rollout']['rounds'] == cfg['rollout']['branching'] == 4
    assert cfg['two_gpu']['batch_cases'] == 5
    assert cfg['agent_reward'] == {'w_outcome': 0., 'w_rubric': 1., 'r_a_weighting': 'equal'}
    assert cfg['fixed_accuracy']['enabled'] and not cfg['prompt_update']['enabled']
    assert cfg['intervention']['reruns_per_intervention'] == 3
    assert cfg['execution']['global_api_max_in_flight'] == 50
    assert all(e.startswith('Missing file:') for e in preflight(cfg, resources=False))
    bad = copy.deepcopy(cfg)
    bad['five_gpu']['opening_identity']['private_need'] = {**cfg['doctor_need'], 'enabled': True}
    assert any('need-enabled D1 cache' in e for e in preflight(bad, resources=False))
    gen = (ROOT/cfg['rubric']['generator_prompt_file']).read_text()
    assert 'No prospective private Doctor need records are supplied or required' in gen
    assert 'do not generate a duplicate generic medical-accuracy criterion' in gen


def test_hf_doctor_plain_turn_status_then_reflection_without_need(local_model):
    cfg = load_config()
    d = make_doctor(cfg['rollout'], (ROOT/cfg['rollout']['doctor_system_prompt_file']).read_text())
    sent = []
    def chat(user, seed, max_tokens=None, system_override=None):
        sent.append({'user': user, 'system': system_override or d.system, 'budget': max_tokens or d.max_tokens})
        if system_override:
            return json.dumps({'helpful': [{'agent_reply_id': 1, 'experience': 'I could examine the comparison.'}], 'unresolved': []})
        return 'Please compare the alternatives.' + ('\nSTATUS: '+json.dumps({'current_option': 'A', 'open_issue': True,
            'collaborator_objection': False, 'more_info_could_change_decision': True}) if 'STATUS:' in user else '')
    d._chat = chat
    opening = d.turn(CASE, [], 1, 4, 0)
    assert opening['visible'] == 'Please compare the alternatives.' and 'private_need' not in opening
    history = [{'speaker':'doctor','stage':'D1','round':1,'text':opening['visible']},
               {'speaker':'agent','stage':'A1','round':1,'text':'Here is the comparison.'}]
    follow = d.turn(CASE, history, 2, 4, 1)
    assert follow['status']['current_option'] == 'A' and 'STATUS:' not in follow['visible']
    assert 'private_need' not in follow and sent[0]['budget'] == sent[1]['budget'] == 256
    f = make_feedback(cfg, SimpleNamespace(doctor=d), 'unused')
    assert not f.accepts_private_needs
    with pytest.raises(ValueError, match='completed dialogue'):
        f(CASE, history, 1.)
    history.append({'speaker':'doctor','stage':'FINAL','round':2,'text':'Final answer: option A'})
    result = f({**CASE, 'correct_option':'GOLD_SENTINEL'}, history, 0.)
    assert result['valid'] and result['reflection_need_links'] == []
    assert set(result['reflection']) == {'helpful','unresolved'}
    assert sent[-1]['budget'] == 512
    for request in sent:
        assert not any(s in json.dumps(request) for s in ('private_need','need_assessment','GOLD_SENTINEL','ground_truth'))


def test_disabled_need_diagnostics_are_not_missing_records():
    from test_b200_runner import records
    from src.online_rl.tree.b200_diagnostics import batch_diagnostics
    batch = records()
    for r in batch: r['doctor_need_enabled'] = False
    diag = batch_diagnostics(batch, {'optimizer_step': True}, .1)
    for case in diag['cases']:
        assert case['needs']['enabled'] is False
        assert case['needs']['missing_node_records'] == case['needs']['unique_records'] == 0
    assert diag['needs']['missing_node_records'] == 0


def test_old_protocol_cannot_resume_into_new_run():
    cfg = load_config(); old = copy.deepcopy(cfg)
    old['doctor_need']['enabled'] = True
    old['five_gpu']['opening_identity']['private_need'] = old['doctor_need']
    state = {'config_sha256':sha256_json(old), 'source_sha256':'same'}
    with pytest.raises(RuntimeError, match='Config/code changed'):
        validate_resume_identity(state, cfg, 'same')
