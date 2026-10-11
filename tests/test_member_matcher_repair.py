"""A malformed assignment is retried once, never promoted to a new rubric."""
import json

import pytest

from src.online_rl.tree import backends as B
from src.online_rl.tree.cache import CallCache
from test_member_intervention_matcher import JOINT, bank, decision, member


def scripted(monkeypatch, outputs):
    sent = []
    def call(key, model, system, user, settings, schema, name, role):
        sent.append({'model': model, 'system': system, 'user': user, 'settings': settings,
                     'schema': schema, 'name': name, 'role': role})
        value = outputs[len(sent) - 1]
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value), {'usage': {'total_tokens': 10}}
    monkeypatch.setattr(B, '_openrouter_json', call)
    return sent


def test_partial_json_corrected_with_same_input_and_member_instruction(monkeypatch):
    partial = '```json\n{"decision":"CREATE_NEW","rationale":"In this trajectory, '
    corrected = decision()
    sent = scripted(monkeypatch, [partial, corrected])
    matcher = B.api_rubric_matcher('secret', 'fixed-generator', JOINT)
    result = matcher('Generator prompt requiring analysis and candidate rubric.', 'ORIGINAL_MEMBER_AND_BANK')
    assert result['decision'] == 'REUSE_EXISTING'
    assert result['intervention_instruction'] == corrected['intervention_instruction']
    assert result['selected_agent_turn_id'] == 2
    assert len(sent) == 2
    assert all(c['settings']['max_tokens'] == JOINT['max_tokens'] for c in sent)
    assert all(c['model'] == 'fixed-generator' for c in sent)
    assert sent[0]['system'].endswith(B.MATCH_SINGLE_INSTRUCTIONS)
    assert sent[1]['system'] == sent[0]['system']
    assert sent[1]['user'].startswith('ORIGINAL_MEMBER_AND_BANK')
    assert 'Do not turn an output-format error into CREATE_NEW' in sent[1]['user']
    audit = result['match_correction_audit']
    assert audit['attempts'][0]['raw'] == partial
    assert audit['attempts'][1]['error'] is None
    assert matcher.settings['match_correction_contract'] == B.MATCH_CORRECTION_CONTRACT
    assert 'match_correction_contract' not in JOINT


def test_persistent_schema_failure_is_error_and_is_never_cached(monkeypatch, tmp_path):
    sent = scripted(monkeypatch, [decision(intervention_instruction=None)] * 2)
    matcher = B.api_rubric_matcher('secret', 'fixed-generator', JOINT)
    cache = CallCache(tmp_path / 'calls.jsonl')
    with pytest.raises(B.RubricOutputIncomplete, match='schema correction failed'):
        cache.call('rubric_match', {'fixed': True}, lambda: matcher('S', 'U'))
    assert len(sent) == 2 and cache.store == {}
    failure = json.loads((tmp_path / 'calls.errors.jsonl').read_text())
    assert len(failure['audit']['attempts']) == 2
    assert failure['audit']['system'].endswith(B.MATCH_SINGLE_INSTRUCTIONS)
    assert 'secret' not in json.dumps(failure)


def test_batch_correction_preserves_exact_member_id_coverage(monkeypatch):
    bad = {'decisions': [{'trajectory_id': 'first', 'peer_representative_id': None, 'peer_checks': None, **decision()}]}
    good = {'decisions': [
        {'trajectory_id': 'first', 'peer_representative_id': None, 'peer_checks': None, **decision()},
        {'trajectory_id': 'second', 'peer_representative_id': 'second', 'peer_checks': {k:True for k in B.MATCH_CHECKS}, **decision('CREATE_NEW')},
    ]}
    sent = scripted(monkeypatch, [bad, good])
    matcher = B.api_rubric_matcher('secret', 'fixed-generator', JOINT, match_batch_size=8)
    result = matcher.batch('S', [member('first'), member('second')], bank())
    assert set(result['decisions']) == {'first', 'second'}
    assert result['decisions']['first']['selected_agent_turn_id'] == 2
    assert result['decisions']['second']['intervention_instruction'] is None
    assert len(sent) == 2
    assert sent[1]['user'].startswith(sent[0]['user'])
    assert result['match_correction_audit']['attempts'][0]['error'] == 'Batch match ID coverage differs'


def test_provider_incomplete_has_only_one_bounded_correction(monkeypatch):
    failure = B.RubricOutputIncomplete('finish_reason=length')
    sent = scripted(monkeypatch, [failure, decision('CREATE_NEW')])
    result = B.api_rubric_matcher('secret', 'fixed-generator', JOINT)('S', 'U')
    assert result['decision'] == 'CREATE_NEW' and len(sent) == 2
    assert result['match_correction_audit']['attempts'][0]['raw'] is None
    assert result['match_correction_audit']['attempts'][0]['error'] == 'finish_reason=length'


def test_network_failure_is_not_treated_as_a_schema_correction(monkeypatch):
    sent = scripted(monkeypatch, [TimeoutError('offline')])
    with pytest.raises(TimeoutError):
        B.api_rubric_matcher('secret', 'fixed-generator', JOINT)('S', 'U')
    assert len(sent) == 1
