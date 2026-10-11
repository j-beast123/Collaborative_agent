"""Joint REUSE matching selects a member's own turn and instruction in the same request."""
import copy
import json

import pytest

from src.online_rl.tree import backends as B
from src.online_rl.tree.records import TrajectoryRubric, TrajectoryRubricInput


JOINT = {'output_contract': 'trajectory_analysis_candidate_v2_with_intervention',
         'max_tokens': 4096, 'temperature': 0.7, 'response_format': None}


def decision(kind='REUSE_EXISTING', **overrides):
    return {'decision': kind, 'rubric_id': 'canonical' if kind == 'REUSE_EXISTING' else '',
            'checks': {key: True for key in B.MATCH_CHECKS}, 'rationale': 'The same visible request recurs.',
            'selected_agent_turn_id': 2 if kind == 'REUSE_EXISTING' else None,
            'intervention_instruction': 'Explain how the two stated candidates compare on duration.' if kind == 'REUSE_EXISTING' else None,
            **overrides}


def member(leaf_id='member', outcome='failure'):
    return TrajectoryRubricInput(leaf_id, 'case', '', 'Which candidate?', (('A', 'alpha'), ('B', 'beta')),
        (('doctor', 'D1', 1, 'I lean toward one candidate.'), ('agent', 'A1', 1, 'A prior position.'),
         ('doctor', 'D2', 2, 'Compare their duration.'), ('agent', 'A2', 2, 'The original member reply.'),
         ('doctor', 'FINAL', 0, 'Final answer: option A')),
        outcome, 'The comparison could improve.', 'member-hash', target_reply_id=1)


def bank():
    rubric = {'rubric_id': 'canonical', 'name': 'Compare duration', 'applicability_condition': 'A duration comparison is requested.',
              'required_behavior': 'Compare the two candidates on duration.',
              'scores': {'0': 'No useful duration comparison.', '1': 'Partial comparison.', '2': 'Complete duration comparison.'},
              'source_provenance': {'secret': 'CREATOR_PRIVATE_PROVENANCE'}}
    return [TrajectoryRubric('canonical', 'creator', 'prompt', None, rubric, True, (), None,
                             intervention_instruction='CREATOR_PRIVATE_INSTRUCTION')]


def fake_response(monkeypatch, output):
    calls = []
    def fake(key, model, system, user, settings, schema, name, role):
        calls.append(copy.deepcopy({'system': system, 'user': user, 'settings': settings, 'schema': schema,
                                    'name': name, 'role': role}))
        return json.dumps(output), {'usage': {'calls': 1}}
    monkeypatch.setattr(B, '_openrouter_json', fake)
    return calls


def test_joint_match_returns_member_instruction_in_the_existing_single_call(monkeypatch):
    raw = decision()
    calls = fake_response(monkeypatch, raw)
    settings = copy.deepcopy(JOINT)
    matcher = B.api_rubric_matcher('key', 'generator', settings)
    result = matcher('Package', 'Member context and frozen canonical rubric')
    assert result == {**raw, 'usage': {'calls': 1}} and len(calls) == 1
    assert calls[0]['schema'] == B.MEMBER_MATCH_JSON_SCHEMA
    assert matcher.member_interventions is True
    assert matcher.member_intervention_version == matcher.batch.member_intervention_version == B.MEMBER_INTERVENTION_VERSION
    assert matcher.settings['member_intervention_version'] == B.MEMBER_INTERVENTION_VERSION
    assert 'member_intervention_version' not in settings
    for boundary in ('immutable', 'BEFORE', 'same', 'three reruns', 'gold/ground-truth'):
        assert boundary in calls[0]['system']


@pytest.mark.parametrize('changes', [
    {'selected_agent_turn_id': 0}, {'selected_agent_turn_id': True}, {'selected_agent_turn_id': '2'},
    {'selected_agent_turn_id': None}, {'intervention_instruction': ''}, {'intervention_instruction': '  '},
    {'intervention_instruction': None}, {'intervention_instruction': ['Explain']},
    {'decision': 'CREATE_NEW'}, {'extra': 'forbidden'},
])
def test_joint_single_match_rejects_malformed_member_fields(monkeypatch, changes):
    fake_response(monkeypatch, decision(**changes))
    with pytest.raises(B.RubricOutputIncomplete, match='schema'):
        B.api_rubric_matcher('key', 'generator', JOINT)('Package', 'Member')


def test_joint_create_requires_both_explicit_null_fields(monkeypatch):
    output = decision('CREATE_NEW')
    fake_response(monkeypatch, output)
    assert B.api_rubric_matcher('key', 'generator', JOINT)('Package', 'Member')['selected_agent_turn_id'] is None
    output.pop('intervention_instruction')
    fake_response(monkeypatch, output)
    with pytest.raises(B.RubricOutputIncomplete, match='schema'):
        B.api_rubric_matcher('key', 'generator', JOINT)('Package', 'Member')


def test_joint_batch_removes_forced_feedback_locus_and_preserves_per_member_decisions(monkeypatch):
    raw = {'decisions': [{'trajectory_id': 'first', 'peer_representative_id': None, **decision()},
                         {'trajectory_id': 'second', 'peer_representative_id': 'second', **decision('CREATE_NEW')}]}
    calls = fake_response(monkeypatch, raw)
    matcher = B.api_rubric_matcher('key', 'generator', JOINT, match_batch_size=8)
    result = matcher.batch('Package', [member('first'), member('second', outcome='success')], bank())
    assert result['decisions']['first']['selected_agent_turn_id'] == 2
    assert result['decisions']['second']['intervention_instruction'] is None
    assert len(calls) == 1
    packet = json.loads(calls[0]['user'])
    assert all('behavioral_locus' not in row for row in packet['trajectories'])
    for row in packet['trajectories']:
        assert 'specified behavioral locus' not in row['input']
        assert 'Agent reply 1 [turn_id=1]' in row['input'] and 'Agent reply 2 [turn_id=2]' in row['input']
    for hidden in ('CREATOR_PRIVATE_INSTRUCTION', 'CREATOR_PRIVATE_PROVENANCE'):
        assert hidden not in calls[0]['user']
    schema = calls[0]['schema']['properties']['decisions']['items']
    assert set(schema['required']) == {'trajectory_id', 'peer_representative_id', *B.MEMBER_MATCH_JSON_SCHEMA['required']}


@pytest.mark.parametrize('override', [{'selected_agent_turn_id': 3}, {'selected_agent_turn_id': True},
                                    {'intervention_instruction': ''}, {'decision': 'CREATE_NEW'}])
def test_joint_batch_rejects_invalid_or_nonexistent_member_turn(monkeypatch, override):
    d=decision(**override)
    peer=None if d['decision']=='REUSE_EXISTING' else 'member'
    fake_response(monkeypatch, {'decisions': [{'trajectory_id': 'member', 'peer_representative_id': peer, **d}]})
    with pytest.raises(B.RubricOutputIncomplete):
        B.api_rubric_matcher('key', 'generator', JOINT).batch('Package', [member()], bank())


def test_legacy_single_schema_is_unchanged_and_batch_adds_peer_grouping(monkeypatch):
    output = {key: value for key, value in decision().items() if key in B.MATCH_JSON_SCHEMA['required']}
    calls = fake_response(monkeypatch, output)
    matcher = B.api_rubric_matcher('key', 'generator', {'response_format': 'json_schema_strict'})
    assert matcher('Package', 'Member')['rubric_id'] == 'canonical'
    assert calls[0]['schema'] == B.MATCH_JSON_SCHEMA
    assert not matcher.member_interventions and matcher.member_intervention_version is None
    assert 'JOINT MEMBER' not in calls[0]['system']
    calls = fake_response(monkeypatch, {'decisions': [{'trajectory_id': 'member', 'peer_representative_id': None, **output}]})
    matcher.batch('Package', [member()], bank())
    assert json.loads(calls[0]['user'])['trajectories'][0]['behavioral_locus'] == 1
    schema = calls[0]['schema']['properties']['decisions']['items']
    assert set(schema['required']) == {'trajectory_id', 'peer_representative_id', *B.MATCH_JSON_SCHEMA['required']}
