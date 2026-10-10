"""Synthetic contract tests; no clinical model, network, GPU or optimizer calls."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.judges.rubric_schema import (PUBLIC_CRITERION_PROTOCOL as PROTOCOL,
    PUBLIC_CRITERION_FIELDS, validate_public_criterion)
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import backends as B, cumulative_bank as CB
from src.online_rl.tree.agent_state import visible_rubric, flat_rubric, rubric_problems
from src.online_rl.tree.cache import CallCache
from test_cumulative_bank import record, tin, decision
from test_instruction_alignment_generation import example
from src.online_rl.tree.instruction_audit import ANALYSIS_PROTOCOL

BASE = Path('prompts/sherpa')
SCHEMA = BASE / 'trajectory_rubric_generator_v2_4_public_mc.schema.json'
PROMPT = BASE / 'trajectory_rubric_generator_v2_4_public_mc.txt'


def public_fields():
    return {'required_behavior': 'Compare the stated considerations in relation to the requested criterion.',
            'minimum_performance': 'Express one concrete relation between the stated considerations.',
            'completion_condition': 'Connect that relation to the requested criterion without leaving that connection unresolved.',
            'criterion_version': PROTOCOL}


def public_output(outcome='success'):
    source, obj = example(outcome)
    obj['candidate_rubric'].update(public_fields())
    obj['analysis']['score_construction_check']['minimal_contribution'] = public_fields()['minimum_performance']
    return source, obj


def normalize(source, obj, **kwargs):
    return B.normalize_trajectory_analysis_output(obj, None, source, require_intervention=True,
        instruction_alignment_protocol=ANALYSIS_PROTOCOL, public_criterion_protocol=PROTOCOL, **kwargs)


def public_record(case='public', **kwargs):
    row = record(case, **kwargs)
    raw = row['trajectory_rubrics'][0]
    raw['rubric'].update(public_fields())
    raw['gate'].update(protocol='public_mc_gate_v1', public_criterion_version=PROTOCOL,
                       criterion_sha256=CB.content_hash(raw['rubric']))
    for run in row['rubric_bank']['witness'][0]['tested'][0]['rerun_records']:
        run['intervention_spec'] = {'rubric_sha256': sha256_json(raw['rubric'])}
    return row


def public_decision(payload, reuse=True):
    result = decision(payload, reuse)
    result['checks'] = {key: reuse for key in CB.PUBLIC_CHECKS}
    return result


def test_public_fields_survive_generation_visibility_and_flat_transport_without_analysis_promotion():
    source, obj = public_output()
    before = copy.deepcopy(obj)
    rubric, error = normalize(source, obj)
    assert error is None and obj == before
    assert rubric_problems(rubric) == []
    for field, value in public_fields().items():
        assert rubric[field] == value
        assert visible_rubric(rubric)[field] == value
        assert flat_rubric(rubric)[field] == value
    assert rubric['required_behavior'] != rubric['scores']['2']['behavior']
    assert 'source_provenance' not in visible_rubric(rubric)
    assert rubric['source_provenance']['public_criterion_protocol'] == PROTOCOL
    assert rubric['source_provenance']['analysis']['source_score'] == 2


@pytest.mark.parametrize('field', ['required_behavior', *PUBLIC_CRITERION_FIELDS])
@pytest.mark.parametrize('bad', [None, '', '  ', {}, 2])
def test_bad_public_field_is_not_repaired_from_hidden_analysis(field, bad):
    source, obj = public_output()
    obj['candidate_rubric'][field] = bad
    rubric, error = normalize(source, obj)
    assert rubric is None and error
    assert validate_public_criterion(obj['candidate_rubric'], required=True)


def test_missing_public_fields_and_analysis_m_drift_fail_closed():
    source, obj = public_output()
    obj['candidate_rubric'].pop('minimum_performance')
    assert normalize(source, obj)[0] is None
    source, obj = public_output()
    obj['analysis']['score_construction_check']['minimal_contribution'] = 'Bare agreement.'
    assert 'exactly repeat' in normalize(source, obj)[1]
    # A legacy normalization call must not drop a new public field silently.
    assert B.normalize_trajectory_analysis_output(obj, None, source, require_intervention=True,
        instruction_alignment_protocol=ANALYSIS_PROTOCOL)[0] is None


def test_no_public_fields_added_to_legacy_visible_rubric_or_hash():
    r = record()['trajectory_rubrics'][0]['rubric']
    original = copy.deepcopy(r)
    expected = {'applicability': {k: r['applicability'][k] for k in sorted(r['applicability'])},
                'required_behavior': r['required_behavior'],
                'scores': {k: r['scores'][k]['behavior'] for k in '012'}}
    assert CB.content_hash(r) == sha256_json(expected)
    assert not any(k in visible_rubric(r) for k in PUBLIC_CRITERION_FIELDS)
    r['source_provenance']['analysis'] = {'score_construction_check': {
        'minimal_contribution': 'PRIVATE CANARY M', 'complete_behavior': 'PRIVATE CANARY C'}}
    assert CB.content_hash(r) == CB.content_hash(original)
    assert 'PRIVATE CANARY' not in json.dumps(visible_rubric(r))
    assert not validate_public_criterion(r)


@pytest.mark.parametrize('field', PUBLIC_CRITERION_FIELDS)
def test_public_criterion_changes_cannot_inherit_gate_or_witness(field):
    row = public_record()
    raw = row['trajectory_rubrics'][0]
    old_hash = CB.content_hash(raw['rubric'])
    raw['rubric'][field] += ' changed'
    assert CB.content_hash(raw['rubric']) != old_hash
    with pytest.raises(ValueError, match='versioned gate proof'):
        CB.extend_snapshot(CB.empty_snapshot(), [row])
    raw['gate'].update(criterion_sha256=CB.content_hash(raw['rubric']),
                       public_criterion_version=raw['rubric']['criterion_version'])
    with pytest.raises(ValueError, match='another criterion witness'):
        CB.extend_snapshot(CB.empty_snapshot(), [row])


def test_adding_public_fields_does_not_upgrade_legacy_keep():
    row = record()
    row['trajectory_rubrics'][0]['rubric'].update(public_fields())
    with pytest.raises(ValueError, match='quality gate'):
        CB.extend_snapshot(CB.empty_snapshot(), [row])


def test_frozen_bank_and_reuse_preserve_mc_and_proof_without_new_generation():
    row = public_record()
    original = copy.deepcopy(row)
    bank = CB.extend_snapshot(CB.empty_snapshot(), [row])
    score_rubric = CB.scoring_rubrics(bank)[0]
    for field, value in public_fields().items():
        assert score_rubric.rubric[field] == value
    calls = []
    def matcher(payload):
        calls.append(copy.deepcopy(payload))
        assert 'source_provenance' not in json.dumps(payload['candidates'])
        for field, value in public_fields().items():
            assert payload['candidates'][0]['rubric'][field] == value
        return public_decision(payload)
    cache = CallCache(None)
    first = CB.match_existing(tin(), bank, matcher, cache)
    assert first == CB.match_existing(tin(), bank, matcher, cache)
    assert len(calls) == 1 and set(cache.calls) == {'cumulative_bank_match'}
    assert CB.validate_reuse(tin(), bank, first) == first
    assert first['sourceproof'] == bank['entries'][0]['sources']
    assert row == original


def test_public_matcher_requires_both_mc_equivalence_checks():
    bank = CB.extend_snapshot(CB.empty_snapshot(), [public_record()])
    with pytest.raises(ValueError, match='incomplete decision'):
        CB.match_existing(tin(), bank, decision, CallCache(None))
    def mismatched(payload):
        d = public_decision(payload)
        d['checks']['same_completion_condition'] = False
        return d
    with pytest.raises(ValueError, match='every equivalence check'):
        CB.match_existing(tin(), bank, mismatched, CallCache(None))


def test_new_generation_local_schema_and_cache_identity_are_explicit(monkeypatch):
    import jsonschema
    from src.online_rl.tree.feedback_grounding import PROTOCOL as FEEDBACK, REFLECTION_KIND
    source, obj = public_output()
    source = replace(source, feedback_kind=REFLECTION_KIND, feedback=json.dumps({'helpful': [], 'unresolved': []}))
    obj['analysis']['feedback_check']['claim_reviews'] = []
    schema = json.loads(SCHEMA.read_text())
    jsonschema.validate(obj, schema)
    calls = []
    def fake(key, model, system, user, settings, schema, name, role):
        calls.append({'settings': settings, 'schema': schema, 'system': system})
        return json.dumps(obj), {}
    monkeypatch.setattr(B, '_openrouter_json', fake)
    cfg = {'generation': {'public_criterion_protocol': PROTOCOL,
        'output_contract': 'trajectory_analysis_candidate_v2_with_intervention',
        'instruction_alignment_protocol': ANALYSIS_PROTOCOL, 'feedback_check_protocol': FEEDBACK,
        'local_schema_file': str(SCHEMA), 'local_schema_sha256': hashlib.sha256(SCHEMA.read_bytes()).hexdigest(),
        'selected_turn_policy': 'generator_v1'}}
    settings = B.generation_settings(cfg)
    gen = B.api_rubric_generator('unused', 'synthetic', settings)
    result = gen(PROMPT.read_text(), source)
    assert result['parse_error'] is None and len(calls) == 1
    assert result['intervention_instruction'] == obj['intervention_instruction']
    assert calls[0]['settings']['public_criterion_protocol'] == PROTOCOL
    assert calls[0]['settings']['effective_output_schema_sha256'] == sha256_json(schema)
    assert 'public_criterion_protocol' not in B.generation_settings({})
    with pytest.raises(ValueError, match='Unknown public'):
        B.generation_settings({'generation': {'public_criterion_protocol': 'unknown'}})
    with pytest.raises(ValueError, match='pinned local schema'):
        B.api_rubric_generator('unused', 'synthetic', {'public_criterion_protocol': PROTOCOL})


def test_new_bank_matcher_uses_public_schema_but_does_not_change_legacy_schema(monkeypatch):
    public_bank = CB.extend_snapshot(CB.empty_snapshot(), [public_record()])
    calls = []
    def fake(key, model, system, user, settings, schema, name, role):
        payload = json.loads(user)
        calls.append(schema['properties']['checks']['required'])
        checks = CB._decision_checks(payload)
        result = decision(payload)
        result['checks'] = {c: True for c in checks}
        return json.dumps(result), {}
    monkeypatch.setattr(B, '_openrouter_json', fake)
    matcher = CB.api_bank_matcher('unused', 'synthetic', {'max_tokens': 800, 'temperature': 0},
                                 (BASE / 'cumulative_bank_match_public_mc_v1.txt').read_text())
    CB.match_existing(tin(), public_bank, matcher, CallCache(None))
    legacy = CB.extend_snapshot(CB.empty_snapshot(), [record()])
    CB.match_existing(tin(), legacy, matcher, CallCache(None))
    assert calls == [list(CB.PUBLIC_CHECKS), list(CB.CHECKS)]


def test_same_case_matcher_mc_wording_and_cache_identity_are_opt_in(monkeypatch):
    calls = []
    def fake(key, model, system, user, settings, schema, name, role):
        calls.append({'system': system, 'settings': settings})
        return json.dumps({'decision': 'CREATE_NEW', 'rubric_id': '',
            'checks': {c: False for c in B.MATCH_CHECKS}, 'rationale': 'Synthetic unrelated criterion.',
            'selected_agent_turn_id': None, 'intervention_instruction': None}), {}
    monkeypatch.setattr(B, '_openrouter_json', fake)
    settings = {'max_tokens': 1000, 'temperature': 0,
                'output_contract': 'trajectory_analysis_candidate_v2_with_intervention',
                'instruction_alignment_protocol': ANALYSIS_PROTOCOL}
    old = B.api_rubric_matcher('unused', 'synthetic', settings)
    new = B.api_rubric_matcher('unused', 'synthetic', {**settings, 'public_criterion_protocol': PROTOCOL})
    old('old package', 'old synthetic context')
    new('new package', 'new synthetic context')
    assert B.PUBLIC_MEMBER_ALIGNMENT_INSTRUCTIONS not in calls[0]['system']
    assert B.MEMBER_ALIGNMENT_INSTRUCTIONS in calls[0]['system']
    assert B.PUBLIC_MEMBER_ALIGNMENT_INSTRUCTIONS in calls[1]['system']
    assert B.MEMBER_ALIGNMENT_INSTRUCTIONS not in calls[1]['system']
    assert old.settings['member_intervention_prompt_sha256'] != new.settings['member_intervention_prompt_sha256']
    assert 'public_criterion_protocol' not in old.settings


def test_new_criterion_version_adds_new_bank_entry_without_relabeling_old_proof():
    first = public_record('a')
    bank = CB.extend_snapshot(CB.empty_snapshot(), [first])
    before = copy.deepcopy(bank)
    second = public_record('b')
    raw = second['trajectory_rubrics'][0]
    raw['rubric']['criterion_version'] = 'public_behavior_mc_v2'
    raw['gate'].update(public_criterion_version='public_behavior_mc_v2',
                       criterion_sha256=CB.content_hash(raw['rubric']))
    for run in second['rubric_bank']['witness'][0]['tested'][0]['rerun_records']:
        run['intervention_spec']['rubric_sha256'] = sha256_json(raw['rubric'])
    extended = CB.extend_snapshot(bank, [second])
    assert bank == before and len(extended['entries']) == 2
    old_entry = next(e for e in extended['entries'] if e['bank_id'] == bank['entries'][0]['bank_id'])
    assert old_entry == bank['entries'][0]
    new_entry = next(e for e in extended['entries'] if e['bank_id'] != bank['entries'][0]['bank_id'])
    assert [p['case_id'] for p in new_entry['sources']] == ['b']
