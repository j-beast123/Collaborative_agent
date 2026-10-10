import copy
import json
from pathlib import Path

import pytest

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import cumulative_bank as cb
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import TrajectoryRubricInput


def record(case='source', keyword='comparison', outcome='failure'):
    app = {'user_side_context': f'Doctor requests {keyword}.',
           'agent_side_state': 'No additional Agent state is required.'}
    rid, leaf, policy, prompt = 'rubric-' + case, case + '/0', 'policy-' + case, 'prompt'
    rubric = {'rubric_id': rid, 'name': keyword, 'applicability': app,
              'required_behavior': 'Perform ' + keyword,
              'scores': {str(i): {**app, 'behavior': f'{keyword} boundary {i}'} for i in range(3)},
              'source_provenance': {'source_case_id': case, 'source_trajectory_id': leaf,
                                    'source_outcome': outcome}}
    raw = {'rubric_id': rid, 'leaf_id': leaf, 'prompt_sha256': prompt, 'raw': 'original generation',
           'rubric': rubric, 'valid': True, 'problems': [], 'intervention_instruction': 'SOURCE GUIDE CANARY',
           'gate': {'protocol': 'evidence_gate_v2', 'passed': True, 'observable': True, 'score_boundaries_clear': True,
                    'source_anchor': {'matches': True}},
           'source_check': {'status': 'PASS', 'first': {'applicable': True}}}
    target = 1.0 if outcome == 'failure' else 0.0
    runs = [{'outcome': target, 'rubric_id': rid, 'leaf_id': leaf,
             'continuation_policy_sha256': policy, 'level': '2' if outcome == 'failure' else '0',
             'seed': i, 'agent_seed': i + 100, 'rerun_index': i} for i in range(2)]
    witness = {'rubric_id': rid, 'decision': 'KEEP', 'stratum': outcome,
               'validated_by_trajectory_id': leaf,
               'tested': [{'leaf_id': leaf, 'result': 'PASS', 'stage1_applicable': True,
                           'reruns': 2, 'outcomes': [target, target], 'rerun_records': runs,
                           'intervention_audit': {'protocol': 'intervention_instruction_audit_v1', 'passed': True,
                               'answer': {'instruction_aligned': True, 'unsupported_context_premise': False,
                                          'conclusion_prescribed': False}}}]}
    return {'case_id': case, 'case_cursor': 1, 'policy_sha256': policy, 'prompt_sha256': prompt,
            'trajectory_rubrics': [raw], 'selection': [{'rubric_id': rid, 'retained': True}],
            'rubric_bank': {'witness': [witness]}}


def tin():
    return TrajectoryRubricInput('new/0', 'new', '', 'Which candidate?', (('A', 'a'), ('B', 'b')),
        (('doctor', 'D1', 1, 'Please compare these choices.'),
         ('agent', 'A1', 1, 'I only name one choice.'),
         ('doctor', 'FINAL', 1, 'My final choice is A.')), 'failure',
         'Please check the missing comparison.', 'tin-sha')


def decision(payload, reuse=True):
    return {'reuse': reuse, 'bank_id': payload['candidates'][0]['bank_id'] if reuse else None,
            'selected_agent_turn_id': 1 if reuse else None,
            'doctor_quote': 'Please compare these choices.' if reuse else None,
            'agent_quote': 'I only name one choice.' if reuse else None,
            'checks': {key: reuse for key in cb.CHECKS}, 'rationale': 'Same specific condition and all score boundaries.'}


def test_exact_content_deduplication_preserves_every_original_source():
    empty = cb.empty_snapshot()
    a, b = record('a'), record('b', outcome='success')
    b['trajectory_rubrics'][0]['rubric']['name'] = 'Other name'
    before = copy.deepcopy([a, b])
    bank = cb.extend_snapshot(empty, [b, a])
    assert [a, b] == before
    assert empty == cb.empty_snapshot() and bank['version'] == 1
    assert bank['parent_sha256'] == empty['sha256']
    assert len(bank['entries']) == 1 and len(bank['entries'][0]['sources']) == 2
    assert cb.extend_snapshot(empty, [a, b]) == bank
    assert cb.extend_snapshot(bank, [])['version'] == 2
    copies = cb.scoring_rubrics(bank)
    assert copies[0].rubric_id == copies[0].rubric['rubric_id'] == bank['entries'][0]['bank_id']
    assert bank['entries'][0]['rubric']['rubric_id'] == 'rubric-a'
    copies[0].rubric['scores']['0']['behavior'] = 'mutated copy'
    cb.validate_snapshot(bank)


def test_same_name_different_score_or_agent_condition_is_not_deduplicated():
    a, b, c = record('a'), record('b'), record('c')
    b['trajectory_rubrics'][0]['rubric']['scores']['1']['behavior'] = 'Different partial boundary'
    for place in [c['trajectory_rubrics'][0]['rubric']['applicability'],
                  *c['trajectory_rubrics'][0]['rubric']['scores'].values()]:
        place['agent_side_state'] = 'Agent has already made a comparison.'
    bank = cb.extend_snapshot(cb.empty_snapshot(), [a, b, c])
    assert len(bank['entries']) == 3


@pytest.mark.parametrize('fault', ['gate', 'observable', 'boundaries', 'anchor', 'source', 'keep', 'audit', 'audit_protocol',
                                   'onehit', 'none', 'duplicate', 'wrongpolicy', 'wrongcase', 'missingrun'])
def test_unverified_or_corrupt_retained_criterion_never_enters_bank(fault):
    r = record(); raw = r['trajectory_rubrics'][0]; w = r['rubric_bank']['witness'][0]; t = w['tested'][0]
    if fault == 'gate': raw['gate']['passed'] = False
    elif fault == 'observable': raw['gate']['observable'] = False
    elif fault == 'boundaries': raw['gate']['score_boundaries_clear'] = False
    elif fault == 'anchor': raw['gate']['source_anchor']['matches'] = False
    elif fault == 'source': raw['source_check']['first']['applicable'] = False
    elif fault == 'keep': w['decision'] = 'DROP'
    elif fault == 'audit': t['intervention_audit']['passed'] = False
    elif fault == 'audit_protocol': t['intervention_audit']['protocol'] = 'old'
    elif fault == 'onehit': t['outcomes'][0] = 0.0
    elif fault == 'none': t['outcomes'][0] = None
    elif fault == 'duplicate': t['rerun_records'][1] = copy.deepcopy(t['rerun_records'][0])
    elif fault == 'wrongpolicy': t['rerun_records'][0]['continuation_policy_sha256'] = 'other'
    elif fault == 'wrongcase': raw['rubric']['source_provenance']['source_case_id'] = 'other'
    elif fault == 'missingrun': t['rerun_records'].pop()
    with pytest.raises(ValueError):
        cb.extend_snapshot(cb.empty_snapshot(), [r])


def test_nonretained_candidate_is_not_admitted():
    r = record(); r['selection'][0]['retained'] = False
    assert cb.extend_snapshot(cb.empty_snapshot(), [r])['entries'] == []


def test_snapshot_validates_contents_even_after_outer_hash_is_recomputed():
    bank = cb.extend_snapshot(cb.empty_snapshot(), [record()])
    bank['entries'][0]['sources'][0]['witness']['decision'] = 'DROP'
    bank['sha256'] = sha256_json({k: v for k, v in bank.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='KEEP'):
        cb.validate_snapshot(bank)


def test_retrieval_deterministic_bounded_and_small_bank_exhaustive():
    bank = cb.extend_snapshot(cb.empty_snapshot(), [record(str(i), keyword=f'concept{i}') for i in range(12)])
    rows = cb.shortlist(bank, 'concept7 concept7', 3)
    assert len(rows) == 3 and rows[0]['rubric']['rubric']['name'] == 'concept7'
    assert rows == cb.shortlist(bank, 'concept7 concept7', 3)
    assert len(cb.shortlist(bank, 'concept7', 64)) == 12
    assert cb.shortlist(bank, '', 3) == bank['entries'][:3]


def test_crosscase_crossoutcome_reuse_has_no_generation_gate_or_rerun_and_keeps_source_proof():
    bank = cb.extend_snapshot(cb.empty_snapshot(), [record(outcome='success')])
    cache, calls = CallCache(None), []
    def matcher(payload):
        calls.append(payload)
        text = json.dumps(payload)
        assert 'SOURCE GUIDE CANARY' not in text and 'source_provenance' not in text
        assert 'ground_truth' not in text and 'original generation' not in text
        return decision(payload)
    result = cb.match_existing(tin(), bank, matcher, cache)
    assert result['selected_agent_turn_id'] == 1
    assert result['sourceproof'][0]['rubric']['rubric']['source_provenance']['source_outcome'] == 'success'
    assert cb.validate_reuse(tin(), bank, result) == result
    assert cb.match_existing(tin(), bank, matcher, cache) == result and len(calls) == 1
    assert set(cache.calls) == {'cumulative_bank_match'}


@pytest.mark.parametrize('fault', ['turn', 'future', 'quote', 'boundary', 'unknown'])
def test_bad_reuse_evidence_is_not_cached(fault):
    bank = cb.extend_snapshot(cb.empty_snapshot(), [record()]); cache = CallCache(None)
    def matcher(payload):
        row = decision(payload)
        if fault == 'turn': row['selected_agent_turn_id'] = 2
        elif fault == 'future': row['doctor_quote'] = 'My final choice is A.'
        elif fault == 'quote': row['agent_quote'] = 'Invented explanation'
        elif fault == 'boundary': row['checks']['same_score_boundaries'] = False
        elif fault == 'unknown': row['bank_id'] = 'unknown'
        return row
    with pytest.raises(ValueError): cb.match_existing(tin(), bank, matcher, cache)
    assert not cache.store


def test_saved_reuse_proof_and_snapshot_must_remain_immutable():
    bank = cb.extend_snapshot(cb.empty_snapshot(), [record()])
    result = cb.match_existing(tin(), bank, decision, CallCache(None))
    result['sourceproof'][0]['policy_sha256'] = 'changed'
    with pytest.raises(ValueError, match='proof'): cb.validate_reuse(tin(), bank, result)


def test_saved_reuse_cannot_be_reassigned_to_changed_context():
    from dataclasses import replace
    bank = cb.extend_snapshot(cb.empty_snapshot(), [record()])
    result = cb.match_existing(tin(), bank, decision, CallCache(None))
    with pytest.raises(ValueError, match='dialogue/feedback'):
        cb.validate_reuse(replace(tin(), feedback='Different retrospective need.'), bank, result)


def test_api_matcher_bounded_schema_repair_records_messages_and_forces_plain_json(monkeypatch):
    from src.online_rl.tree import backends
    bank = cb.extend_snapshot(cb.empty_snapshot(), [record()]); calls = []
    def request(key, model, system, user, settings, schema, name, role):
        calls.append((user, settings))
        payload = json.loads(user.split('\n\nCorrect only')[0])
        return ('{}' if len(calls) == 1 else json.dumps(decision(payload))), {}
    monkeypatch.setattr(backends, '_openrouter_json', request)
    matcher = cb.api_bank_matcher('key', 'same-generator', {'max_tokens': 1000, 'response_format': 'json_schema_strict'}, 'system')
    cache = CallCache(None)
    result = cb.match_existing(tin(), bank, matcher, cache)
    assert result and len(calls) == 2 and calls[0][1]['response_format'] is None
    stored = next(iter(cache.store.values()))['value']
    assert len(stored['audit']['attempts']) == 2 and stored['audit']['system'] == 'system'


def test_api_matcher_second_failure_is_operational_with_audit(monkeypatch, tmp_path):
    from src.online_rl.tree import backends
    bank = cb.extend_snapshot(cb.empty_snapshot(), [record()]); calls = []
    def request(*args):
        calls.append(args)
        return '{}', {}
    monkeypatch.setattr(backends, '_openrouter_json', request)
    matcher = cb.api_bank_matcher('key', 'model', {'max_tokens': 1000}, 'system')
    cache = CallCache(tmp_path/'cache.jsonl')
    with pytest.raises(backends.RubricOutputIncomplete):
        cb.match_existing(tin(), bank, matcher, cache)
    assert len(calls) == 2 and not cache.store
    assert (tmp_path/'cache.errors.jsonl').is_file()


def test_prompt_contract_contains_all_equivalence_fields():
    path = Path(__file__).resolve().parents[1]/'prompts/sherpa/cumulative_bank_match_v1.txt'
    text = path.read_text()
    assert all(field in text for field in cb.CHECKS)
    assert 'No new intervention instruction' in text and 'prior' in text.lower()
