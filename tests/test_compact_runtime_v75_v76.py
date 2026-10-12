"""Offline integration checks for the isolated v7.5/v7.6 compact scorer.

Synthetic judge replies check transport and contracts, not clinical accuracy.
"""
import copy
import json
from pathlib import Path

import pytest

from scripts.online_rl.run_tree import evidence_gate_identity, make_gate, make_scorer
from src.online_rl.tree.applicability import ApplicabilityUncertain
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.compact_audit import _app, _validate_relation, app_schema, prepare
from src.online_rl.tree.compact_runtime import upgrade
from src.online_rl.tree.experiment import load_config
from src.online_rl.tree.luna import LunaDecisionClient
from src.online_rl.tree.scoring import score_cumulative_nodes, score_nodes
from tests.test_cumulative_trainer import snapshot
from tests.test_scoring_batch import CASE, KEPT, node


CFG = 'configs/experiments/train120_v7_fresh.json'


def _cfg():
    cfg = load_config(CFG)
    cfg['judges']['compact_runtime'] = 'v7_5_v7_6'
    return cfg


def _relation(kind='none', *, status='not_required', lifecycle='open'):
    return {'condition_relation': {'relation_type': kind, 'subject': '', 'object': '',
                                    'temporal_target': ''},
            'visible_relation': {'relation_type': 'none', 'subject': '', 'object': '',
                                 'temporal_target': ''},
            'relation_status': status, 'relation_evidence': [],
            'active_request_required': False, 'request_lifecycle': lifecycle,
            'explanation': 'The visible request and declared condition were compared.'}


def test_no_required_relation_allows_an_evidenced_unrelated_visible_relation():
    quote = 'What are your thoughts on the alternative?'
    audit = _relation(lifecycle='not_required')
    audit['visible_relation']['relation_type'] = 'other'
    audit['relation_evidence'] = [{'turn_id': 'prefix:0', 'quote': quote}]
    state = {'dialogue_before_target_reply': [
        {'turn_id': 'prefix:0', 'speaker': 'doctor', 'text': quote}]}
    answer = {'doctor_condition': [{'status': 'supported'}]}
    _validate_relation(audit, state, {'applicable': True, 'request_state': 'not_required'}, answer)


def test_absent_request_discards_redundant_exact_visible_different_request():
    quote = 'Could you weigh in on whether another option has been implicated?'
    packet = {'state': {'dialogue_before_target_reply': [
                  {'turn_id': 'prefix:0', 'speaker': 'doctor', 'text': quote}]},
              'items': {'criterion': {
                  'doctor_condition': 'The Doctor asks about nerve supply.',
                  'agent_condition': 'No additional recorded Agent state is required.'}}}
    raw = {'answers': {'criterion': {
        'doctor_condition': [{
            'clause': 'The Doctor asks about nerve supply.', 'status': 'unmet',
            'evidence': [], 'inspected': ['prefix:0'],
            'note': 'The visible request concerns a different relation.'}],
        'agent_condition': [{
            'clause': 'No additional recorded Agent state is required.',
            'status': 'not_required', 'evidence': [], 'inspected': [],
            'note': 'No prior Agent state is required.'}],
        'request_state': 'absent',
        'request_evidence': [{'turn_id': 'prefix:0', 'quote': quote}],
        'remaining_request': '',
        'relation_audit': {
            'condition_relation': {'relation_type': 'other', 'subject': 'structures',
                                   'object': 'nerve supply', 'temporal_target': 'current'},
            'visible_relation': {'relation_type': 'other', 'subject': 'another option',
                                 'object': 'implication', 'temporal_target': 'historical'},
            'relation_status': 'different',
            'relation_evidence': [{'turn_id': 'prefix:0', 'quote': quote}],
            'active_request_required': False, 'request_lifecycle': 'absent',
            'explanation': 'The visible request is about a different relation.'}}}}
    result = _app(raw, packet)
    row = result['applicability']['criterion']
    assert row['status'] == 'not_applicable'
    assert row['request_state'] == 'absent'
    assert row['request_evidence'] == []


def test_absent_request_still_rejects_invented_quote_before_discarding():
    raw_quote = 'This quote was never said.'
    packet = {'state': {'dialogue_before_target_reply': [
                  {'turn_id': 'prefix:0', 'speaker': 'doctor', 'text': 'Actual request.'}]},
              'items': {'criterion': {
                  'doctor_condition': 'The Doctor asks about nerve supply.',
                  'agent_condition': 'No additional recorded Agent state is required.'}}}
    raw = {'answers': {'criterion': {
        'doctor_condition': [{
            'clause': 'The Doctor asks about nerve supply.', 'status': 'unmet',
            'evidence': [], 'inspected': ['prefix:0'], 'note': 'Not present.'}],
        'agent_condition': [{
            'clause': 'No additional recorded Agent state is required.',
            'status': 'not_required', 'evidence': [], 'inspected': [],
            'note': 'No state required.'}],
        'request_state': 'absent',
        'request_evidence': [{'turn_id': 'prefix:0', 'quote': raw_quote}],
        'remaining_request': ''}}}
    with pytest.raises(ValueError, match='Request-state quote is not exact prefix evidence'):
        _app(raw, packet)


def test_exact_quote_accepts_only_typographic_unicode_equivalents():
    from src.online_rl.tree.compact_audit import _quote_in_text
    source = "I'm leaning toward HBV - compare HBV versus HCV."
    assert _quote_in_text("I’m leaning toward HBV – compare HBV versus HCV.", source)
    assert not _quote_in_text("I'm leaning toward HAV - compare HBV versus HCV.", source)


class Transport:
    def __init__(self, *, uncertain=False):
        self.calls = []
        self.uncertain = uncertain

    def __call__(self, url, payload, key, timeout):
        self.calls.append(copy.deepcopy(payload))
        packet = json.loads(payload['messages'][1]['content'])
        assert 'correct_option' not in json.dumps(packet)
        if 'responses' not in packet:
            assert 'target_agent_reply' not in json.dumps(packet)
            assert 'minimum_performance' not in json.dumps(packet['items'])
            prefix = packet['state']['dialogue_before_target_reply']
            first = prefix[0]
            span = {'turn_id': first['turn_id'], 'quote': first['text']}
            answers = {}
            for rid, item in packet['items'].items():
                answers[rid] = {
                    'doctor_condition': [{
                        'clause': item['doctor_condition'],
                        'status': 'uncertain' if self.uncertain else 'supported',
                        'evidence': [] if self.uncertain else [span],
                        'inspected': [turn['turn_id'] for turn in prefix],
                        'note': 'Synthetic pre-reply condition check.'}],
                    'agent_condition': [{
                        'clause': item['agent_condition'], 'status': 'not_required',
                        'evidence': [], 'inspected': [],
                        'note': 'No earlier Agent state is required.'}],
                    'request_state': 'uncertain' if self.uncertain else 'unresolved',
                    'request_evidence': [span],
                    'remaining_request': 'Whether the request remains open is unclear.'
                        if self.uncertain else 'Please explain the reasoning.',
                    'relation_audit': _relation('other', status='uncertain', lifecycle='uncertain')
                        if self.uncertain else _relation(),
                }
        else:
            answers = {}
            for response in packet['responses']:
                reply = response['reply']
                level = reply[-1] if reply[-1] in '012' else '2'
                answers[response['response_id']] = {}
                for rid, rubric in packet['rubric_items'].items():
                    refs = ['score_anchors.0', 'score_anchors.1', 'score_anchors.2']
                    refs += [key for key in ('minimum_performance', 'completion_condition')
                             if key in rubric]
                    checks = []
                    for ref in refs:
                        met = (ref == f'score_anchors.{level}')
                        checks.append({'ref': ref, 'met': met,
                                       'cites': [0] if met and level != '0' else [],
                                       'why': 'Synthetic direct anchor check.'})
                    answers[response['response_id']][rid] = {
                        'level': level, 'quotes': [reply], 'checks': checks,
                        'gap': '' if level == '2' else 'Written completion is absent.',
                        'defect': ''}
        return {'choices': [{'finish_reason': 'stop',
                             'message': {'content': json.dumps({'answers': answers})}}],
                'usage': {}}


def _scorer(*, uncertain=False):
    cfg = _cfg()
    transport = Transport(uncertain=uncertain)
    client = LunaDecisionClient(model=cfg['judges']['scorer_model'], key='OFFLINE',
        system=Path(cfg['judges']['scorer_prompt_file']).read_text(),
        max_tokens=4096, post=transport)
    return cfg, transport, client, make_scorer(cfg, client)


def test_v75_v76_wire_changes_only_stage_contract_and_system_supplements():
    state = {'case_context': {}, 'dialogue_before_target_reply': [], 'case_evidence': []}
    app = {'stage': 'applicability', 'state': state,
           'items': {'r': {'rubric': 'criterion', 'doctor_condition': 'request',
                           'agent_condition': ''}}}
    behavior = {'stage': 'level', 'shared_pre_reply_state': state,
                'rubric_items': {'r': {'rubric': 'criterion',
                                     'score_anchors': {'0': 'none', '1': 'partial', '2': 'done'}}},
                'responses': [{'response_id': 'n', 'reply': 'observed'}]}
    for packet, kind in ((app, 'applicability'), (behavior, 'behavior')):
        base = {'model': 'offline', 'max_tokens': 4096,
                'messages': [{'role': 'system', 'content': 'Base system'},
                             {'role': 'user', 'content': json.dumps(packet)}]}
        prior = upgrade(base, revision='v7_3')
        current = upgrade(base, revision='v7_5_v7_6')
        assert json.loads(current['messages'][1]['content']) == json.loads(prior['messages'][1]['content'])
        assert current['model'] == prior['model'] == 'offline'
        assert current['max_tokens'] == prior['max_tokens'] == 4096
        semantics = Path(f'prompts/sherpa/compact_{kind}_semantics_v3.txt').read_text()
        assert semantics in current['messages'][0]['content']
        items = json.loads(current['messages'][1]['content'])[
            'rubric_items' if kind == 'behavior' else 'items']
        assert all(semantics in item['instructions'] for item in items.values())
        schema = current['response_format']['json_schema']['schema']
        if kind == 'applicability':
            answer = schema['properties']['answers']['properties']['r']
            assert 'relation_audit' in answer['required']
            assert 'relation_audit' not in prior['response_format']['json_schema']['schema'][
                'properties']['answers']['properties']['r']['required']
            assert Path('prompts/sherpa/compact_applicability_relation_v7_4.txt').read_text() in current['messages'][0]['content']
            assert Path('prompts/sherpa/compact_applicability_referent_v7_5.txt').read_text() in current['messages'][0]['content']
        else:
            assert schema == prior['response_format']['json_schema']['schema']
            assert Path('prompts/sherpa/compact_behavior_referent_v7_6.txt').read_text() in current['messages'][0]['content']


def test_selector_gate_and_sibling_scoring_preserve_real_zero():
    cfg, transport, client, scorer = _scorer()
    assert scorer.compact_revision == 'v7_5_v7_6'
    assert scorer.review_identity['contract'] == 'compact_condition_v7_5'
    gate_client = LunaDecisionClient(model=cfg['judges']['gate_model'], key='OFFLINE',
        system=Path(cfg['judges']['gate_prompt_file']).read_text(),
        max_tokens=cfg['judges']['gate_max_tokens'], post=transport)
    gate = make_gate(cfg, gate_client, [CASE])
    assert gate.settings['blind_source_protocol']
    assert evidence_gate_identity(cfg)['contract_sha256'] == gate.settings['contract_sha256']

    nodes = [node(0), node(2)]
    scores = score_nodes(CASE, nodes, [KEPT], scorer, CallCache(None))
    assert [row.score for row in scores] == [0, 1]
    app_calls = [p for p in transport.calls if 'responses' not in json.loads(p['messages'][1]['content'])]
    behavior_calls = [p for p in transport.calls if 'responses' in json.loads(p['messages'][1]['content'])]
    # One shared prefix produces one applicability request and one fused
    # sibling behavior request.
    assert len(app_calls) == 1 and len(behavior_calls) == 1
    assert all(all('relation_audit' in item['required'] for item in
                   p['response_format']['json_schema']['schema']['properties']
                    ['answers']['properties'].values()) for p in app_calls)


def test_uncertain_stage1_blocks_scoring_without_na_or_behavior_call():
    _, transport, _, scorer = _scorer(uncertain=True)
    rows = score_nodes(CASE, [node(0), node(1)], [KEPT], scorer, CallCache(None))
    assert all(row.scoring_status == 'applicability_error' and row.score is None
               and row.match is False for row in rows)
    assert len(transport.calls) == 1
    assert 'responses' not in json.loads(transport.calls[0]['messages'][1]['content'])


def test_conjunctive_definite_unmet_outweighs_other_uncertainty():
    state = {'case_context': {}, 'case_evidence': [], 'dialogue_before_target_reply': [
        {'turn_id': 'prefix:0', 'speaker': 'doctor', 'text': 'Please explain.'}]}
    packet = {'state': state, 'items': {'r': {'doctor_condition': 'request',
                                             'agent_condition': 'earlier behavior'}}}
    raw = {'answers': {'r': {
        'doctor_condition': [{'clause': 'request', 'status': 'unmet', 'evidence': [],
                              'inspected': ['prefix:0'], 'note': 'Different request.'}],
        'agent_condition': [{'clause': 'earlier behavior', 'status': 'uncertain',
                             'evidence': [], 'inspected': ['prefix:0'], 'note': 'Unclear.'}],
        'request_state': 'absent', 'request_evidence': [], 'remaining_request': '',
        'relation_audit': _relation('other', status='different', lifecycle='absent')}}}
    assert _app(raw, packet)['applicability']['r']['applicable'] is False


def test_full_bank_path_uses_same_versioned_scorer_and_checks_every_item():
    _, transport, _, scorer = _scorer()
    bank = snapshot('one', 'two')
    scores, retrieval = score_cumulative_nodes(CASE, [node(0), node(2)], bank,
        scorer, CallCache(None), scoring_scope='all')
    assert len(scores) == 4
    assert retrieval[0]['bank_size'] == 2
    assert retrieval[0]['candidate_selection'] == 'full_cumulative_bank'
    assert not any(row.scoring_status == 'not_retrieved' for row in scores)
    assert [row.score for row in scores if row.response_node_id == 'c1/0'] == [0, 0]
    app_calls = [p for p in transport.calls if 'responses' not in json.loads(p['messages'][1]['content'])]
    # All bank criteria for the same visible state are judged together.
    assert len(app_calls) == 1
    packet = json.loads(app_calls[0]['messages'][1]['content'])
    assert len(packet['items']) == 2
