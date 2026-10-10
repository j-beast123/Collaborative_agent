"""CPU-only transport and public-criterion -> sibling reward structural checks.

Judges below return synthetic expected labels. They test plumbing and masking,
not clinical semantics or the quality of model-generated criteria.
"""
import copy
import hashlib
import importlib.util
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import cumulative_bank as CB
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.pilot_evidence_scorer import evidence_scorer, PUBLIC_EVIDENCE_CONTRACT
from src.online_rl.tree.posthoc import case_view
from src.online_rl.tree.records import to_dialogue
from src.online_rl.tree.scoring import score_cumulative_nodes
from tests.test_not_retrieved_reward_pipeline import reward_and_capture
from tests.test_public_mc_contract import public_record
from tests.test_tree_v1 import CASES, Log, tree_for


class SyntheticPublicClient:
    model = 'synthetic-no-api'
    system = 'Synthetic public MC test client.'
    protocol_sha256 = 'synthetic-public-contract'
    max_tokens = 2048
    timeout_s = 10

    def __init__(self, roles, response_ids, *, fail=False):
        self.roles, self.response_ids, self.fail = roles, response_ids, fail
        self.calls = []
        self.log = []

    def judge(self, stage, state, items, fields, purpose):
        assert stage == 'applicability'
        assert 'target_agent_reply' not in json.dumps(state)
        assert all('minimum_performance' not in item and 'score_anchors' not in item for item in items.values())
        self.calls.append((stage, copy.deepcopy(state), copy.deepcopy(items)))
        prefix = state['dialogue_before_target_reply']
        values = {}
        for rid, item in items.items():
            yes = self.roles[item['rubric']] != 'na'
            values[rid] = {'applicable': yes, 'evidence_basis': 'Synthetic trigger.' if yes else '',
                'missing_requirement': '' if yes else 'Synthetic unmet condition.',
                'doctor_condition': {'condition': item['doctor_condition'], 'status': 'supported' if yes else 'unmet',
                    'quotes': [{'turn_id': prefix[0]['turn_id'], 'speaker': 'doctor', 'quote': prefix[0]['text']}] if yes else [],
                    'inspected_turn_ids': [r['turn_id'] for r in prefix], 'entailment': 'Fixture-specified trigger status.',
                    'unmet_condition': '' if yes else 'Synthetic unmet condition.'},
                'agent_condition': {'condition': item['agent_condition'], 'status': 'not_required', 'quotes': [],
                    'inspected_turn_ids': [], 'entailment': 'Written condition requires no additional state.', 'unmet_condition': ''}}
        return values, {}

    def judge_batch(self, stage, state, responses, items, fields, purpose):
        assert stage == 'level' and len(responses) == 4
        self.calls.append((stage, copy.deepcopy(state), copy.deepcopy(items)))
        if self.fail:
            raise IntegrationUnavailable('Synthetic execution failure; not a behavioral zero.')
        values = {}
        for response in responses:
            idx = self.response_ids.index(response['response_id'])
            values[response['response_id']] = {}
            for rid, item in items.items():
                assert all(k in item for k in ('required_behavior', 'minimum_performance', 'completion_condition', 'criterion_version'))
                role = self.roles[item['rubric']]
                level = ('UNSCORABLE' if idx == 0 else '2') if role == 'unscorable' else (
                    '0' if role == 'zero' else ('0', '1', '2', '2')[idx])
                ledger = []
                for field, met in [('minimum_performance', level in ('1', '2')), ('completion_condition', level == '2')]:
                    ledger.append({'condition': field, 'definition_quote': item[field], 'met': met,
                        'evidence_type': 'response_spans' if met else 'absence',
                        'response_quotes': [response['target_agent_reply']] if met else [],
                        'entailment': 'Synthetic expected behavior evidence for this response only.'})
                values[response['response_id']][rid] = {'level': level,
                    'anchor_quote': '' if level == 'UNSCORABLE' else item['score_anchors'][level],
                    'anchor_basis': 'Synthetic frozen fixture label, not clinical gold.', 'requirements': ledger,
                    'remaining_gap': '' if level == '2' else 'The synthetic mandatory completion is absent.',
                    'criterion_defect': 'Synthetic uncovered response.' if level == 'UNSCORABLE' else ''}
        return values, {}


def pipeline(*, all_unscored=False, fail=False):
    tree, _ = tree_for(Log(), rounds=1, branching=4)
    outcomes = (0.0, 0.0, 1.0, 1.0) if all_unscored else (0.0, 1.0, 0.0, 1.0)
    tree = replace(tree, leaves=tuple(replace(leaf, outcome=value) for leaf, value in zip(tree.leaves, outcomes)))
    bank = CB.extend_snapshot(CB.empty_snapshot(), [public_record(f'public{i}', keyword=f'criterion{i}') for i in range(9)])
    query = json.dumps({'case': case_view(CASES[0]), 'prefix': to_dialogue(tree.nodes[0].prefix)}, sort_keys=True, ensure_ascii=False)
    selected = [e['bank_id'] for e in CB.shortlist(bank, query, 8)]
    roles = dict.fromkeys(selected, 'na')
    roles[selected[0]] = 'unscorable' if all_unscored else 'zero'
    if not all_unscored:
        roles[selected[1]], roles[selected[2]] = 'varying', 'unscorable'
    roles_by_name = {e['rubric']['rubric']['name']: roles[e['bank_id']] for e in bank['entries'] if e['bank_id'] in roles}
    client = SyntheticPublicClient(roles_by_name, [n.node_id for n in tree.nodes], fail=fail)
    scorer = evidence_scorer(client, response_batch_size=4, evidence_contract=PUBLIC_EVIDENCE_CONTRACT)
    cache = CallCache(None)
    scores, retrieval = score_cumulative_nodes(CASES[0], list(tree.nodes), bank, scorer, cache, top_k=8)
    # This production cumulative function delegates its selected public criteria to score_nodes.
    assert len(retrieval) == 1 and len(scores) == 36
    assert sum(c[0] == 'applicability' for c in client.calls) == 8  # once per criterion/state, not4x
    assert sum(c[0] == 'level' for c in client.calls) == 1
    before = len(client.calls)
    assert score_cumulative_nodes(CASES[0], list(tree.nodes), bank, scorer, cache, top_k=8)[0] == scores
    assert len(client.calls) == before
    return tree, bank, scores, client


def test_public_rubric_reaches_shared_applicability_independent_scores_reward_and_advantages():
    tree, bank, scores, client = pipeline()
    assert sum(s.scoring_status == 'not_retrieved' for s in scores) == 4
    assert sum(s.scoring_status == 'not_applicable' for s in scores) == 20
    assert sum(s.scoring_status == 'group_unscorable' for s in scores) == 4
    assert sum(s.scoring_status == 'scored' and s.score == 0 for s in scores) == 5
    assert all(s.score is None for s in scores if s.scoring_status != 'scored')
    targets, capture = reward_and_capture(tree, bank, scores, (0.5, 0.5))
    assert [targets['nodes'][n.node_id]['R_A'] for n in tree.nodes] == [0.0, 0.25, 0.5, 0.5]
    assert [r['reward'] for r in capture.groups[0]] == [0.0, 0.625, 0.25, 0.75]
    assert capture.advantages == [[-0.40625, 0.21875, -0.15625, 0.34375]]
    wire = json.dumps(client.calls)
    assert 'minimum_performance' in wire and 'criterion_version' in wire
    for hidden in ('source_provenance', 'source_outcome', 'intervention_instruction', 'ground_truth'):
        assert hidden not in wire


@pytest.mark.parametrize('weights,rewards,advantages', [
    ((0.5, 0.5), [0.0, 0.0, 0.5, 0.5], [-0.25, -0.25, 0.25, 0.25]),
    ((0.0, 1.0), [None] * 4, [None] * 4),
])
def test_all_unscored_public_criteria_never_become_behavioral_zero(weights, rewards, advantages):
    tree, bank, scores, _ = pipeline(all_unscored=True)
    assert all(s.score is None for s in scores)
    targets, capture = reward_and_capture(tree, bank, scores, weights)
    assert all(targets['nodes'][n.node_id]['R_A'] is None for n in tree.nodes)
    assert [r['reward'] for r in capture.groups[0]] == rewards
    assert capture.advantages == [advantages]


def test_execution_error_aborts_before_reward_or_training_handoff():
    with pytest.raises(IntegrationUnavailable, match='Synthetic execution failure'):
        pipeline(fail=True)


def pilot_module():
    path = Path(__file__).resolve().parents[2] / 'pilot_support.py'
    spec = importlib.util.spec_from_file_location('isolated_semantic_pilot_transport_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_records_native_payload_and_metadata_without_reimplementing_call(monkeypatch, tmp_path):
    from src.collaboration import generator as G
    module = pilot_module()
    monkeypatch.setattr(module, 'OUT', tmp_path)
    monkeypatch.setenv('API_USAGE_LOG', str(tmp_path / 'usage.jsonl'))
    sent = []
    response = {'model': 'resolved-model', 'provider': 'synthetic-provider', 'usage': {'prompt_tokens': 5, 'cost': 0},
                'choices': [{'message': {'content': '  completed  '}, 'finish_reason': 'stop', 'native_finish_reason': 'end'}]}
    def fake(request, **kwargs):
        sent.append((request, kwargs))
        return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(module.urllib.request, 'urlopen', fake)
    meta = {}
    result = module.recorded_doctor_call(module.Transport('synthetic-only'), G.doctor_call,
        ' synthetic-secret ', 'requested-model', 'system', 'user', 1234, 0.7,
        extra={'reasoning': {'effort': 'low'}, 'response_format': {'type': 'json_object'}}, meta=meta)
    assert result == 'completed'
    assert module.urllib.request.urlopen is fake
    request, kwargs = sent[0]
    assert kwargs == {'timeout': 180}
    payload = json.loads(request.data)
    assert payload == {'model': 'requested-model', 'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'user'}],
        'temperature': 0.7, 'max_tokens': 1234, 'usage': {'include': True},
        'reasoning': {'effort': 'low'}, 'response_format': {'type': 'json_object'}}
    assert meta['model'] == 'resolved-model' and meta['provider'] == 'synthetic-provider'
    assert meta['finish_reason'] == 'stop' and meta['usage'] == response['usage']
    written = json.loads((tmp_path / 'http/0001.request.json').read_text())
    assert written['payload'] == payload
    assert written['request_body_sha256'] == hashlib.sha256(request.data).hexdigest()
    assert 'synthetic-secret' not in json.dumps(written)
    assert json.loads((tmp_path / 'usage.jsonl').read_text())['purpose'] == 'collaboration'


def test_pilot_preserves_native_provider_error_and_restores_transport(monkeypatch, tmp_path):
    from src.collaboration import generator as G
    module = pilot_module()
    monkeypatch.setattr(module, 'OUT', tmp_path)
    attempts = []
    def fake(request, **kwargs):
        return io.BytesIO(json.dumps({'error': {'code': 503, 'message': 'synthetic unavailable'}}).encode())
    monkeypatch.setattr(module.urllib.request, 'urlopen', fake)
    def no_wait(self, kind, reason, retry_after=None):
        attempts.append((kind, reason))
        return False
    monkeypatch.setattr(G._retry.Retry, 'wait', no_wait)
    with pytest.raises(G._ProviderError):
        module.recorded_doctor_call(module.Transport('synthetic-error'), G.doctor_call,
            'unused', 'synthetic-model', 'system', 'user', 128)
    assert module.urllib.request.urlopen is fake
    assert attempts and attempts[0][0] == 'transient'
    assert json.loads((tmp_path / 'http/0001.response.json').read_text())['error']['code'] == 503
