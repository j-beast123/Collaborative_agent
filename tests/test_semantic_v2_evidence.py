"""Offline protocol tests. Synthetic decisions do not establish semantic accuracy."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from src.common.config import IntegrationUnavailable
from src.online_rl.tree.applicability import (
    CONDITION_EVIDENCE_CONTRACT_V2, ScoringContractError, condition_evidence_fields,
    condition_evidence_reviewer, condition_items, condition_state,
    materialize_condition_evidence, state_applicability,
)
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.pilot_evidence_scorer import (
    PUBLIC_EVIDENCE_CONTRACT, PUBLIC_EVIDENCE_CONTRACT_V2, evidence_scorer,
    materialize_public_evidence, public_evidence_fields,
)
from src.online_rl.tree.quality_gate import (
    PUBLIC_GATE_PROTOCOL_V2, PUBLIC_FIELD_REFERENCE_PROTOCOL,
    materialize_public_quality, public_gate_state, public_mc_posthoc_gate,
)
from tests.test_public_mc_evidence import (
    CASE, REPLY, Client, condition_answer, ledger, legacy_rubric, node, quality, rubric, tin,
)


def referenced_ledger(level='2', definition=None, *, minimum=None, completion=None):
    definition = definition or rubric()
    value = ledger(level, definition)
    value.pop('anchor_quote')
    value['anchor_ref'] = 'none' if level == 'UNSCORABLE' else f'score_anchors.{level}'
    value['anchor_matches'] = [] if level == 'UNSCORABLE' else [level]
    value['criterion_status'] = 'ambiguous' if level == 'UNSCORABLE' else 'coherent'
    for row in value['requirements']:
        row['definition_ref'] = row['condition']
        row.pop('definition_quote')
        met = minimum if row['condition'] == 'minimum_performance' else completion
        if met is not None:
            row.update(met=met, evidence_type='response_spans' if met else 'absence',
                       response_quotes=['Its onset is immediate.'] if met else [])
    return value


def referenced_quality():
    value = quality()
    for check in value['checks'].values():
        check.pop('rubric_quotes')
        check['rubric_refs'] = ['required_behavior']
    for probe in value['boundary_probes']:
        probe['anchor_levels'] = [row['level'] for row in probe.pop('anchor_quotes')]
    return value


def referenced_condition(item, state, applicable=True):
    value = condition_answer(item, state, applicable)
    for side in ('doctor_condition', 'agent_condition'):
        value[side].pop('condition')
        value[side]['condition_ref'] = side
    return value


class V2Client(Client):
    def __init__(self, **kwargs):
        kwargs.setdefault('result', referenced_ledger())
        super().__init__(**kwargs)

    def judge(self, stage, state, items, fields, purpose):
        self.calls.append(deepcopy(dict(stage=stage, state=state, items=items, fields=fields)))
        values = {rid: (referenced_condition(item, state, self.app) if stage == 'applicability'
                       else referenced_quality() if stage == PUBLIC_GATE_PROTOCOL_V2 else deepcopy(self.result))
                  for rid, item in items.items()}
        if self.mutate:
            self.mutate(stage, values)
        record = {'model': self.model, 'parsed': deepcopy(values), 'raw': [json.dumps(values)]}
        self.log.append(record)
        return values, record


def scorer(client=None, **kwargs):
    return evidence_scorer(client or V2Client(), evidence_contract=PUBLIC_EVIDENCE_CONTRACT_V2, **kwargs)


@pytest.mark.parametrize(('level', 'minimum', 'completion'), [
    ('0', False, False), ('0', False, True), ('1', True, False), ('2', True, True),
])
def test_public_mc_mapping_retains_exact_level(level, minimum, completion):
    raw = referenced_ledger(level, minimum=minimum, completion=completion)
    resolved = materialize_public_evidence(raw, REPLY, rubric())
    assert resolved['level'] == level
    assert resolved['requirements'][0]['definition_quote'] == rubric()['minimum_performance']
    assert 'definition_ref' in raw['requirements'][0]  # retain unmodified transport evidence


def test_completed_minimum_with_conflicting_completion_is_one_not_zero():
    raw = referenced_ledger('1')
    raw['requirements'][1].update(evidence_type='conflict', response_quotes=['their timing differs.'],
                                  entailment='This synthetic example marks only completion as conflicting.')
    assert materialize_public_evidence(raw, REPLY, rubric())['level'] == '1'
    raw.update(level='0', anchor_ref='score_anchors.0', anchor_matches=['0'])
    with pytest.raises(ValueError, match='M/C'):
        materialize_public_evidence(raw, REPLY, rubric())


@pytest.mark.parametrize('status,matches', [('contradictory', ['2']), ('ambiguous', ['1', '2']), ('coherent', [])])
def test_numeric_claim_cannot_override_ambiguous_or_contradictory_anchors(status, matches):
    raw = referenced_ledger()
    raw.update(criterion_status=status, anchor_matches=matches)
    with pytest.raises(ValueError, match='unchanged coherent written anchor'):
        materialize_public_evidence(raw, REPLY, rubric())


def test_anchor_mc_contradiction_stays_unscorable_despite_completed_mc():
    raw = referenced_ledger('UNSCORABLE', minimum=True, completion=True)
    raw.update(criterion_status='contradictory', anchor_matches=['0'],
               criterion_defect='The written zero anchor also covers this completed M/C response.')
    assert materialize_public_evidence(raw, REPLY, rubric())['level'] == 'UNSCORABLE'
    raw.update(criterion_status='coherent')
    with pytest.raises(ValueError, match='actual scale'):
        materialize_public_evidence(raw, REPLY, rubric())


def test_legacy_declaration_anchor_remains_one_without_public_minimum():
    definition = legacy_rubric()
    definition['scores']['1']['behavior'] = 'Declares the proposed option appropriate.'
    raw = {'level': '1', 'anchor_ref': 'score_anchors.1', 'anchor_matches': ['1'],
        'criterion_status': 'coherent', 'anchor_basis': 'The legacy anchor explicitly grants this declaration.',
        'requirements': [{'condition': 'written_anchor', 'definition_ref': 'score_anchors.1',
            'met': True, 'evidence_type': 'response_spans', 'response_quotes': ['That option is appropriate.'],
            'entailment': 'The response performs exactly the declaration this legacy anchor requires.'}],
        'remaining_gap': 'The fuller comparison required by2 remains absent.', 'criterion_defect': ''}
    result = scorer(V2Client(result=raw)).score_behavior(CASE, [], [definition], 'That option is appropriate.')
    assert result['candidate-id']['hard'] == 1
    fields = public_evidence_fields([definition], evidence_contract=PUBLIC_EVIDENCE_CONTRACT_V2)
    assert fields['requirements']['items']['properties']['condition']['enum'] == ['written_anchor']
    assert fields['requirements']['items']['properties']['definition_ref']['enum'] == [f'score_anchors.{i}' for i in range(3)]


@pytest.mark.parametrize('mutate', [
    lambda row: row.update(anchor_ref='score_anchors.0'),
    lambda row: row['requirements'][0].update(definition_ref='completion_condition'),
    lambda row: row['requirements'][0].update(response_quotes=['A span from the sibling response.']),
])
def test_invalid_refs_or_foreign_evidence_remain_execution_errors(mutate):
    raw = referenced_ledger()
    mutate(raw)
    with pytest.raises(IntegrationUnavailable):
        scorer(V2Client(result=raw)).score_behavior(CASE, [], [rubric()], REPLY)


def test_batch_evidence_is_bound_to_each_response():
    with pytest.raises(IntegrationUnavailable, match='own target response'):
        scorer(response_batch_size=2).score_behavior_batch(CASE, [], [rubric()],
            [{'response_id': 'first', 'reply': REPLY}, {'response_id': 'second', 'reply': 'No such evidence.'}])


def test_applicability_references_preserve_missing_premise_decision():
    definition = rubric()
    definition['applicability']['user_side_context'] = 'The Doctor misinterprets dates whose administrative origin is visible.'
    state = condition_state(CASE, [{'speaker': 'doctor', 'text': 'The listed years describe disease timing.'}])
    item = condition_items([definition])['candidate-id']
    raw = referenced_condition(item, state, False)
    raw['doctor_condition'].update(quotes=[], unmet_condition='Administrative origin is not established.',
                                   entailment='The timing claim supplies no evidence of the dates origin.')
    resolved = materialize_condition_evidence(raw, state, item)
    assert not resolved['applicable']
    assert resolved['doctor_condition']['condition'] == item['doctor_condition']
    raw['doctor_condition']['condition_ref'] = 'agent_condition'
    with pytest.raises(ScoringContractError, match='wrong side'):
        materialize_condition_evidence(raw, state, item)


def test_v2_app_cache_aliases_and_hidden_belief_identity_without_transmission(tmp_path):
    review = condition_evidence_reviewer(V2Client(), contract=CONDITION_EVIDENCE_CONTRACT_V2)
    first = rubric()
    second = {**first, 'rubric_id': 'bank-alias'}
    saved = node('n')
    cache = CallCache(tmp_path/'calls.jsonl')
    a = state_applicability(CASE, saved, first, review, cache)
    b = state_applicability(CASE, saved, second, review, cache)
    assert a['provenance']['cache_key'] == b['provenance']['cache_key']
    assert a['stage1_contract'] == CONDITION_EVIDENCE_CONTRACT_V2
    assert 'target_agent_reply' not in json.dumps(a['evidence_validation']['requests'])
    changed = replace(saved, trace={'pre_response_belief': {'hidden': 'SECOND_HIDDEN_SENTINEL'}})
    c = state_applicability(CASE, changed, first, review, cache)
    assert a['provenance']['cache_key'] != c['provenance']['cache_key']
    assert 'SECOND_HIDDEN_SENTINEL' not in json.dumps(c['evidence_validation']['requests'])


def test_gate_field_references_materialize_and_reject_unknown_evidence():
    state = public_gate_state(tin(), rubric())
    raw = referenced_quality()
    resolved, contradictions = materialize_public_quality(raw, state)
    assert not contradictions
    assert resolved['checks']['observability']['rubric_quotes'] == [rubric()['required_behavior']]
    raw['checks']['observability']['rubric_refs'] = ['source_provenance.analysis']
    with pytest.raises(ValueError, match='unknown visible'):
        materialize_public_quality(raw, state)


def test_gate_boundary_self_contradiction_blocks_even_if_flags_pass():
    raw = referenced_quality()
    raw['boundary_probes'][0]['matching_levels'] = []
    resolved, reasons = materialize_public_quality(raw, public_gate_state(tin(), rubric()))
    assert resolved['score_boundaries_clear'] is True
    assert 'boundary_flag_contradicts_evidence' in reasons


def test_v2_gate_preserves_failure_source_one_and_does_not_send_expected_score():
    definition = rubric()
    definition['source_provenance']['analysis']['source_score'] = 1
    client = V2Client(result=referenced_ledger('1'))
    result = public_mc_posthoc_gate(client, {CASE['case_id']: CASE},
        protocol=PUBLIC_GATE_PROTOCOL_V2, rubric_quote_protocol=PUBLIC_FIELD_REFERENCE_PROTOCOL)(tin('failure'), definition)
    assert result['protocol'] == PUBLIC_GATE_PROTOCOL_V2
    assert result['source_anchor']['observed_level'] == '1'
    assert result['source_anchor']['generator_declared_level'] == 1
    assert result['source_anchor']['expected_level'] == '0'
    assert not result['passed'] and 'source_anchor_mismatch' in result['reasons']
    source_request = client.calls[-1]
    blob = json.dumps(source_request)
    for forbidden in ('source_score', 'expected_level', 'source_provenance', 'outcome', 'boundary_probes'):
        assert forbidden not in json.dumps(source_request['state'])
        assert forbidden not in json.dumps({k: {a: b for a, b in v.items() if a != 'instructions'}
                                           for k, v in source_request['items'].items()})
    assert 'rubric_field_catalog' not in blob


def test_v2_and_v1_identities_are_distinct_and_model_settings_are_bound():
    old = evidence_scorer(Client(), evidence_contract=PUBLIC_EVIDENCE_CONTRACT)
    new = scorer()
    changed = scorer(V2Client(max_tokens=3072))
    assert len({old.protocol, new.protocol, changed.protocol}) == 3
    assert old.review_state.protocol != new.review_state.protocol
    assert new.behavior_cache_contract == 'public_behavior_content_alias_v1'
    assert condition_evidence_fields()['doctor_condition']['properties'].get('condition')
    assert condition_evidence_fields(CONDITION_EVIDENCE_CONTRACT_V2)['doctor_condition']['properties']['condition_ref']['enum'] == ['doctor_condition']
