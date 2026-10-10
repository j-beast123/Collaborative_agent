"""Scope contracts use synthetic judgments, not an accuracy oracle or heldout data."""
from copy import deepcopy

import pytest

from src.common.config import IntegrationUnavailable
from src.online_rl.tree.applicability import (
    CONDITION_EVIDENCE_CONTRACT_V2, CONDITION_EVIDENCE_CONTRACT_V3, ScoringContractError,
    condition_evidence_reviewer, condition_items, condition_state, materialize_condition_evidence,
)
from src.online_rl.tree.pilot_evidence_scorer import (
    PUBLIC_EVIDENCE_CONTRACT_V2, PUBLIC_EVIDENCE_CONTRACT_V3, evidence_scorer,
    materialize_public_evidence, validate_task_frame,
)
from src.online_rl.tree.quality_gate import PUBLIC_GATE_PROTOCOL_V3, public_mc_posthoc_gate
from tests.test_semantic_v2_evidence import (
    CASE, REPLY, V2Client, referenced_condition, referenced_ledger, referenced_quality, rubric, tin,
)

PREFIX = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'Compare the timing of alpha and beta.'}]
STATE = condition_state(CASE, PREFIX)


def frame(state=None):
    state = state or STATE
    source = state['dialogue_before_target_reply'][-1]
    return {'requested_operator': 'comparison', 'requested_targets': ['alpha', 'beta'], 'reference_targets': [],
        'request_quotes': [{'turn_id': source['turn_id'], 'speaker': source['speaker'], 'quote': source['text']}],
        'target_assessments': [{'target': target, 'coverage': 'addressed', 'response_operators': ['comparison'],
            'response_quotes': [quote]} for target, quote in [('alpha', 'Its onset is immediate.'),
                ('beta', 'The alternative develops later; their timing differs.')]],
        'scope_met': True, 'reason': 'Both requested alternatives are compared on the requested dimension.'}


def answer(level='2', state=None):
    row = referenced_ledger(level)
    row['task_frame'] = frame(state)
    return row


class V3Client(V2Client):
    def __init__(self, **kwargs):
        kwargs.setdefault('result', answer())
        super().__init__(**kwargs)

    def judge(self, stage, state, items, fields, purpose):
        self.calls.append(deepcopy(dict(stage=stage, state=state, items=items, fields=fields)))
        if stage == PUBLIC_GATE_PROTOCOL_V3:
            values = {rid: referenced_quality() for rid in items}
        elif stage == 'applicability':
            values = {rid: referenced_condition(item, state, self.app) for rid, item in items.items()}
        else:
            values = {rid: deepcopy(self.result) for rid in items}
            for value in values.values():
                value['task_frame']['request_quotes'] = frame(state)['request_quotes']
        if self.mutate:
            self.mutate(stage, values)
        record = {'model': self.model, 'parsed': deepcopy(values)}
        self.log.append(record)
        return values, record


def scorer(client=None, **kwargs):
    return evidence_scorer(client or V3Client(), evidence_contract=PUBLIC_EVIDENCE_CONTRACT_V3, **kwargs)


def test_v3_records_requested_scope_separately_from_literal_anchor_evidence():
    result = scorer().score_behavior(CASE, PREFIX, [rubric()], REPLY)['candidate-id']
    assert result['hard'] == 2
    assert result['evidence_validation']['task_frame']['requested_targets'] == ['alpha', 'beta']
    assert result['stage2']['anchor_quote'] == rubric()['scores']['2']['behavior']
    assert result['evidence_validation']['ledger_schema_revision'] == 'requested_task_scope_ledger_v4'


def test_frequency_only_cannot_certify_requested_possibility():
    evidence = frame()
    evidence['requested_operator'] = 'possibility_any'
    for row in evidence['target_assessments']:
        row['response_operators'] = ['frequency_typicality']
    with pytest.raises(ValueError, match='scope flag contradicts'):
        validate_task_frame(evidence, REPLY, STATE, level='2')
    evidence['scope_met'] = False
    with pytest.raises(ValueError, match='Level2'):
        validate_task_frame(evidence, REPLY, STATE, level='2')
    # Recognized incomplete scope alone does not choose a numeric level.
    validate_task_frame(evidence, REPLY, STATE, level='1')
    validate_task_frame(evidence, REPLY, STATE, level='UNSCORABLE')


def test_scope_operator_checks_use_declared_semantics_not_reply_keyword_rules():
    evidence = frame()
    evidence['requested_operator'] = 'possibility_any'
    for row in evidence['target_assessments']:
        row['response_operators'] = ['frequency_typicality', 'possibility_any']
    validate_task_frame(evidence, REPLY, STATE, level='2')
    # This is not semantic approval: the model must justify labels from quotes.


@pytest.mark.parametrize('mutation', [
    lambda f: f['target_assessments'][1].update(target='favored-reference'),
    lambda f: f['target_assessments'][1].update(target='alpha'),
    lambda f: f.update(reference_targets=['alpha']),
])
def test_favored_or_duplicate_entities_cannot_inflate_requested_target_count(mutation):
    evidence = frame()
    mutation(evidence)
    with pytest.raises(ValueError):
        validate_task_frame(evidence, REPLY, STATE, level='1')


@pytest.mark.parametrize('mutation', [
    lambda f: f['request_quotes'][0].update(turn_id='source'),
    lambda f: f['request_quotes'][0].update(quote='A request from future dialogue.'),
    lambda f: f['target_assessments'][0].update(response_quotes=['A sibling-only span.']),
])
def test_task_frame_evidence_stays_on_its_pre_response_and_reply_boundaries(mutation):
    evidence = frame()
    mutation(evidence)
    with pytest.raises(ValueError):
        validate_task_frame(evidence, REPLY, STATE, level='1')


def test_true_information_limit_route_remains_available_without_invented_quotes():
    evidence = frame()
    reply = REPLY + ' The available information does not locate beta.'
    evidence['target_assessments'][1].update(coverage='explicit_limitation',
        response_operators=['information_limit'], response_quotes=['The available information does not locate beta.'])
    validate_task_frame(evidence, reply, STATE, level='2')
    evidence['target_assessments'][1]['response_quotes'] = []
    with pytest.raises(ValueError, match='information limit'):
        validate_task_frame(evidence, reply, STATE, level='2')


def test_collective_comparison_can_support_multiple_requested_targets():
    evidence = frame()
    reply = 'Both alpha and beta differ from the reference on the requested timing dimension.'
    for row in evidence['target_assessments']:
        row['response_quotes'] = [reply]
    validate_task_frame(evidence, reply, STATE, level='2')


def test_unmet_scope_is_never_silently_remapped_to_a_lower_reward():
    raw = answer()
    raw['task_frame']['target_assessments'][1].update(coverage='not_addressed', response_operators=[], response_quotes=[])
    raw['task_frame']['scope_met'] = False
    client = V3Client(result=raw)
    with pytest.raises(IntegrationUnavailable, match='Level2'):
        scorer(client).score_behavior(CASE, PREFIX, [rubric()], REPLY)
    assert len(client.calls) == 2
    assert all(row['parsed']['candidate-id']['level'] == '2' for row in client.log)


def test_legacy_boolean_gap_can_remain_unscorable_with_visible_performed_scope():
    definition = rubric()
    for key in ('criterion_version', 'minimum_performance', 'completion_condition'):
        definition.pop(key)
    definition['scores']['0']['behavior'] = 'Endorses the claim without comparing it.'
    definition['scores']['1']['behavior'] = 'Withholds endorsement but leaves the comparison incomplete.'
    definition['scores']['2']['behavior'] = 'Withholds endorsement and supplies the comparison.'
    raw = answer('UNSCORABLE')
    raw['criterion_defect'] = 'Endorsement with comparison is uncovered; do not discard the without clause.'
    raw['remaining_gap'] = 'The scale omits the observed conjunction.'
    raw['requirements'] = [{'condition': 'written_anchor', 'definition_ref': 'score_anchors.0', 'met': False,
        'evidence_type': 'response_spans', 'response_quotes': ['their timing differs.'],
        'entailment': 'Comparison is displayed, so the without-comparison clause is not met.'}]
    resolved = materialize_public_evidence(raw, REPLY, definition,
        evidence_contract=PUBLIC_EVIDENCE_CONTRACT_V3, state=STATE)
    assert resolved['level'] == 'UNSCORABLE'


def test_uncertain_applicability_is_preserved_instead_of_fabricated_missing_text():
    client = V3Client()
    def uncertainty(stage, values):
        if stage != 'applicability':
            return
        for row in values.values():
            row.update(applicable=False, missing_requirement='The shorthand has two unresolved contextual meanings.')
            row['doctor_condition'].update(status='uncertain', unmet_condition='The shorthand meaning is unresolved.',
                                           entailment='The wording exists, but its intended interpretation is ambiguous.')
    client.mutate = uncertainty
    result = condition_evidence_reviewer(client, contract=CONDITION_EVIDENCE_CONTRACT_V3)(CASE, PREFIX, [rubric()])['candidate-id']
    assert result['applicable'] is False
    assert result['applicability_status'] == 'uncertain'
    assert result['stage1']['doctor_condition']['status'] == 'uncertain'
    # Prior contracts do not silently gain the new status.
    raw = client.log[-1]['parsed']['candidate-id']
    with pytest.raises(ScoringContractError, match='invalid status'):
        materialize_condition_evidence(raw, STATE, condition_items([rubric()])['candidate-id'],
                                       contract=CONDITION_EVIDENCE_CONTRACT_V2)


def test_v3_gate_binds_v3_blind_source_and_preserves_partial_source_mismatch():
    client = V3Client(result=answer('1'))
    result = public_mc_posthoc_gate(client, {CASE['case_id']: CASE}, protocol=PUBLIC_GATE_PROTOCOL_V3)(tin('failure'), rubric())
    assert result['protocol'] == PUBLIC_GATE_PROTOCOL_V3
    assert result['source_anchor']['observed_level'] == '1'
    assert result['source_anchor']['expected_level'] == '0'
    assert not result['passed'] and 'source_anchor_mismatch' in result['reasons']
    assert result['evidence']['blind_source']['protocol'] == PUBLIC_EVIDENCE_CONTRACT_V3


def test_gate_rejects_conflicting_blind_source_as_unscorable_without_aborting():
    class ConflictingBlindScorer:
        protocol = 'test-blind-scorer'

        def score_behavior(self, *args):
            error = IntegrationUnavailable('synthetic anchor contradiction')
            error.audit = {'validation': {'status': 'execution_error', 'kind': 'evidence_contract',
                'error': 'Numeric score contradicts anchor matches',
                'raw_parsed': {'answers': {'target': {'level': '2',
                    'checks': [{'ref': 'score_anchors.1', 'met': True},
                               {'ref': 'score_anchors.2', 'met': True}]}}}}}
            raise error

    result = public_mc_posthoc_gate(V3Client(), {CASE['case_id']: CASE},
        protocol=PUBLIC_GATE_PROTOCOL_V3, blind_scorer_override=ConflictingBlindScorer())(tin(), rubric())
    assert not result['passed']
    assert 'source_unscorable' in result['reasons']
    assert result['source_anchor']['observed_level'] == 'UNSCORABLE'
    assert result['blind_source_assessment']['level'] == 'UNSCORABLE'
    assert result['evidence']['blind_source']['validation']['raw_parsed']


def test_gate_does_not_mask_other_blind_scorer_execution_errors():
    class BrokenBlindScorer:
        protocol = 'test-blind-scorer'

        def score_behavior(self, *args):
            error = IntegrationUnavailable('synthetic truncated output')
            error.audit = {'validation': {'status': 'execution_incomplete', 'kind': 'finish_reason'}}
            raise error

    with pytest.raises(IntegrationUnavailable, match='truncated'):
        public_mc_posthoc_gate(V3Client(), {CASE['case_id']: CASE},
            protocol=PUBLIC_GATE_PROTOCOL_V3, blind_scorer_override=BrokenBlindScorer())(tin(), rubric())


def test_v3_cache_identities_differ_from_v2():
    old = evidence_scorer(V2Client(), evidence_contract=PUBLIC_EVIDENCE_CONTRACT_V2)
    new = scorer()
    assert old.protocol != new.protocol
    assert old.review_state.protocol != new.review_state.protocol


def test_legacy_zero_conflict_is_valid_failure_evidence_only_in_v3():
    definition = rubric()
    for key in ('criterion_version', 'minimum_performance', 'completion_condition'):
        definition.pop(key)
    definition['scores']['0']['behavior'] = 'States a comparison and then draws the opposite conclusion.'
    raw = answer('0')
    raw['requirements'] = [{'condition': 'written_anchor', 'definition_ref': 'score_anchors.0',
        'met': True, 'evidence_type': 'conflict', 'response_quotes': ['their timing differs.'],
        'entailment': 'Synthetic conflict evidence establishes the legacy failure condition, not a positive minimum.'}]
    resolved = materialize_public_evidence(raw, REPLY, definition,
        evidence_contract=PUBLIC_EVIDENCE_CONTRACT_V3, state=STATE)
    assert resolved['level'] == '0' and resolved['requirements'][0]['met'] is True
    old_raw = {k: v for k, v in raw.items() if k != 'task_frame'}
    with pytest.raises(ValueError, match='conflicting evidence'):
        materialize_public_evidence(old_raw, REPLY, definition, evidence_contract=PUBLIC_EVIDENCE_CONTRACT_V2)
    public = answer('2')
    public['requirements'][0]['evidence_type'] = 'conflict'
    with pytest.raises(ValueError, match='conflicting evidence'):
        materialize_public_evidence(public, REPLY, rubric(), evidence_contract=PUBLIC_EVIDENCE_CONTRACT_V3, state=STATE)
