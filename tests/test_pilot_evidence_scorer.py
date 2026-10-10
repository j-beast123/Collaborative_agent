import copy
import json

import pytest

from src.common.config import IntegrationUnavailable
from src.online_rl.tree.pilot_evidence_scorer import evidence_scorer, validate_evidence


def rubric():
    context = {'user_side_context': 'The physician requests a comparison.',
               'agent_side_state': 'No additional recorded Agent state is required.'}
    return {'rubric_id': 'r', 'name': 'Compare explicitly', 'applicability': context,
            'required_behavior': 'Explain both candidates and their difference.',
            'scores': {k: {**context, 'behavior': behavior} for k, behavior in {
                '0': 'No candidate property is explained.',
                '1': 'Explain one candidate property, leaving the comparison missing.',
                '2': 'Explain both properties and explicitly contrast them.'}.items()},
            'intervention_instruction': 'PRIVATE_INSTRUCTION',
            'source_outcome': 'SECRET_OUTCOME', 'analysis': {'source_score': 2}}


REPLY = 'A has short duration. B lasts longer, unlike A.'
VIEW = {'question': 'Which candidate?', 'options': {'A': 'alpha', 'B': 'beta'},
        'gold_option': 'PRIVATE_GOLD', 'intervention_instruction': 'PRIVATE_INSTRUCTION'}
PREFIX = [{'speaker': 'doctor', 'text': 'Compare their duration.'}]


def answer(level='2', **kw):
    value = {'level': level, 'anchor_basis': 'The reply gives properties and an explicit contrast.',
             'minimum_contribution_quote': 'A has short duration.',
             'completion_quote': 'B lasts longer, unlike A.', 'missing_or_conflicting': ''}
    value.update(kw)
    return value


class Client:
    model = 'fake'
    protocol_sha256 = 'fake'

    def __init__(self, responses, applicable=True):
        self.responses = iter(responses)
        self.calls = []
        self.log = []
        self.applicable = applicable

    def judge(self, stage, state, items, fields, purpose):
        self.calls.append(copy.deepcopy({'stage': stage, 'state': state, 'items': items, 'fields': fields}))
        values = ({'r': {'applicable': self.applicable, 'evidence_basis': 'The request is visible.',
                         'missing_requirement': ''}} if stage == 'applicability'
                  else {'r': next(self.responses)})
        rec = {'stage': stage, 'parsed': copy.deepcopy(values)}
        self.log.append(rec)
        return values, rec


def test_valid_completion_requires_literal_target_evidence_and_no_private_inputs():
    client = Client([answer()])
    scorer = evidence_scorer(client)
    result = scorer(VIEW, PREFIX, [rubric()], REPLY)['r']
    assert result['hard'] == 2 and result['scoring_status'] == 'scored'
    assert result['evidence_validation']['attempts'] == 1
    assert client.calls[0]['stage'] == 'applicability'
    assert 'target_agent_reply' not in client.calls[0]['state']
    packet = json.dumps(client.calls)
    for secret in ('PRIVATE_INSTRUCTION', 'PRIVATE_GOLD', 'SECRET_OUTCOME', 'source_score'):
        assert secret not in packet


def test_fabricated_quote_gets_one_logged_correction_without_downscoring():
    client = Client([answer(completion_quote='An invented comparison.'), answer()])
    result = evidence_scorer(client).score_behavior(VIEW, PREFIX, [rubric()], REPLY)['r']
    assert result['hard'] == 2
    assert result['evidence_validation']['attempts'] == 2
    assert len(client.log) == 2
    assert 'Previous answer:' in client.calls[1]['items']['r']['instructions']
    assert client.calls[0]['state'] == client.calls[1]['state']


def test_repeated_fabricated_quote_is_error_not_zero_or_na():
    bad = answer(minimum_contribution_quote='A property found only in the prefix.')
    with pytest.raises(IntegrationUnavailable, match='after one correction'):
        evidence_scorer(Client([bad, bad])).score_behavior(VIEW, PREFIX, [rubric()], REPLY)


@pytest.mark.parametrize('kwargs', [
    {'missing_or_conflicting': 'The comparison is missing.'},
    {'minimum_contribution_quote': ''}, {'completion_quote': ''},
])
def test_two_rejects_self_contradiction_or_missing_evidence(kwargs):
    with pytest.raises(ValueError):
        validate_evidence(answer(**kwargs), REPLY)


def test_one_needs_minimum_evidence_and_remaining_gap():
    good = answer('1', completion_quote='', missing_or_conflicting='No property of B or contrast is explained.')
    validate_evidence(good, 'A has short duration.')
    with pytest.raises(ValueError, match='remaining gap'):
        validate_evidence({**good, 'missing_or_conflicting': ''}, 'A has short duration.')


def test_zero_can_have_no_quote_but_explains_missing_contribution():
    result = answer('0', minimum_contribution_quote='', completion_quote='',
                    missing_or_conflicting='Only a candidate is named; no property is explained.')
    validate_evidence(result, 'I choose A.')


@pytest.mark.parametrize('quote_field', ['minimum_contribution_quote', 'completion_quote'])
def test_zero_rejects_claimed_contribution_or_completion_even_with_literal_quote(quote_field):
    result = answer('0', minimum_contribution_quote='', completion_quote='',
                    missing_or_conflicting='The minimum contribution is absent.')
    result[quote_field] = 'A has short duration.'
    with pytest.raises(ValueError, match='Level 0 cannot simultaneously report'):
        validate_evidence(result, REPLY)


def test_unscorable_is_not_applicability_na():
    result = answer('UNSCORABLE', minimum_contribution_quote='', completion_quote='',
                    missing_or_conflicting='None of the complete anchors covers this response.')
    value = evidence_scorer(Client([result]))(VIEW, PREFIX, [rubric()], 'Not enough detail.')['r']
    assert value['hard'] is None and value['applicable'] is True
    assert value['scoring_status'] == 'rubric_unscorable'


def test_unscorable_can_preserve_actual_partial_evidence():
    result = answer('UNSCORABLE', completion_quote='',
                    missing_or_conflicting='The partial contribution does not match any entire anchor.')
    validate_evidence(result, REPLY)


def test_nonapplicable_skips_behavior_and_preserves_na():
    client = Client([], applicable=False)
    value = evidence_scorer(client)(VIEW, PREFIX, [rubric()], REPLY)['r']
    assert value['hard'] is None and value['applicable'] is False
    assert value['scoring_status'] == 'not_applicable'
    assert len(client.calls) == 1


class BatchClient(Client):
    def judge_batch(self, stage, state, responses, items, fields, purpose):
        self.calls.append(copy.deepcopy({'stage': stage, 'state': state, 'responses': responses,
                                         'items': items, 'fields': fields}))
        values = next(self.responses)
        self.log.append({'stage': stage, 'parsed': copy.deepcopy(values)})
        return values, self.log[-1]


def test_training_batch_keeps_separate_evidence_and_unscorable_status():
    uncovered = answer('UNSCORABLE', minimum_contribution_quote='', completion_quote='',
                       missing_or_conflicting='This reply is uncovered by the supplied anchors.')
    client = BatchClient([{'left': {'r': answer()}, 'right': {'r': uncovered}}])
    scorer = evidence_scorer(client, response_batch_size=4)
    values = scorer.score_behavior_batch(VIEW, PREFIX, [rubric()], [
        {'response_id': 'left', 'reply': REPLY}, {'response_id': 'right', 'reply': 'Another reply.'}])
    assert values['left']['r']['hard'] == 2
    assert values['right']['r']['hard'] is None
    assert values['right']['r']['scoring_status'] == 'rubric_unscorable'
    assert len(client.calls) == 1 and scorer.response_batch_size == 4
    assert scorer.applicability.protocol == scorer.review_identity['protocol']
    packet = json.dumps(client.calls)
    for secret in ('PRIVATE_GOLD', 'PRIVATE_INSTRUCTION', 'SECRET_OUTCOME', 'source_score'):
        assert secret not in packet


def test_batch_cannot_borrow_literal_evidence_from_a_sibling_response():
    wrong = {'left': {'r': answer()}, 'right': {'r': answer()}}
    zero = answer('0', minimum_contribution_quote='', completion_quote='',
                  missing_or_conflicting='No candidate property is explained.')
    fixed = {'left': {'r': answer()}, 'right': {'r': zero}}
    client = BatchClient([wrong, fixed])
    values = evidence_scorer(client, response_batch_size=4).score_behavior_batch(VIEW, PREFIX, [rubric()], [
        {'response_id': 'left', 'reply': REPLY}, {'response_id': 'right', 'reply': 'I choose A.'}])
    assert [values[key]['r']['hard'] for key in ('left', 'right')] == [2, 0]
    assert len(client.calls) == 2
    assert values['right']['r']['evidence_validation']['attempts'] == 2
    assert 'right:' in values['right']['r']['evidence_validation']['errors'][0]
    assert client.calls[0]['state'] == client.calls[1]['state']
    assert client.calls[0]['responses'] == client.calls[1]['responses']


def test_batch_repeated_evidence_error_or_incomplete_ids_raises():
    for invalid in ({'left': {'r': answer()}},
                    {'left': {'r': answer()}, 'right': {'r': answer()}}):
        client = BatchClient([invalid, invalid])
        with pytest.raises(IntegrationUnavailable, match='after one correction'):
            evidence_scorer(client, response_batch_size=4).score_behavior_batch(VIEW, PREFIX, [rubric()], [
                {'response_id': 'left', 'reply': REPLY}, {'response_id': 'right', 'reply': 'I choose A.'}])


def test_applicability_sanitizes_private_prefix_metadata():
    client = Client([], applicable=False)
    prefix = [{**PREFIX[0], 'intervention_instruction': 'PRIVATE_INSTRUCTION', 'outcome': 'SECRET_OUTCOME'}]
    evidence_scorer(client).applicability(VIEW, prefix, [rubric()])
    assert client.calls[0]['state']['dialogue_before_target_reply'] == PREFIX
    for secret in ('PRIVATE_INSTRUCTION', 'SECRET_OUTCOME', 'PRIVATE_GOLD'):
        assert secret not in json.dumps(client.calls)
