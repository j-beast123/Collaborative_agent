"""Gate decisions need local evidence, blind source anchoring, and new cache identity."""
import copy
import json
from dataclasses import replace

import pytest

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_text
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.posthoc import generate_rubric
from src.online_rl.tree.quality_gate import evidence_posthoc_gate
from src.online_rl.tree.records import TrajectoryRubricInput, to_dialogue


REPLY = 'Its onset is immediate. The alternative develops later; their timing differs.'
CASE = {'case_id': 'gate-case', 'scenario': 'A visible clinical scenario.',
        'question': 'Which consideration matters?', 'options': {'A': 'alpha', 'B': 'beta'},
        'ground_truth': {'correct_option': 'A', 'secret': 'PRIVATE_GOLD'},
        'source_outcome': 'PRIVATE_CASE_OUTCOME', 'hidden_belief': 'PRIVATE_BELIEF'}


def rubric():
    context = {'user_side_context': 'The Doctor asks to compare onset timing.',
               'agent_side_state': 'No additional prior Agent behavior is required.'}
    return {'rubric_id': 'candidate-id', 'name': 'Compare onset timing',
        'applicability': context,
        'required_behavior': 'Compare the onset timing of the two considerations.',
        'scores': {level: {**context, 'behavior': behavior} for level, behavior in {
            '0': 'Does not explain either onset timing.',
            '1': 'Explains one onset timing but leaves the comparison absent.',
            '2': 'Explains both onset timings and explicitly compares them.'}.items()},
        'source_provenance': {'analysis': {'selected_agent_turn_id': 2,
                                          'source_score': 2,
                                          'private_reason': 'PRIVATE_GENERATOR_ANALYSIS'},
                              'source_outcome': 'success'},
        'intervention_instruction': 'PRIVATE_INTERVENTION_GUIDE',
        'source_score': 2, 'delta_k': 'PRIVATE_EFFECTIVENESS'}


def tin(outcome='success'):
    return TrajectoryRubricInput('gate-case/0.1', 'gate-case', CASE['scenario'],
        CASE['question'], tuple(CASE['options'].items()),
        (('doctor', 'D1', 1, 'State your initial view.'),
         ('agent', 'A1', 1, 'I have an initial view.'),
         ('doctor', 'D2', 2, 'Compare onset timing.'),
         ('agent', 'A2', 2, REPLY),
         ('doctor', 'D3', 3, 'FUTURE_DOCTOR_SENTINEL'),
         ('agent', 'A3', 3, 'FUTURE_AGENT_SENTINEL'),
         ('doctor', 'FINAL', 0, 'PRIVATE_FINAL_SENTINEL')),
        outcome, 'PRIVATE_FEEDBACK_SENTINEL', 'trajectory-hash',
        'physician_post_final_reflection_v1', 1)


def answer(level='2', **changes):
    out = {
        'answer_presupposition': False, 'answer_presupposition_quote': '',
        'answer_presupposition_reason': 'The comparison leaves the selected conclusion open.',
        'new_domain_premise': False, 'new_domain_premise_quote': '',
        'new_domain_premise_reason': 'It requires comparison without stipulating a clinical fact.',
        'observable': True, 'observable_quote': '',
        'observable_reason': 'An explicit onset comparison can be identified in the reply.',
        'score_boundaries_clear': True, 'score_boundaries_quote': '',
        'score_boundaries_other_quote': '',
        'score_boundaries_reason': 'No explanation, one explanation, and explicit comparison are distinct.',
        'source_level': level, 'source_minimum_quote': 'Its onset is immediate.',
        'source_completion_quote': 'The alternative develops later; their timing differs.',
        'source_missing_or_conflicting': '',
        'source_reason': 'Both timings and their explicit difference are present.'}
    if level == '0':
        out.update(source_minimum_quote='', source_completion_quote='',
                   source_missing_or_conflicting='Neither onset timing is explained.',
                   source_reason='The minimum contribution is absent.')
    elif level == '1':
        out.update(source_completion_quote='',
                   source_missing_or_conflicting='The explicit comparison remains absent.',
                   source_reason='One onset is explained but the requested contrast is absent.')
    elif level == 'UNSCORABLE':
        out.update(source_minimum_quote='', source_completion_quote='',
                   source_missing_or_conflicting='The response does not fit any complete anchor.',
                   source_reason='No complete score definition covers the response.')
    out.update(changes)
    return out


class Client:
    def __init__(self, responses, *, model='gate-model', system='Gate system.', max_tokens=4096):
        self.responses = iter(responses)
        self.calls = []
        self.log = []
        self.model, self.system, self.max_tokens = model, system, max_tokens
        self.protocol_sha256 = sha256_text(system)

    def judge(self, stage, state, items, fields, purpose):
        self.calls.append(copy.deepcopy({'stage': stage, 'state': state,
                                       'items': items, 'fields': fields, 'purpose': purpose}))
        values = {'candidate': next(self.responses)}
        record = {'raw': [json.dumps({'answers': values})], 'parsed': copy.deepcopy(values),
                  'model': self.model, 'purpose': purpose}
        self.log.append(record)
        return values, record


def gate(client):
    return evidence_posthoc_gate(client, {CASE['case_id']: CASE}, observed_context=True)


def test_selected_source_turn_is_separate_from_strict_prefix_and_future_is_hidden():
    client = Client([answer()])
    result = gate(client)(tin(), rubric())
    assert result['passed']
    state = client.calls[0]['state']
    assert state['dialogue_before_target_turn'] == to_dialogue(tin().dialogue[:3])
    assert state['source_agent_reply']['text'] == REPLY
    assert state['selected_agent_turn_id'] == 2  # The feedback locus is 1.
    assert state['case_context'] == {k: CASE[k] for k in ('scenario', 'question', 'options')}
    packet = json.dumps(client.calls)
    for hidden in ('FUTURE_DOCTOR_SENTINEL', 'FUTURE_AGENT_SENTINEL', 'PRIVATE_FINAL_SENTINEL',
                   'PRIVATE_GOLD', 'PRIVATE_BELIEF', 'PRIVATE_CASE_OUTCOME',
                   'PRIVATE_FEEDBACK_SENTINEL', 'PRIVATE_GENERATOR_ANALYSIS',
                   'PRIVATE_INTERVENTION_GUIDE', 'PRIVATE_EFFECTIVENESS',
                   'source_outcome', 'source_score', 'ground_truth', 'delta_k'):
        assert hidden not in packet
    assert set(client.calls[0]['items']) == {'candidate'}


def test_same_source_judgement_is_blind_to_outcome_but_local_anchor_direction_changes():
    success_client, failure_client = Client([answer()]), Client([answer()])
    success = gate(success_client)(tin('success'), rubric())
    failed_rubric = rubric()
    failed_rubric['source_provenance']['source_outcome'] = 'failure'
    failed_rubric['source_provenance']['analysis']['source_score'] = 0
    failure = gate(failure_client)(tin('failure'), failed_rubric)
    assert success_client.calls == failure_client.calls
    assert success['passed'] and not failure['passed']
    assert str(success['source_anchor']['expected_level']) == '2'
    assert str(failure['source_anchor']['expected_level']) == '0'
    assert str(failure['source_anchor']['observed_level']) == '2'
    assert failure['source_anchor']['matches'] is False


def test_failure_source_zero_can_pass_without_becoming_an_outcome_score():
    source = tin('failure')
    source = replace(source, dialogue=source.dialogue[:3] +
                     (('agent', 'A2', 2, 'I maintain my initial view.'),) + source.dialogue[4:])
    result = gate(Client([answer('0')]))(source, rubric())
    assert result['passed']
    assert str(result['source_anchor']['observed_level']) == '0'


@pytest.mark.parametrize('level', ['1', 'UNSCORABLE'])
def test_partial_or_unscorable_success_source_does_not_pass_anchor(level):
    result = gate(Client([answer(level)]))(tin(), rubric())
    assert not result['passed'] and result['source_anchor']['matches'] is False
    reason = 'source_unscorable' if level == 'UNSCORABLE' else 'source_anchor_mismatch'
    assert reason in result['reasons']


@pytest.mark.parametrize('check, changes', [
    ('answer_presupposition', {'answer_presupposition': True,
        'answer_presupposition_quote': 'Compare the onset timing',
        'answer_presupposition_reason': 'This quoted requirement forces a conclusion in this test.'}),
    ('new_domain_premise', {'new_domain_premise': True,
        'new_domain_premise_quote': 'onset timing',
        'new_domain_premise_reason': 'This quoted requirement adds a factual premise in this test.'}),
    ('observable', {'observable': False, 'observable_quote': 'Compare onset timing',
        'observable_reason': 'The named internal operation is unobservable in this test.'}),
    ('score_boundaries_clear', {'score_boundaries_clear': False,
        'score_boundaries_quote': 'Does not explain either onset timing.',
        'score_boundaries_other_quote': 'Explains one onset timing but leaves the comparison absent.',
        'score_boundaries_reason': 'The two quoted conditions overlap in this test.'}),
])
def test_each_quality_failure_blocks_with_preserved_specific_evidence(check, changes):
    client = Client([answer(**changes)])
    result = gate(client)(tin(), rubric())
    assert not result['passed'] and result['reasons']
    assert len(client.calls) == 1
    saved = json.dumps(result)
    assert changes[check + '_reason' if check != 'score_boundaries_clear' else 'score_boundaries_reason'] in saved


def test_fabricated_blocking_quote_gets_one_correction_not_a_rejection():
    bad = answer(answer_presupposition=True,
                 answer_presupposition_quote='Invented instruction absent from every rubric field.')
    client = Client([bad, answer()])
    result = gate(client)(tin(), rubric())
    assert result['passed'] and len(client.calls) == 2
    corrected = dict(client.calls[1]['state'])
    correction = corrected.pop('evidence_correction')
    assert client.calls[0]['state'] == corrected
    assert correction['previous_answer'] == bad and correction['error']
    assert json.dumps(bad) in json.dumps(result) or bad['answer_presupposition_quote'] in json.dumps(result)


@pytest.mark.parametrize('changes', [
    {'observable_reason': '   '},
    {'source_completion_quote': 'An invented comparison.'},
    {'source_minimum_quote': 'Compare onset timing.'},  # Real prefix text, not source reply.
    {'source_minimum_quote': ''},
    {'source_missing_or_conflicting': 'The requested comparison is absent.'},
    {'observable': False, 'observable_quote': ''},
    {'score_boundaries_clear': False,
     'score_boundaries_quote': 'Does not explain either onset timing.',
     'score_boundaries_other_quote': 'An invented conflicting score anchor.'},
])
def test_invalid_or_self_contradictory_evidence_is_operational_error_after_one_correction(changes):
    bad = answer(**changes)
    client = Client([bad, bad])
    with pytest.raises(IntegrationUnavailable):
        gate(client)(tin(), rubric())
    assert len(client.calls) == 2


@pytest.mark.parametrize('selected_turn', [None, True, 99])
def test_missing_or_invalid_generator_selected_turn_never_falls_back_to_feedback(selected_turn):
    candidate = rubric()
    candidate['source_provenance']['analysis']['selected_agent_turn_id'] = selected_turn
    client = Client([])
    with pytest.raises((ValueError, IntegrationUnavailable)):
        gate(client)(tin(), candidate)
    assert not client.calls


def test_gate_protocol_fingerprint_binds_model_prompt_budget_and_observed_context():
    baseline = gate(Client([]))
    variants = [gate(Client([], model='other-model')),
                gate(Client([], system='A revised gate system.')),
                gate(Client([], max_tokens=8192)),
                evidence_posthoc_gate(Client([]), {CASE['case_id']: CASE}, observed_context=False)]
    assert all(other.protocol != baseline.protocol for other in variants)
    assert baseline.model == 'gate-model'


def test_existing_generator_cache_can_be_reused_but_legacy_gate_cannot(tmp_path):
    calls = []
    def generate(prompt, source):
        calls.append('generation')
        return {'rubric': rubric(), 'raw': 'generated rubric'}
    def legacy(source, candidate):
        calls.append('legacy gate')
        return {'passed': True, 'reasons': []}
    legacy.model, legacy.protocol = 'gate-model', 'legacy-gate-protocol'
    cache_path = tmp_path / 'calls.jsonl'
    source = tin()
    generate_rubric('fixed prompt', 'prompt-hash', source, generate, CallCache(cache_path), legacy)
    client = Client([answer()])
    revised = gate(client)
    result = generate_rubric('fixed prompt', 'prompt-hash', source, generate, CallCache(cache_path), revised)
    assert result.valid and calls == ['generation', 'legacy gate'] and len(client.calls) == 1
    resumed = generate_rubric('fixed prompt', 'prompt-hash', source, generate, CallCache(cache_path), revised)
    assert resumed == result and len(client.calls) == 1
    records = [json.loads(line) for line in cache_path.read_text().splitlines()]
    assert len([row for row in records if row['kind'] == 'rubric_generation']) == 1
    assert len([row for row in records if row['kind'] == 'posthoc_gate']) == 2
