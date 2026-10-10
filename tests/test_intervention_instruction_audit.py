"""Pre-rerun guide validity is separate from response adherence and outcomes.

Recorded inputs exercise admission/transport with manually authored judge stubs;
these tests do not claim to measure a model judge's semantic accuracy.
"""
import copy
import json
from pathlib import Path

import pytest

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json, sha256_text
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.instruction_audit import (
    cached_instruction_auditor,
    instruction_state,
    make_instruction_auditor,
    validate_evidence,
)
from src.online_rl.tree.member_intervention import VERSION


CASE = {
    'case_id': 'instruction-case', 'scenario': 'A description includes several dates.',
    'question': 'Which interpretation follows from the supplied description?',
    'options': {'A': 'First interpretation', 'B': 'Second interpretation'},
}
PREFIX = [
    {'speaker': 'doctor', 'stage': 'D1', 'round': 1,
     'text': 'Compare the dates with the stated course.'},
    {'speaker': 'agent', 'stage': 'A1', 'round': 1,
     'text': 'The text lists dates but does not label their purpose.'},
    {'speaker': 'doctor', 'stage': 'D2', 'round': 2,
     'text': 'Please check what supports my interpretation.'},
]
GUIDE = ('Identify the text relevant to the clinician\'s interpretation. '
         'Explain whether that text supports the interpretation, without fixing a conclusion.')


def rubric():
    context = {'user_side_context': 'The Doctor requests evaluation of an interpretation.',
               'agent_side_state': 'No additional recorded Agent state is required.'}
    return {
        'rubric_id': 'evidence-relation', 'name': 'Evaluate the expressed interpretation',
        'applicability': context,
        'required_behavior': 'Identify relevant text and explain its relation to the interpretation.',
        'scores': {level: {**context, 'behavior': behavior} for level, behavior in {
            '0': 'Does not identify text relevant to the interpretation.',
            '1': 'Identifies relevant text but leaves its relation to the interpretation unexplained.',
            '2': 'Identifies relevant text and explains whether it supports the interpretation.',
        }.items()},
    }


def member_spec(*, origin='generated', guide=GUIDE, level='2', prefix=None, candidate=None):
    prefix = PREFIX if prefix is None else prefix
    candidate = rubric() if candidate is None else candidate
    return {
        'version': VERSION, 'leaf_id': 'instruction-case/0.0', 'case_id': CASE['case_id'],
        'canonical_rubric_id': candidate['rubric_id'], 'rubric_sha256': sha256_json(candidate),
        'selected_agent_turn_id': 2, 'intervention_instruction': guide,
        'requested_level': level, 'prefix_sha256': sha256_json(prefix), 'origin': origin,
    }


def answer(**changes):
    result = {
        'instruction_aligned': True, 'instruction_alignment_quote': GUIDE,
        'instruction_alignment_reason': 'M identifies relevant text; C explains its relation. The level-2 guide requires both.',
        'unsupported_context_premise': False, 'unsupported_context_premise_quote': '',
        'unsupported_context_premise_reason': 'The guide requests evaluation without supplying an ungrounded factual premise.',
        'unsupported_quote_source': 'none',
        'conclusion_prescribed': False, 'conclusion_quote': '',
        'conclusion_reason': 'The guide leaves the interpretation open to the supplied evidence.',
        'conclusion_quote_source': 'none',
    }
    result.update(changes)
    return result


class Client:
    def __init__(self, responses, *, model='instruction-model', system='Independent guide audit.',
                 max_tokens=3072, timeout_s=180):
        self.responses = iter(responses)
        self.calls = []
        self.log = []
        self.model, self.system, self.max_tokens = model, system, max_tokens
        self.timeout_s = timeout_s
        self.protocol_sha256 = sha256_text(system)

    def judge(self, stage, state, items, fields, purpose):
        self.calls.append(copy.deepcopy({'stage': stage, 'state': state, 'items': items,
                                        'fields': fields, 'purpose': purpose}))
        values = {'candidate': copy.deepcopy(next(self.responses))}
        record = {'raw': [json.dumps({'answers': values})], 'parsed': copy.deepcopy(values),
                  'model': self.model, 'purpose': purpose}
        self.log.append(record)
        return values, record


def test_real_judge_schema_failure_keeps_its_own_raw_evidence_without_extra_vote(tmp_path):
    from src.online_rl.tree.luna import LunaDecisionClient
    sent = []

    def post(url, payload, key, timeout):
        sent.append(copy.deepcopy(payload))
        return {'model': 'fixture', 'choices': [{'message': {'content': 'not-json'}}]}

    client = LunaDecisionClient(model='fixture', key='unused', system='Independent audit.', post=post)
    cache_path = tmp_path / 'calls.jsonl'
    cache = CallCache(cache_path)
    auditor = cached_instruction_auditor(make_instruction_auditor(client), cache)
    with pytest.raises(IntegrationUnavailable):
        auditor(CASE, PREFIX, rubric(), member_spec())
    assert len(sent) == 2  # One client format repair; no second semantic review.
    assert not cache.store and not cache_path.exists()
    failure = json.loads(cache_path.with_suffix('.errors.jsonl').read_text())
    evidence = failure['audit']['client_error_audit']
    assert evidence['decisions'][0]['raw'] == ['not-json', 'not-json']
    assert [r['messages'] for r in evidence['requests']] == [p['messages'] for p in sent]
    assert len(evidence['requests']) == 2


def test_visible_state_discards_private_context_and_does_not_mutate_inputs():
    case, prefix, candidate = copy.deepcopy(CASE), copy.deepcopy(PREFIX), rubric()
    case.update(ground_truth='PRIVATE_GOLD', outcome='PRIVATE_OUTCOME',
                feedback='PRIVATE_FEEDBACK', target_agent_reply='PRIVATE_TARGET')
    prefix[0].update(hidden_belief='PRIVATE_BELIEF', future_dialogue='PRIVATE_FUTURE',
                     target_response='PRIVATE_PREFIX_TARGET')
    candidate.update(source_provenance={'source_outcome': 'PRIVATE_SOURCE_OUTCOME',
                                      'analysis': 'PRIVATE_ANALYSIS'},
                     intervention_instruction='PRIVATE_CANONICAL_GUIDE',
                     minimum_contribution='PRIVATE_MINIMUM_HINT')
    spec = member_spec(candidate=candidate)
    spec.update(outcome='PRIVATE_MEMBER_OUTCOME', source_response='PRIVATE_SOURCE_RESPONSE',
                final='PRIVATE_FINAL', complete_behavior='PRIVATE_COMPLETION_HINT')
    before = copy.deepcopy((case, prefix, candidate, spec))
    client = Client([answer()])
    result = make_instruction_auditor(client)(case, prefix, candidate, spec)
    state = client.calls[0]['state']
    assert set(state) == {'case_context', 'dialogue_before_target_turn', 'rubric',
                          'selected_agent_turn_id', 'requested_level', 'intervention_instruction'}
    assert state['case_context'] == {key: CASE[key] for key in ('scenario', 'question', 'options')}
    assert state['dialogue_before_target_turn'] == PREFIX
    assert state['rubric'] == rubric()
    assert state['selected_agent_turn_id'] == 2
    assert type(state['requested_level']) is int and state['requested_level'] == 2
    assert state['intervention_instruction'] == GUIDE
    assert 'PRIVATE_' not in json.dumps(client.calls)
    assert (case, prefix, candidate, spec) == before
    assert result['passed'] is True
    assert result['state_sha256'] == sha256_json(state)


def test_generated_and_matched_members_receive_identical_independent_review():
    generated, matched = Client([answer()]), Client([answer()])
    first = make_instruction_auditor(generated)(CASE, PREFIX, rubric(), member_spec())
    spec = member_spec(origin='matched')
    spec['leaf_id'] = 'instruction-case/3.2'
    second = make_instruction_auditor(matched)(CASE, PREFIX, rubric(), spec)
    assert generated.calls == matched.calls
    assert first['passed'] is second['passed'] is True
    assert first['state_sha256'] == second['state_sha256']


def test_tuple_and_dictionary_prefixes_describe_the_same_visible_state():
    tuples = [tuple(row[key] for key in ('speaker', 'stage', 'round', 'text')) for row in PREFIX]
    assert instruction_state(CASE, tuples, rubric(), member_spec()) == instruction_state(
        CASE, PREFIX, rubric(), member_spec())


@pytest.mark.parametrize('extra', [
    {'speaker': 'doctor', 'stage': 'FINAL', 'round': 0, 'text': 'PRIVATE_FINAL'},
    {'speaker': 'agent', 'stage': 'A2', 'round': 2, 'text': 'PRIVATE_TARGET_RESPONSE'},
])
def test_final_or_selected_response_in_prefix_is_rejected_before_judging(extra):
    client = Client([])
    with pytest.raises(ValueError):
        make_instruction_auditor(client)(CASE, PREFIX + [extra], rubric(), member_spec())
    assert client.calls == []


@pytest.mark.parametrize('bad_level', ['1', 'NA', '', None, True])
def test_only_real_zero_or_two_target_levels_are_admissible(bad_level):
    client = Client([])
    with pytest.raises(ValueError):
        make_instruction_auditor(client)(CASE, PREFIX, rubric(), member_spec(level=bad_level))
    assert client.calls == []


@pytest.mark.parametrize('bad_turn', [0, -1, True, '2', None])
def test_selected_turn_requires_a_positive_integer(bad_turn):
    spec = member_spec()
    spec['selected_agent_turn_id'] = bad_turn
    with pytest.raises(ValueError):
        instruction_state(CASE, PREFIX, rubric(), spec)


@pytest.mark.parametrize('changes', [
    {'instruction_aligned': 1},
    {'unsupported_context_premise': 'false'},
    {'instruction_alignment_quote': 'Invented instruction passage'},
    {'instruction_alignment_reason': '   '},
    {'unsupported_context_premise': True, 'unsupported_context_premise_quote': '',
     'unsupported_quote_source': 'instruction'},
    {'conclusion_prescribed': True, 'conclusion_quote': 'Invented rubric passage',
     'conclusion_quote_source': 'rubric'},
    {'unsupported_context_premise': False, 'unsupported_context_premise_quote': GUIDE,
     'unsupported_quote_source': 'instruction'},
    {'conclusion_prescribed': False, 'conclusion_quote': '', 'conclusion_quote_source': 'rubric'},
])
def test_invalid_evidence_has_one_correction_then_an_operational_error(changes):
    invalid = answer(**changes)
    client = Client([invalid, invalid])
    cache = CallCache(None)
    audit = cached_instruction_auditor(make_instruction_auditor(client), cache)
    with pytest.raises(IntegrationUnavailable):
        audit(CASE, PREFIX, rubric(), member_spec())
    assert len(client.calls) == 2
    assert not cache.store  # Invalid evidence never becomes a cached negative judgment.


def test_one_evidence_correction_preserves_input_and_records_both_decisions():
    client = Client([answer(instruction_alignment_quote='Not in the instruction'), answer()])
    result = make_instruction_auditor(client)(CASE, PREFIX, rubric(), member_spec())
    assert result['passed'] is True
    assert len(client.calls) == len(result['requests']) == len(result['decisions']) == 2
    assert len(result['validation_errors']) == 1
    original = client.calls[0]['state']
    assert {key: client.calls[1]['state'][key] for key in original} == original
    assert result['state_sha256'] == sha256_json(original)


def test_semantic_rejection_is_final_without_retry_or_outcome_compliance_score():
    rejected = answer(instruction_aligned=False,
                      instruction_alignment_reason='The written target lacks a unique M/C boundary; alignment cannot be established.')
    client = Client([rejected])
    result = make_instruction_auditor(client)(CASE, PREFIX, rubric(), member_spec())
    assert result['passed'] is False
    assert result['reasons'] == ['instruction_alignment']
    assert result['answer'] == rejected
    assert result['validation_errors'] == [] and len(client.calls) == 1
    assert not ({'reward', 'outcome', 'compliance_score', 'keep', 'source_level'} & result.keys())


def test_blocking_quotes_can_reference_a_rubric_field_but_not_another_source():
    state = instruction_state(CASE, PREFIX, rubric(), member_spec())
    clause = state['rubric']['scores']['2']['behavior']
    rejected = answer(unsupported_context_premise=True, unsupported_context_premise_quote=clause,
                      unsupported_quote_source='rubric')
    validate_evidence(rejected, state)
    with pytest.raises(ValueError):
        validate_evidence({**rejected, 'unsupported_quote_source': 'instruction'}, state)


def test_cache_shares_equal_visible_inputs_despite_member_bookkeeping():
    client, cache = Client([answer()]), CallCache(None)
    audit = cached_instruction_auditor(make_instruction_auditor(client), cache)
    first = audit(CASE, PREFIX, rubric(), member_spec())
    spec = member_spec(origin='matched')
    spec['leaf_id'] = 'instruction-case/3.3'
    second = audit(CASE, PREFIX, rubric(), spec)
    assert first == second
    assert len(client.calls) == len(cache.store) == 1


@pytest.mark.parametrize('changed', ['guide', 'rubric', 'prefix', 'level'])
def test_visible_input_changes_invalidate_cache(changed):
    client, cache = Client([answer(), answer()]), CallCache(None)
    audit = cached_instruction_auditor(make_instruction_auditor(client), cache)
    audit(CASE, PREFIX, rubric(), member_spec())
    prefix, candidate, spec = copy.deepcopy(PREFIX), rubric(), member_spec()
    if changed == 'guide':
        spec['intervention_instruction'] += ' Preserve the stated uncertainty.'
    elif changed == 'rubric':
        candidate['scores']['2']['behavior'] += ' State any unresolved support explicitly.'
        spec['rubric_sha256'] = sha256_json(candidate)
    elif changed == 'prefix':
        prefix[-1]['text'] += ' Include the stated uncertainty.'
        spec['prefix_sha256'] = sha256_json(prefix)
    else:
        spec['requested_level'] = '0'
    audit(CASE, prefix, candidate, spec)
    assert len(client.calls) == len(cache.store) == 2


@pytest.mark.parametrize('settings', [
    {'model': 'other-instruction-model'}, {'system': 'Changed independent audit prompt.'},
    {'max_tokens': 4096}, {'timeout_s': 240},
])
def test_judge_configuration_changes_invalidate_cache(settings):
    cache = CallCache(None)
    first, changed = Client([answer()]), Client([answer()], **settings)
    cached_instruction_auditor(make_instruction_auditor(first), cache)(CASE, PREFIX, rubric(), member_spec())
    cached_instruction_auditor(make_instruction_auditor(changed), cache)(CASE, PREFIX, rubric(), member_spec())
    assert len(first.calls) == len(changed.calls) == 1
    assert len(cache.store) == 2


def test_optional_policy_identity_is_frozen_in_each_cache_wrapper():
    client, cache = Client([answer(), answer()]), CallCache(None)
    auditor = make_instruction_auditor(client)
    before = cached_instruction_auditor(auditor, cache, policy_sha256='policy-before')
    after = cached_instruction_auditor(auditor, cache, policy_sha256='policy-after')
    before(CASE, PREFIX, rubric(), member_spec())
    after(CASE, PREFIX, rubric(), member_spec())
    before(CASE, PREFIX, rubric(), member_spec())
    assert len(client.calls) == len(cache.store) == 2


RECORDED = json.loads((Path(__file__).parent / 'fixtures' /
                       'intervention_instruction_audit_batch1.json').read_text())['cases']


@pytest.mark.parametrize('record', RECORDED, ids=[row['id'] for row in RECORDED])
def test_recorded_keep_guides_cannot_bypass_a_negative_instruction_review(record):
    client = Client([record['expected_review']])
    result = make_instruction_auditor(client)(record['case'], record['prefix'],
                                             record['rubric'], record['member_spec'])
    assert result['passed'] is False
    assert result['answer'] == record['expected_review']
    assert result['reasons']
    assert len(client.calls) == 1
    state = client.calls[0]['state']
    assert state['requested_level'] == int(record['member_spec']['requested_level'])
    assert state['intervention_instruction'] == record['member_spec']['intervention_instruction']
    assert not ({'source_agent_reply', 'target_agent_reply', 'source_outcome', 'feedback', 'gold'} & state.keys())
