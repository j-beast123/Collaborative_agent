"""Opt-in literal rubric evidence selection preserves the gate's admission rules."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json, sha256_text
from src.online_rl.tree.luna import LunaDecisionClient, parse_objects
from src.online_rl.tree.quality_gate import (
    FIELDS, ITEM, INPUT_VERSION, PROTOCOL, RUBRIC_QUOTE_FIELDS,
    RUBRIC_QUOTE_PROTOCOL, evidence_posthoc_gate,
)
from test_quality_gate_v2 import CASE, REPLY, Client, answer, rubric, tin


def new_gate(client):
    return evidence_posthoc_gate(client, {CASE['case_id']: CASE}, observed_context=True,
                                 rubric_quote_protocol=RUBRIC_QUOTE_PROTOCOL)


def catalog(candidate):
    context = candidate['applicability']
    return list(dict.fromkeys(['', candidate['name'], context['user_side_context'],
                              context['agent_side_state'], candidate['required_behavior'],
                              *(candidate['scores'][str(i)]['behavior'] for i in range(3))]))


def test_legacy_default_settings_and_messages_are_unchanged():
    client = Client([answer()])
    gate = evidence_posthoc_gate(client, {CASE['case_id']: CASE}, observed_context=True)
    expected = {
        'protocol': PROTOCOL, 'input_version': INPUT_VERSION,
        'model': client.model, 'prompt_sha256': sha256_text(client.system),
        'client_protocol_sha256': client.protocol_sha256,
        'max_tokens': client.max_tokens, 'timeout_s': None,
        'observed_context': True,
        'contract_sha256': sha256_json({'fields': FIELDS, 'item': ITEM}),
        'evidence_corrections': 1,
    }
    assert gate.settings == expected
    assert gate.protocol == sha256_json(expected)
    result = gate(tin(), rubric())
    call = client.calls[0]
    assert call['fields'] == FIELDS and call['items'] == {'candidate': ITEM}
    request = result['evidence']['requests'][0]
    assert set(request) == {'messages', 'messages_sha256'}
    assert json.loads(request['messages'][1]['content']) == {
        'stage': call['stage'], 'state': call['state'], 'items': call['items']}


def test_only_five_rubric_quote_fields_receive_complete_unique_visible_field_enums():
    before_fields, before_item = copy.deepcopy(FIELDS), copy.deepcopy(ITEM)
    candidate = rubric()
    # Rubric-local private strings must not enter the schema or evidence request.
    candidate['name'] = 'Compare onset timing — α'
    client = Client([answer()])
    result = new_gate(client)(tin(), candidate)
    fields = client.calls[0]['fields']
    for name in RUBRIC_QUOTE_FIELDS:
        assert fields[name] == {'type': 'string', 'enum': catalog(candidate)}
    for name in set(FIELDS) - set(RUBRIC_QUOTE_FIELDS):
        assert fields[name] == FIELDS[name]
    assert fields['source_minimum_quote'] == {'type': 'string'}
    assert fields['source_completion_quote'] == {'type': 'string'}
    packet = json.dumps(client.calls)
    for private in ('PRIVATE_', 'FUTURE_', 'source_outcome', 'source_score',
                    'ground_truth', 'delta_k', candidate['rubric_id']):
        # rubric_id remains visible in the input but may not enter any quote enum.
        if private != candidate['rubric_id']:
            assert private not in packet
    assert candidate['rubric_id'] not in fields['observable_quote']['enum']
    assert FIELDS == before_fields and ITEM == before_item
    request = result['evidence']['requests'][0]
    assert request['output_fields'] == fields
    assert request['output_fields_sha256'] == sha256_json(fields)
    assert request['messages_sha256'] == sha256_json(request['messages'])
    assert 'complete' in client.calls[0]['items']['candidate']['evidence_contract']


def test_new_mode_binds_cache_settings_without_changing_legacy_identity():
    legacy = evidence_posthoc_gate(Client([]), {CASE['case_id']: CASE}, observed_context=True)
    explicit_none = evidence_posthoc_gate(Client([]), {CASE['case_id']: CASE}, observed_context=True,
                                          rubric_quote_protocol=None)
    updated = new_gate(Client([]))
    assert legacy.settings == explicit_none.settings
    assert updated.settings['rubric_quote_protocol'] == RUBRIC_QUOTE_PROTOCOL
    assert updated.settings['contract_sha256'] != legacy.settings['contract_sha256']
    assert updated.protocol != legacy.protocol
    unchanged = set(legacy.settings) - {'contract_sha256'}
    assert all(updated.settings[key] == legacy.settings[key] for key in unchanged)


@pytest.mark.parametrize('mode', ['', True, 'other_mode'])
def test_unknown_quote_modes_are_rejected(mode):
    with pytest.raises(ValueError, match='Unsupported rubric quote protocol'):
        evidence_posthoc_gate(Client([]), {CASE['case_id']: CASE}, rubric_quote_protocol=mode)


def test_exact_rubric_quotes_preserve_rejection_and_blind_source_anchor():
    candidate = rubric()
    decision = answer(score_boundaries_clear=False,
        score_boundaries_quote=candidate['scores']['0']['behavior'],
        score_boundaries_other_quote=candidate['scores']['1']['behavior'],
        score_boundaries_reason='The two quoted anchors overlap in this test.')
    client = Client([decision])
    result = new_gate(client)(tin(), candidate)
    assert not result['passed']
    assert result['reasons'] == ['not_score_boundaries_clear']
    assert result['source_anchor'] == {'expected_level': '2', 'observed_level': '2', 'matches': True}
    assert result['assessment'] == decision
    assert len(client.calls) == 1  # A quality rejection never triggers a re-vote.
    failure_client = Client([answer()])
    mismatch = new_gate(failure_client)(tin('failure'), candidate)
    assert mismatch['reasons'] == ['source_anchor_mismatch']
    assert mismatch['source_anchor']['expected_level'] == '0'


@pytest.mark.parametrize('quote', ['Invented absent instruction.', 'onset timing'])
def test_bad_rubric_quotes_remain_operational_errors_after_one_correction(quote):
    bad = answer(observable=False, observable_quote=quote)
    client = Client([bad, bad])
    with pytest.raises(IntegrationUnavailable) as caught:
        new_gate(client)(tin(), rubric())
    assert len(client.calls) == 2
    assert len(caught.value.audit['validation_errors']) == 2
    assert caught.value.audit['requests'][1]['output_fields'] == client.calls[1]['fields']


def test_corrected_quote_saves_exact_messages_for_both_attempts():
    bad = answer(observable=False, observable_quote='Invented text.')
    good = answer(observable=False, observable_quote=rubric()['required_behavior'])
    client = Client([bad, good])
    result = new_gate(client)(tin(), rubric())
    assert result['reasons'] == ['not_observable']
    assert len(client.calls) == 2
    for request, call in zip(result['evidence']['requests'], client.calls):
        assert request['messages'] == [
            {'role': 'system', 'content': client.system},
            {'role': 'user', 'content': json.dumps({
                'stage': call['stage'], 'state': call['state'], 'items': call['items']}, ensure_ascii=False)}]


@pytest.mark.parametrize('source_quote', ['Compare onset timing.', 'FUTURE_AGENT_SENTINEL'])
def test_source_evidence_cannot_be_borrowed_from_prefix_future_or_rubric(source_quote):
    bad = answer(source_completion_quote=source_quote)
    client = Client([bad, bad])
    with pytest.raises(IntegrationUnavailable, match='source Agent reply'):
        new_gate(client)(tin(), rubric())
    assert len(client.calls) == 2


def test_concurrent_gate_calls_use_isolated_catalogs():
    barrier = Barrier(2)
    before_fields, before_item = copy.deepcopy(FIELDS), copy.deepcopy(ITEM)

    class ConcurrentClient(Client):
        def judge(self, stage, state, items, fields, purpose):
            barrier.wait(timeout=5)
            expected = state['candidate_rubric']['required_behavior']
            assert expected in fields['observable_quote']['enum']
            opposite = 'SECOND unique behavior.' if expected.startswith('FIRST') else 'FIRST unique behavior.'
            assert opposite not in fields['observable_quote']['enum']
            decision = answer(observable_quote=expected)
            return {'candidate': decision}, {'parsed': decision}

    first, second = rubric(), rubric()
    first['required_behavior'], second['required_behavior'] = 'FIRST unique behavior.', 'SECOND unique behavior.'
    gate = new_gate(ConcurrentClient([]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(gate, tin(), candidate) for candidate in (first, second)]
        results = [future.result(timeout=10) for future in futures]
    assert all(result['passed'] for result in results)
    assert results[0]['evidence']['requests'][0]['output_fields'] != results[1]['evidence']['requests'][0]['output_fields']
    assert FIELDS == before_fields and ITEM == before_item


def test_real_luna_provider_schema_parser_and_saved_messages_match(monkeypatch):
    monkeypatch.delenv('API_USAGE_LOG', raising=False)
    payloads = []
    decision = answer(observable_quote=rubric()['required_behavior'])

    def post(url, payload, key, timeout):
        payloads.append(copy.deepcopy(payload))
        return {'choices': [{'message': {'content': json.dumps({'answers': {'candidate': decision}})}}]}

    client = LunaDecisionClient(model='test-luna', key='', system='Gate system.', post=post)
    result = new_gate(client)(tin(), rubric())
    assert result['passed'] and len(payloads) == 1
    fields = payloads[0]['response_format']['json_schema']['schema']['properties']['answers']['properties']['candidate']['properties']
    for name in RUBRIC_QUOTE_FIELDS:
        assert fields[name]['enum'] == catalog(rubric())
    parsed, error = parse_objects(json.dumps({'answers': {'candidate': decision}}), ['candidate'], fields)
    assert error is None and parsed['candidate'] == decision
    invalid = {**decision, 'observable_quote': 'onset timing'}
    parsed, error = parse_objects(json.dumps({'candidate': invalid}), ['candidate'], fields)
    assert parsed is None and 'not one of' in error
    assert result['evidence']['requests'][0]['messages'] == payloads[0]['messages']


def test_real_luna_invalid_enum_is_not_mapped_to_rejection_or_zero(monkeypatch):
    monkeypatch.delenv('API_USAGE_LOG', raising=False)
    payloads = []
    invalid = answer(observable_quote='Invented evidence.')

    def post(url, payload, key, timeout):
        payloads.append(copy.deepcopy(payload))
        return {'choices': [{'message': {'content': json.dumps({'candidate': invalid})}}]}

    client = LunaDecisionClient(model='test-luna', key='', system='Gate system.', post=post)
    with pytest.raises(IntegrationUnavailable, match='after one repair'):
        new_gate(client)(tin(), rubric())
    assert len(payloads) == 2
    assert client.log[-1]['parsed'] is None


@pytest.mark.parametrize('literal', ['  Literal quoted field.\n', 'Clinical Fact', 'CLINICAL FACT'])
def test_luna_enum_preserves_exact_whitespace_and_case_distinct_strings(literal):
    fields = {'quote': {'type': 'string', 'enum': ['', '  Literal quoted field.\n',
                                                 'Clinical Fact', 'CLINICAL FACT']}}
    parsed, error = parse_objects(json.dumps({'candidate': {'quote': literal}}), ['candidate'], fields)
    assert error is None
    assert parsed == {'candidate': {'quote': literal}}


def test_luna_existing_choice_canonicalisation_still_works():
    fields = {'choice': {'type': 'string', 'enum': ['Y', 'N']}}
    parsed, error = parse_objects(json.dumps({'candidate': {'choice': ' y '}}), ['candidate'], fields)
    assert error is None
    assert parsed == {'candidate': {'choice': 'Y'}}


def test_catalog_full_field_with_surrounding_whitespace_reaches_real_luna_unchanged(monkeypatch):
    monkeypatch.delenv('API_USAGE_LOG', raising=False)
    candidate = rubric()
    candidate['required_behavior'] = '  Compare the onset timing of the two considerations.\n'
    literal = candidate['required_behavior']

    def post(url, payload, key, timeout):
        decision = answer(observable_quote=literal)
        return {'choices': [{'message': {'content': json.dumps({'candidate': decision})}}]}

    client = LunaDecisionClient(model='test-luna', key='', system='Gate system.', post=post)
    result = new_gate(client)(tin(), candidate)
    assert result['passed']
    assert result['assessment']['observable_quote'] == literal
    assert result['evidence']['attempts'] == 1
