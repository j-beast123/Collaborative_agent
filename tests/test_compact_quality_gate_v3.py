"""v3 offline assembly tests. Fake stage answers are structure fixtures, not gold."""
from __future__ import annotations

import json
import glob
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.online_rl.integrated.records import sha256_text, sha256_json
from src.online_rl.tree.compact_quality_gate_v3 import (
    GateExecutionError, SCHEMA_PATH, assemble_v3, validate_v3_audit_checks,
    validate_v3_bundle,
    _source_behavior, _applicability,
)
from src.online_rl.tree.quality_gate import public_gate_state
from src.online_rl.tree.quality_gate import QUALITY_CATEGORIES
from tests.test_public_mc_evidence import rubric, tin


def inputs(*, outcome='success', level='2', request='unresolved', quality_codes=(),
           aligned=True):
    state = public_gate_state(tin(outcome), rubric())
    doctor = next(r for r in reversed(state['dialogue_before_target_turn']) if r['speaker'] == 'doctor')
    source_reply = state['source_agent_reply']['text']
    q = {'decision': 'reject' if quality_codes else 'pass', 'failure_codes': list(quality_codes),
         'checks': {name: name.upper() not in quality_codes for name in QUALITY_CATEGORIES},
         'evidence': ([{'turn_id': 'criterion:required_behavior',
                        'quote': state['candidate_rubric']['required_behavior']}]
                      if quality_codes else []),
         'reason': ('; '.join(quality_codes) if quality_codes else 'No identified quality defect.')}
    part = {'clause': state['candidate_rubric']['applicability']['user_side_context'],
            'status': 'supported', 'evidence': [{'turn_id': doctor['turn_id'],
                                               'quote': doctor['text'], 'speaker': 'doctor'}]}
    no_prior = {'clause': state['candidate_rubric']['applicability']['agent_side_state'],
                'status': 'not_required', 'evidence': []}
    app = {'status': 'applicable', 'applicable': True, 'request_state': request,
           'request_evidence': [{'turn_id': doctor['turn_id'], 'quote': doctor['text'],
                                 'speaker': 'doctor'}],
           'remaining_request': 'Compare onset timing.' if request in ('unresolved', 'partial') else '',
           'conditions': {'doctor_condition': [part], 'agent_condition': [no_prior]}}
    if request == 'resolved':
        app['request_evidence'].append({'turn_id': state['dialogue_before_target_turn'][1]['turn_id'],
                                         'quote': state['dialogue_before_target_turn'][1]['text'],
                                         'speaker': 'agent'})
    checks = [{'ref': 'score_anchors.' + str(i), 'met': str(i) == level} for i in range(3)]
    source = {'stage2': {'level': level,
                         'quotes': ['Its onset is immediate.'] if level != 'UNSCORABLE' else [],
                         'checks': checks, 'defect': 'Scale ambiguity.' if level == 'UNSCORABLE' else ''},
              'evidence_validation': {'protocol': 'independent_source_v3',
                                      'exact_quotes_validated': True, 'requests': [{'prefix_only': True}]}}
    assert 'Its onset is immediate.' in source_reply
    answer = {'instruction_aligned': aligned,
              'instruction_alignment_quote': 'guide' if aligned else '',
              'instruction_alignment_reason': 'The guide is aligned.' if aligned else 'M is not removed.',
              'unsupported_context_premise': False,
              'unsupported_context_premise_quote': '',
              'unsupported_context_premise_reason': 'None.',
              'unsupported_quote_source': 'none', 'conclusion_prescribed': False,
              'conclusion_quote': '', 'conclusion_reason': 'None.',
              'conclusion_quote_source': 'none'}
    instruction_state = {'intervention_instruction': 'guide',
                         'rubric': state['candidate_rubric']}
    instruction = {'protocol': 'intervention_instruction_audit_v1', 'answer': answer,
                   'state_sha256': sha256_json(instruction_state)}
    return {'state': state, 'source_outcome': outcome, 'quality': q,
            'source': source, 'applicability': app, 'instruction': instruction,
            'instruction_state': instruction_state,
            'quality_prompt_sha256': sha256_text('pinned-v5-plus-output-only'),
            'candidate_namespace': 'gemini_gate_v3',
            'candidate_generator_model': 'gemini-pilot',
            'model_settings': {'model': 'gpt-6-luna', 'max_tokens': 8192}}


def assembled(**kw):
    return assemble_v3(**inputs(**kw))


def test_valid_pass_has_all_checks_and_local_decision():
    result = assembled()
    wire = result['compact_result']
    assert wire['decision'] == 'pass' and wire['source_level'] == 2 and not wire['failure_codes']
    assert result['passed'] and result['audit_checks']['decision_computed_locally']
    assert result['audit_checks']['applicability']['request_state'] == 'active'
    assert all(result['audit_checks']['rubric_quality'].values())
    assert all(result['audit_checks']['intervention_quality']['quality'].values())
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(wire)


def test_success_one_and_failure_two_reject_without_revote():
    one = assembled(level='1')['compact_result']
    two = assembled(level='2', outcome='failure')['compact_result']
    assert one['decision'] == two['decision'] == 'reject'
    assert one['source_level'] == 1 and two['source_level'] == 2
    assert one['failure_codes'] == two['failure_codes'] == ['SOURCE_ANCHOR_MISMATCH']


def test_unscorable_overrides_reject_but_preserves_other_failure_codes():
    wire = assembled(level='UNSCORABLE', quality_codes=('BOUNDARY_COVERAGE',))['compact_result']
    assert wire['decision'] == 'unscorable' and wire['source_level'] is None
    assert wire['failure_codes'] == ['BOUNDARY_COVERAGE', 'SOURCE_ANCHOR_MISMATCH', 'SOURCE_UNSCORABLE']
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(wire)


def test_request_resolved_or_instruction_weak_rejects_even_with_quality_model_pass():
    resolved = assembled(request='resolved')['compact_result']
    weak = assembled(aligned=False)['compact_result']
    assert resolved['decision'] == weak['decision'] == 'reject'
    assert 'REQUEST_NOT_ACTIVE' in resolved['failure_codes']
    assert 'INSTRUCTION_ALIGNMENT' in weak['failure_codes']


def test_missing_or_corrupt_stages_are_execution_errors_not_rejects():
    for mutate in (
        lambda p: p['source']['evidence_validation'].update(exact_quotes_validated=False),
        lambda p: p['source']['stage2'].update(level=''),
        lambda p: p['applicability'].update(request_evidence=[]),
        lambda p: p['applicability'].update(remaining_request=''),
        lambda p: p['instruction'].pop('answer'),
        lambda p: p['quality'].pop('checks'),
    ):
        payload = inputs()
        mutate(payload)
        with pytest.raises(GateExecutionError):
            assemble_v3(**payload)


def test_v3_identity_is_distinct_from_v2_and_source_input_hides_outcome():
    result = assembled()
    audit = result['audit_checks']
    assert result['protocol'] == 'public_mc_gate_compact_v3_1'
    assert audit['candidate_content_sha256'] and result['cache_identity_sha256']
    assert 'outcome' not in json.dumps(audit['applicability'])
    assert 'source_agent_reply' not in json.dumps(audit['applicability'])
    assert 'PRIVATE_FINAL_SENTINEL' not in json.dumps(audit['applicability'])


def test_sidecar_missing_required_checks_cannot_validate():
    checks = assembled()['audit_checks']
    for key in ('source_outcome', 'expected_source_level', 'rubric_quality', 'source_behavior',
                'applicability', 'intervention_quality'):
        changed = dict(checks)
        changed.pop(key)
        with pytest.raises(GateExecutionError):
            validate_v3_audit_checks(changed)
    changed = dict(checks)
    changed['rubric_quality'] = dict(changed['rubric_quality'])
    changed['rubric_quality'].pop('observability')
    with pytest.raises(GateExecutionError):
        validate_v3_audit_checks(changed)


def test_saved_real_v9_source_and_v75_prefix_shapes_offline():
    paths = glob.glob('/home/kai/audits/direct_luna_v9_20261011/repo/outputs/'
                      'train120_v9_1*/r05*/batches/*/case_*/calls.jsonl')
    if not paths:
        pytest.skip('Local paid-run evidence is unavailable in this checkout')
    found = False
    for path in paths:
        for line in Path(path).open():
            row = json.loads(line)
            if row.get('kind') != 'posthoc_gate':
                continue
            gate = row['value']
            state = json.loads(gate['evidence']['requests'][0]['messages'][1]['content'])['state']
            blind = {'stage2': gate['blind_source_assessment'],
                     'evidence_validation': gate['evidence']['blind_source']}
            actual = _source_behavior(blind, state)
            assert actual['independent_level'] in ('0', '1', '2', 'UNSCORABLE')
            found = True
            break
        if found:
            break
    assert found

    base = Path('/home/kai/audits/scorer_quality_isolated_20261011/harness/v7_5_isolate/run')
    if not (base / 'request_template.jsonl').is_file() or not (base / 'paid_attempts_v75.jsonl').is_file():
        pytest.skip('Local v7.5 paid-run evidence is unavailable in this checkout')
    requests = {row['request_id']: row for row in
                map(json.loads, (base / 'request_template.jsonl').open())}
    for row in map(json.loads, (base / 'paid_attempts_v75.jsonl').open()):
        if row.get('event') != 'attempt_result' or row.get('status') != 'valid_execution_contract':
            continue
        record = row['validation']['applicability']
        rid = next(iter(record))
        packet = json.loads(requests[row['request_id']]['request']['messages'][1]['content'])
        raw = packet['state']
        state = {'case_context': raw['case_context'], 'case_evidence': raw['case_evidence'],
                 'dialogue_before_target_turn': raw['dialogue_before_target_reply'],
                 'candidate_rubric': {'applicability': {
                     'user_side_context': packet['items'][rid]['doctor_condition'],
                     'agent_side_state': packet['items'][rid]['agent_condition']}}}
        actual, _ = _applicability(record[rid], state)
        assert actual['request_state'] in ('active', 'resolved', 'absent', 'uncertain')
        break
    else:
        pytest.fail('No saved valid v7.5 applicability response')


def test_witness_boundary_rejects_forged_pass_and_identity_swap():
    bundle = assembled()
    audit = bundle['audit_checks']
    expected = {'expected_candidate_sha': audit['candidate_content_sha256'],
                'expected_input_sha': audit['input_sha256'],
                'namespace': audit['candidate_namespace'],
                'generator_model': audit['candidate_generator_model']}
    assert validate_v3_bundle(bundle, **expected)['passed']
    with pytest.raises(GateExecutionError):
        validate_v3_bundle(bundle, **{**expected, 'expected_candidate_sha': '0'*64})
    forged = json.loads(json.dumps(bundle))
    forged['audit_checks']['rubric_quality']['observability'] = False
    with pytest.raises(GateExecutionError):
        validate_v3_bundle(forged, **expected)
    forged = json.loads(json.dumps(bundle))
    forged['compact_result'].pop('reason')
    with pytest.raises(GateExecutionError):
        validate_v3_bundle(forged, **expected)
    forged = json.loads(json.dumps(bundle))
    forged['passed'] = False
    with pytest.raises(GateExecutionError):
        validate_v3_bundle(forged, **expected)


def test_actual_first_gemini_pilot_blind_source_ledger_replay():
    base = Path('/home/kai/audits/compact_gate_contract_v1/gate_v3_pilot/candidates/C1_2.0')
    if not (base/'blind_source.json').exists():
        pytest.skip('Paid pilot raw response has not completed yet')
    quality = json.loads((base/'quality.json').read_text())
    state = json.loads(quality['request']['messages'][1]['content'])['state']
    source = json.loads((base/'blind_source.json').read_text())['response']
    actual = _source_behavior(source, state)
    assert actual['independent_level'] == '1'
    assert actual['score_unique'] is True
    assert actual['anchor_matches'] == ['1']
