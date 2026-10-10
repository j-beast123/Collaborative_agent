"""Isolated v3 Gate decision assembly; not registered in the training runner.

The quality judge, prefix-only applicability reviewer, frozen-instruction auditor,
and independent blind source scorer remain separate. Their validated records are
assembled here. A model-supplied ``pass`` is never an admission decision.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator

from src.online_rl.integrated.records import sha256_json, sha256_text
from src.online_rl.tree.compact_quality_gate import QUALITY_CODES, QUALITY_FIELDS, validate_quality
from src.online_rl.tree.instruction_audit import validate_evidence as validate_instruction_evidence
from src.online_rl.tree.quality_gate import QUALITY_CATEGORIES

PROTOCOL = 'public_mc_gate_compact_v3_1'
ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / 'prompts/sherpa/gate_public_mc_compact_result_v3_1.schema.json'
AUDIT_SCHEMA_PATH = ROOT / 'prompts/sherpa/gate_public_mc_audit_checks_v3_1.schema.json'
PROMPT_PATH = ROOT / 'prompts/sherpa/gate_public_mc_compact_v3_1.txt'
MODULE_PATH = Path(__file__)
DECISION_CODES = tuple(QUALITY_CODES) + (
    'SOURCE_ANCHOR_MISMATCH', 'SOURCE_UNSCORABLE', 'APPLICABILITY_UNSUPPORTED',
    'REQUEST_NOT_ACTIVE', 'INSTRUCTION_ALIGNMENT', 'UNSUPPORTED_CONTEXT_PREMISE',
    'CONCLUSION_PRESCRIBED',
)
INTERVENTION_KEYS = ('instruction_aligned', 'no_unsupported_context_premise',
                     'no_conclusion_prescribed')
QUALITY_FIELDS_V3 = {
    **copy.deepcopy(QUALITY_FIELDS),
    'checks': {'type': 'object', 'additionalProperties': False,
               'required': list(QUALITY_CATEGORIES),
               'properties': {name: {'type': 'boolean'} for name in QUALITY_CATEGORIES}},
}


class GateExecutionError(ValueError):
    """Missing, truncated or invalid evidence; never counted as rubric rejection."""


def validate_quality_v3(answer: dict, state: dict) -> dict:
    """Every named category gets its own model boolean before local admission."""
    if not isinstance(answer, dict) or set(answer) != {
            'decision', 'checks', 'failure_codes', 'evidence', 'reason'}:
        raise GateExecutionError('Missing v3 quality category checks')
    flags = answer['checks']
    if (not isinstance(flags, dict) or set(flags) != set(QUALITY_CATEGORIES)
            or any(type(value) is not bool for value in flags.values())):
        raise GateExecutionError('Incomplete v3 quality boolean map')
    codes = answer['failure_codes']
    if not isinstance(codes, list):
        raise GateExecutionError('Missing v3 quality failure codes')
    for category in QUALITY_CATEGORIES:
        if flags[category] == (category.upper() in codes):
            raise GateExecutionError('Quality boolean contradicts category code: ' + category)
    if 'CRITERION_UNCERTAIN' in codes and all(flags.values()):
        raise GateExecutionError('Material criterion uncertainty needs a failed named check')
    try:
        validate_quality({key: copy.deepcopy(value) for key, value in answer.items()
                          if key != 'checks'}, state)
    except ValueError as exc:
        raise GateExecutionError(str(exc)) from exc
    return copy.deepcopy(answer)


def _literal_quote(row: dict, catalog: dict[str, str]) -> None:
    if (not isinstance(row, dict) or set(row) != {'turn_id', 'quote'}
            or not isinstance(row['turn_id'], str) or row['turn_id'] not in catalog
            or not isinstance(row['quote'], str) or not row['quote'].strip()
            or row['quote'] not in catalog[row['turn_id']]):
        raise GateExecutionError('Nonliteral or unknown Gate evidence quote')


def _prefix_catalog(state: dict) -> dict[str, str]:
    return {row['turn_id']: row['text'] for row in
            state['case_evidence'] + state['dialogue_before_target_turn']}


def _applicability(app: dict, state: dict) -> tuple[dict, list[str]]:
    """Adapt a validated v7.5 prefix-only result; preserve lifecycle evidence."""
    if not isinstance(app, dict) or app.get('status') not in ('applicable', 'not_applicable', 'uncertain'):
        raise GateExecutionError('Missing validated v7.5 applicability result')
    if type(app.get('applicable')) not in (bool, type(None)):
        raise GateExecutionError('Invalid applicability value')
    raw_state = app.get('request_state')
    if raw_state not in ('not_required', 'absent', 'unresolved', 'resolved', 'partial', 'uncertain'):
        raise GateExecutionError('Missing v7.5 request lifecycle')
    request_evidence = app.get('request_evidence')
    if not isinstance(request_evidence, list):
        raise GateExecutionError('Missing request lifecycle evidence')
    catalog = _prefix_catalog(state)
    evidence = []
    for row in request_evidence:
        quote = {'turn_id': row.get('turn_id'), 'quote': row.get('quote')} if isinstance(row, dict) else row
        _literal_quote(quote, catalog)
        evidence.append(quote)
    remaining = app.get('remaining_request')
    if not isinstance(remaining, str):
        raise GateExecutionError('Missing remaining request record')
    if raw_state in ('unresolved', 'partial', 'resolved', 'uncertain') and not evidence:
        raise GateExecutionError('Claimed request lifecycle has no prefix quote')
    if raw_state in ('absent', 'not_required') and evidence:
        raise GateExecutionError('Absent/non-required request has invented quote')
    if raw_state in ('unresolved', 'partial') and not remaining.strip():
        raise GateExecutionError('Active request has no concrete remaining request')
    if raw_state in ('resolved', 'absent', 'not_required') and remaining.strip():
        raise GateExecutionError('Closed request cannot have remaining request')
    conditions = app.get('conditions')
    if not isinstance(conditions, dict) or set(conditions) != {'doctor_condition', 'agent_condition'}:
        raise GateExecutionError('Missing condition-by-condition evidence')
    statuses = []
    for side in ('doctor_condition', 'agent_condition'):
        if not isinstance(conditions[side], list) or not conditions[side]:
            raise GateExecutionError('Missing applicability clause audit')
        for part in conditions[side]:
            if not isinstance(part, dict) or part.get('status') not in (
                    'supported', 'not_required', 'absent', 'unmet', 'uncertain'):
                raise GateExecutionError('Invalid applicability clause status')
            statuses.append(part['status'])
            for quote in part.get('evidence', []):
                _literal_quote({'turn_id': quote.get('turn_id'), 'quote': quote.get('quote')}, catalog)
    supported = app['applicable'] is True and all(s in ('supported', 'not_required') for s in statuses)
    if (app['applicable'] is True) != all(s in ('supported', 'not_required') for s in statuses):
        raise GateExecutionError('Applicability status contradicts grounded clauses')
    # `partial` is active only when the assessor names a concrete residual;
    # the local check does not invent such a residual from historical requests.
    request_state = 'active' if raw_state in ('unresolved', 'partial') else raw_state
    if request_state == 'not_required':
        request_state = 'absent'
    checks = {'supported': supported, 'request_state': request_state,
              'raw_request_state': raw_state, 'remaining_request': remaining,
              'evidence': evidence, 'conditions': copy.deepcopy(conditions),
              'prefix_sha256': sha256_json({
                  'case_context': state['case_context'],
                  'dialogue_before_target_turn': state['dialogue_before_target_turn'],
                  'candidate_applicability': state['candidate_rubric']['applicability'],
              })}
    codes = ([] if supported else ['APPLICABILITY_UNSUPPORTED']) + (
        [] if request_state == 'active' else ['REQUEST_NOT_ACTIVE'])
    return checks, codes


def _source_behavior(source: dict, state: dict) -> dict:
    if not isinstance(source, dict) or not isinstance(source.get('stage2'), dict):
        raise GateExecutionError('Blind source result is missing')
    validation = source.get('evidence_validation')
    if not isinstance(validation, dict) or validation.get('exact_quotes_validated') is not True:
        raise GateExecutionError('Blind source result lacks validated independent evidence')
    stage = source['stage2']
    level = stage.get('level')
    if level not in ('0', '1', '2', 'UNSCORABLE'):
        raise GateExecutionError('Blind source level is missing or invalid')
    checks = stage.get('checks')
    if isinstance(checks, list):
        by_ref = {row.get('ref'): row for row in checks if isinstance(row, dict)}
        if not all('score_anchors.' + str(i) in by_ref for i in range(3)):
            raise GateExecutionError('Blind source did not check all three anchors')
        matches = [str(i) for i in range(3) if by_ref['score_anchors.' + str(i)].get('met') is True]
        uncertain = any(by_ref['score_anchors.' + str(i)].get('met') is None for i in range(3))
        unique = level != 'UNSCORABLE' and matches == [level] and not uncertain
        quotes = stage.get('quotes')
    else:
        # The public v3 blind scorer stores its complete unique-anchor ledger
        # in the independently validated raw decision, while normalized stage2
        # retains literal M/C response evidence but drops ledger bookkeeping.
        decisions = validation.get('decisions')
        if not isinstance(decisions, list) or not decisions:
            raise GateExecutionError('Blind source raw unique-anchor ledger is missing')
        parsed = decisions[-1].get('parsed') if isinstance(decisions[-1], dict) else None
        raw = parsed.get('source_criterion') if isinstance(parsed, dict) else None
        if not isinstance(raw, dict) or raw.get('level') != level:
            raise GateExecutionError('Blind source raw and normalized levels differ')
        matches = raw.get('anchor_matches')
        if not isinstance(matches, list) or len(matches) != len(set(matches)) or any(
                x not in ('0', '1', '2') for x in matches):
            raise GateExecutionError('Blind source raw anchor matches are invalid')
        status = raw.get('criterion_status')
        ref = raw.get('anchor_ref')
        unique = (level != 'UNSCORABLE' and status == 'coherent'
                  and matches == [level] and ref == 'score_anchors.' + level)
        if level == 'UNSCORABLE' and (status == 'coherent' or ref != 'none'):
            raise GateExecutionError('Blind UNSCORABLE contradicts raw criterion status')
        checks = [{'ref': 'raw.anchor_matches', 'matches': matches,
                   'criterion_status': status, 'anchor_ref': ref}]
        requirements = stage.get('requirements')
        if not isinstance(requirements, list):
            raise GateExecutionError('Blind source M/C evidence is missing')
        quotes = list(dict.fromkeys(q for row in requirements if isinstance(row, dict)
                                    for q in row.get('response_quotes', [])
                                    if isinstance(q, str) and q))
    if level != 'UNSCORABLE' and not unique:
        raise GateExecutionError('Numeric blind source level lacks unique anchor evidence')
    reply = state['source_agent_reply']['text']
    if not isinstance(quotes, list) or any(not isinstance(q, str) or not q or q not in reply for q in quotes):
        raise GateExecutionError('Blind source quotes are not literal')
    raw_task_frame = validation.get('task_frame')
    if raw_task_frame is not None and not isinstance(raw_task_frame, dict):
        raise GateExecutionError('Blind source task frame has invalid shape')
    return {'independent_level': level, 'score_unique': unique,
            'anchor_matches': matches, 'anchor_checks': copy.deepcopy(checks),
            'evidence': [{'turn_id': state['source_agent_reply']['turn_id'], 'quote': q} for q in quotes],
            'task_frame': copy.deepcopy(raw_task_frame),
            'raw_decision_sha256': sha256_json(validation.get('decisions', [])),
            'normalized_stage2_sha256': sha256_json(stage),
            'blind_protocol': validation.get('protocol'),
            'blind_requests_sha256': sha256_json(validation.get('requests', []))}


def _intervention(instruction: dict, instruction_state: dict) -> tuple[dict, list[str]]:
    if not isinstance(instruction, dict) or instruction.get('protocol') != 'intervention_instruction_audit_v1':
        raise GateExecutionError('Missing independent frozen-instruction audit')
    answer = instruction.get('answer')
    if not isinstance(answer, dict) or set(answer) != {
            'instruction_aligned', 'instruction_alignment_quote', 'instruction_alignment_reason',
            'unsupported_context_premise', 'unsupported_context_premise_quote',
            'unsupported_context_premise_reason', 'unsupported_quote_source',
            'conclusion_prescribed', 'conclusion_quote', 'conclusion_reason',
            'conclusion_quote_source'}:
        raise GateExecutionError('Incomplete instruction audit answer')
    if not isinstance(instruction_state, dict) or instruction.get('state_sha256') != sha256_json(instruction_state):
        raise GateExecutionError('Instruction audit belongs to another frozen candidate/state')
    try:
        validate_instruction_evidence(answer, instruction_state)
    except ValueError as exc:
        raise GateExecutionError('Invalid instruction evidence: ' + str(exc)) from exc
    if any(type(answer[key]) is not bool for key in (
            'instruction_aligned', 'unsupported_context_premise', 'conclusion_prescribed')):
        raise GateExecutionError('Instruction quality flags are not booleans')
    quality = {'instruction_aligned': answer['instruction_aligned'],
               'no_unsupported_context_premise': not answer['unsupported_context_premise'],
               'no_conclusion_prescribed': not answer['conclusion_prescribed']}
    codes = (["INSTRUCTION_ALIGNMENT"] if not quality['instruction_aligned'] else []) + (
        ['UNSUPPORTED_CONTEXT_PREMISE'] if not quality['no_unsupported_context_premise'] else []) + (
        ['CONCLUSION_PRESCRIBED'] if not quality['no_conclusion_prescribed'] else [])
    return {'quality': quality, 'answer': copy.deepcopy(answer),
            'state_sha256': instruction.get('state_sha256'),
            'audit_protocol': instruction.get('protocol')}, codes


def assemble_v3(*, state: dict, source_outcome: str, quality: dict, source: dict,
                applicability: dict, instruction: dict, instruction_state: dict,
                quality_prompt_sha256: str, candidate_namespace: str,
                candidate_generator_model: str,
                model_settings: dict) -> dict:
    """Assemble v3 only from validated independent stage results.

    Missing/invalid stage evidence raises GateExecutionError. It cannot become
    a numeric zero, NA, UNSCORABLE, or a rubric-level rejection.
    """
    if source_outcome not in ('success', 'failure'):
        raise GateExecutionError('Observed source outcome is missing')
    if (not isinstance(candidate_namespace, str) or not candidate_namespace.strip()
            or not isinstance(candidate_generator_model, str) or not candidate_generator_model.strip()):
        raise GateExecutionError('Candidate namespace and generator model are required')
    try:
        quality = validate_quality_v3(quality, state)
        app, app_codes = _applicability(applicability, state)
        behavior = _source_behavior(source, state)
        intervention, instruction_codes = _intervention(instruction, instruction_state)
    except (ValueError, KeyError, TypeError) as exc:
        raise GateExecutionError(str(exc)) from exc
    expected = 0 if source_outcome == 'failure' else 2
    actual = behavior['independent_level']
    rubric_quality = copy.deepcopy(quality['checks'])
    codes = list(dict.fromkeys(quality['failure_codes'] + app_codes + instruction_codes))
    if actual == 'UNSCORABLE' or int(actual) != expected:
        codes.append('SOURCE_ANCHOR_MISMATCH')
    if actual == 'UNSCORABLE':
        codes.append('SOURCE_UNSCORABLE')
    codes = list(dict.fromkeys(codes))
    decision = 'unscorable' if actual == 'UNSCORABLE' else 'reject' if codes else 'pass'
    if decision == 'pass' and not (
            not codes and app['supported'] and app['request_state'] == 'active'
            and behavior['score_unique'] and all(rubric_quality.values())
            and all(intervention['quality'].values())):
        raise GateExecutionError('v3 pass lacks one or more mandatory grounded checks')
    evidence = list(quality['evidence'])
    for row in app['evidence'] + behavior['evidence']:
        if row not in evidence:
            evidence.append(row)
    details = []
    if 'REQUEST_NOT_ACTIVE' in codes:
        details.append('REQUEST_NOT_ACTIVE: prior request state is ' + app['raw_request_state'])
    if 'APPLICABILITY_UNSUPPORTED' in codes:
        details.append('APPLICABILITY_UNSUPPORTED: one or more written pre-turn conditions lack support')
    if 'SOURCE_ANCHOR_MISMATCH' in codes:
        details.append(f'SOURCE_ANCHOR_MISMATCH: blind source {actual} versus expected {expected}')
    if 'SOURCE_UNSCORABLE' in codes:
        details.append('SOURCE_UNSCORABLE: no unique written source level')
    for code in instruction_codes:
        details.append(code + ': ' + intervention['answer'][{
            'INSTRUCTION_ALIGNMENT': 'instruction_alignment_reason',
            'UNSUPPORTED_CONTEXT_PREMISE': 'unsupported_context_premise_reason',
            'CONCLUSION_PRESCRIBED': 'conclusion_reason'}[code]])
    labels = '; '.join(details) if details else ', '.join(codes) if codes else 'all required checks passed'
    result = {'decision': decision, 'source_level': None if actual == 'UNSCORABLE' else int(actual),
              'failure_codes': codes, 'evidence': evidence,
              'reason': (quality['reason'].strip() + '; ' + labels)[:4000]}
    validate_v3_result(result)
    audit_checks = {'protocol': PROTOCOL, 'source_outcome': source_outcome,
                    'expected_source_level': expected, 'applicability': app,
                    'source_behavior': behavior, 'rubric_quality': rubric_quality,
                    'intervention_quality': intervention, 'quality_model_output': quality,
                    'quality_prompt_sha256': quality_prompt_sha256,
                    'model_settings': copy.deepcopy(model_settings),
                    'final_schema_sha256': sha256_text(SCHEMA_PATH.read_text()),
                    'audit_schema_sha256': sha256_text(AUDIT_SCHEMA_PATH.read_text()),
                    'module_sha256': sha256_text(MODULE_PATH.read_text()),
                    'candidate_namespace': candidate_namespace,
                    'candidate_generator_model': candidate_generator_model,
                    'candidate_content_sha256': sha256_json(state['candidate_rubric']),
                    'input_sha256': sha256_json(state),
                    'decision_computed_locally': True,
                    'semantic_accuracy': 'independent_human_review_pending'}
    validate_v3_audit_checks(audit_checks)
    bundle = {'protocol': PROTOCOL, 'passed': decision == 'pass',
            'compact_result': result, 'audit_checks': audit_checks,
            'cache_identity_sha256': _cache_identity(audit_checks)}
    return validate_v3_bundle(bundle,
        expected_candidate_sha=audit_checks['candidate_content_sha256'],
        expected_input_sha=audit_checks['input_sha256'],
        namespace=candidate_namespace, generator_model=candidate_generator_model)


def _cache_identity(checks: dict) -> str:
    return sha256_json({
                'protocol': PROTOCOL, 'candidate': checks['candidate_content_sha256'],
                'candidate_namespace': checks['candidate_namespace'],
                'candidate_generator_model': checks['candidate_generator_model'],
                'input': checks['input_sha256'], 'schema': checks['final_schema_sha256'],
                'audit_schema': checks['audit_schema_sha256'],
                'module': checks['module_sha256'],
                'quality_prompt': checks['quality_prompt_sha256'],
                'model_settings': checks['model_settings'],
                'applicability_contract': 'compact_condition_v7_5',
                'blind_protocol': checks['source_behavior']['blind_protocol'],
                'instruction_protocol': checks['intervention_quality']['audit_protocol'],
            })


def validate_v3_result(result: dict) -> None:
    if not isinstance(result, dict) or set(result) != {
            'decision', 'source_level', 'failure_codes', 'evidence', 'reason'}:
        raise GateExecutionError('v3 final fields differ')
    codes = result['failure_codes']
    if not isinstance(codes, list) or len(codes) != len(set(codes)) or any(c not in DECISION_CODES for c in codes):
        raise GateExecutionError('Unknown/duplicate v3 failure code')
    level, decision = result['source_level'], result['decision']
    if level is None:
        if decision != 'unscorable' or 'SOURCE_UNSCORABLE' not in codes:
            raise GateExecutionError('Null level is reserved for blind UNSCORABLE')
    elif type(level) is not int or level not in (0, 1, 2) or decision != ('reject' if codes else 'pass'):
        raise GateExecutionError('v3 decision contradicts source level or failure codes')
    if not isinstance(result['reason'], str) or not result['reason'].strip():
        raise GateExecutionError('Missing v3 reason')
    if not isinstance(result['evidence'], list):
        raise GateExecutionError('Missing v3 evidence')
    for row in result['evidence']:
        if not isinstance(row, dict) or set(row) != {'turn_id', 'quote'} or not all(
                isinstance(row[k], str) and row[k].strip() for k in row):
            raise GateExecutionError('Invalid v3 evidence shape')


def validate_v3_audit_checks(checks: dict) -> None:
    try:
        Draft202012Validator(json.loads(AUDIT_SCHEMA_PATH.read_text())).validate(checks)
    except Exception as exc:
        raise GateExecutionError('Incomplete v3 audit checks: ' + str(exc)) from exc
    if (not isinstance(checks['quality_model_output'], dict)
            or 'checks' not in checks['quality_model_output']
            or checks['expected_source_level'] != (2 if checks['source_outcome'] == 'success' else 0)
            or checks['rubric_quality'] != checks['quality_model_output']['checks']):
        raise GateExecutionError('v3 audit checks contradict independent stage output')


def validate_v3_bundle(bundle: dict, *, expected_candidate_sha: str,
                       expected_input_sha: str, namespace: str,
                       generator_model: str | None = None) -> dict:
    """Fail-closed Gate→witness boundary; validates the whole admission record."""
    if not isinstance(bundle, dict) or bundle.get('protocol') != PROTOCOL:
        raise GateExecutionError('Wrong v3 Gate bundle protocol')
    result = bundle.get('compact_result')
    checks = bundle.get('audit_checks')
    validate_v3_result(result)
    validate_v3_audit_checks(checks)
    if (checks['candidate_content_sha256'] != expected_candidate_sha
            or checks['input_sha256'] != expected_input_sha
            or checks['candidate_namespace'] != namespace
            or (generator_model is not None and checks['candidate_generator_model'] != generator_model)
            or checks['final_schema_sha256'] != sha256_text(SCHEMA_PATH.read_text())
            or checks['audit_schema_sha256'] != sha256_text(AUDIT_SCHEMA_PATH.read_text())
            or checks['module_sha256'] != sha256_text(MODULE_PATH.read_text())
            or bundle.get('cache_identity_sha256') != _cache_identity(checks)):
        raise GateExecutionError('v3 Gate candidate/input/version/cache identity mismatch')
    if type(bundle.get('passed')) is not bool or bundle['passed'] != (result['decision'] == 'pass'):
        raise GateExecutionError('Outer PASS contradicts final decision')
    source = checks['source_behavior']
    level = source['independent_level']
    if result['source_level'] != (None if level == 'UNSCORABLE' else int(level)):
        raise GateExecutionError('Final source level differs from independent blind source')
    codes = set(result['failure_codes'])
    expected = checks['expected_source_level']
    if (level == 'UNSCORABLE' or int(level) != expected) != ('SOURCE_ANCHOR_MISMATCH' in codes):
        raise GateExecutionError('Source anchor mismatch code is missing or invented')
    if (level == 'UNSCORABLE') != ('SOURCE_UNSCORABLE' in codes):
        raise GateExecutionError('Blind UNSCORABLE code is missing or invented')
    app = checks['applicability']
    if (not app['supported']) != ('APPLICABILITY_UNSUPPORTED' in codes):
        raise GateExecutionError('Applicability support code contradicts audit')
    if (app['request_state'] != 'active') != ('REQUEST_NOT_ACTIVE' in codes):
        raise GateExecutionError('Request lifecycle code contradicts audit')
    quality = checks['rubric_quality']
    for name, passed in quality.items():
        if (not passed) != (name.upper() in codes):
            raise GateExecutionError('Rubric quality code contradicts audit: ' + name)
    instruction = checks['intervention_quality']['quality']
    for name, code in (
        ('instruction_aligned', 'INSTRUCTION_ALIGNMENT'),
        ('no_unsupported_context_premise', 'UNSUPPORTED_CONTEXT_PREMISE'),
        ('no_conclusion_prescribed', 'CONCLUSION_PRESCRIBED'),
    ):
        if (not instruction[name]) != (code in codes):
            raise GateExecutionError('Intervention quality code contradicts audit: ' + name)
    if set(checks['quality_model_output']['failure_codes']) - codes:
        raise GateExecutionError('Quality model failure code omitted in final result')
    if result['decision'] == 'pass' and not (
            app['supported'] and app['request_state'] == 'active'
            and source['score_unique'] and all(quality.values())
            and all(instruction.values()) and not codes):
        raise GateExecutionError('Forged v3 PASS: mandatory check failed')
    return copy.deepcopy(bundle)
