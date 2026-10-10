"""Versioned, opt-in compact public Gate output; never used by the active v9.1 run.

The quality judge does not see the source outcome or expected score. Its compact
answer has no source_level field. A separate blind behavior scorer supplies
that field only after both independent decisions have completed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json, sha256_text
from src.online_rl.tree.agent_state import rubric_problems
from src.online_rl.tree.quality_gate import (
    QUALITY_CATEGORIES, public_gate_state, public_rubric_field_catalog,
)


PROTOCOL = 'public_mc_gate_compact_v2'
PROMPT_PATH = Path(__file__).resolve().parents[3] / 'prompts/sherpa/gate_public_mc_compact_v2.txt'
QUALITY_CODES = tuple(category.upper() for category in QUALITY_CATEGORIES) + ('CRITERION_UNCERTAIN',)
FINAL_CODES = QUALITY_CODES + ('SOURCE_ANCHOR_MISMATCH', 'SOURCE_UNSCORABLE')
QUALITY_FIELDS = {
    'decision': {'type': 'string', 'enum': ['pass', 'reject', 'uncertain']},
    'failure_codes': {'type': 'array', 'items': {'type': 'string', 'enum': list(QUALITY_CODES)}},
    'evidence': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
        'required': ['turn_id', 'quote'], 'properties': {
            'turn_id': {'type': 'string'}, 'quote': {'type': 'string'}}}},
    'reason': {'type': 'string'},
}
FINAL_SCHEMA_PATH = Path(__file__).resolve().parents[3] / 'prompts/sherpa/gate_public_mc_compact_result_v2.schema.json'


def _quality_decision(codes: list[str]) -> str:
    definite = [x for x in codes if x != 'CRITERION_UNCERTAIN']
    return 'reject' if definite else ('uncertain' if codes else 'pass')


def _evidence_catalog(state: dict) -> dict[str, str]:
    result = {'criterion:' + path: text for path, text in
              public_rubric_field_catalog(state['candidate_rubric']).items()}
    for row in state['case_evidence'] + state['dialogue_before_target_turn'] + [state['source_agent_reply']]:
        result[row['turn_id']] = row['text']
    return result


def validate_quality(answer: dict, state: dict) -> dict:
    """Validate traceability and internal consistency, not semantic truth."""
    if not isinstance(answer, dict) or set(answer) != set(QUALITY_FIELDS):
        raise ValueError('Compact Gate output fields differ')
    codes = answer['failure_codes']
    if (not isinstance(codes, list) or len(codes) != len(set(codes))
            or any(not isinstance(code, str) or code not in QUALITY_CODES for code in codes)):
        raise ValueError('Unknown or duplicate compact Gate failure code')
    if answer['decision'] != _quality_decision(codes):
        raise ValueError('Compact Gate decision contradicts failure codes')
    reason = answer['reason']
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2500:
        raise ValueError('Compact Gate needs one concise substantive reason')
    if any(code not in reason for code in codes):
        raise ValueError('Compact Gate reason must name every failure code')
    evidence = answer['evidence']
    if not isinstance(evidence, list) or len(evidence) > 12 or (codes and not evidence):
        raise ValueError('Compact Gate rejection/uncertainty needs bounded literal evidence')
    catalog = _evidence_catalog(state)
    seen = set()
    for row in evidence:
        if not isinstance(row, dict) or set(row) != {'turn_id', 'quote'}:
            raise ValueError('Compact Gate evidence fields differ')
        turn_id, quote = row['turn_id'], row['quote']
        if (not isinstance(turn_id, str) or turn_id not in catalog
                or not isinstance(quote, str) or not quote.strip() or quote not in catalog[turn_id]):
            raise ValueError('Compact Gate evidence must be an exact local quote')
        if (turn_id, quote) in seen:
            raise ValueError('Compact Gate duplicate evidence')
        seen.add((turn_id, quote))
    if codes and not any(row['turn_id'].startswith('criterion:') for row in evidence):
        raise ValueError('Compact Gate failure needs its written criterion clause')
    return copy.deepcopy(answer)


def validate_compact_result(result: dict) -> dict:
    """Check the assembled public wire record without reinterpreting scores."""
    if not isinstance(result, dict) or set(result) != {
            'decision', 'source_level', 'failure_codes', 'evidence', 'reason'}:
        raise ValueError('Compact Gate final fields differ')
    codes = result['failure_codes']
    if (not isinstance(codes, list) or len(codes) != len(set(codes))
            or any(not isinstance(code, str) or code not in FINAL_CODES for code in codes)):
        raise ValueError('Compact Gate final failure codes differ')
    if result['decision'] != _quality_decision(codes):
        raise ValueError('Compact Gate final decision contradicts failure codes')
    level = result['source_level']
    if (level is None) != ('SOURCE_UNSCORABLE' in codes):
        raise ValueError('Null source level requires SOURCE_UNSCORABLE and vice versa')
    if level is not None and (type(level) is not int or level not in (0, 1, 2)):
        raise ValueError('Compact Gate source level must be a JSON number 0, 1, or 2')
    if not isinstance(result['reason'], str) or not result['reason'].strip():
        raise ValueError('Compact Gate final reason is empty')
    evidence = result['evidence']
    if not isinstance(evidence, list) or any(
            not isinstance(row, dict) or set(row) != {'turn_id', 'quote'}
            or not all(isinstance(row[key], str) and row[key].strip() for key in ('turn_id', 'quote'))
            for row in evidence):
        raise ValueError('Compact Gate final evidence shape differs')
    return copy.deepcopy(result)


def compact_public_mc_posthoc_gate(client, cases: dict, *, blind_scorer_override=None):
    """Return a legacy-compatible Gate record plus an exact five-field summary.

    This is deliberately not registered in run_tree/backends/config. Activation
    requires a separate versioned experiment and an actual semantic API pilot.
    """
    from src.online_rl.tree.backends import with_role
    from src.online_rl.tree.cumulative_bank import content_hash
    prompt = PROMPT_PATH.read_text()
    if client.system != prompt:
        raise ValueError('Compact Gate requires its pinned compact system prompt')
    if blind_scorer_override is None:
        raise ValueError('Compact Gate requires an independently configured blind source scorer')
    blind = blind_scorer_override
    settings = {'protocol': PROTOCOL, 'quality_prompt_sha256': sha256_text(prompt),
                'quality_fields_sha256': sha256_json(QUALITY_FIELDS), 'model': client.model,
                'final_schema_sha256': sha256_text(FINAL_SCHEMA_PATH.read_text()),
                'client_protocol_sha256': getattr(client, 'protocol_sha256', None),
                'max_tokens': client.max_tokens, 'timeout_s': getattr(client, 'timeout_s', None),
                'blind_source_protocol': blind.protocol, 'source_anchor_policy': 'success2_failure0_unchanged',
                'quality_calls': 1, 'blind_source_calls': 1}

    def gate(tin, rubric):
        if tin.case_id not in cases or tin.outcome not in ('success', 'failure'):
            raise ValueError('Compact Gate requires a declared source case and observed outcome')
        problems = rubric_problems(rubric)
        if problems:
            return {'protocol': PROTOCOL, 'schema': False, 'passed': False,
                    'reasons': ['schema: ' + str(problems)], 'unresolved': [],
                    'leakage_deterministic': [], 'compact_result': None}
        state = public_gate_state(tin, rubric)
        evidence_ids = list(_evidence_catalog(state))
        quality_state = copy.deepcopy(state)
        quality_state['evidence_ids'] = evidence_ids
        item = {'instructions': 'Assess every named public Gate category. Return only failures and exact local evidence. '
                                'Do not judge source outcome or infer the expected source score.',
                'criterion_contract': 'public_mc_definitions' if 'criterion_version' in rubric
                                      else 'legacy_written_anchors'}
        fields = copy.deepcopy(QUALITY_FIELDS)
        fields['evidence']['items']['properties']['turn_id']['enum'] = evidence_ids
        current, requests, decisions, errors = quality_state, [], [], []
        for attempt in range(2):
            messages = [{'role': 'system', 'content': client.system}, {'role': 'user', 'content': json.dumps(
                {'stage': PROTOCOL, 'state': current, 'items': {'candidate': item}}, ensure_ascii=False)}]
            requests.append({'messages': messages, 'messages_sha256': sha256_json(messages),
                             'output_fields': copy.deepcopy(fields)})
            values, record = with_role('rubric_gate', lambda: client.judge(
                PROTOCOL, current, {'candidate': item}, fields, 'public_criterion_quality_compact'))
            decisions.append(copy.deepcopy(record))
            try:
                if not isinstance(values, dict) or set(values) != {'candidate'}:
                    raise ValueError('Compact Gate must return exactly candidate')
                quality = validate_quality(values['candidate'], state)
                break
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
                if attempt:
                    error = IntegrationUnavailable('Invalid compact Gate evidence after one correction: ' + str(exc))
                    error.audit = {'requests': requests, 'decisions': decisions, 'validation_errors': errors}
                    raise error from exc
                current = {**copy.deepcopy(quality_state), 'evidence_correction': {
                    'previous_answer': values.get('candidate') if isinstance(values, dict) else values,
                    'error': str(exc)}}

        # This is a different request with no quality decision, outcome, expected
        # level, Doctor reflection, or generator analysis in its input.
        visible = {**state['candidate_rubric'], 'rubric_id': 'source_criterion'}
        try:
            source = blind.score_behavior(state['case_context'], state['dialogue_before_target_turn'],
                                          [visible], state['source_agent_reply']['text'])['source_criterion']
        except IntegrationUnavailable as exc:
            audit = getattr(exc, 'audit', None) or {}
            validation = audit.get('validation') or {}
            if (validation.get('status') != 'execution_error' or validation.get('kind') != 'evidence_contract'
                    or validation.get('error') != 'Numeric score contradicts anchor matches'):
                raise
            source = {'stage2': {'level': 'UNSCORABLE', 'quotes': [],
                                 'defect': 'Blind source answer contradicts its anchor matches.'},
                      'evidence_validation': audit}
        source_level = source['stage2']['level']
        if source_level not in ('0', '1', '2', 'UNSCORABLE'):
            raise IntegrationUnavailable('Blind source scorer returned an invalid level')
        expected = '2' if tin.outcome == 'success' else '0'
        codes = list(quality['failure_codes'])
        if source_level == 'UNSCORABLE':
            codes.append('SOURCE_UNSCORABLE')
        elif source_level != expected:
            codes.append('SOURCE_ANCHOR_MISMATCH')
        decision = 'reject' if any(code != 'CRITERION_UNCERTAIN' for code in codes) else (
            'uncertain' if codes else 'pass')
        source_quotes = [{'turn_id': state['source_agent_reply']['turn_id'], 'quote': quote}
                         for quote in source['stage2'].get('quotes', [])
                         if isinstance(quote, str) and quote in state['source_agent_reply']['text']]
        evidence = list(quality['evidence'])
        for row in source_quotes:
            if row not in evidence:
                evidence.append(row)
        source_note = ('; SOURCE_UNSCORABLE: blind source cannot receive a unique written score' if source_level == 'UNSCORABLE'
                       else f'; SOURCE_ANCHOR_MISMATCH: blind source {source_level} versus required {expected}'
                            if source_level != expected else '')
        compact_result = validate_compact_result({'decision': decision,
                          'source_level': None if source_level == 'UNSCORABLE' else int(source_level),
                          'failure_codes': codes, 'evidence': evidence,
                          'reason': quality['reason'] + source_note})
        return {'protocol': PROTOCOL, 'schema': True, 'passed': decision == 'pass',
                'public_criterion_version': rubric.get('criterion_version'),
                'criterion_sha256': content_hash(rubric), 'reasons': [code.lower() for code in codes],
                'unresolved': ['criterion_uncertain'] if 'CRITERION_UNCERTAIN' in codes else [],
                'leakage_deterministic': [], 'assessment': quality,
                'observable': 'OBSERVABILITY' not in codes,
                'score_boundaries_clear': not any(code in codes for code in
                    ('MC_ANCHOR_ALIGNMENT', 'BOUNDARY_COVERAGE', 'CRITERION_UNCERTAIN')),
                'source_anchor': {'expected_level': expected, 'observed_level': source_level,
                                  'matches': source_level == expected,
                                  'generator_declared_level': ((rubric.get('source_provenance') or {}).get('analysis') or {}).get('source_score')},
                'blind_source_assessment': source['stage2'],
                'selected_agent_turn_id': state['selected_agent_turn_id'],
                'input_sha256': sha256_json(state), 'compact_result': compact_result,
                'evidence': {'requests': requests, 'validation_errors': errors, 'attempts': len(decisions),
                             'blind_source': source['evidence_validation'],
                             'exact_quotes_are_semantic_proof': False}, 'decisions': decisions}

    gate.model, gate.protocol, gate.settings = client.model, sha256_json(settings), settings
    return gate
