"""Independent, prefix-only checks of frozen member instructions before reruns.

Admission of a written instruction is separate from Agent adherence and the
outcome-based KEEP rule. A malformed review is an operational error, not a
negative intervention result. No source reply or outcome is sent to this judge.
"""
from __future__ import annotations

import copy
import json

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json, sha256_text
from src.online_rl.tree.agent_state import visible_rubric
from src.online_rl.tree.records import to_dialogue

PROTOCOL = 'intervention_instruction_audit_v1'
ANALYSIS_PROTOCOL = 'intervention_alignment_analysis_v1'
FIELDS = {
    'instruction_aligned': {'type': 'boolean'},
    'instruction_alignment_quote': {'type': 'string'},
    'instruction_alignment_reason': {'type': 'string'},
    'unsupported_context_premise': {'type': 'boolean'},
    'unsupported_context_premise_quote': {'type': 'string'},
    'unsupported_context_premise_reason': {'type': 'string'},
    'unsupported_quote_source': {'type': 'string', 'enum': ['instruction', 'rubric', 'none']},
    'conclusion_prescribed': {'type': 'boolean'},
    'conclusion_quote': {'type': 'string'},
    'conclusion_reason': {'type': 'string'},
    'conclusion_quote_source': {'type': 'string', 'enum': ['instruction', 'rubric', 'none']},
}
ITEM = {
    'task': 'Independently review the exact frozen intervention instruction against the visible rubric anchors and pre-response evidence.',
    'checks': {
        'instruction_aligned': 'Derive minimum contribution M and completion C from the literal anchors. Level 0 must omit M, not merely C; level 2 must perform M+C. Do not repair overlapping anchors or permit an instruction that also satisfies another anchor.',
        'unsupported_context_premise': 'Identify unsupported factual premises in either the instruction or rubric, including metadata/provenance classifications as well as medical facts. A participant claim is not automatically established fact.',
        'conclusion_prescribed': 'Check whether the instruction or rubric fixes an answer or factual conclusion instead of an open-ended operation.',
    },
    'evidence_contract': 'Return all requested fields with actual booleans and nonempty reasons. Cite literal contiguous instruction/rubric text. A pass requires a nonempty instruction alignment quote; premise/conclusion rejection requires a quote and its correct source. No generated response, outcome, source reply or future is available.',
}


def _texts(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key != 'rubric_id':
                yield from _texts(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _texts(item)


def instruction_state(case, prefix, rubric, member_spec):
    """Whitelist judge inputs; member provenance and post-prefix evidence stay local."""
    if not isinstance(member_spec, dict):
        raise ValueError('instruction audit requires a frozen member intervention')
    level = member_spec.get('requested_level')
    if type(level) not in (str, int) or str(level) not in ('0', '2'):
        raise ValueError('instruction audit requested_level must be 0 or 2')
    instruction = member_spec.get('intervention_instruction')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction audit requires a nonblank instruction')
    turn = member_spec.get('selected_agent_turn_id')
    if type(turn) is not int or turn < 1:
        raise ValueError('instruction audit requires a positive integer selected turn')
    dialogue = []
    for row in prefix:
        if isinstance(row, dict):
            if row.get('speaker') not in ('doctor', 'agent') or not isinstance(row.get('text'), str):
                raise ValueError('instruction audit prefix requires visible speaker/text rows')
            dialogue.append({k: copy.deepcopy(row.get(k, '' if k == 'stage' else 0))
                             for k in ('speaker', 'stage', 'round', 'text')})
        else:
            if not isinstance(row, (tuple, list)) or len(row) != 4:
                raise ValueError('instruction audit prefix requires visible dialogue rows')
            dialogue.extend(to_dialogue([row]))
    if any(row['stage'] == 'FINAL' for row in dialogue):
        raise ValueError('instruction audit prefix cannot contain FINAL')
    # Selected turn is Agent-local: precisely its prior Agent replies must exist.
    if sum(row['speaker'] == 'agent' for row in dialogue) != turn - 1:
        raise ValueError('instruction audit prefix does not precede its selected Agent turn')
    return {
        'case_context': {k: copy.deepcopy(case.get(k, '' if k != 'options' else {}))
                         for k in ('scenario', 'question', 'options')},
        'selected_agent_turn_id': turn,
        'dialogue_before_target_turn': dialogue,
        'rubric': copy.deepcopy(visible_rubric(rubric)),
        'requested_level': int(level),
        'intervention_instruction': instruction,
    }


def validate_evidence(answer, state):
    """Check evidence integrity, not the semantic truth of the model judgement."""
    if not isinstance(answer, dict) or set(answer) != set(FIELDS):
        raise ValueError('instruction audit answer fields differ from the contract')
    for key, spec in FIELDS.items():
        if spec['type'] == 'boolean':
            if type(answer[key]) is not bool:
                raise ValueError(f'{key} must be a boolean')
        elif not isinstance(answer[key], str):
            raise ValueError(f'{key} must be a string')
        if 'enum' in spec and answer[key] not in spec['enum']:
            raise ValueError(f'{key} has an invalid source')
    for key in ('instruction_alignment_reason', 'unsupported_context_premise_reason', 'conclusion_reason'):
        if not answer[key].strip():
            raise ValueError(f'{key} must be nonempty')
    quote = answer['instruction_alignment_quote']
    guide = state['intervention_instruction']
    if quote and (not quote.strip() or quote not in guide):
        raise ValueError('instruction_alignment_quote must be an exact instruction quote')
    if answer['instruction_aligned'] and not quote.strip():
        raise ValueError('instruction_aligned requires a nonempty instruction quote')
    for flag, quote_key, source_key in (
        ('unsupported_context_premise', 'unsupported_context_premise_quote', 'unsupported_quote_source'),
        ('conclusion_prescribed', 'conclusion_quote', 'conclusion_quote_source'),
    ):
        quote, source = answer[quote_key], answer[source_key]
        if not answer[flag]:
            if quote or source != 'none':
                raise ValueError(f'{flag}=false requires an empty quote and source=none')
            continue
        if not quote.strip() or source == 'none':
            raise ValueError(f'{flag}=true requires quoted evidence and its source')
        texts = [guide] if source == 'instruction' else list(_texts(state['rubric']))
        if not any(quote in text for text in texts):
            raise ValueError(f'{quote_key} must be an exact quote from its stated source')


def instruction_audit_settings(client):
    if not callable(getattr(client, 'judge', None)):
        raise ValueError('instruction audit requires a structured judge client')
    return {'protocol': PROTOCOL, 'input_version': 'frozen_member_prefix_only_v1',
            'model': client.model, 'prompt_sha256': sha256_text(client.system),
            'client_protocol_sha256': getattr(client, 'protocol_sha256', None),
            'max_tokens': client.max_tokens, 'timeout_s': getattr(client, 'timeout_s', None),
            'contract_sha256': sha256_json({'fields': FIELDS, 'item': ITEM}),
            'evidence_corrections': 1}


def _passed(answer):
    return (answer['instruction_aligned'] and not answer['unsupported_context_premise']
            and not answer['conclusion_prescribed'])


def make_instruction_auditor(client):
    settings = instruction_audit_settings(client)

    def audit(case, prefix, rubric, member_spec):
        from src.online_rl.tree.backends import with_role
        state = instruction_state(case, prefix, rubric, member_spec)
        current = copy.deepcopy(state)
        attempts, requests, errors = [], [], []
        for attempt in range(2):
            user = json.dumps({'stage': PROTOCOL, 'state': current, 'items': {'candidate': ITEM}}, ensure_ascii=False)
            messages = [{'role': 'system', 'content': client.system}, {'role': 'user', 'content': user}]
            requests.append({'messages': messages, 'messages_sha256': sha256_json(messages)})
            try:
                answers, record = with_role('intervention_audit', lambda: client.judge(
                    PROTOCOL, current, {'candidate': ITEM}, FIELDS, 'tree_intervention_instruction_audit_v1'))
            except IntegrationUnavailable as exc:
                client_audit = getattr(exc, 'audit', None)
                exc.audit = {'requests': requests, 'decisions': attempts,
                             'validation_errors': errors + [str(exc)], 'client_error_audit': client_audit}
                raise
            attempts.append(copy.deepcopy(record))
            answer = answers.get('candidate') if isinstance(answers, dict) else None
            try:
                if not isinstance(answers, dict) or set(answers) != {'candidate'}:
                    raise ValueError('instruction audit must return exactly candidate')
                validate_evidence(answer, state)
                break
            except ValueError as exc:
                errors.append(str(exc))
                if attempt == 1:
                    error = IntegrationUnavailable('Invalid instruction audit evidence after one correction: ' + str(exc))
                    error.audit = {'requests': requests, 'decisions': attempts, 'validation_errors': errors}
                    raise error from exc
                current = {**copy.deepcopy(state), 'evidence_correction': {
                    'previous_answer': answer, 'error': str(exc),
                    'instruction': 'Repair only the malformed evidence/output fields against the same visible input. Do not force approval or rewrite the frozen instruction/rubric. Reassess unsupported claims independently.',
                }}
        reasons = []
        if not answer['instruction_aligned']: reasons.append('instruction_alignment')
        if answer['unsupported_context_premise']: reasons.append('unsupported_context_premise')
        if answer['conclusion_prescribed']: reasons.append('conclusion_prescribed')
        return {'protocol': PROTOCOL, 'passed': _passed(answer), 'reasons': reasons,
                'answer': copy.deepcopy(answer), 'state_sha256': sha256_json(state),
                'requests': requests, 'decisions': attempts, 'validation_errors': errors}

    audit.settings = settings
    audit.model, audit.protocol = client.model, sha256_json(settings)
    return audit


def cached_instruction_auditor(auditor, cache, policy_sha256=None):
    """Identical frozen inputs share one decision; changed guides cannot hit it."""
    def cached(case, prefix, rubric, member_spec):
        state = instruction_state(case, prefix, rubric, member_spec)
        identity = {'contract': PROTOCOL, 'state_sha256': sha256_json(state),
                    'settings': auditor.settings, 'policy_sha256': policy_sha256}

        def validate(result):
            if (not isinstance(result, dict) or result.get('protocol') != PROTOCOL
                    or result.get('state_sha256') != identity['state_sha256']):
                raise ValueError('instruction audit result identity mismatch')
            validate_evidence(result.get('answer'), state)
            if type(result.get('passed')) is not bool or result['passed'] != _passed(result['answer']):
                raise ValueError('instruction audit passed flag differs from its decisions')
            return result

        return validate(cache.call('intervention_instruction_audit', identity,
                                   lambda: validate(auditor(case, prefix, rubric, member_spec))))

    cached.settings, cached.model, cached.protocol = auditor.settings, auditor.model, auditor.protocol
    return cached
