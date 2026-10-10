"""Evidence-backed admission checks for a generated interaction rubric.

This gate does not measure outcome effectiveness. It grades the source reply
without receiving its outcome; the expected source anchor is compared locally.
Invalid judge output is an operational error, never a negative quality label.
"""
from __future__ import annotations

import copy
import json

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json, sha256_text
from src.online_rl.tree.agent_state import rubric_problems, visible_rubric
from src.online_rl.tree.records import to_dialogue


PROTOCOL = 'evidence_gate_v2'
INPUT_VERSION = 'selected_prefix_and_separate_source_reply_v1'
FIELDS = {
    'answer_presupposition': {'type': 'boolean'},
    'answer_presupposition_quote': {'type': 'string'},
    'answer_presupposition_reason': {'type': 'string'},
    'new_domain_premise': {'type': 'boolean'},
    'new_domain_premise_quote': {'type': 'string'},
    'new_domain_premise_reason': {'type': 'string'},
    'observable': {'type': 'boolean'},
    'observable_quote': {'type': 'string'},
    'observable_reason': {'type': 'string'},
    'score_boundaries_clear': {'type': 'boolean'},
    'score_boundaries_quote': {'type': 'string'},
    'score_boundaries_other_quote': {'type': 'string'},
    'score_boundaries_reason': {'type': 'string'},
    'source_level': {'type': 'string', 'enum': ['0', '1', '2', 'UNSCORABLE']},
    'source_minimum_quote': {'type': 'string'},
    'source_completion_quote': {'type': 'string'},
    'source_missing_or_conflicting': {'type': 'string'},
    'source_reason': {'type': 'string'},
}
ITEM = {
    'task': 'Review the visible candidate rubric and independently assess the separate source Agent reply.',
    'checks': {
        'answer_presupposition': 'True only if satisfying an anchor requires a predetermined answer, factual conclusion, agreement or disagreement. An open-ended examination is not itself a predetermined conclusion.',
        'new_domain_premise': 'True if the rubric asserts or requires a particular domain fact not present in its admissible evidence. Asking to evaluate a relationship or identify a missing fact without supplying the answer is not itself a new factual assertion.',
        'observable': 'Every anchor must be decidable from visible local text without gold, hidden beliefs, final outcome or later turns.',
        'score_boundaries_clear': 'The actual written 0/1/2 anchors must separate absent minimum contribution, partial performance with a named remaining gap, and adequate completion of that same behavior, without overlap, uncovered coherent responses, or adding an unrelated requirement.',
        'source_level': 'Independently assess the source reply from its text and prefix against the written anchors. No expected score is provided. Use UNSCORABLE if a unique supported score cannot be assigned.',
    },
    'evidence_contract': 'Give nonempty reasons for all checks. Nonempty rubric quotes must be verbatim contiguous substrings of one visible rubric field. Every blocking check needs a rubric quote. Quote both conflicting anchors for overlap. Source minimum/completion quotes must be verbatim substrings of source_agent_reply.text, never substituted from the prefix. Grade the whole reply; do not repair missing relations yourself.',
}
RUBRIC_QUOTE_PROTOCOL = 'visible_field_enum_v1'
RUBRIC_QUOTE_FIELDS = (
    'answer_presupposition_quote', 'new_domain_premise_quote', 'observable_quote',
    'score_boundaries_quote', 'score_boundaries_other_quote',
)
RUBRIC_QUOTE_POLICY = {
    'protocol': RUBRIC_QUOTE_PROTOCOL,
    'fields': list(RUBRIC_QUOTE_FIELDS),
    'catalog': 'Empty string followed by unique nonempty complete string field values of visible_rubric, excluding rubric_id, in traversal order; preserve literal text.',
    'validation': 'Require exact catalog membership, then apply unchanged local evidence and source-anchor validation.',
}
RUBRIC_QUOTE_INSTRUCTION = (
    ' For the five rubric quote fields, select either the empty string or exactly one complete '
    'visible rubric field value from the output schema enum. Do not rewrite, shorten, join, '
    'or normalize its text. This changes only how rubric evidence is quoted; keep the same '
    'independent checks, decisions, and reasons. Source reply evidence quotes remain verbatim '
    'substrings of source_agent_reply.text and do not use this rubric catalog.'
)


def _text_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key != 'rubric_id':
                yield from _text_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _text_values(item)


def _quote_catalog_contract(state):
    """Each call owns its schema; concurrent candidates cannot share quote enums."""
    fields, item = copy.deepcopy(FIELDS), copy.deepcopy(ITEM)
    catalog = list(dict.fromkeys([''] + [text for text in _text_values(state['candidate_rubric'])
                                       if text.strip()]))
    for name in RUBRIC_QUOTE_FIELDS:
        fields[name]['enum'] = list(catalog)
    item['evidence_contract'] += RUBRIC_QUOTE_INSTRUCTION
    return fields, item


def gate_state(tin, rubric):
    """Whitelist source-local evidence; never expose provenance, feedback or outcome."""
    turn = ((rubric.get('source_provenance') or {}).get('analysis') or {}).get('selected_agent_turn_id')
    if type(turn) is not int or turn < 1:
        raise ValueError('evidence gate requires an actual integer selected_agent_turn_id')
    prefix, target, number = [], None, 0
    for item in to_dialogue(tin.dialogue):
        if item['stage'] == 'FINAL':
            break
        if item['speaker'] == 'agent':
            number += 1
            if number == turn:
                target = item
                break
        prefix.append(item)
    if target is None or not target['text'].strip():
        raise ValueError('evidence gate selected Agent reply is missing or empty')
    return {
        'case_context': {'scenario': tin.scenario, 'question': tin.question, 'options': dict(tin.options)},
        'selected_agent_turn_id': turn,
        'dialogue_before_target_turn': prefix,
        'source_agent_reply': target,
        'candidate_rubric': visible_rubric(rubric),
    }


def validate_evidence(answer, state):
    """Validate traceability and output consistency; not the semantic judgement."""
    if not isinstance(answer, dict) or set(answer) != set(FIELDS):
        raise ValueError('gate answer fields differ from the requested contract')
    for key, spec in FIELDS.items():
        if spec['type'] == 'boolean':
            if type(answer[key]) is not bool:
                raise ValueError(f'{key} must be a boolean')
        elif not isinstance(answer[key], str):
            raise ValueError(f'{key} must be a string')
    for key in ('answer_presupposition_reason', 'new_domain_premise_reason', 'observable_reason',
                'score_boundaries_reason', 'source_reason'):
        if not answer[key].strip():
            raise ValueError(f'{key} must be nonempty')
    rubric_strings = list(_text_values(state['candidate_rubric']))
    for key in ('answer_presupposition_quote', 'new_domain_premise_quote', 'observable_quote',
                'score_boundaries_quote', 'score_boundaries_other_quote'):
        quote = answer[key]
        if quote and (not quote.strip() or not any(quote in text for text in rubric_strings)):
            raise ValueError(f'{key} must be an exact quote from one visible rubric field')
    for check, blocked in (('answer_presupposition', True), ('new_domain_premise', True),
                           ('observable', False), ('score_boundaries_clear', False)):
        quote_key = 'score_boundaries_quote' if check == 'score_boundaries_clear' else check + '_quote'
        if answer[check] is blocked and not answer[quote_key].strip():
            raise ValueError(f'{check} requires a quoted rubric clause supporting rejection')
    level = answer['source_level']
    if level not in FIELDS['source_level']['enum']:
        raise ValueError('invalid source_level')
    reply = state['source_agent_reply']['text']
    for key in ('source_minimum_quote', 'source_completion_quote'):
        quote = answer[key]
        if quote and (not quote.strip() or quote not in reply):
            raise ValueError(f'{key} must be an exact quote from the source Agent reply')
    if level in ('1', '2') and not answer['source_minimum_quote'].strip():
        raise ValueError('source levels 1/2 require a minimum contribution quote')
    if level == '0' and (answer['source_minimum_quote'] or answer['source_completion_quote']):
        raise ValueError('source level 0 cannot include minimum/completion evidence')
    if level == '2':
        if not answer['source_completion_quote'].strip() or answer['source_missing_or_conflicting'].strip():
            raise ValueError('source level 2 requires completion evidence and no remaining gap')
    elif not answer['source_missing_or_conflicting'].strip():
        raise ValueError('source below 2 or UNSCORABLE requires a missing/conflicting explanation')


def evidence_posthoc_gate(client, cases, observed_context=False, rubric_quote_protocol=None):
    """One quality+blind-anchor call, with at most one evidence correction.

The client separately has one JSON-format repair. A quality rejection never
triggers a re-vote. The correction only repairs a malformed evidence record.
"""
    if not callable(getattr(client, 'judge', None)):
        raise ValueError('evidence_gate_v2 requires a structured judge client')
    if rubric_quote_protocol not in (None, RUBRIC_QUOTE_PROTOCOL):
        raise ValueError('Unsupported rubric quote protocol: ' + str(rubric_quote_protocol))
    settings = {
        'protocol': PROTOCOL, 'input_version': INPUT_VERSION,
        'model': client.model, 'prompt_sha256': sha256_text(client.system),
        'client_protocol_sha256': getattr(client, 'protocol_sha256', None),
        'max_tokens': client.max_tokens, 'timeout_s': getattr(client, 'timeout_s', None),
        'observed_context': bool(observed_context),
        'contract_sha256': sha256_json({'fields': FIELDS, 'item': ITEM}),
        'evidence_corrections': 1,
    }
    if rubric_quote_protocol is not None:
        item = copy.deepcopy(ITEM)
        item['evidence_contract'] += RUBRIC_QUOTE_INSTRUCTION
        settings['rubric_quote_protocol'] = rubric_quote_protocol
        settings['contract_sha256'] = sha256_json({
            'fields': FIELDS, 'item': item, 'dynamic_rubric_quote_policy': RUBRIC_QUOTE_POLICY,
        })

    def gate(tin, rubric):
        from src.online_rl.tree.backends import with_role
        if tin.case_id not in cases:
            raise ValueError('Gate source case is not in this run')
        problems = rubric_problems(rubric)
        result = {
            'protocol': PROTOCOL, 'schema': not problems, 'passed': False,
            'reasons': [], 'unresolved': [], 'leakage_deterministic': [],
        }
        if problems:
            result['reasons'] = ['schema: ' + str(problems)]
            return result
        if tin.outcome not in ('success', 'failure'):
            raise ValueError('Gate source outcome must be an observed success or failure')
        state = gate_state(tin, rubric)
        # No unconditional option-string veto: neutral references are distinguishable
        # from answer steering by the evidence-backed answer_presupposition check.
        attempts, requests, errors = [], [], []
        current = copy.deepcopy(state)
        for attempt in range(2):
            fields, item = (_quote_catalog_contract(state) if rubric_quote_protocol is not None
                            else (FIELDS, ITEM))
            user = json.dumps({'stage': PROTOCOL, 'state': current, 'items': {'candidate': item}}, ensure_ascii=False)
            messages = [{'role': 'system', 'content': client.system}, {'role': 'user', 'content': user}]
            requests.append({'messages': messages, 'messages_sha256': sha256_json(messages)})
            if rubric_quote_protocol is not None:
                requests[-1].update(output_fields=copy.deepcopy(fields), output_fields_sha256=sha256_json(fields))
            answers, record = with_role('rubric_gate', lambda: client.judge(
                PROTOCOL, current, {'candidate': item}, fields, 'tree_posthoc_rubric_gate_v2'))
            attempts.append(copy.deepcopy(record))
            answer = answers.get('candidate') if isinstance(answers, dict) else None
            try:
                if not isinstance(answers, dict) or set(answers) != {'candidate'}:
                    raise ValueError('gate response must contain exactly candidate')
                validate_evidence(answer, state)
                if rubric_quote_protocol is not None:
                    for name in RUBRIC_QUOTE_FIELDS:
                        if answer[name] not in fields[name]['enum']:
                            raise ValueError(f'{name} must select one complete literal visible rubric field from its enum')
                break
            except ValueError as exc:
                errors.append(str(exc))
                if attempt == 1:
                    error = IntegrationUnavailable('Invalid evidence_gate_v2 evidence after one correction: ' + str(exc))
                    error.audit = {'requests': requests, 'decisions': attempts, 'validation_errors': errors}
                    raise error from exc
                current = {**copy.deepcopy(state), 'evidence_correction': {
                    'previous_answer': answer, 'error': str(exc),
                    'instruction': 'Correct only the evidence/output-contract error against the same input. Do not infer an expected source score or force acceptance. Use exact quotes and independently reassess any claim lacking support.',
                }}
        expected = '2' if tin.outcome == 'success' else '0'
        reasons = []
        for key, block in (('answer_presupposition', True), ('new_domain_premise', True),
                           ('observable', False), ('score_boundaries_clear', False)):
            if answer[key] is block:
                reasons.append(key if block else 'not_' + key)
        if answer['source_level'] == 'UNSCORABLE':
            reasons.append('source_unscorable')
        elif answer['source_level'] != expected:
            reasons.append('source_anchor_mismatch')
        result.update({
            **{k: answer[k] for k in ('answer_presupposition', 'new_domain_premise', 'observable', 'score_boundaries_clear')},
            'passed': not reasons, 'reasons': reasons, 'assessment': answer,
            'source_anchor': {'expected_level': expected, 'observed_level': answer['source_level'],
                              'matches': answer['source_level'] == expected},
            'selected_agent_turn_id': state['selected_agent_turn_id'],
            'input_sha256': sha256_json(state),
            'evidence': {'requests': requests, 'validation_errors': errors, 'attempts': len(attempts)},
            'decisions': attempts,
        })
        return result

    gate.model, gate.protocol, gate.settings = client.model, sha256_json(settings), settings
    return gate
