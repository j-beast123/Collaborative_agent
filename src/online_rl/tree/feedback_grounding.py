"""Opt-in local contract for a generator's audit of Doctor reflection claims.

This checks coverage, attribution, quotations and chronology. Whether a cited
passage actually supports an interpretation remains a model judgment; exact
substring checks are not a substitute for that judgment. The original Doctor
experience is preserved even when it cannot support a pre-response requirement.
"""
from __future__ import annotations

import copy
import json


PROTOCOL = 'all_reflection_claims_v1'
REFLECTION_KIND = 'physician_post_final_reflection_v1'
REVIEW_FIELDS = {
    'feedback_kind', 'feedback_index', 'reported_agent_turn_id', 'claim',
    'decision', 'accepted_part', 'rejected_part', 'reason',
    'actual_agent_turn_id', 'agent_evidence', 'prior_context', 'retrospective_only',
}


def validate_protocol(protocol):
    if protocol is not None and protocol != PROTOCOL:
        raise ValueError(f'Unknown rubric feedback_check_protocol: {protocol!r}')
    return protocol


def extend_schema(schema: dict) -> dict:
    """Return a separate schema; legacy global schemas must never be mutated."""
    result = copy.deepcopy(schema)
    feedback = result['properties']['analysis']['properties']['feedback_check']
    evidence = {'type': 'object', 'additionalProperties': False,
        'required': ['speaker', 'turn_id', 'quote'],
        'properties': {'speaker': {'type': 'string', 'enum': ['doctor', 'agent']},
                       'turn_id': {'type': 'integer', 'minimum': 1},
                       'quote': {'type': 'string', 'minLength': 1}}}
    agent_evidence = copy.deepcopy(evidence)
    agent_evidence['properties']['speaker']['enum'] = ['agent']
    nullable_string = {'anyOf': [{'type': 'string', 'minLength': 1}, {'type': 'null'}]}
    feedback['required'].append('claim_reviews')
    feedback['properties']['claim_reviews'] = {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False, 'required': sorted(REVIEW_FIELDS),
        'properties': {
            'feedback_kind': {'type': 'string', 'enum': ['helpful', 'unresolved']},
            'feedback_index': {'type': 'integer', 'minimum': 0},
            'reported_agent_turn_id': {'type': 'integer', 'minimum': 1},
            'claim': {'type': 'string', 'minLength': 1},
            'decision': {'type': 'string', 'enum': ['accept', 'partly_accept', 'reject']},
            'accepted_part': copy.deepcopy(nullable_string),
            'rejected_part': copy.deepcopy(nullable_string),
            'reason': {'type': 'string', 'minLength': 1},
            'actual_agent_turn_id': {'anyOf': [{'type': 'integer', 'minimum': 1}, {'type': 'null'}]},
            'agent_evidence': {'type': 'array', 'items': agent_evidence},
            'prior_context': {'type': 'array', 'items': evidence},
            'retrospective_only': {'type': 'boolean'},
        }}}
    return result


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_claim_reviews(feedback_check: dict, tin) -> str | None:
    """Validate without changing or synthesizing any reported claim/evidence."""
    prefix = 'analysis.feedback_check'
    if getattr(tin, 'feedback_kind', None) != REFLECTION_KIND:
        return f'{prefix}.claim_reviews requires feedback_kind={REFLECTION_KIND}'
    try:
        reflection = json.loads(tin.feedback)
    except (ValueError, TypeError):
        return f'{prefix}.claim_reviews requires a valid JSON Doctor reflection'
    if not isinstance(reflection, dict) or set(reflection) != {'helpful', 'unresolved'}:
        return f'{prefix}.claim_reviews requires helpful and unresolved source arrays'

    # Speaker-scoped IDs are labels, not global timestamps. Retain positions for
    # the strict "before this Agent reply" test and exclude FINAL everywhere.
    turns = {}
    counts = {'doctor': 0, 'agent': 0}
    for position, (speaker, stage, _round, body) in enumerate(tin.dialogue):
        if speaker not in counts or not isinstance(body, str):
            return f'{prefix}.claim_reviews requires valid Doctor/Agent dialogue'
        counts[speaker] += 1
        turns[(speaker, counts[speaker])] = (position, stage, body)

    source_claims = {}
    for kind in ('helpful', 'unresolved'):
        items = reflection[kind]
        if not isinstance(items, list):
            return f'{prefix}.claim_reviews requires source {kind} to be an array'
        for index, item in enumerate(items):
            if (not isinstance(item, dict) or set(item) != {'agent_reply_id', 'experience'}
                    or not _integer(item['agent_reply_id']) or not _text(item['experience'])
                    or ('agent', item['agent_reply_id']) not in turns
                    or turns[('agent', item['agent_reply_id'])][1] == 'FINAL'):
                return f'{prefix}.claim_reviews has an invalid source {kind}[{index}]'
            source_claims[(kind, index)] = item

    reviews = feedback_check.get('claim_reviews')
    if not isinstance(reviews, list):
        return f'{prefix}.claim_reviews must be an array'
    if source_claims and feedback_check.get('assessment') == 'not_provided':
        return f'{prefix}.assessment cannot be not_provided when reflection claims exist'
    seen = set()
    for index, review in enumerate(reviews):
        path = f'{prefix}.claim_reviews[{index}]'
        if not isinstance(review, dict) or set(review) != REVIEW_FIELDS:
            return f'{path} must contain exactly {sorted(REVIEW_FIELDS)}'
        if review['feedback_kind'] not in ('helpful', 'unresolved') or not _integer(review['feedback_index']) or review['feedback_index'] < 0:
            return f'{path} must identify a valid feedback_kind and nonnegative integer feedback_index'
        identity = (review['feedback_kind'], review['feedback_index'])
        if identity in seen:
            return f'{path} duplicates a reflection claim'
        if identity not in source_claims:
            return f'{path} does not identify a supplied reflection claim'
        seen.add(identity)
        original = source_claims[identity]
        if not _integer(review['reported_agent_turn_id']) or review['reported_agent_turn_id'] != original['agent_reply_id']:
            return f'{path}.reported_agent_turn_id must preserve the original reflection attribution'
        if review['claim'] != original['experience']:
            return f'{path}.claim must exactly preserve the original reflection experience'
        decision = review['decision']
        if decision not in ('accept', 'partly_accept', 'reject'):
            return f'{path}.decision must be accept, partly_accept or reject'
        for field in ('accepted_part', 'rejected_part'):
            if review[field] is not None and not _text(review[field]):
                return f'{path}.{field} must be a substantive string or null'
        if not _text(review['reason']):
            return f'{path}.reason must be a substantive non-empty string'
        required_parts = {'accept': (True, False), 'partly_accept': (True, True), 'reject': (False, True)}[decision]
        if tuple(review[field] is not None for field in ('accepted_part', 'rejected_part')) != required_parts:
            return f'{path}.accepted_part/rejected_part are inconsistent with decision={decision}'
        if not isinstance(review['retrospective_only'], bool):
            return f'{path}.retrospective_only must be boolean'
        if review['retrospective_only'] and (decision != 'reject' or review['accepted_part'] is not None):
            return f'{path}: retrospective-only requirements must be rejected as pre-response evidence'
        actual = review['actual_agent_turn_id']
        if actual is not None and (not _integer(actual) or ('agent', actual) not in turns or turns[('agent', actual)][1] == 'FINAL'):
            return f'{path}.actual_agent_turn_id must identify an existing non-FINAL Agent reply or be null'
        accepted = decision in ('accept', 'partly_accept')
        if accepted and actual is None:
            return f'{path}: accepted claims require actual_agent_turn_id'
        if decision == 'accept' and actual != review['reported_agent_turn_id']:
            return (f'{path}: a changed Agent attribution cannot be fully accepted; '
                    'use partly_accept and reject the original attribution, or reject the claim')
        for field in ('agent_evidence', 'prior_context'):
            evidence = review[field]
            if not isinstance(evidence, list):
                return f'{path}.{field} must be an array'
            if evidence and actual is None:
                return f'{path}.{field} requires actual_agent_turn_id to establish temporal scope'
            for ei, span in enumerate(evidence):
                ep = f'{path}.{field}[{ei}]'
                if (not isinstance(span, dict) or set(span) != {'speaker', 'turn_id', 'quote'}
                        or span['speaker'] not in ('agent', 'doctor') or not _integer(span['turn_id'])
                        or not _text(span['quote'])):
                    return f'{ep} must contain valid speaker, integer turn_id and substantive quote'
                source = turns.get((span['speaker'], span['turn_id']))
                if source is None or source[1] == 'FINAL' or span['quote'] not in source[2]:
                    return f'{ep} must quote exactly from its supplied non-FINAL speaker/turn'
                if field == 'agent_evidence' and (span['speaker'] != 'agent' or span['turn_id'] != actual):
                    return f'{ep} must identify the same actual Agent reply'
                if field == 'prior_context' and source[0] >= turns[('agent', actual)][0]:
                    return f'{ep} must occur strictly before the actual Agent reply'
        if accepted and not review['agent_evidence']:
            return f'{path}: accepted claims require exact Agent evidence'
        if accepted and review['feedback_kind'] == 'unresolved' and not review['prior_context']:
            return f'{path}: accepted unresolved claims require observed prior context'
    if seen != set(source_claims):
        missing = sorted(set(source_claims) - seen)
        return f'{prefix}.claim_reviews must cover every reflection claim; missing {missing}'
    return None
