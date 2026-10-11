"""Post-witness semantic duplicate guard for cumulative KEEP registration.

Workers never use this decision while generating, gating, or verifying a rubric.
The coordinator calls it once, after KEEP evidence is complete and before the
next immutable bank snapshot is published.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cumulative_bank import criteria

VERSION = 'semantic_keep_guardian_v1'
CHECKS = ('same_user_context', 'same_agent_state', 'same_minimum_performance',
          'same_completion_condition', 'same_score_boundaries')
DECISIONS = ('SAME_MEANING', 'DISTINCT', 'UNCERTAIN')

DECISION_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['decisions'],
    'properties': {
        'decisions': {
            'type': 'array',
            'items': {
                'type': 'object', 'additionalProperties': False,
                'required': ['candidate_id', 'decision', 'representative_id', 'checks', 'reason'],
                'properties': {
                    'candidate_id': {'type': 'string'},
                    'decision': {'type': 'string', 'enum': list(DECISIONS)},
                    'representative_id': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
                    'checks': {'type': 'object', 'additionalProperties': False,
                               'required': list(CHECKS),
                               'properties': {key: {'type': 'boolean'} for key in CHECKS}},
                    'reason': {'type': 'string'},
                },
            },
        },
    },
}


def public_contract(rubric: dict) -> dict:
    """Only scorer-visible criterion content may influence consolidation."""
    return criteria(rubric)


def exact_key(rubric: dict) -> str:
    return sha256_json(public_contract(rubric))


def request(existing: list[dict], candidates: list[dict]) -> dict:
    return {
        'contract': VERSION,
        'rules': {
            'same_requires_all_checks': True,
            'partial_overlap_is_distinct': True,
            'uncertain_is_not_merged': True,
            'topic_or_name_similarity_is_insufficient': True,
        },
        'existing': [{'representative_id': row['representative_id'],
                      'criterion': public_contract(row['rubric'])} for row in existing],
        'candidates': [{'candidate_id': row['candidate_id'],
                        'criterion': public_contract(row['rubric'])} for row in candidates],
    }


def validate_response(value: dict, payload: dict) -> list[dict]:
    rows = value.get('decisions') if isinstance(value, dict) and set(value) == {'decisions'} else None
    expected = [row['candidate_id'] for row in payload['candidates']]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('Guardian response must cover every candidate exactly once')
    known = {row['representative_id'] for row in payload['existing']}
    seen, out = set(), []
    for position, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) != {'candidate_id', 'decision', 'representative_id', 'checks', 'reason'}
                or row.get('candidate_id') not in expected or row['candidate_id'] in seen
                or row.get('decision') not in DECISIONS or not isinstance(row.get('reason'), str) or not row['reason'].strip()
                or not isinstance(row.get('checks'), dict) or set(row['checks']) != set(CHECKS)
                or any(type(v) is not bool for v in row['checks'].values())):
            raise ValueError('Malformed Guardian decision')
        seen.add(row['candidate_id'])
        representative = row.get('representative_id')
        if row['decision'] == 'SAME_MEANING':
            # A new candidate may point only to the frozen bank or an earlier
            # accepted candidate, preventing cycles and order-dependent aliases.
            if representative not in known or not all(row['checks'].values()):
                raise ValueError('SAME_MEANING requires an earlier representative and all checks')
        elif representative is not None:
            raise ValueError('DISTINCT/UNCERTAIN cannot assert a representative')
        if row['decision'] != 'SAME_MEANING':
            known.add(row['candidate_id'])
        out.append(copy.deepcopy(row))
    if seen != set(expected):
        raise ValueError('Guardian response has missing or duplicate candidate IDs')
    # Preserve deterministic input order even if the provider reordered rows.
    by_id = {row['candidate_id']: row for row in out}
    return [by_id[candidate_id] for candidate_id in expected]


def decide(existing: list[dict], candidates: list[dict], model_call: Callable[[dict], dict]) -> dict:
    """Exact duplicates are local; only remaining candidates reach Gemini."""
    representatives = {exact_key(row['rubric']): row['representative_id'] for row in existing}
    exact, pending = [], []
    for row in candidates:
        key = exact_key(row['rubric'])
        if key in representatives:
            exact.append({'candidate_id': row['candidate_id'], 'decision': 'EXACT_SAME',
                          'representative_id': representatives[key], 'content_sha256': key})
        else:
            pending.append(row)
            representatives[key] = row['candidate_id']
    payload = request(existing, pending)
    semantic = validate_response(model_call(payload), payload) if pending else []
    return {'contract': VERSION, 'request_sha256': sha256_json(payload), 'request': payload,
            'exact': exact, 'semantic': semantic,
            'aliases': {row['candidate_id']: row['representative_id'] for row in exact + semantic
                        if row['decision'] in ('EXACT_SAME', 'SAME_MEANING')},
            'accepted': [row['candidate_id'] for row in semantic if row['decision'] != 'SAME_MEANING']}


def gemini_model_call(key: str, *, model: str = 'google/gemini-3.8-flash-20260902', max_tokens: int = 4096):
    """Pinned OpenRouter adapter. API/schema failures propagate; callers fail open by registering separately."""
    import json
    from pathlib import Path
    from src.online_rl.tree.backends import _generator_json
    root = Path(__file__).resolve().parents[3]
    system = (root / 'prompts/sherpa/bank_guardian_semantic_v1.txt').read_text()
    settings = {'max_tokens': max_tokens, 'temperature': 0.0, 'reasoning_effort': 'low',
                'response_format': None, 'api_provider': 'openrouter',
                'output_contract': VERSION}
    def call(payload):
        raw, meta = _generator_json(key, model, system, json.dumps(payload, ensure_ascii=False), settings,
                                    DECISION_SCHEMA, 'bank_guardian_decisions', 'rubric_bank_guardian')
        value = json.loads(raw)
        value['_usage'] = meta.get('usage')
        # Usage is evidence, not part of the semantic response contract.
        usage = value.pop('_usage')
        result = {'decisions': validate_response(value, payload)}
        call.last_usage = usage
        return result
    call.model = model
    call.settings = settings
    call.last_usage = None
    return call


def filter_registration_records(snapshot: dict, records: list[dict], model_call: Callable[[dict], dict]):
    """Return copies used only for bank extension plus a durable alias audit."""
    existing = [{'representative_id': entry['bank_id'], 'rubric': entry['rubric']['rubric']}
                for entry in snapshot['entries']]
    candidates, locations = [], {}
    for record_index, record in enumerate(records):
        rubrics = {row['rubric_id']: row for row in record.get('trajectory_rubrics', [])}
        for selection_index, selection in enumerate(record.get('selection', [])):
            if selection.get('retained') is not True:
                continue
            rid = selection['rubric_id']
            if rid not in rubrics:
                raise ValueError('Guardian candidate lacks its retained rubric')
            candidate_id = f"{record.get('case_cursor', record_index)}:{record['case_id']}:{rid}"
            candidates.append({'candidate_id': candidate_id, 'rubric': rubrics[rid]['rubric']})
            locations[candidate_id] = (record_index, selection_index)
    result = decide(existing, candidates, model_call)
    filtered = copy.deepcopy(records)
    for candidate_id, representative_id in result['aliases'].items():
        record_index, selection_index = locations[candidate_id]
        selection = filtered[record_index]['selection'][selection_index]
        selection['retained'] = False
        selection['bank_guardian_alias'] = representative_id
    return filtered, result
