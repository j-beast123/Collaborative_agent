"""Immutable, evidence-backed KEEP criteria shared across training cases.

Reuse is a membership decision, never a new causal-effect claim. Original source
rubrics and witness proofs remain immutable; only scorer-facing IDs are aliased.
"""
from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import replace

from src.online_rl.integrated.records import from_dict, sha256_json, sha256_text
from src.online_rl.tree.agent_state import visible_rubric
from src.online_rl.tree.records import TrajectoryRubric, to_dialogue
from src.judges.rubric_schema import PUBLIC_CRITERION_FIELDS, has_public_criterion, validate_public_criterion

VERSION = 'cumulative_keep_bank_v1'
RETRIEVAL_VERSION = 'deterministic_lexical_bm25_v1'
CHECKS = ('same_user_context', 'same_agent_state', 'same_behavior',
          'same_score_boundaries', 'observed_source_anchor', 'no_new_premise')
PUBLIC_CHECKS = CHECKS + ('same_minimum_performance', 'same_completion_condition')
DECISION_FIELDS = {'reuse', 'bank_id', 'selected_agent_turn_id', 'doctor_quote',
                   'agent_quote', 'checks', 'rationale'}
SNAPSHOT_FIELDS = {'protocol', 'version', 'parent_sha256', 'entries', 'sha256'}


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Cumulative bank requires nonblank criterion/evidence strings')
    return ' '.join(unicodedata.normalize('NFC', value).split())


def criteria(rubric):
    """Exact normalized behavioral identity; no names, IDs, provenance or guides."""
    problems = validate_public_criterion(rubric)
    if problems: raise ValueError('; '.join(problems))
    app = rubric.get('applicability')
    if not isinstance(app, dict) or set(app) != {'user_side_context', 'agent_side_state'}:
        raise ValueError('Cumulative bank requires both applicability sides')
    normalized = {key: _text(app[key]) for key in sorted(app)}
    scores = rubric.get('scores')
    if not isinstance(scores, dict) or set(scores) != {'0', '1', '2'}:
        raise ValueError('Cumulative bank requires all three score boundaries')
    behaviors = {}
    for level in ('0', '1', '2'):
        row = scores[level]
        if not isinstance(row, dict):
            raise ValueError('Cumulative bank requires structured score boundaries')
        if any(_text(row.get(key)) != normalized[key] for key in normalized):
            raise ValueError('Score-level applicability differs from canonical condition')
        behaviors[level] = _text(row.get('behavior'))
    # Absent public fields are not filled with nulls or inferred from analysis:
    # legacy content hashes and existing immutable bank IDs remain identical.
    return {'applicability': normalized, 'required_behavior': _text(rubric.get('required_behavior')),
            'scores': behaviors, **{key: _text(rubric[key]) for key in PUBLIC_CRITERION_FIELDS if key in rubric}}


def content_hash(rubric):
    return sha256_json(criteria(rubric))


def _seal(payload):
    return {**payload, 'sha256': sha256_json(payload)}


def empty_snapshot():
    return _seal({'protocol': VERSION, 'version': 0, 'parent_sha256': None, 'entries': []})


def _validate_proof(proof):
    raw = proof['rubric']
    if raw.get('valid') is not True or raw.get('problems'):
        raise ValueError('Only valid rubrics may enter the cumulative bank')
    rubric = raw.get('rubric') or {}
    criteria(rubric)
    if raw.get('rubric_id') != rubric.get('rubric_id'):
        raise ValueError('Source rubric IDs disagree')
    gate = raw.get('gate') or {}
    public = has_public_criterion(rubric)
    expected_gates = ('public_mc_gate_v1', 'public_mc_gate_v2', 'public_mc_gate_v3') if public else ('evidence_gate_v2',)
    if (gate.get('protocol') not in expected_gates or gate.get('passed') is not True or gate.get('observable') is not True or
            gate.get('score_boundaries_clear') is not True or
            (gate.get('source_anchor') or {}).get('matches') is not True):
        raise ValueError('Cumulative registration requires the evidence quality gate')
    if rubric.get('criterion_version') == 'public_behavior_mc_semantic_v2' and gate.get('protocol') != 'public_mc_gate_v2':
        raise ValueError('Semantic v2 criterion requires its own v2 gate evidence')
    if rubric.get('criterion_version') == 'public_behavior_mc_semantic_v3' and gate.get('protocol') != 'public_mc_gate_v3':
        raise ValueError('Semantic v3 criterion requires its own v3 gate evidence')
    if public and (gate.get('criterion_sha256') != content_hash(rubric) or
                   gate.get('public_criterion_version') != rubric['criterion_version']):
        raise ValueError('Public criterion requires its own versioned gate proof')
    source_check = raw.get('source_check') or {}
    stage = source_check.get('second') if source_check.get('status') == 'REPAIRED_PASS' else source_check.get('first')
    if source_check.get('status') not in ('PASS', 'REPAIRED_PASS') or (stage or {}).get('applicable') is not True:
        raise ValueError('Cumulative registration requires verified source applicability')
    origin = rubric.get('source_provenance') or {}
    if (origin.get('source_case_id') != proof['case_id'] or
            origin.get('source_trajectory_id') != raw.get('leaf_id') or
            raw.get('prompt_sha256') != proof['prompt_sha256']):
        raise ValueError('Cumulative source provenance mismatch')
    for key in ('case_id', 'policy_sha256', 'prompt_sha256', 'record_sha256'):
        _text(proof[key])
    witness = proof['witness']
    expected_stratum = origin.get('source_outcome')
    if (witness.get('decision') != 'KEEP' or witness.get('rubric_id') != raw['rubric_id'] or
            expected_stratum not in ('success', 'failure') or witness.get('stratum') != expected_stratum):
        raise ValueError('Cumulative registration requires original KEEP witness')
    target = 0.0 if expected_stratum == 'success' else 1.0
    tests = [row for row in witness.get('tested', [])
             if row.get('leaf_id') == witness.get('validated_by_trajectory_id') and row.get('result') == 'PASS']
    if len(tests) != 1:
        raise ValueError('Cumulative KEEP must identify one passing witness member')
    test = tests[0]
    audit = test.get('intervention_audit') or {}
    answer = audit.get('answer') or {}
    if (audit.get('protocol') != 'intervention_instruction_audit_v1' or audit.get('passed') is not True or
            answer.get('instruction_aligned') is not True or answer.get('unsupported_context_premise') is not False or
            answer.get('conclusion_prescribed') is not False):
        raise ValueError('Cumulative registration requires the recorded passing intervention audit')
    outcomes = test.get('outcomes') or []
    runs = test.get('rerun_records') or []
    if (test.get('stage1_applicable') is not True or not 2 <= len(outcomes) <= 3 or
            len(runs) != len(outcomes) or test.get('reruns') != len(runs) or
            sum(type(o) in (int, float) and o == target for o in outcomes) < 2):
        raise ValueError('Cumulative KEEP lacks two actual valid target hits')
    seeds = set()
    for outcome, run in zip(outcomes, runs):
        if (run.get('outcome') != outcome or run.get('rubric_id') != raw['rubric_id'] or
                run.get('leaf_id') != test['leaf_id'] or
                run.get('continuation_policy_sha256') != proof['policy_sha256'] or
                run.get('level') != ('0' if target == 0.0 else '2')):
            raise ValueError('Witness rerun identity differs from its source proof')
        if public and (run.get('intervention_spec') or {}).get('rubric_sha256') != sha256_json(rubric):
            raise ValueError('Public criterion cannot inherit another criterion witness')
        seed = (run.get('seed'), run.get('agent_seed'))
        if seed in seeds or seed[0] is None:
            raise ValueError('Duplicate/missing independent witness rerun identity')
        seeds.add(seed)
    return proof


def validate_snapshot(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != SNAPSHOT_FIELDS or snapshot.get('protocol') != VERSION:
        raise ValueError('Invalid cumulative bank snapshot contract')
    if type(snapshot['version']) is not int or snapshot['version'] < 0:
        raise ValueError('Invalid cumulative bank version')
    if (snapshot['version'] == 0) != (snapshot['parent_sha256'] is None):
        raise ValueError('Invalid cumulative bank parent version')
    if snapshot['sha256'] != sha256_json({k: v for k, v in snapshot.items() if k != 'sha256'}):
        raise ValueError('Cumulative bank snapshot hash mismatch')
    if not isinstance(snapshot['entries'], list):
        raise ValueError('Cumulative bank entries must be a list')
    if snapshot['version'] == 0 and snapshot['entries']:
        raise ValueError('Initial cumulative bank version must be empty')
    seen = set()
    for entry in snapshot['entries']:
        if set(entry) != {'bank_id', 'content_sha256', 'rubric', 'sources'}:
            raise ValueError('Invalid cumulative bank entry')
        key = content_hash(entry['rubric']['rubric'])
        if entry['content_sha256'] != key or entry['bank_id'] != 'bank_' + key or key in seen:
            raise ValueError('Cumulative bank content identity or duplicate mismatch')
        seen.add(key)
        if not isinstance(entry['sources'], list) or not entry['sources']:
            raise ValueError('Cumulative bank entry lacks source evidence')
        if entry['rubric'] != entry['sources'][0]['rubric']:
            raise ValueError('Canonical source rubric changed')
        source_ids = set()
        for proof in entry['sources']:
            _validate_proof(proof)
            if content_hash(proof['rubric']['rubric']) != key:
                raise ValueError('Deduplicated source has different criteria')
            proof_id = sha256_json(proof)
            if proof_id in source_ids:
                raise ValueError('Duplicate cumulative source proof')
            source_ids.add(proof_id)
    if [entry['bank_id'] for entry in snapshot['entries']] != sorted(seen_id for seen_id in ('bank_' + k for k in seen)):
        raise ValueError('Cumulative bank entries must have deterministic ID order')
    return snapshot


def extend_snapshot(snapshot, records):
    """Freeze one batch extension. Caller publishes its hash with the checkpoint."""
    validate_snapshot(snapshot)
    by_key = {entry['content_sha256']: copy.deepcopy(entry) for entry in snapshot['entries']}
    for record in sorted(records, key=lambda r: (r.get('case_cursor', -1), r['case_id'])):
        canonical = {row['rubric_id']: row for row in record.get('trajectory_rubrics', [])}
        witnesses = {row['rubric_id']: row for row in (record.get('rubric_bank') or {}).get('witness', [])}
        for selection in record.get('selection', []):
            if selection.get('retained') is not True:
                continue
            rid = selection['rubric_id']
            if rid not in canonical or rid not in witnesses:
                raise ValueError('Retained rubric lacks its original generation/witness record')
            proof = copy.deepcopy({'case_id': record['case_id'], 'policy_sha256': record['policy_sha256'],
                'prompt_sha256': record['prompt_sha256'], 'record_sha256': sha256_json(record),
                'case_cursor': record.get('case_cursor'), 'rubric': canonical[rid], 'witness': witnesses[rid]})
            _validate_proof(proof)
            key = content_hash(canonical[rid]['rubric'])
            if key not in by_key:
                by_key[key] = {'bank_id': 'bank_' + key, 'content_sha256': key,
                               'rubric': copy.deepcopy(canonical[rid]), 'sources': []}
            if proof not in by_key[key]['sources']:
                by_key[key]['sources'].append(proof)
    return validate_snapshot(_seal({'protocol': VERSION, 'version': snapshot['version'] + 1,
        'parent_sha256': snapshot['sha256'], 'entries': [by_key[key] for key in sorted(by_key)]}))


def scoring_rubrics(snapshot):
    validate_snapshot(snapshot)
    result = []
    for entry in snapshot['entries']:
        original = from_dict(TrajectoryRubric, copy.deepcopy(entry['rubric']))
        result.append(replace(original, rubric_id=entry['bank_id'],
                              rubric={**original.rubric, 'rubric_id': entry['bank_id']}))
    return result


def _tokens(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text).casefold(), re.UNICODE)


def shortlist(snapshot, query, top_k=8):
    """Local retrieval only. Similarity is never proof of applicability/reuse."""
    validate_snapshot(snapshot)
    if not isinstance(query, str) or type(top_k) is not int or not 1 <= top_k <= 64:
        raise ValueError('Cumulative retrieval requires a string and top_k in 1..64')
    entries = snapshot['entries']
    if len(entries) <= top_k:
        return copy.deepcopy(entries)
    documents = [Counter(_tokens(json.dumps(criteria(e['rubric']['rubric']), ensure_ascii=False))) for e in entries]
    terms = set(_tokens(query)); n = len(documents)
    average = sum(sum(d.values()) for d in documents) / n or 1.0
    frequency = Counter(term for d in documents for term in d)
    def score(index):
        d = documents[index]; length = sum(d.values())
        return sum(math.log(1 + (n - frequency[t] + .5) / (frequency[t] + .5)) *
                   d[t] * 2.2 / (d[t] + 1.2 * (.25 + .75 * length / average))
                   for t in terms if d[t])
    ranked = sorted(range(n), key=lambda i: (-score(i), entries[i]['bank_id']))[:top_k]
    return [copy.deepcopy(entries[i]) for i in ranked]


def _payload(tin, candidates):
    return {'case_id': tin.case_id, 'trajectory_id': tin.leaf_id, 'scenario': tin.scenario,
            'question': tin.question, 'options': dict(tin.options), 'source_outcome': tin.outcome,
            'dialogue': to_dialogue(tin.dialogue), 'doctor_feedback': tin.feedback,
            'candidates': [{'bank_id': e['bank_id'], 'rubric': {
                k: v for k, v in visible_rubric(e['rubric']['rubric']).items() if k != 'rubric_id'}} for e in candidates]}


def _validate_decision(value, payload):
    checks = _decision_checks(payload)
    if (not isinstance(value, dict) or set(value) != DECISION_FIELDS or type(value.get('reuse')) is not bool or
            not isinstance(value.get('checks'), dict) or set(value['checks']) != set(checks) or
            any(type(v) is not bool for v in value['checks'].values())):
        raise ValueError('Cumulative matcher returned an incomplete decision')
    _text(value.get('rationale'))
    fields = ('bank_id', 'selected_agent_turn_id', 'doctor_quote', 'agent_quote')
    if not value['reuse']:
        if any(value.get(k) is not None for k in fields):
            raise ValueError('Nonreuse must not assert a reused target or evidence')
        return value
    if not all(value['checks'].values()):
        raise ValueError('Cumulative reuse requires every equivalence check')
    if value['bank_id'] not in {e['bank_id'] for e in payload['candidates']}:
        raise ValueError('Cumulative reuse named an unknown bank ID')
    selected = next(e for e in payload['candidates'] if e['bank_id'] == value['bank_id'])
    if checks == PUBLIC_CHECKS and not has_public_criterion(selected['rubric']):
        raise ValueError('Public M/C reuse cannot silently upgrade a legacy criterion')
    turn = value['selected_agent_turn_id']
    positions = [i for i, row in enumerate(payload['dialogue']) if row['speaker'] == 'agent']
    if type(turn) is not int or not 1 <= turn <= len(positions):
        raise ValueError('Cumulative reuse selected an absent Agent turn')
    position = positions[turn - 1]
    aq, dq = value['agent_quote'], value['doctor_quote']
    _text(aq); _text(dq)
    if aq not in payload['dialogue'][position]['text']:
        raise ValueError('Cumulative reuse Agent quote is not literal at selected turn')
    if not any(row['speaker'] == 'doctor' and dq in row['text'] for row in payload['dialogue'][:position]):
        raise ValueError('Cumulative reuse Doctor quote must precede selected Agent turn')
    return value


def _decision_checks(payload):
    return (PUBLIC_CHECKS if any(has_public_criterion(row['rubric']) for row in payload['candidates']) else CHECKS)


def validate_reuse(tin, snapshot, match):
    validate_snapshot(snapshot)
    if match.get('leaf_id', tin.leaf_id) != tin.leaf_id or match.get('snapshot_sha256') != snapshot['sha256']:
        raise ValueError('Saved cumulative reuse snapshot/trajectory differs')
    if match.get('trajectory_sha256') != sha256_json(_payload(tin, [])):
        raise ValueError('Saved cumulative reuse dialogue/feedback changed')
    ids = match.get('retrieval_ids')
    by_id = {e['bank_id']: e for e in snapshot['entries']}
    if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(rid not in by_id for rid in ids):
        raise ValueError('Saved cumulative reuse retrieval IDs differ')
    value = {key: match.get(key) for key in DECISION_FIELDS}
    _validate_decision(value, _payload(tin, [by_id[rid] for rid in ids]))
    positions = [i for i, row in enumerate(tin.dialogue) if row[0] == 'agent']
    if value['reuse'] and match.get('prefix_sha256') != sha256_json(to_dialogue(tin.dialogue[:positions[value['selected_agent_turn_id'] - 1]])):
        raise ValueError('Saved cumulative reuse prefix changed')
    if not value['reuse'] or match.get('sourceproof') != by_id[value['bank_id']]['sources']:
        raise ValueError('Saved cumulative reuse source proof differs')
    return match


def match_existing(tin, snapshot, matcher, cache, top_k=8):
    query = '\n'.join([tin.question, tin.feedback, *(row[3] for row in tin.dialogue)])
    candidates = shortlist(snapshot, query, top_k)
    if not candidates:
        return None
    payload = _payload(tin, candidates)
    identity = {'protocol': VERSION, 'retrieval': RETRIEVAL_VERSION, 'top_k': top_k,
                'snapshot_sha256': snapshot['sha256'], 'input_sha256': sha256_json(payload),
                'model': getattr(matcher, 'model', None), 'settings': getattr(matcher, 'settings', None),
                'prompt_sha256': getattr(matcher, 'prompt_sha256', None)}
    def call():
        response = matcher(payload)
        decision = {key: response.get(key) for key in DECISION_FIELDS}
        _validate_decision(decision, payload)
        return response
    response = cache.call('cumulative_bank_match', identity, call)
    decision = _validate_decision({key: response.get(key) for key in DECISION_FIELDS}, payload)
    if not decision['reuse']:
        return None
    selected = next(e for e in candidates if e['bank_id'] == decision['bank_id'])
    positions = [i for i, row in enumerate(tin.dialogue) if row[0] == 'agent']
    result = {**copy.deepcopy(decision), 'leaf_id': tin.leaf_id,
              'sourceproof': copy.deepcopy(selected['sources']), 'snapshot_sha256': snapshot['sha256'],
              'retrieval_ids': [e['bank_id'] for e in candidates],
              'trajectory_sha256': sha256_json(_payload(tin, [])),
              'prefix_sha256': sha256_json(to_dialogue(tin.dialogue[:positions[decision['selected_agent_turn_id'] - 1]]))}
    return validate_reuse(tin, snapshot, result)


def api_bank_matcher(key, model, settings, prompt_text):
    """Same generator provider, plain JSON plus strict local checks and one repair."""
    from src.online_rl.tree.backends import RubricOutputIncomplete, _generator_json
    settings = {**settings, 'response_format': None, 'cumulative_bank_protocol': VERSION,
                'matcher_prompt_sha256': sha256_text(prompt_text)}
    schema = {'type': 'object', 'additionalProperties': False, 'required': sorted(DECISION_FIELDS),
        'properties': {'reuse': {'type': 'boolean'}, 'bank_id': {'type': ['string', 'null']},
            'selected_agent_turn_id': {'type': ['integer', 'null'], 'minimum': 1},
            'doctor_quote': {'type': ['string', 'null']}, 'agent_quote': {'type': ['string', 'null']},
            'rationale': {'type': 'string'}, 'checks': {'type': 'object', 'additionalProperties': False,
                'required': list(CHECKS), 'properties': {c: {'type': 'boolean'} for c in CHECKS}}}}
    def decide(payload):
        call_schema = copy.deepcopy(schema)
        checks = _decision_checks(payload)
        call_schema['properties']['checks'] = {'type': 'object', 'additionalProperties': False,
            'required': list(checks), 'properties': {c: {'type': 'boolean'} for c in checks}}
        original = json.dumps(payload, ensure_ascii=False)
        user = original; attempts = []
        for attempt in range(2):
            raw = None; meta = {}
            try:
                raw, meta = _generator_json(key, model, prompt_text, user, settings, call_schema,
                                            'cumulative_bank_match', 'rubric_matcher')
                parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()))
                value = _validate_decision(parsed, payload)
            except (ValueError, TypeError, AttributeError, RubricOutputIncomplete) as exc:
                if getattr(exc, 'audit', None) and not meta:
                    meta = exc.audit
                attempts.append({'user': user, 'raw': raw, 'meta': meta, 'error': str(exc)})
                if attempt:
                    error = RubricOutputIncomplete(f'Cumulative bank matcher failed after one correction: {exc}')
                    error.audit = {'contract': VERSION, 'system': prompt_text, 'user': original,
                                   'settings': settings, 'schema': call_schema, 'attempts': attempts}
                    raise error from exc
                user = original + '\n\nCorrect only the invalid decision using the SAME original input. Do not invent evidence or return nonreuse merely to conceal a formatting error. Return complete JSON.\n' + json.dumps({'schema': call_schema, 'error': str(exc), 'previous_output': raw}, ensure_ascii=False)
                continue
            attempts.append({'user': user, 'raw': raw, 'meta': meta, 'error': None})
            return {**value, 'audit': {'contract': VERSION, 'system': prompt_text,
                    'user': original, 'settings': settings, 'attempts': attempts}}
    decide.model = model
    decide.settings = settings
    decide.prompt_sha256 = sha256_text(prompt_text)
    return decide
