"""One immutable Stage-1 decision per visible state, rubric and judge contract.

Source checks, witness checks and response scoring use the same singleton judge
request. Node IDs, target replies and downstream evidence never distinguish
otherwise identical applicability inputs. The opt-in public contract also binds
hidden pre-response belief in its cache identity without exposing it to the judge.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Callable

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.agent_state import visible_rubric
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.posthoc import case_view
from src.online_rl.tree.records import AgentNode, to_dialogue


APPLICABILITY_CONTRACT = 'shared_state_rubric_applicability_v2'
APPLICABILITY_CACHE_KIND = 'rubric_state_applicability'


class ScoringContractError(RuntimeError):
    """An incomplete/invalid judge response must not become NA or a reward."""


class ApplicabilityUncertain(RuntimeError):
    """A valid but unresolved Stage-1 judgment blocks scoring without becoming NA."""

    scoring_status = 'applicability_uncertain'


from src.common.config import IntegrationUnavailable
UNCERTAIN_REJUDGMENTS = 2
UNRESOLVED_STATUS = 'applicability_unresolved'


def resolve_uncertain(first: dict, rejudge, retries: int = UNCERTAIN_REJUDGMENTS) -> tuple[dict, dict]:
    """`first` is a valid Stage-1 row whose verdict is uncertain (applicable is None).
    rejudge(n) asks the SAME judge again (same prompt, model, input, parser) and returns its validated row. A fresh TRUE or
    FALSE is used normally. If every re-judgment is also uncertain, only this rubric / state pair is left unresolved
    (applicable stays None, unresolved_abstain = True): callers skip the pair; it is never read as FALSE."""
    trail = []
    for n in range(1, retries + 1):
        row = rejudge(n); verdict = row['applicable']
        trail.append({'retry': n, 'result': 'true' if verdict is True else 'false' if verdict is False else 'uncertain'})
        if verdict is not None:
            return row, {'initial': 'uncertain', 'retries': trail,
                         'final': 'retry_resolved_true' if verdict else 'retry_resolved_false'}
    return {**first, 'unresolved_abstain': True}, {'initial': 'uncertain', 'retries': trail, 'final': 'final_unresolved_abstain'}


EXECUTION_UNRESOLVED = 'execution_unresolved'      # execution-invalid judge response after every local retry: no verdict at all


def execution_invalid(exc) -> bool:
    """An IntegrationUnavailable whose audit shows an execution-invalid / truncated judge response (never a transport or
    credential failure)."""
    validation = (getattr(exc, 'audit', None) or {}).get('validation') or {}
    return validation.get('status') in ('execution_error', 'execution_incomplete')


def log_execution_unresolved(cache, *, stage, case_id, node_id=None, rubric_id=None, attempts=None, reasons=None, reference=None):
    """One line per terminal EXECUTION_UNRESOLVED judgment in <case cache>.execution_unresolved.jsonl (never mixed with UNCERTAIN)."""
    path = getattr(cache, 'path', None)
    if not path: return
    import json
    from pathlib import Path
    with Path(path).with_suffix('.execution_unresolved.jsonl').open('a') as f:
        f.write(json.dumps({'final_status': 'EXECUTION_UNRESOLVED', 'stage': stage, 'case_id': case_id, 'node_id': node_id,
                            'rubric_id': rubric_id, 'total_attempts': attempts, 'reasons_by_attempt': reasons, 'raw_reference': reference},
                           ensure_ascii=False) + '\n')


def _log_uncertain(cache, case, node, rubric_id, audit, cache_key):
    """One line per uncertain first verdict in <case cache>.applicability_uncertain.jsonl (repeated on cache replays; same cache_key)."""
    path = getattr(cache, 'path', None)
    if not path: return
    import json
    from pathlib import Path
    with Path(path).with_suffix('.applicability_uncertain.jsonl').open('a') as f:
        f.write(json.dumps({'case_id': case.get('case_id'), 'trajectory_node_id': node.node_id, 'rubric_id': rubric_id,
                            'cache_key': cache_key, **audit}, ensure_ascii=False) + '\n')


def _reviewer(reviewer):
    """Prefer the exact callable exposed to the source and witness paths."""
    callback = getattr(reviewer, 'applicability', None)
    if callable(callback):
        return callback, getattr(callback, 'model', None), getattr(callback, 'protocol', None), getattr(callback, 'review_identity', None)
    callback = getattr(reviewer, 'review_state', None)
    if callable(callback):
        review_identity = getattr(reviewer, 'review_identity', None)
        protocol = (review_identity.get('protocol') if isinstance(review_identity, dict) and 'protocol' in review_identity
                    else getattr(reviewer, 'protocol', None))
        return callback, getattr(reviewer, 'model', None), protocol, review_identity
    if not callable(reviewer):
        raise ScoringContractError('Stage-1 reviewer must be callable')
    return reviewer, getattr(reviewer, 'model', None), getattr(reviewer, 'protocol', None), getattr(reviewer, 'review_identity', None)


def _validate(value, rubric_id):
    if not isinstance(value, dict) or set(value) != {rubric_id}:
        raise ScoringContractError(f'{rubric_id}: state applicability must contain exactly the requested rubric ID')
    row = value[rubric_id]
    if not isinstance(row, dict) or row.get('error'):
        raise ScoringContractError(f'{rubric_id}: invalid state applicability result')
    quality_uncertain = (row.get('stage1_contract') in ('compact_condition_v7_3', 'compact_condition_v7_5') and
                         row.get('applicable') is None and row.get('applicability_status') == 'uncertain')
    if type(row.get('applicable')) is not bool and not quality_uncertain:
        raise ScoringContractError(f'{rubric_id}: invalid state applicability result')
    if row.get('stage1_contract') in ('compact_condition_v7_1', 'compact_condition_v7_3',
                                      'compact_condition_v7_5'):
        if not isinstance(row.get('stage1'), dict):
            raise ScoringContractError('Missing compact condition evidence')
        return value
    if 'stage1' in row:
        stage1 = row['stage1']
        if not isinstance(stage1, dict):
            raise ScoringContractError(f'{rubric_id}: invalid Stage-1 evidence')
        fields = {'applicable', 'evidence_basis', 'missing_requirement'}
        # Generic deterministic adapters may omit evidence or provide an empty
        # mapping. Native structured judge evidence must keep its full schema.
        if row.get('stage1_contract') in (CONDITION_EVIDENCE_CONTRACT, CONDITION_EVIDENCE_CONTRACT_V2, CONDITION_EVIDENCE_CONTRACT_V3):
            validate_condition_evidence(stage1, contract=row['stage1_contract'])
            if stage1['applicable'] != row['applicable']:
                raise ScoringContractError(f'{rubric_id}: contradictory applicability evidence')
        elif stage1:
            if (set(stage1) != fields or type(stage1.get('applicable')) is not bool
                    or stage1['applicable'] != row['applicable']
                    or any(not isinstance(stage1.get(key), str) for key in fields - {'applicable'})):
                raise ScoringContractError(f'{rubric_id}: malformed or contradictory Stage-1 evidence')
    return value


def state_applicability(case: dict, node: AgentNode, rubric: dict, reviewer: Callable,
                        cache: CallCache) -> dict:
    """Return a validated decision and its stable cache provenance.

The cache supplies single-flight behavior, including simultaneous calls from
different phases. Each missing rubric is requested alone: rubric bundles must
not change the judge context or create competing decisions for the same item.
"""
    visible = visible_rubric(rubric)
    rubric_id = visible.get('rubric_id')
    if not isinstance(rubric_id, str) or not rubric_id:
        raise ScoringContractError('Stage-1 rubric requires a nonempty rubric ID')
    view, prefix = case_view(case), to_dialogue(node.prefix)
    callback, model, protocol, review_identity = _reviewer(reviewer)
    evidence_contract = getattr(callback, 'applicability_contract', None)
    if evidence_contract in (CONDITION_EVIDENCE_CONTRACT, CONDITION_EVIDENCE_CONTRACT_V2, CONDITION_EVIDENCE_CONTRACT_V3):
        # Stable singleton alias: names assigned by source, bank, or a pilot
        # cannot produce competing Stage-1 decisions for identical criteria.
        canonical = {**visible, 'rubric_id': 'criterion'}
        identity = {
            'contract': evidence_contract,
            'state_sha256': sha256_json([view, prefix, (node.trace or {}).get('pre_response_belief')]),
            'policy': node.policy_sha256,
            'rubric_sha256': sha256_json({k: v for k, v in visible.items() if k != 'rubric_id'}),
            'model': model, 'protocol': protocol, 'review_identity': review_identity,
        }
        def validate(value):
            _validate(value, 'criterion')
            row = value['criterion']
            if row.get('stage1_contract') != evidence_contract:
                raise ScoringContractError('Public applicability evidence contract is missing')
            validate_condition_evidence(row['stage1'], condition_state(view, prefix), condition_items([canonical])['criterion'],
                                        contract=evidence_contract)
            return value
        value = cache.call(APPLICABILITY_CACHE_KIND, identity,
                           lambda: validate(callback(view, prefix, [canonical])))
        row = deepcopy(validate(value)['criterion'])
        row['provenance'] = {'cache_kind': APPLICABILITY_CACHE_KIND,
                             'cache_key': sha256_json([APPLICABILITY_CACHE_KIND, identity]), 'identity': identity}
        return row
    if evidence_contract in ('compact_condition_v7_1', 'compact_condition_v7_3',
                             'compact_condition_v7_5'):
        from src.online_rl.tree.compact_audit import _app, app_schema
        from jsonschema import validate as validate_schema
        canonical = {**visible, 'rubric_id': 'criterion'}
        state = condition_state(view, prefix)
        items = condition_items([canonical])
        identity = {'contract': evidence_contract,
            'state_sha256': sha256_json([view, prefix] if evidence_contract in (
                                        'compact_condition_v7_3', 'compact_condition_v7_5')
                                        else [view, prefix, (node.trace or {}).get('pre_response_belief')]),
            'rubric_sha256': sha256_json({k:v for k,v in visible.items() if k != 'rubric_id'}),
            'model': model, 'protocol': protocol, 'review_identity': review_identity}
        if evidence_contract == 'compact_condition_v7_1':
            identity['policy'] = node.policy_sha256
        def checked(value):
            _validate(value, 'criterion')
            row = value['criterion']
            if row.get('stage1_contract') != evidence_contract:
                raise ScoringContractError('Missing compact applicability contract')
            raw = {'answers': {'criterion': row['stage1']}}
            validate_schema(raw, app_schema(items,
                quality=evidence_contract in ('compact_condition_v7_3', 'compact_condition_v7_5'),
                relation=evidence_contract == 'compact_condition_v7_5'))
            judged = _app(raw, {'state': state, 'items': items})['applicability']['criterion']
            if judged['applicable'] != row['applicable'] or (
                    evidence_contract in ('compact_condition_v7_3', 'compact_condition_v7_5') and
                    judged['status'] != row.get('applicability_status')):
                raise ScoringContractError('Contradictory compact applicability')
            return value
        def judged():
            try:
                return checked(callback(view, prefix, [canonical]))
            except IntegrationUnavailable as exc:
                # Every configured local retry returned an execution-invalid response: the judgment is
                # EXECUTION_UNRESOLVED (no TRUE / FALSE / UNCERTAIN) and the run goes on.
                if not getattr(callback, 'execution_unresolved_terminal', False) or not execution_invalid(exc): raise
                audit = getattr(exc, 'audit', None) or {}
                tries = [*audit.get('prior_failed_attempts', []), audit]
                return {'criterion': {'applicable': None, 'applicability_status': EXECUTION_UNRESOLVED, 'execution_unresolved': True,
                                      'stage1_contract': evidence_contract, 'stage1': {}, 'total_attempts': len(tries),
                                      'reasons_by_attempt': [{k: str((a.get('validation') or {}).get(k))[:160] for k in ('status', 'kind', 'error')} for a in tries]}}
        value = cache.call(APPLICABILITY_CACHE_KIND, identity, judged)
        terminal = isinstance(value, dict) and isinstance(value.get('criterion'), dict) and value['criterion'].get('execution_unresolved') is True
        row = deepcopy(value['criterion'] if terminal else checked(value)['criterion'])
        row['provenance'] = {'cache_kind': APPLICABILITY_CACHE_KIND,
            'cache_key': sha256_json([APPLICABILITY_CACHE_KIND, identity]), 'identity': identity}
        if terminal:
            log_execution_unresolved(cache, stage='applicability', case_id=case.get('case_id'), node_id=node.node_id, rubric_id=rubric_id,
                                     attempts=row['total_attempts'], reasons=row['reasons_by_attempt'], reference=row['provenance']['cache_key'])
            return row
        if row['applicable'] is None:                       # uncertain: up to two fresh re-judgments, then abstain on this pair only
            first_key = row['provenance']['cache_key']
            def rejudge(n):
                again = {**identity, 'uncertain_rejudgment': n}    # a new cache key = a fresh call; the request itself is unchanged
                fresh = deepcopy(checked(cache.call(APPLICABILITY_CACHE_KIND, again,
                                                    lambda: checked(callback(view, prefix, [canonical]))))['criterion'])
                fresh['provenance'] = {'cache_kind': APPLICABILITY_CACHE_KIND,
                                       'cache_key': sha256_json([APPLICABILITY_CACHE_KIND, again]), 'identity': again}
                return fresh
            row, audit = resolve_uncertain(row, rejudge)
            row['uncertain_rejudgment'] = audit
            _log_uncertain(cache, case, node, rubric_id, audit, first_key)
        return row
    identity = {
        'contract': APPLICABILITY_CONTRACT,
        'state_sha256': sha256_json([view, prefix]),
        'policy': node.policy_sha256,
        'rubric_id': rubric_id,
        'rubric_sha256': sha256_json(visible),
        'model': model,
        'protocol': protocol,
        'review_identity': review_identity,
    }
    value = cache.call(APPLICABILITY_CACHE_KIND, identity,
                       lambda: _validate(callback(view, prefix, [visible]), rubric_id))
    row = deepcopy(_validate(value, rubric_id)[rubric_id])
    row['provenance'] = {
        'cache_kind': APPLICABILITY_CACHE_KIND,
        'cache_key': sha256_json([APPLICABILITY_CACHE_KIND, identity]),
        'identity': identity,
    }
    return row


CONDITION_EVIDENCE_CONTRACT = 'public_condition_evidence_v1'
CONDITION_EVIDENCE_CONTRACT_V2 = 'public_condition_evidence_v2'
CONDITION_EVIDENCE_CONTRACT_V3 = 'public_condition_evidence_v3'
_QUOTE_FIELDS = {'turn_id': {'type': 'string'}, 'speaker': {'type': 'string'}, 'quote': {'type': 'string'}}
_CONDITION_FIELDS = {
    'condition': {'type': 'string'},
    'status': {'type': 'string', 'enum': ['supported', 'absent', 'unmet', 'not_required']},
    'quotes': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                                       'required': list(_QUOTE_FIELDS), 'properties': _QUOTE_FIELDS}},
    'inspected_turn_ids': {'type': 'array', 'items': {'type': 'string'}},
    'entailment': {'type': 'string'}, 'unmet_condition': {'type': 'string'},
}
CONDITION_FIELDS = {
    'applicable': {'type': 'boolean'}, 'evidence_basis': {'type': 'string'},
    'missing_requirement': {'type': 'string'},
    **{side: {'type': 'object', 'additionalProperties': False, 'required': list(_CONDITION_FIELDS),
              'properties': _CONDITION_FIELDS} for side in ('doctor_condition', 'agent_condition')},
}


def condition_state(view, prefix):
    """Only visible pre-response facts, with stable IDs for quote/absence scope."""
    turns = [{**{k: turn[k] for k in ('speaker', 'stage', 'round', 'text') if k in turn},
              'turn_id': f'prefix:{index}'} for index, turn in enumerate(prefix)]
    case = {k: view[k] for k in ('scenario', 'question', 'options') if k in view}
    evidence = [{'turn_id': f'case:{key}', 'speaker': 'case', 'text': value}
                for key, value in case.items() if isinstance(value, str) and value]
    evidence += [{'turn_id': f'option:{key}', 'speaker': 'case', 'text': value}
                 for key, value in case.get('options', {}).items()]
    return {'case_context': case, 'dialogue_before_target_reply': turns, 'case_evidence': evidence}


def condition_items(rubrics):
    """Do not transmit M/C, scores, source data, or the target response."""
    items = {}
    for rubric in rubrics:
        app = rubric.get('applicability')
        if isinstance(app, dict):
            doctor, agent = app['user_side_context'], app['agent_side_state']
        else:
            doctor, agent = rubric['applicability_condition'], ''
        items[rubric['rubric_id']] = {'rubric': rubric['name'], 'doctor_condition': doctor,
                                      'agent_condition': agent,
                                      **({'criterion_version': rubric['criterion_version']}
                                         if 'criterion_version' in rubric else {})}
    return items


def validate_condition_evidence(answer, state=None, item=None, *, contract=CONDITION_EVIDENCE_CONTRACT):
    if not isinstance(answer, dict) or set(answer) != set(CONDITION_FIELDS):
        raise ScoringContractError('Condition evidence fields differ')
    if type(answer['applicable']) is not bool or any(not isinstance(answer[k], str)
            for k in ('evidence_basis', 'missing_requirement')):
        raise ScoringContractError('Condition applicability/types are invalid')
    catalog = ({row['turn_id']: row for row in state['case_evidence'] + state['dialogue_before_target_reply']}
               if state is not None else None)
    prefix_ids = [row['turn_id'] for row in state['dialogue_before_target_reply']] if state is not None else None
    met = []
    for side in ('doctor_condition', 'agent_condition'):
        row = answer[side]
        if not isinstance(row, dict) or set(row) != set(_CONDITION_FIELDS):
            raise ScoringContractError(f'{side}: invalid condition evidence object')
        if any(not isinstance(row[k], str) for k in ('condition', 'status', 'entailment', 'unmet_condition')):
            raise ScoringContractError(f'{side}: condition text must be strings')
        if item is not None and row['condition'] != item[side]:
            raise ScoringContractError(f'{side}: condition changed from the declared rubric')
        allowed_statuses = _CONDITION_FIELDS['status']['enum'] + (['uncertain'] if contract == CONDITION_EVIDENCE_CONTRACT_V3 else [])
        if row['status'] not in allowed_statuses or not row['entailment'].strip():
            raise ScoringContractError(f'{side}: invalid status or empty entailment')
        if not isinstance(row['quotes'], list) or not isinstance(row['inspected_turn_ids'], list):
            raise ScoringContractError(f'{side}: quote/scope arrays required')
        if any(not isinstance(t, str) for t in row['inspected_turn_ids']) or len(set(row['inspected_turn_ids'])) != len(row['inspected_turn_ids']):
            raise ScoringContractError(f'{side}: invalid inspected turn IDs')
        if catalog is not None and any(t not in catalog for t in row['inspected_turn_ids']):
            raise ScoringContractError(f'{side}: inspected evidence outside the visible prefix')
        for quote in row['quotes']:
            if (not isinstance(quote, dict) or set(quote) != set(_QUOTE_FIELDS)
                    or any(not isinstance(value, str) or not value.strip() for value in quote.values())):
                raise ScoringContractError(f'{side}: malformed source quote')
            if catalog is not None:
                source = catalog.get(quote['turn_id'])
                if source is None or quote['speaker'] != source['speaker'] or quote['quote'] not in source['text']:
                    raise ScoringContractError(f'{side}: quote speaker/turn/text is not exact')
        if row['status'] == 'supported' and not row['quotes']:
            raise ScoringContractError(f'{side}: positive support requires an exact quote')
        if row['status'] == 'absent':
            if row['quotes'] or (prefix_ids is not None and row['inspected_turn_ids'] != prefix_ids):
                raise ScoringContractError(f'{side}: absence requires full inspected prefix and no invented quote')
        if row['status'] == 'not_required' and row['quotes']:
            raise ScoringContractError(f'{side}: no additional condition needs no invented quote')
        if (row['status'] in ('unmet', 'uncertain')) != bool(row['unmet_condition'].strip()):
            raise ScoringContractError(f'{side}: unmet condition contradicts its status')
        met.append(row['status'] not in ('unmet', 'uncertain'))
    if answer['applicable'] != all(met):
        raise ScoringContractError('Overall applicability contradicts the two condition decisions')
    if answer['applicable']:
        if not answer['evidence_basis'].strip() or answer['missing_requirement'].strip():
            raise ScoringContractError('Applicable condition evidence needs support and no missing condition')
    elif not answer['missing_requirement'].strip():
        raise ScoringContractError('NA requires an explicit unmet condition')


def public_client_identity(client):
    """Actual Luna request settings; omitted decoding controls use provider defaults."""
    return {'model': getattr(client, 'model', None),
            'system_sha256': sha256_json(getattr(client, 'system', None)),
            'client_protocol_sha256': getattr(client, 'protocol_sha256', None),
            'max_tokens': getattr(client, 'max_tokens', None),
            'timeout_s': getattr(client, 'timeout_s', None),
            'decoding': {'temperature': 'not_sent_provider_default',
                         'top_p': 'not_sent_provider_default', 'seed': 'not_sent_provider_default'},
            'response_format': 'strict_json_schema'}


def condition_evidence_fields(contract=CONDITION_EVIDENCE_CONTRACT):
    fields = deepcopy(CONDITION_FIELDS)
    if contract in (CONDITION_EVIDENCE_CONTRACT_V2, CONDITION_EVIDENCE_CONTRACT_V3):
        for side in ('doctor_condition', 'agent_condition'):
            row = fields[side] = deepcopy(fields[side])
            row['properties'].pop('condition')
            row['properties']['condition_ref'] = {'type': 'string', 'enum': [side]}
            if contract == CONDITION_EVIDENCE_CONTRACT_V3:
                row['properties']['status']['enum'].append('uncertain')
            row['required'] = list(row['properties'])
    elif contract != CONDITION_EVIDENCE_CONTRACT:
        raise ValueError('Unknown applicability evidence contract')
    return fields


def materialize_condition_evidence(answer, state, item, *, contract=CONDITION_EVIDENCE_CONTRACT_V2):
    """Bind each side to its unchanged condition, retaining literal turn evidence."""
    fields = condition_evidence_fields(contract)
    if not isinstance(answer, dict) or set(answer) != set(fields):
        raise ScoringContractError('V2 condition evidence fields differ')
    resolved = deepcopy(answer)
    for side in ('doctor_condition', 'agent_condition'):
        row = resolved[side]
        if not isinstance(row, dict) or set(row) != set(fields[side]['properties']):
            raise ScoringContractError(f'{side}: V2 condition evidence fields differ')
        if row.pop('condition_ref') != side:
            raise ScoringContractError(f'{side}: condition reference names the wrong side')
        row['condition'] = item[side]
    validate_condition_evidence(resolved, state, item, contract=contract)
    return resolved


def condition_evidence_reviewer(client, *, contract=CONDITION_EVIDENCE_CONTRACT):
    """Opt-in shared Stage 1; one evidence correction, never an error-to-NA fallback."""
    from src.common.config import IntegrationUnavailable
    from src.online_rl.tree.backends import with_role
    fields = condition_evidence_fields(contract)
    v2 = contract in (CONDITION_EVIDENCE_CONTRACT_V2, CONDITION_EVIDENCE_CONTRACT_V3)
    v3 = contract == CONDITION_EVIDENCE_CONTRACT_V3
    prompt = (Path(__file__).resolve().parents[3] /
              f'prompts/sherpa/applicability_public_mc_v{3 if v3 else 2 if v2 else 1}.txt').read_text()
    identity = {'contract': contract, 'prompt': sha256_json(prompt),
                'fields': sha256_json(fields), 'client': public_client_identity(client)}

    def review(view, prefix, rubrics):
        state, items = condition_state(view, prefix), condition_items(rubrics)
        for item in items.values():
            item['instructions'] = prompt
        requests, decisions, errors = [], [], []
        current = deepcopy(state)
        for attempt in range(1):
            messages = [{'role': 'system', 'content': client.system}, {'role': 'user', 'content': json.dumps(
                {'stage': 'applicability', 'state': current, 'items': items}, ensure_ascii=False)}]
            requests.append({'messages': messages, 'messages_sha256': sha256_json(messages),
                             'output_fields': deepcopy(fields)})
            values, record = with_role('rubric_scorer', lambda: client.judge(
                'applicability', current, items, fields, 'public_condition_evidence'))
            decisions.append(deepcopy(record))
            try:
                if not isinstance(values, dict) or set(values) != set(items):
                    raise ScoringContractError('Applicability rubric IDs differ')
                normalized = deepcopy(values)
                for rid, answer in values.items():
                    if v2:
                        normalized[rid] = materialize_condition_evidence(answer, state, items[rid], contract=contract)
                    else:
                        validate_condition_evidence(answer, state, items[rid])
                return {rid: {'applicable': answer['applicable'], 'stage1': answer,
                              'stage1_contract': contract,
                              **({'applicability_status': 'uncertain' if any(answer[side]['status'] == 'uncertain'
                                      for side in ('doctor_condition', 'agent_condition')) else 'decided'} if v3 else {}),
                              'evidence_validation': {'requests': requests, 'decisions': decisions,
                                  'errors': errors, 'exact_quotes_validated': True,
                                  'semantic_entailment': 'model_assessment_not_independent_proof'}}
                        for rid, answer in normalized.items()}
            except (ValueError, TypeError, ScoringContractError) as exc:
                errors.append(str(exc))
                error = IntegrationUnavailable('Invalid public applicability evidence; no retry: ' + str(exc))
                error.audit = {'requests': requests, 'decisions': decisions, 'validation_errors': errors}
                raise error from exc
    review.model = getattr(client, 'model', None)
    review.protocol = sha256_json(identity)
    review.review_identity = identity
    review.applicability_contract = contract
    return review
