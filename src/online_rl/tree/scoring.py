"""Response-level rubric scoring with the two-stage scorer (applicability on the pre-reply dialogue, then the 0/1/2 level;
non-applicable = No Match, kept as a record without a score). Scores stay raw, one record per (response, rubric)."""
from __future__ import annotations
from dataclasses import replace
import math
from typing import Callable
from src.online_rl.integrated.metrics import hard_value, r_a
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.posthoc import case_view
from src.online_rl.tree.records import AgentNode, ResponseRubricScore, TrajectoryRubric, to_dialogue
from src.online_rl.tree.agent_state import visible_rubric
from src.online_rl.tree.applicability import ScoringContractError, state_applicability
from src.online_rl.integrated.records import sha256_json

VISIBLE = ('rubric_id', 'name', 'applicability_condition', 'required_behavior', 'scores')
SCORE_RECORD_CONTRACT = 'complete_state_scoring_with_shared_applicability_v2'


def _mapping(value, expected, label, metadata=()):
    if not isinstance(value, dict):
        raise ScoringContractError(f'{label}: expected a mapping, not duplicate/list rows')
    actual = set(value) - set(metadata)
    if actual != set(expected):
        raise ScoringContractError(f'{label}: IDs differ; missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}')
    return value


def _score_record(node, rid, source, applicable, result=None):
    if not applicable:
        return ResponseRubricScore(node.node_id, rid, source, False, None, None, None, 'not_applicable')
    if not isinstance(result, dict):
        raise ScoringContractError(f'{node.node_id}/{rid}: missing behavior result for applicable rubric')
    if result.get('applicable', True) is not True or result.get('error'):
        raise ScoringContractError(f'{node.node_id}/{rid}: behavior result contradicts applicability or contains an error')
    status = result.get('scoring_status', 'scored')
    evidence = {key: result[key] for key in ('stage1', 'stage2', 'evidence_validation') if key in result}
    hard, probabilities = result.get('hard'), result.get('probabilities')
    if status == 'execution_unresolved':                         # behavior judgment invalid after every local retry: no score at all
        if hard is not None or probabilities is not None:
            raise ScoringContractError(f'{node.node_id}/{rid}: EXECUTION_UNRESOLVED must not carry a numeric score')
        return ResponseRubricScore(node.node_id, rid, source, True, None, None, None, status, evidence)
    if status == 'rubric_unscorable':
        if hard is not None or probabilities is not None:
            raise ScoringContractError(f'{node.node_id}/{rid}: UNSCORABLE must not carry a numeric score')
        return ResponseRubricScore(node.node_id, rid, source, True, None, None, None, status, evidence)
    if status != 'scored':
        raise ScoringContractError(f'{node.node_id}/{rid}: invalid scoring status {status!r}')
    if isinstance(hard, bool) or hard not in (0, 1, 2, '0', '1', '2'):
        raise ScoringContractError(f'{node.node_id}/{rid}: scored behavior requires a 0/1/2 hard label')
    hard_score = hard_value(hard)
    if probabilities is not None:
        if not isinstance(probabilities, dict) or set(probabilities) != {'0', '1', '2'}:
            raise ScoringContractError(f'{node.node_id}/{rid}: malformed score probabilities')
        try:
            values = [float(probabilities[k]) for k in ('0', '1', '2')]
        except (TypeError, ValueError) as exc:
            raise ScoringContractError(f'{node.node_id}/{rid}: nonnumeric score probabilities') from exc
        if any(not math.isfinite(x) or x < 0 for x in values) or sum(values) <= 0:
            raise ScoringContractError(f'{node.node_id}/{rid}: invalid score probabilities')
        score = r_a(probabilities)
    else:
        score = hard_score
    return ResponseRubricScore(node.node_id, rid, source, True, score, hard_score, probabilities, 'scored', evidence)


def _state_key(view, node):
    return sha256_json([view, to_dialogue(node.prefix), (node.trace or {}).get('pre_response_belief')])


def _mask_unscorable(scores, view, nodes):
    states = {n.node_id: _state_key(view, n) for n in nodes}
    excluded = {(states[s.response_node_id], s.rubric_id) for s in scores if s.scoring_status == 'rubric_unscorable'}
    return [replace(s, score=None, hard=None, probabilities=None, scoring_status='group_unscorable',
                    individual_assessment={'scoring_status': s.scoring_status, 'score': s.score, 'hard': s.hard,
                                           'probabilities': s.probabilities, 'evidence': s.scoring_evidence})
            if s.match and (states[s.response_node_id], s.rubric_id) in excluded else s for s in scores]


def score_nodes(case: dict, nodes: list[AgentNode], kept: list[TrajectoryRubric], scorer: Callable, cache: CallCache,
                workers: int = 1) -> list[ResponseRubricScore]:
    """scorer(case_view, prefix, rubrics, reply) -> {rubric_id: {'applicable': bool, 'probabilities': dict|None, 'hard': int|None}};
    workers > 1 (execution only): structured scorers run independent states concurrently,
    each reviewing applicability once then scoring its response batches. Legacy scorers
    parallelize responses. Results are always collected in node order."""
    if not kept: return []
    if len({n.node_id for n in nodes}) != len(nodes):
        raise ScoringContractError('Duplicate response node IDs')
    if len({r.rubric_id for r in kept}) != len(kept):
        raise ScoringContractError('Duplicate kept rubric IDs')
    rubrics = [visible_rubric(c.rubric) for c in kept]; src = {c.rubric_id: c.leaf_id for c in kept}; out = []
    if [r['rubric_id'] for r in rubrics] != [r.rubric_id for r in kept]:
        raise ScoringContractError('Kept rubric IDs disagree with rubric content')
    view = case_view(case)

    # Structured scorers can split state applicability from reply-level behavior.
    # Sibling replies with identical pre-reply context share Stage 1, while their
    # behavior scores remain separate (and can be sent in a small response batch).
    if callable(getattr(scorer, 'review_state', None)) and callable(getattr(scorer, 'score_behavior', None)):
        state_groups = {}
        for n in nodes:
            state_sha = _state_key(view, n)
            state_groups.setdefault((state_sha, n.policy_sha256), []).append(n)
        batch_fn = getattr(scorer, 'score_behavior_batch', None)
        batch_size = int(getattr(scorer, 'response_batch_size', 1)) if callable(batch_fn) else 1
        if batch_size < 1 or batch_size > 8:
            raise ValueError('response_batch_size must be between 1 and 8')

        def validate_match(value):
            _mapping(value, src, 'state match')
            for rid, row in value.items():
                if not isinstance(row, dict) or str(row.get('value')).upper() not in ('Y', 'N') or row.get('error'):
                    raise ScoringContractError(f'{rid}: invalid state match result')
            return value

        def score_state(item):
            # A state owns its review and response batches. No cross-state review
            # barrier or nested pool: at most `workers` API calls are in flight.
            (state_sha, _policy), group = item
            n = group[0]; prefix = to_dialogue(n.prefix)
            identity = {'state_sha256': state_sha, 'policy': n.policy_sha256,
                        'rubrics': sha256_json(rubrics), 'model': getattr(scorer, 'model', None),
                        'review_identity': getattr(scorer, 'review_identity', None),
                        'protocol': getattr(scorer, 'protocol', None), 'score_record_contract': SCORE_RECORD_CONTRACT}
            matched = None
            if callable(getattr(scorer, 'match_state', None)):
                matched = validate_match(cache.call('rubric_state_match', identity,
                    lambda: validate_match(scorer.match_state(view, prefix, rubrics))))
            # One independent Stage-1 request per rubric (singleton requests, exact single-flight cache): bounded concurrency,
            # results keyed in the original rubric order. SHERPA_SCORE_RUBRIC_WORKERS (default 1) is execution only.
            import os
            rubric_workers = int(os.environ.get('SHERPA_SCORE_RUBRIC_WORKERS', '1'))
            if rubric_workers > 1 and len(rubrics) > 1:
                from concurrent.futures import ThreadPoolExecutor
                from contextvars import copy_context
                with ThreadPoolExecutor(min(rubric_workers, len(rubrics))) as rubric_pool:
                    pending = [rubric_pool.submit(copy_context().run, state_applicability, case, n, r, scorer, cache) for r in rubrics]
                    review = {r['rubric_id']: future.result() for r, future in zip(rubrics, pending)}
            else:
                review = {r['rubric_id']: state_applicability(case, n, r, scorer, cache) for r in rubrics}
            on = [r for r in rubrics if review[r['rubric_id']]['applicable'] and
                  (matched is None or str(matched[r['rubric_id']]['value']).upper() == 'Y')]
            mask = tuple(r['rubric_id'] for r in on)
            # Opt-in content aliases let a source ID and bank ID share the same
            # judgment. Historical scorer/cache contracts remain byte-for-byte
            # unchanged. Request evidence keeps the canonical transmitted IDs.
            content_aliases = getattr(scorer, 'behavior_cache_contract', None) == 'public_behavior_content_alias_v1'
            aliases = {r['rubric_id']: 'criterion_' + sha256_json({k: v for k, v in r.items() if k != 'rubric_id'})
                       for r in on} if content_aliases else {}
            canonical_by_id = {aliases[r['rubric_id']]: {**r, 'rubric_id': aliases[r['rubric_id']]}
                               for r in on} if content_aliases else {}
            request_rubrics = [canonical_by_id[k] for k in sorted(canonical_by_id)] if content_aliases else on

            def restore_aliases(rows):
                if not content_aliases:
                    return rows
                _mapping(rows, canonical_by_id, 'canonical behavior criteria')
                return {rid: rows[cid] for rid, cid in aliases.items()}

            state_results = {}
            if not on:
                return group, matched, review, state_results

            def validate_behavior(node, rows):
                _mapping(rows, mask, f'{node.node_id} behavior')
                for rid in mask:
                    _score_record(node, rid, src[rid], True, rows[rid])
                return rows

            for start in range(0, len(group), batch_size):
                part = group[start:start + batch_size]
                first = part[0]; prefix = to_dialogue(first.prefix)
                responses = [{'response_id': node.node_id, 'reply': node.response} for node in part]
                identity = {'state_sha256': sha256_json([view, prefix, (first.trace or {}).get('pre_response_belief')]),
                            'policy': first.policy_sha256, 'rubrics': sha256_json(request_rubrics),
                            'responses': [{'id': node.node_id, 'reply_sha256': sha256_json(node.response)} for node in part],
                            'model': getattr(scorer, 'model', None), 'protocol': getattr(scorer, 'protocol', None),
                            'batch_size': batch_size, 'score_record_contract': SCORE_RECORD_CONTRACT}
                if content_aliases:
                    identity['behavior_cache_contract'] = scorer.behavior_cache_contract
                if callable(batch_fn):
                    def validate_batch(rows):
                        _mapping(rows, [node.node_id for node in part], 'behavior response batch')
                        for node in part:
                            validate_behavior(node, restore_aliases(rows[node.node_id]))
                        return rows
                    rows = validate_batch(cache.call('rubric_behavior_batch', identity,
                        lambda: validate_batch(batch_fn(view, prefix, request_rubrics, responses))))
                    state_results.update({nid: restore_aliases(values) for nid, values in rows.items()})
                else:
                    for node in part:
                        single_identity = {**identity, 'response_id': node.node_id}
                        def validate_single(rows):
                            validate_behavior(node, restore_aliases(rows))
                            return rows
                        rows = cache.call('rubric_behavior', single_identity,
                            lambda: validate_single(scorer.score_behavior(view, to_dialogue(node.prefix), request_rubrics,
                                                                                 node.response, matched)))
                        state_results[node.node_id] = validate_behavior(node, restore_aliases(rows))
            return group, matched, review, state_results

        items = list(state_groups.items())
        if workers > 1 and len(items) > 1:
            from concurrent.futures import ThreadPoolExecutor
            from contextvars import copy_context
            with ThreadPoolExecutor(min(workers, len(items))) as pool:
                # Preserve caller case attribution in the API usage ledger.
                futures = [pool.submit(copy_context().run, score_state, item) for item in items]
                results = [future.result() for future in futures]
        else:
            results = [score_state(item) for item in items]
        # Workers keep request state local; assembly and reward masking happen
        # only after every state succeeded, in the original response order.
        reviewed, result_by_node = {}, {}
        for group, matched, review, rows in results:
            for node in group:
                reviewed[node.node_id] = (matched, review)
            result_by_node.update(rows)
        for n in nodes:
            matched, review = reviewed[n.node_id]
            behavior = result_by_node.get(n.node_id, {})
            for rid in sorted(src):
                applicable = bool(review[rid].get('applicable')) and (matched is None or
                    str(matched[rid].get('value', 'Y')).upper() == 'Y')
                r = behavior.get(rid) if applicable else None
                record = _score_record(n, rid, src[rid], applicable, r)
                if review[rid].get('applicable') is None:        # no verdict (abstain or execution-unresolved): no score, reward or penalty
                    record = replace(record, scoring_status='execution_unresolved' if review[rid].get('execution_unresolved')
                                     else 'applicability_unresolved')
                if record.scoring_status == 'execution_unresolved' and review[rid].get('applicable') is not None:
                    from src.online_rl.tree.applicability import log_execution_unresolved
                    log_execution_unresolved(cache, stage='behavior_scoring', case_id=case.get('case_id'), node_id=n.node_id, rubric_id=rid,
                                             attempts=(r or {}).get('total_attempts'), reasons=(r or {}).get('reasons_by_attempt'))
                out.append(replace(record, scoring_evidence={**(record.scoring_evidence or {}), 'applicability': review[rid]}))
        return _mask_unscorable(out, view, nodes)

    def one(n):
        def validate_result(value):
            _mapping(value, src, f'{n.node_id} rubric scores', metadata=('_decisions',))
            for rid in src:
                row = value[rid]
                if not isinstance(row, dict) or not isinstance(row.get('applicable'), bool) or row.get('error'):
                    raise ScoringContractError(f'{n.node_id}/{rid}: invalid applicability or scoring error')
                _score_record(n, rid, src[rid], row['applicable'], row)
            return value
        return validate_result(cache.call('rubric_score', {'node': n.node_id, 'policy': n.policy_sha256, 'rubrics': sha256_json(rubrics), 'model': getattr(scorer, 'model', None),
                                           'protocol': getattr(scorer, 'protocol', None), 'score_record_contract': SCORE_RECORD_CONTRACT,
                                           'state_sha256': _state_key(view, n), 'reply_sha256': sha256_json(n.response)},
                          lambda: validate_result(scorer(view, to_dialogue(n.prefix), rubrics, n.response))))
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as pool: results = list(pool.map(one, nodes))
    else: results = [one(n) for n in nodes]
    for n, res in zip(nodes, results):
        for rid in sorted(src):
            r = res[rid]
            out.append(_score_record(n, rid, src[rid], r['applicable'], r))
    return _mask_unscorable(out, view, nodes)


def scores_by_response(scores: list[ResponseRubricScore]) -> dict[str, list[ResponseRubricScore]]:
    out: dict[str, list[ResponseRubricScore]] = {}
    for s in scores: out.setdefault(s.response_node_id, []).append(s)
    return out


def score_cumulative_nodes(case: dict, nodes: list[AgentNode], snapshot: dict, scorer: Callable,
                           cache: CallCache, *, top_k: int = 8, workers: int = 1,
                           scoring_scope: str = 'top_k', prefetched_scores=None):
    """Share applicability across sibling replies for the configured scoring set.

    With scope ``all``, every bank criterion receives an applicability check;
    ``top_k`` preserves legacy prefix retrieval. In either mode only applicable
    criteria receive behavior scores. Legacy omitted criteria remain explicitly
    unevaluated (`not_retrieved`, score=None), never behavior zero.
    """
    import json
    from src.online_rl.tree import cumulative_bank as CB
    if scoring_scope not in ('top_k', 'all'):
        raise ScoringContractError('Cumulative scoring_scope must be top_k or all')
    CB.validate_snapshot(snapshot)
    rubrics = CB.scoring_rubrics(snapshot)
    prefetched = {}
    for raw in prefetched_scores or ():
        row = raw if isinstance(raw, ResponseRubricScore) else ResponseRubricScore(**raw)
        key = (row.response_node_id, row.rubric_id)
        if key in prefetched:
            raise ScoringContractError('Prefetched score contains a duplicate response/rubric pair')
        prefetched[key] = row
    valid_pairs = {(node.node_id, rubric.rubric_id) for node in nodes for rubric in rubrics}
    if not set(prefetched) <= valid_pairs:
        raise ScoringContractError('Prefetched score contains a response or rubric outside the final bank')
    if len({node.node_id for node in nodes}) != len(nodes):
        raise ScoringContractError('Duplicate response node IDs')
    by_id = {rubric.rubric_id: rubric for rubric in rubrics}
    view = case_view(case)
    groups = {}
    for node in nodes:
        groups.setdefault((_state_key(view, node), node.policy_sha256), []).append(node)

    def score_state(item):
        (state_sha, policy_sha), group = item
        first = group[0]
        # Neither current reply, later dialogue nor outcome can influence retrieval.
        query = json.dumps({'case': view, 'prefix': to_dialogue(first.prefix)},
                           sort_keys=True, ensure_ascii=False)
        shortlist = (snapshot['entries'] if scoring_scope == 'all'
                     else CB.shortlist(snapshot, query, top_k=top_k))
        selected_ids = [entry['bank_id'] for entry in shortlist]
        if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids) <= set(by_id):
            raise ScoringContractError('Cumulative retrieval returned duplicate or unknown bank criteria')
        selected = [by_id[rid] for rid in selected_ids]
        missing = [rubric for rubric in selected
                   if any((node.node_id, rubric.rubric_id) not in prefetched for node in group)]
        fresh = score_nodes(case, group, missing, scorer, cache, workers=1)
        fresh_by_pair = {(row.response_node_id, row.rubric_id): row for row in fresh}
        rows = []
        for node in group:
            for rubric in selected:
                key = (node.node_id, rubric.rubric_id)
                row = prefetched.get(key, fresh_by_pair.get(key))
                if row is None:
                    raise ScoringContractError(f'Missing cumulative score for {key[0]}/{key[1]}')
                rows.append(row)
        for node in group:
            for rid in sorted(set(by_id) - set(selected_ids)):
                rows.append(ResponseRubricScore(node.node_id, rid, by_id[rid].leaf_id, False, None, None, None,
                    'not_retrieved', {'retrieval': {'evaluated': False, 'snapshot_sha256': snapshot['sha256']}}))
        retrieval = {'state_sha256': state_sha, 'policy_sha256': policy_sha,
                     'response_ids': [node.node_id for node in group], 'snapshot_sha256': snapshot['sha256'],
                     'query_sha256': sha256_json(query), 'retrieval_ids': selected_ids,
                     'retrieval_content_sha256': {entry['bank_id']: entry['content_sha256'] for entry in shortlist},
                     'bank_size': len(rubrics), 'top_k': top_k}
        if scoring_scope == 'all':
            retrieval.update(scoring_scope='all', candidate_selection='full_cumulative_bank', top_k=None)
        return rows, retrieval

    items = list(groups.items())
    if workers > 1 and len(items) > 1:
        from concurrent.futures import ThreadPoolExecutor
        from contextvars import copy_context
        with ThreadPoolExecutor(min(workers, len(items))) as pool:
            futures = [pool.submit(copy_context().run, score_state, item) for item in items]
            results = [future.result() for future in futures]
    else:
        results = [score_state(item) for item in items]
    rows = [row for scores, _ in results for row in scores]
    node_order = {node.node_id: i for i, node in enumerate(nodes)}
    rows.sort(key=lambda row: (node_order[row.response_node_id], row.rubric_id))
    return rows, [retrieval for _, retrieval in results]
