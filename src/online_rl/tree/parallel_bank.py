"""Independent per-trajectory candidates, then one immutable case bank.

Consolidation may partition candidates, never write or synthesize rubric text.
The lexicographically smallest candidate ID in each cluster is the canonical
creator (and verification source); no trial outcome participates in that choice.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.posthoc import generate_rubric
from src.online_rl.tree.records import TrajectoryRubric, RubricAssignment
from src.online_rl.tree.rubric_bank import _member_spec

VERSION = 'parallel_candidates_consolidate_v1'
CHECKS = ('same_need', 'same_applicability', 'same_required_behavior',
          'same_score_semantics', 'public_evidence_clear', 'no_new_premise')
ROOT = Path(__file__).resolve().parents[3]
CONSOLIDATE_REPAIR_ATTEMPTS = 2
_validation_audit_lock = threading.Lock()

def validation_audit(record):
    path = os.environ.get('SHERPA_CONSOLIDATE_AUDIT')
    if path:
        with _validation_audit_lock:
            target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def membership_report(result, candidates):
    """Exact ID accounting for model correction; never changes membership."""
    ids = {c['candidate_id']: c for c in candidates}
    locations = {}
    malformed = []
    def note(value, location):
        if not isinstance(value, str):
            malformed.append({'location': location, 'value': value})
            return
        locations.setdefault(value, []).append(location)
    if isinstance(result, dict):
        clusters = result.get('clusters')
        if isinstance(clusters, list):
            for i, cluster in enumerate(clusters):
                if isinstance(cluster, dict) and isinstance(cluster.get('candidate_ids'), list):
                    for j, cid in enumerate(cluster['candidate_ids']): note(cid, f'clusters[{i}].candidate_ids[{j}]')
        excluded = result.get('excluded')
        if isinstance(excluded, list):
            for i, item in enumerate(excluded):
                if isinstance(item, dict): note(item.get('candidate_id'), f'excluded[{i}].candidate_id')
    return {'expected_count': len(ids), 'returned_occurrences': sum(map(len, locations.values())),
        'unique_known_count': len(set(locations) & set(ids)),
        'missing_ids': sorted(set(ids) - set(locations)),
        'duplicate_ids': {cid: locations[cid] for cid in sorted(locations) if len(locations[cid]) > 1},
        'unknown_ids': {cid: locations[cid] for cid in sorted(set(locations) - set(ids))},
        'invalid_memberships': malformed}


class PartitionCoverageError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('Consolidate must partition all candidates exactly once: '
            f"expected={report['expected_count']}, returned={report['returned_occurrences']}, "
            f"missing={len(report['missing_ids'])}, duplicate_ids={len(report['duplicate_ids'])}, "
            f"unknown_ids={len(report['unknown_ids'])}. "
            'Missing IDs: ' + json.dumps(report['missing_ids']) +
            '; duplicate locations: ' + json.dumps(report['duplicate_ids'], sort_keys=True))


def validate_partition(result, candidates):
    """No missing, invented, duplicated, cross-stratum or uncertain membership."""
    if not isinstance(result, dict) or set(result) != {'clusters', 'excluded'}:
        raise ValueError('Consolidate must return clusters and excluded')
    if not isinstance(result['clusters'], list) or not isinstance(result['excluded'], list):
        raise ValueError('Consolidate clusters/excluded must be arrays')
    ids = {c['candidate_id']: c for c in candidates}
    seen = []
    for cluster in result['clusters']:
        if not isinstance(cluster, dict) or set(cluster) != {'candidate_ids', 'checks', 'reason'}:
            raise ValueError('Consolidate cluster cannot rewrite a rubric')
        members = cluster['candidate_ids']
        if not isinstance(members, list) or not members or any(not isinstance(m, str) or m not in ids for m in members):
            raise ValueError('Unknown/empty consolidate cluster')
        if len({ids[m]['stratum'] for m in members}) != 1:
            raise ValueError('Consolidate cannot merge success and failure strata')
        if not isinstance(cluster['checks'], dict) or set(cluster['checks']) != set(CHECKS) or any(cluster['checks'][k] is not True for k in CHECKS):
            raise ValueError('Uncertain equivalence/evidence must be split or explicitly excluded')
        if not isinstance(cluster['reason'], str) or not cluster['reason'].strip():
            raise ValueError('Consolidate needs an evidence-grounded reason')
        seen.extend(members)
    for item in result['excluded']:
        if not isinstance(item, dict) or set(item) != {'candidate_id', 'reason'} or item['candidate_id'] not in ids or not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError('Every consolidate exclusion needs an original ID and reason')
        seen.append(item['candidate_id'])
    if sorted(seen) != sorted(ids):
        raise PartitionCoverageError(membership_report(result, candidates))
    return result


def api_consolidator(key, cfg):
    from src.online_rl.tree.backends import _openrouter_json
    import hashlib
    text = (ROOT / cfg['prompt_file']).read_text()
    if hashlib.sha256(text.encode()).hexdigest() != cfg['prompt_sha256']:
        raise ValueError('Consolidate prompt hash mismatch')
    settings = {k: cfg[k] for k in ('max_tokens', 'temperature', 'reasoning_effort', 'strict_provider_parameters')}
    if 'api_provider' in cfg: settings['api_provider'] = cfg['api_provider']
    # Successful input/output and cache identity stay unchanged. Only invalid
    # output takes the bounded repair path; never split/drop candidates in code.
    def consolidate(candidates):
        inventory = sorted([{'candidate_id': c['candidate_id'], 'stratum': c['stratum']} for c in candidates], key=lambda c: c['candidate_id'])
        if len({c['candidate_id'] for c in inventory}) != len(inventory):
            raise ValueError('Duplicate input candidate IDs; cannot consolidate')
        original = json.dumps(candidates, ensure_ascii=False, sort_keys=True)
        user, attempts = original, []
        for attempt in range(CONSOLIDATE_REPAIR_ATTEMPTS + 1):
            raw, meta = _openrouter_json(key, cfg['model'], text, user,
                                         settings, {}, 'case_consolidation', 'rubric_consolidate')
            error = None; result = None
            try:
                result = json.loads(raw)
                validate_partition(result, candidates)
            except (ValueError, TypeError, KeyError) as exc: error = str(exc)
            coverage = membership_report(result, candidates) if result is not None else None
            record = {'protocol': 'consolidate_validation_repair_v1',
                      'time': datetime.now(timezone.utc).isoformat(),
                      'input_sha256': sha256_json(candidates), 'attempt': attempt + 1,
                      'model': cfg['model'], 'api_provider': settings.get('api_provider', 'openrouter'),
                      'request_sha256': sha256_json(user), 'raw': raw, 'validation_error': error,
                      'accepted': error is None, 'repair_attempts_limit': CONSOLIDATE_REPAIR_ATTEMPTS,
                      'membership_report': coverage, 'usage': meta.get('usage'),
                      'returned_model': meta.get('model'), 'candidate_count': len(candidates)}
            attempts.append(record)
            validation_audit(record)
            if error is None: return result
            # Fixed originals, source IDs and strata remain authoritative. No
            # reward/verification outcome is used to choose a replacement source.
            user = json.dumps({'original_candidates': candidates, 'previous_response': raw,
                'validation_error': error, 'membership_report': coverage,
                'candidate_inventory': inventory,
                'instruction': 'Correct the JSON partition against the SAME original candidates. '
                    'Return the COMPLETE clusters and excluded, with every original candidate exactly once; not a partial patch. '
                    'Use candidate_inventory as the authoritative checklist. membership_report gives exact missing IDs '
                    'and duplicate locations in your previous response. Evaluate each missing candidate against its '
                    'original evidence and existing clusters; do not automatically exclude it or assume singleton checks pass. '
                    'For duplicates choose membership based on original equivalence/evidence, never just the first occurrence. '
                    'Before returning, check that the union equals candidate_inventory and all IDs have count one. '
                    'Never merge success and failure strata. A mixed cluster must be reconsidered separately '
                    'within each stratum; merge only when every equivalence check is actually supported. '
                    'Keep distinct criteria separate; singletons are allowed. Do not invent IDs, rewrite criteria, '
                    'change source/anchor/stratum, or exclude a candidate merely to repair JSON. '
                    'Exclusion still requires an actual evidence/anchor problem with an explicit reason.'},
                ensure_ascii=False, sort_keys=True)
        failure = RuntimeError('Invalid consolidate JSON/partition after two repair requests; '
                               'stop and resume, never silently drop candidates: ' + error)
        failure.audit = {'protocol': 'consolidate_validation_repair_v1', 'attempts': attempts,
                         'input_sha256': sha256_json(candidates)}
        raise failure
    consolidate.model = cfg['model']
    consolidate.settings = {**settings, 'prompt_sha256': cfg['prompt_sha256'], 'version': VERSION}
    return consolidate


def build_bank(case, leaves, tins, prompt, prompt_sha, generate, consolidate, gate, cache,
               *, chunk_size=8, source_stage1_for=None):
    if type(chunk_size) is not int or chunk_size < 1: raise ValueError('candidate_batch_size must be positive')
    ordered = sorted((l for l in leaves if l.outcome in (0., 1.)), key=lambda l: l.leaf_id)
    identity = {'version': VERSION, 'case': case['case_id'], 'prompt': prompt_sha,
                'sources': [[l.leaf_id, tins[l.leaf_id].sha256] for l in ordered],
                'generator': [getattr(generate, 'model', None), getattr(generate, 'settings', None)],
                'consolidate': [getattr(consolidate, 'model', None), getattr(consolidate, 'settings', None)],
                'gate': [getattr(gate, 'model', None), getattr(gate, 'settings', None), getattr(gate, 'protocol', None)],
                'candidate_batch_size': chunk_size}

    def finalize():
        # Only the leader of this case's cache key reaches here. Candidate tasks
        # write their own immutable call records, never a shared canonical bank.
        generated = []
        def candidate(leaf):
            return generate_rubric(prompt, prompt_sha, tins[leaf.leaf_id], generate, cache)
        with ThreadPoolExecutor(chunk_size) as pool:
            for offset in range(0, len(ordered), chunk_size):
                tasks = [pool.submit(copy_context().run, candidate, leaf) for leaf in ordered[offset:offset + chunk_size]]
                generated.extend(f.result() for f in tasks)
        candidates, exclusions = [], []
        originals = {r.rubric_id: r for r in generated}
        for leaf, rub in zip(ordered, generated):
            if not rub.valid:
                exclusions.append({'candidate_id': rub.rubric_id, 'leaf_id': leaf.leaf_id,
                                   'reason': list(rub.problems), 'stage': 'candidate_validation'})
                continue
            tin = tins[leaf.leaf_id]
            candidates.append({'candidate_id': rub.rubric_id, 'stratum': 'success' if leaf.outcome else 'failure',
                               'source_trajectory': leaf.leaf_id, 'rubric': rub.rubric,
                               'anchor': rub.rubric['source_provenance']['analysis']['selected_agent_turn_id'],
                               **({'prospective_needs': list(tin.private_needs)} if tin.private_needs else {}),
                               'intervention_instruction': rub.intervention_instruction})
        candidates.sort(key=lambda c: c['candidate_id'])
        if len(originals) != len(generated):
            raise RuntimeError('Candidate ID collision; cannot establish unique source mapping')
        snapshot = cache.call('consolidate_input', identity, lambda: candidates)
        def decide():
            return validate_partition(consolidate(snapshot), snapshot) if snapshot else {'clusters': [], 'excluded': []}
        partition = cache.call('case_consolidation', {**identity, 'input': sha256_json(snapshot)}, decide)
        validate_partition(partition, snapshot)
        exclusions.extend({**x, 'stage': 'consolidate'} for x in partition['excluded'])
        # Persist bank and preselected source BEFORE gate / verification. Keep
        # exact original rubric and all original provenance, even rejected items.
        clusters = sorted((sorted(c['candidate_ids']) for c in partition['clusters']), key=lambda ids: ids[0])
        bank_plan = {'state': 'canonical_bank_finalized', 'source_selection': 'smallest_candidate_sha256_id_v1',
                     'clusters': [{'canonical_id': ids[0], 'candidate_ids': ids,
                                   'source': originals[ids[0]].leaf_id,
                                   'anchor': originals[ids[0]].rubric['source_provenance']['analysis']['selected_agent_turn_id']}
                                  for ids in clusters],
                     'original_candidates': [asdict(r) for r in generated],
                     'candidate_provenance': snapshot, 'exclusions': exclusions}
        bank_plan = cache.call('canonical_bank_plan', {**identity, 'partition': sha256_json(partition)}, lambda: bank_plan)
        leaf_of = {l.leaf_id: l for l in ordered}
        canon, assigned = {}, []
        mapped = {}
        for ids in clusters:
            original = originals[ids[0]]
            leaf = leaf_of[original.leaf_id]
            # Cached generation; only canonical candidates get source admission
            # and gate. generator_v1 rejects unclear anchors without rewriting.
            rub = generate_rubric(prompt, prompt_sha, tins[leaf.leaf_id], generate, cache, gate,
                                  source_stage1=source_stage1_for(leaf) if source_stage1_for else None)
            canon[rub.rubric_id] = rub
            for cid in ids: mapped[originals[cid].leaf_id] = rub
        for order, leaf in enumerate(ordered):
            own = generated[order]
            rub = mapped.get(leaf.leaf_id)
            if rub is None or not rub.valid:
                # Excluded candidates are audit records, never gate/verification jobs.
                from dataclasses import replace
                if rub is None:
                    rub = replace(own, valid=False, problems=(*own.problems, 'excluded_before_gate: see canonical bank plan'))
                    canon[rub.rubric_id] = rub
                kind, decision = 'CREATE_REJECTED', None
            elif leaf.leaf_id == rub.leaf_id:
                kind, decision = 'CREATED_NEW', None
            else:
                kind = 'REUSED_EXISTING'
                decision = {'selected_agent_turn_id': own.rubric['source_provenance']['analysis']['selected_agent_turn_id'],
                            'intervention_instruction': own.intervention_instruction, 'candidate_id': own.rubric_id}
            assigned.append(RubricAssignment(leaf.leaf_id, case['case_id'], 'success' if leaf.outcome else 'failure', order,
                rub.rubric_id, kind, decision, '', sha256_json(tins[leaf.leaf_id].feedback),
                _member_spec(rub, tins[leaf.leaf_id], kind, decision)))
        report = {'version': VERSION, 'candidate_batch_size': chunk_size,
                  'candidate_chunks': (len(ordered) + chunk_size - 1) // chunk_size,
                  'candidate_tasks': len(ordered), 'generated_candidates': sum(r.rubric is not None for r in generated),
                  'consolidate_candidates': len(candidates), 'excluded_candidates': len(exclusions),
                  'candidate_generation_requests': sum(r.raw is not None for r in generated),
                  'canonical_rubrics': len(clusters), 'duplicates_removed': sum(len(ids) - 1 for ids in clusters),
                  'consolidate_requests': int(bool(snapshot)), 'bank_plan': bank_plan,
                  'candidate_request_note': 'one logical generation per eligible trajectory; physical attempts/repairs in http_attempts.jsonl'}
        return {'canon': {k: asdict(v) for k, v in canon.items()},
                'assignments': [asdict(a) for a in assigned], 'report': report}
    result = cache.call('case_bank_complete', identity, finalize)
    return ({k: TrajectoryRubric(**v) for k, v in result['canon'].items()},
            [RubricAssignment(**a) for a in result['assignments']], result['report'])
