"""Rubric-Generator prompt search on completed trajectories. A prompt p writes ONE rubric per stored trajectory tau_i; it is
matched, intervened on and rerun three times. Fitness is the product of success-source and failure-source mean R_O
ranks the candidates; source-stratified paired bootstrap tests the product gain. The branching skeleton, prompt store,
lineage records and guards are those of src/online_rl/integrated."""
from __future__ import annotations
from dataclasses import dataclass
import random
from src.online_rl.guards import assert_not_protected
from src.online_rl.integrated.prompt_search import PromptStore, SearchBackend, _propose, _quota
from src.online_rl.integrated.records import PromptVersion, sha256_json, atomic_write_json, to_dict
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.effectiveness import evaluate_rubric
from src.online_rl.tree.posthoc import generate_rubric
from src.online_rl.tree.prompt_contract import prompt_violations
from src.online_rl.tree.records import AgentNode, Leaf, TrajectoryRubricInput, to_dialogue


@dataclass(frozen=True)
class EvidenceUnit:
    case_id: str
    leaf: Leaf
    path_nodes: tuple[AgentNode, ...]
    tin: TrajectoryRubricInput
    policy_sha256: str


EVIDENCE_ID_VERSION = 'tree_evidence_v2'


def evidence_id(unit: EvidenceUnit) -> str:
    """Versioned evaluation identity; tree node/leaf IDs themselves remain unchanged.
    Include policy and actual source content, not just a path reusable across runs.
    """
    return EVIDENCE_ID_VERSION + ':' + sha256_json({
        'case_id': unit.case_id, 'leaf_id': unit.leaf.leaf_id,
        'policy_sha256': unit.policy_sha256, 'input_sha256': unit.tin.sha256,
        'dialogue': unit.leaf.dialogue, 'outcome': unit.leaf.outcome,
        'path': unit.leaf.path,
        'path_nodes': [{'node_id': n.node_id, 'policy_sha256': n.policy_sha256,
                        'prefix': n.prefix, 'response': n.response, 'seed': n.seed}
                       for n in unit.path_nodes],
    })


def validate_units(units: list[EvidenceUnit], rerun_policy_sha256: str | None = None) -> None:
    """Fail before API calls on duplicate identities or inconsistent source records.
    A ScoringDeps currently binds one live policy; do not pretend its callbacks
    can rerun another policy version merely by changing a recorded hash.
    """
    seen = set()
    for u in units:
        if u.case_id != u.leaf.case_id or u.case_id != u.tin.case_id or u.leaf.leaf_id != u.tin.leaf_id:
            raise ValueError('Evidence case/leaf/input IDs disagree')
        if tuple(n.node_id for n in u.path_nodes) != u.leaf.path:
            raise ValueError('Evidence nodes must follow the exact source leaf path')
        if u.tin.agent_node_ids != u.leaf.path or u.tin.dialogue != u.leaf.dialogue:
            raise ValueError('Evidence generator input must match the source dialogue and Agent-node mapping')
        if any(n.case_id != u.case_id or n.policy_sha256 != u.policy_sha256 for n in u.path_nodes):
            raise ValueError('Evidence source nodes have inconsistent case/policy versions')
        if rerun_policy_sha256 is not None and u.policy_sha256 != rerun_policy_sha256:
            raise ValueError('ScoringDeps binds a different live policy: partition evidence by policy version and load the matching checkpoint before scoring')
        uid = evidence_id(u)
        if uid in seen:
            raise ValueError(f'Duplicate evidence unit: {uid}; deduplicate before search')
        seen.add(uid)


@dataclass(frozen=True)
class ScoringDeps:
    reruns: int                           # K: reruns per intervention target
    run_seed: int
    generate_fn: object
    gate_fn: object
    scorer: object
    writer: object
    continuation: object
    policy_sha256: str                    # policy that writes interventions and runs continuations
    strategies: Strategies
    cache: CallCache


def score_prompt(text: str, sha: str, units: list[EvidenceUnit], cases: dict, d: ScoringDeps) -> dict[str, dict]:
    """Per trajectory: the ONE rubric this prompt writes and its verification (None if the rubric is invalid)."""
    # Source records stay immutable; the event's callbacks bind the frozen
    # current policy for every parent/candidate comparison.
    validate_units(units)
    out = {}
    for u in units:
        rub = generate_rubric(text, sha, u.tin, d.generate_fn, d.cache, d.gate_fn)
        source_scores, eff = evaluate_rubric(cases[u.case_id], {n.node_id: n for n in u.path_nodes}, u.leaf, rub, scorer=d.scorer, writer=d.writer,
                                 continuation=d.continuation, reruns=d.reruns, run_seed=d.run_seed, policy_sha256=d.policy_sha256,
                                 strategies=d.strategies, cache=d.cache, rerun_policy_mode='current')
        out[evidence_id(u)] = {'rubric': rub, 'effectiveness': eff,
                               'leaf_id': u.leaf.leaf_id, 'case_id': u.case_id,
                               'source_policy_sha256': u.policy_sha256, 'input_sha256': u.tin.sha256, 'source_outcome': u.tin.outcome}
        if d.cache.path is not None:
            owner = getattr(d.writer, '__self__', None)
            record = {'evidence_id': evidence_id(u), 'prompt_sha256': sha,
                      'policy_sha256': d.policy_sha256, 'source_policy_sha256': u.policy_sha256,
                      'case_id': u.case_id, 'leaf_id': u.leaf.leaf_id, 'source_outcome': u.tin.outcome,
                      'rubric': to_dict(rub), 'effectiveness': to_dict(eff) if eff else None,
                      'source_response_scores': [to_dict(score) for score in source_scores],
                      'intervention_instruction': owner.instruction(rub.rubric, eff.level)
                          if owner is not None and hasattr(owner, 'instruction') and eff else None}
            atomic_write_json(d.cache.path.parent / 'evaluations' / sha / sha256_json(d.policy_sha256)
                              / (sha256_json(evidence_id(u)) + '.json'), record)
    return out


def unit_values(results: dict[str, dict], strategies: Strategies) -> dict[str, float]:
    """Versioned evidence_id -> verification value under this prompt."""
    value = strategies.need('prompt_unit_value')
    return {uid: v for uid, r in results.items()
            if not (r['effectiveness'] is not None and r['effectiveness'].status == 'incomplete')
            and (v := value(r)) is not None}


def objective(values: dict, outcomes: dict, minimum_per_source: int = 1) -> dict:
    """Both source groups use all planned trajectories, including rejected rubrics."""
    groups = {name: [key for key, outcome in outcomes.items() if outcome == name]
              for name in ('success', 'failure')}
    missing = sorted(set(outcomes) - set(values))
    ready = not missing and all(len(keys) >= minimum_per_source for keys in groups.values())
    means = {name: sum(values[k] for k in keys) / len(keys)
             if keys and all(k in values for k in keys) else None for name, keys in groups.items()}
    return {'S_success': means['success'], 'S_failure': means['failure'],
            'J': means['success'] * means['failure'] if ready else None,
            'counts': {name: len(keys) for name, keys in groups.items()},
            'status': 'complete' if ready else ('incomplete' if missing else 'insufficient_sources'),
            'missing_evidence_ids': missing}


def prompt_fitness(values: dict, outcomes: dict) -> float | None:
    return objective(values, outcomes)['J']


def stratified_split(units, run_seed, acceptance_fraction, minimum_per_source=1):
    if not 0 < acceptance_fraction < 1 or minimum_per_source < 1:
        raise ValueError('Invalid split fraction or minimum source count')
    split = {}
    for name in ('success', 'failure'):
        group = [u for u in units if u.tin.outcome == name]
        if len(group) < 2 * minimum_per_source:
            return None
        # Stable source identities keep repeated policy versions on the same side.
        group.sort(key=lambda u: sha256_json([run_seed, u.case_id, u.leaf.leaf_id]))
        n = min(len(group) - minimum_per_source,
                max(minimum_per_source, round(len(group) * acceptance_fraction)))
        split.update({evidence_id(u): 'acceptance' if i < n else 'search' for i, u in enumerate(group)})
    return split


def paired_acceptance(parent: dict, cand: dict, acc: dict, outcomes: dict) -> dict:
    minimum = acc.get('minimum_per_source', 1)
    p, c = objective(parent, outcomes, minimum), objective(cand, outcomes, minimum)
    if set(parent) != set(cand) or set(parent) != set(outcomes) or p['J'] is None or c['J'] is None:
        return {'accepted': False, 'checks': {'same_complete_evidence': False},
                'n': len(outcomes), 'parent_objective': p, 'candidate_objective': c,
                'reason': 'incomplete or insufficient paired source evidence', 'status': 'deferred'}
    groups = [[k for k in sorted(outcomes) if outcomes[k] == name] for name in ('success', 'failure')]
    rng = random.Random(acc['bootstrap_seed'])
    wins = 0
    draws = acc['bootstrap_n']
    if draws < 1:
        raise ValueError('bootstrap_n must be positive')
    for _ in range(draws):
        sampled = [[rng.choice(keys) for _ in keys] for keys in groups]
        # Identical sampled trajectories for parent and candidate, separately by source.
        products = []
        for values in (parent, cand):
            means = [sum(values[k] for k in keys) / len(keys) for keys in sampled]
            products.append(means[0] * means[1])
        wins += products[1] > products[0]
    gain = c['J'] - p['J']; p_gain = wins / draws
    checks = {'minimum_trajectories': len(outcomes) >= acc['minimum_trajectories'],
              'gain': gain > 0 and gain >= acc['delta'], 'bootstrap': p_gain >= acc['p_min']}
    return {'accepted': all(checks.values()), 'checks': checks, 'n': len(outcomes),
            'J_gain': gain, 'p_gain': p_gain, 'parent_objective': p, 'candidate_objective': c,
            'reason': 'accepted' if all(checks.values()) else 'failed: ' + ', '.join(k for k, v in checks.items() if not v)}


def _t(s: str, n: int) -> str: return s if len(s) <= n else s[:n] + ' …'


def _gate_problems(gate: dict | None) -> list[str]:
    """Failure reasons from the existing post-hoc gate result, with no gold distinction (gold_option_text and option_text_X
    are both reported as option text)."""
    if not gate or gate.get('passed'): return []
    out = [r for r in gate.get('reasons', []) if r.startswith('schema:')]
    leak = gate.get('leakage_deterministic') or []
    if 'mentions_option_letter' in leak: out.append('deterministic leakage: option letter')
    if any(x == 'gold_option_text' or x.startswith('option_text_') for x in leak): out.append('answer copy: option text')
    out += [k for k in ('answer_presupposition', 'new_domain_premise') if gate.get(k)]
    return out or ['post-hoc gate failed']


def _answer_exposing(gate: dict | None) -> bool:
    """Gate failures whose rubric text may carry answer content (option letters/texts, a presupposed answer)."""
    return bool(gate) and not gate.get('passed') and (bool(gate.get('leakage_deterministic')) or bool(gate.get('answer_presupposition')))


def _interventions(e) -> list[dict]:
    """Selected-target provenance; no-match targets can have no reruns."""
    if e is None: return []
    rows = []
    for target in e.target_node_ids:
        runs = [r for r in e.runs if r.node_id == target]
        rows.append({'response_node_id': target, 'agent_turn': runs[0].turn if runs else None,
                     'level': e.level, 'rerun_outcomes': [r.outcome for r in runs],
                     'rerun_statuses': [r.status for r in runs],
                     'runs': [{'agent_reply': r.intervention_text, 'final_option': r.final_option, 'doctor_final': r.final_text, 'outcome': r.outcome, 'status': r.status, 'error': r.error, 'seed': r.seed} for r in runs],
                     'per_node_R_O': e.per_node_R_O.get(target)})
    return rows


def reflection_view(units: list[EvidenceUnit], results: dict[str, dict]) -> list[dict]:
    """What reflection sees: full trajectory including Doctor FINAL, outcome, feedback, its one
    rubric with structural and gate failure reasons (rubric text withheld when the gate found answer content), and the
    per-target intervention reruns behind s(tau) (field R_O)."""
    rows = []
    for u in units:
        uid = evidence_id(u)
        r = results[uid]; c, e = r['rubric'], r['effectiveness']
        rows.append({'evidence_id': uid, 'source_policy_sha256': u.policy_sha256, 'input_sha256': u.tin.sha256,
                     'trajectory': u.leaf.leaf_id, 'outcome': u.tin.outcome, 'feedback': u.tin.feedback,
                     'dialogue': [{'speaker': d['speaker'], 'text': d['text']} for d in to_dialogue(u.tin.dialogue) if d['stage'] != 'FINAL'],
                     'rubric': None if _answer_exposing(c.gate) else (c.rubric or {}).get('candidate_rubric', c.rubric),
                     'agent_pre_response_states': list(u.tin.agent_pre_response_states), 'full_trajectory': to_dialogue(u.tin.dialogue), 'agent_turn_ids': list(u.tin.agent_node_ids),
                     'analysis': None if _answer_exposing(c.gate) else (c.rubric or {}).get('analysis'),
                     'intervention_behavior': None if _answer_exposing(c.gate) else ((c.rubric or {}).get('candidate_rubric', {}).get('scores', {}).get(e.level, {}) if e else None),
                     'selected_agent_turn_id': c.selected_agent_turn_id,
                     'valid': c.valid, 'problems': list(c.problems) + _gate_problems(c.gate),
                     'verification': e.status if e else 'not_verified', 'interventions': _interventions(e), 'R_O': e.R_O if e else None})
    values = {evidence_id(u): (results[evidence_id(u)]['effectiveness'].R_O if results[evidence_id(u)]['effectiveness'] is not None and results[evidence_id(u)]['effectiveness'].R_O is not None else 0.0) for u in units}
    missing = [evidence_id(u) for u in units if results[evidence_id(u)]['effectiveness'] is not None and results[evidence_id(u)]['effectiveness'].status == 'incomplete']
    for uid in missing:
        values.pop(uid, None)
    totals = objective(values, {evidence_id(u): u.tin.outcome for u in units})
    for row in rows:
        row['prompt_objective'] = totals
    return rows


def cached_backend(backend: SearchBackend, cache: CallCache) -> SearchBackend:
    """Reflection and refinement are Sol calls: cache them so a resumed event repeats no completed call."""
    def fb(parent, view, n):
        identity = {'parent': sha256_json(parent), 'view': sha256_json(view), 'n': n}
        if cache.path is not None:
            atomic_write_json(cache.path.parent / 'prompt_optimization' / 'feedback_inputs' / (sha256_json(identity) + '.json'),
                              {'current_prompt': parent, 'evidence': view, 'directions_requested': n, 'identity': identity})
        return cache.call('reflect', identity, lambda: backend.feedback(parent, view, n))
    rf = lambda parent, diagnosis, direction, history, note: cache.call(
        'refine', {'parent': sha256_json(parent), 'diagnosis': diagnosis, 'direction': direction, 'history': sha256_json(history), 'note': note},
        lambda: backend.refine(parent, diagnosis, direction, history, note))
    return SearchBackend(fb, rf, backend.generate, backend.gate, backend.score)


def run_event(event: int, parent_id: str, parent_sha: str, units: list[EvidenceUnit], cases: dict, store: PromptStore, backend: SearchBackend,
              deps: ScoringDeps, cfg: dict, run_seed: int, case_cursor: int, history: list[dict]) -> dict:
    validate_units(units)
    pu = cfg['prompt_update']; deps.strategies.need('prompt_unit_value')
    outcomes = {evidence_id(u): u.tin.outcome for u in units}
    summary = lambda res: objective(unit_values(res, deps.strategies), {uid: outcomes[uid] for uid in res}, pu.get('minimum_per_source', 1))
    quality = lambda res: summary(res)['J']
    if pu.get('objective', 'success_failure_product_v1') != 'success_failure_product_v1':
        raise ValueError('Unsupported prompt optimization objective')
    def diagnostic_view(selected, results):
        rows = reflection_view(selected, results)
        owner = getattr(deps.writer, '__self__', None)
        for row in rows:
            result = results[row['evidence_id']]
            effect = result['effectiveness']
            rubric = result['rubric']
            row['intervention_instruction'] = (owner.instruction(rubric.rubric, effect.level)
                if owner is not None and hasattr(owner, 'instruction') and effect is not None
                and not _answer_exposing(rubric.gate) else None)
        return rows
    def deferred(reason, results=None):
        return {'event': event, 'case_cursor': case_cursor, 'parent_prompt_id': parent_id, 'parent_sha256': parent_sha,
                'search_leaf_ids': [], 'acceptance_leaf_ids': [], 'versions': [], 'reflections': [],
                'objective_protocol': 'success_failure_product_v1',
                'evaluation_summary': summary(results) if results is not None else None,
                'decision': {'accepted': False, 'status': 'deferred', 'reason': reason}}
    backend = cached_backend(backend, deps.cache)
    assert_not_protected([cases[u.case_id] for u in units], 'tree prompt-search evidence')
    # Keep the original split assignment by source path; versioned IDs are lookup keys.
    # This keeps the same source path in the same split across policy versions.
    split = stratified_split(units, run_seed, pu['acceptance_fraction'], max(pu.get('minimum_per_source', 1), cfg['acceptance'].get('minimum_per_source', 1)))
    if split is None:
        return deferred('success and failure sources are required in both evaluation splits')
    search = [u for u in units if split[evidence_id(u)] == 'search']
    accept_units = [u for u in units if split[evidence_id(u)] == 'acceptance']
    parent_text = store.get(parent_sha); allowed = store.texts(); case_list = [cases[c] for c in sorted({u.case_id for u in units})]
    parent_search = score_prompt(parent_text, parent_sha, search, cases, deps)
    if quality(parent_search) is None:
        return deferred('parent evaluation has technical incompletes', parent_search)
    cands, frontier, reflections = [], [(parent_id, parent_text, parent_search)], []
    for depth in range(1, pu['refinement_depth'] + 1):
        level = []
        for (pid, ptext, presults), n in zip(frontier, _quota(pu['candidate_count'], len(frontier))):
            if n == 0: continue
            prop = _propose(ptext, diagnostic_view(search, presults), n, history, backend, allowed, case_list,
                            prompt_guard=prompt_violations)
            reflections.append({'depth': depth, 'parent_prompt_id': pid, 'reflection': prop['reflection'], 'errors': prop['errors']})
            for p in prop['proposals']:
                c = {'parent_prompt_id': pid, 'depth': depth, 'direction': p['direction'], 'reflection': prop['reflection'], 'violations': p['violations']}
                if p['text'] is None:
                    cands.append({**c, 'prompt_id': None, 'sha': None, 'status': 'rejected', 'reason': 'prompt guard violation after retry'}); continue
                sha = store.put(p['text']); allowed.append(p['text'])
                if sha == parent_sha or any(x['sha'] == sha for x in cands):
                    cands.append({**c, 'prompt_id': None, 'sha': sha, 'status': 'rejected', 'reason': 'duplicate prompt text'}); continue
                res = score_prompt(p['text'], sha, search, cases, deps)
                if quality(res) is None:
                    return deferred('candidate evaluation has technical incompletes; comparison postponed', res)
                c.update(prompt_id=f'E{event}-d{depth}-{sha[:12]}', sha=sha, text=p['text'], results=res, search_score=quality(res), objective=summary(res), status='pruned', reason=None)
                cands.append(c); level.append(c)
        level.sort(key=lambda c: (-c['search_score'], c['sha']))
        for rank, c in enumerate(level): c['rank'] = rank
        frontier = [(c['prompt_id'], c['text'], c['results']) for c in level[:pu['retain_top_k']]]
        if depth < pu['refinement_depth']:
            for c in level[:pu['retain_top_k']]: c['status'] = 'retained'
        if not frontier: break
    scored = sorted((c for c in cands if 'results' in c), key=lambda c: (-c['search_score'], c['sha']))
    decision = {'accepted': False, 'reason': 'no scorable candidate', 'candidate_prompt_id': None}
    if scored:
        best = scored[0]
        parent_acc = score_prompt(parent_text, parent_sha, accept_units, cases, deps)
        best_acc = score_prompt(best['text'], best['sha'], accept_units, cases, deps)
        acc = paired_acceptance(unit_values(parent_acc, deps.strategies), unit_values(best_acc, deps.strategies), cfg['acceptance'], {evidence_id(u): u.tin.outcome for u in accept_units})
        best.update(status='accepted' if acc['accepted'] else 'rejected', acceptance=acc, reason=None if acc['accepted'] else acc.get('reason', 'rejected'))
        decision = {'accepted': bool(acc['accepted']), 'candidate_prompt_id': best['prompt_id'], 'candidate_sha256': best['sha'], 'acceptance': acc,
                    'reason': best['reason'] or 'accepted', 'status': acc.get('status', 'complete')}
    keys = (tuple(evidence_id(u) for u in search), tuple(evidence_id(u) for u in accept_units))
    versions = [PromptVersion(c['prompt_id'] or f'E{event}-d{c["depth"]}-rejected{i}', c['parent_prompt_id'], c['sha'] or '', event, c['depth'], c.get('rank'),
                              c['status'], 'refiner', c['reflection'], c['direction'], keys[0], keys[1],
                              {'search_score': c.get('search_score'), 'objective': c.get('objective'), 'guard_violations': c['violations']}, c.get('acceptance'), c['reason'])
                for i, c in enumerate(cands)]
    return {'event': event, 'case_cursor': case_cursor, 'parent_prompt_id': parent_id, 'parent_sha256': parent_sha,
            'search_leaf_ids': [u.leaf.leaf_id for u in search], 'acceptance_leaf_ids': [u.leaf.leaf_id for u in accept_units],
            'search_evidence_ids': list(keys[0]), 'acceptance_evidence_ids': list(keys[1]),
            'evidence_id_version': EVIDENCE_ID_VERSION,
            'evidence_manifest': {evidence_id(u): {'case_id': u.case_id, 'leaf_id': u.leaf.leaf_id,
                                                 'policy_sha256': u.policy_sha256, 'input_sha256': u.tin.sha256} for u in units},
            'objective_protocol': 'success_failure_product_v1', 'parent_search_score': quality(parent_search), 'parent_search_objective': summary(parent_search),
            'reflections': reflections, 'versions': versions, 'decision': decision}
