"""Rubric-Generator prompt search on stored trajectories under one frozen rerun policy. A prompt P writes one rubric per
trajectory, verified as in training (K reruns per target) to give s(tau). J(P) = S_success(P) * S_failure(P), the mean
s(tau) over source-success and source-failure trajectories, ranks candidates; acceptance is a stratified paired bootstrap
on dJ over held-out trajectories. The branching skeleton, prompt store and guards are those of
src/online_rl/integrated."""
from __future__ import annotations
import random
from dataclasses import dataclass
from src.online_rl.guards import assert_not_protected
from src.online_rl.integrated.prompt_search import PromptStore, SearchBackend, _propose, _quota, split_of
from src.online_rl.integrated.records import PromptVersion, sha256_json
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.effectiveness import evaluate_rubric
from src.online_rl.tree.posthoc import generate_rubric
from src.online_rl.tree.rubric_bank import MARKER
from src.online_rl.tree.records import AgentNode, Leaf, TrajectoryRubricInput, to_dialogue


@dataclass(frozen=True)
class EvidenceUnit:
    case_id: str
    leaf: Leaf
    path_nodes: tuple[AgentNode, ...]
    tin: TrajectoryRubricInput
    policy_sha256: str


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
    assign_fn: object = None              # MATCH-vs-NEW decision (rubric.reuse): per-case same-reason reuse as in training
    source_consistency: bool = False      # v2.1: new rubrics pass source-self-applicability at their locus, as in training


def score_prompt(text: str, sha: str, units: list[EvidenceUnit], cases: dict, d: ScoringDeps) -> dict[str, dict]:
    """Per trajectory: the ONE rubric this prompt gives it and its verification (None if the rubric is invalid). With
    rubric.reuse the trajectories of each case go through the same per-case, per-stratum REUSE-vs-NEW procedure as in
    training (in leaf-id order) and every member keeps its own verification, so J stays trajectory-level."""
    if d.assign_fn is not None: return _score_prompt_reuse(text, sha, units, cases, d)
    out = {}
    for u in units:
        rub = generate_rubric(text, sha, u.tin, d.generate_fn, d.cache, d.gate_fn)
        _, eff = evaluate_rubric(cases[u.case_id], {n.node_id: n for n in u.path_nodes}, u.leaf, rub, scorer=d.scorer, writer=d.writer,
                                 continuation=d.continuation, reruns=d.reruns, run_seed=d.run_seed, policy_sha256=d.policy_sha256,
                                 strategies=d.strategies, cache=d.cache)
        out[u.leaf.leaf_id] = {'rubric': rub, 'effectiveness': eff}
    return out


def _score_prompt_reuse(text: str, sha: str, units: list[EvidenceUnit], cases: dict, d: ScoringDeps) -> dict[str, dict]:
    import dataclasses
    from src.online_rl.tree import rubric_bank as RB
    out = {}
    for cid in sorted({u.case_id for u in units}):
        us = [u for u in units if u.case_id == cid]
        nodes = {n.node_id: n for u in us for n in u.path_nodes}; leaves = {u.leaf.leaf_id: u.leaf for u in us}
        ssf = None
        if d.source_consistency:                                       # same admission as training (v2.1); J itself is unchanged
            from src.online_rl.tree.witness import source_stage1_for
            tin_of = {u.leaf.leaf_id: u.tin for u in us}
            ssf = lambda leaf, cid=cid: source_stage1_for(cases[cid], nodes, leaf, tin_of[leaf.leaf_id].target_reply_id, d.scorer.applicability, d.cache)
        canon, assignments = RB.build_banks(cases[cid], list(leaves.values()), {u.leaf.leaf_id: u.tin for u in us}, text, sha, d.generate_fn,
                                            d.assign_fn, d.gate_fn, d.cache, source_stage1_for=ssf)
        _, member = RB.verify_members(cases[cid], nodes, leaves, canon, assignments, scorer=d.scorer, writer=d.writer, continuation=d.continuation,
                                      reruns=d.reruns, run_seed=d.run_seed, policy_sha256=d.policy_sha256, strategies=d.strategies, cache=d.cache)
        members = {}
        for a in assignments: members.setdefault(a.canonical_rubric_id, []).append(a.leaf_id)
        for a in assignments:
            out[a.leaf_id] = {'rubric': dataclasses.replace(canon[a.canonical_rubric_id], leaf_id=a.leaf_id), 'effectiveness': member[a.leaf_id],
                              'assignment': a, 'members': members[a.canonical_rubric_id]}
    return out


def unit_values(results: dict[str, dict], strategies: Strategies) -> dict[str, float]:
    """leaf_id -> s(tau) of the trajectory's rubric under this prompt (0 when the rubric is invalid or matches nothing)."""
    value = strategies.need('prompt_unit_value')
    return {leaf_id: v for leaf_id, r in results.items() if (v := value(r)) is not None}


def strata(units: list[EvidenceUnit]) -> dict[str, int]:
    return {'success': sum(u.leaf.outcome == 1.0 for u in units), 'failure': sum(u.leaf.outcome == 0.0 for u in units)}


def directional_scores(units: list[EvidenceUnit], results: dict[str, dict], strategies: Strategies) -> dict:
    """S_success / S_failure = mean s(tau) over source-success / source-failure trajectories, J = their product. A missing
    stratum leaves its mean and J undefined (None); it is never imputed."""
    vals = unit_values(results, strategies); out = {}
    for name, o in (('success', 1.0), ('failure', 0.0)):
        side = [vals[u.leaf.leaf_id] for u in units if u.leaf.outcome == o]
        out[f'n_{name}'] = len(side); out[f'S_{name}'] = sum(side) / len(side) if side else None
    effs = [results[u.leaf.leaf_id]['effectiveness'] for u in units]
    out['J'] = None if None in (out['S_success'], out['S_failure']) else out['S_success'] * out['S_failure']
    out['invalid'] = sum(e is None for e in effs); out['no_match'] = sum(e is not None and e.status == 'no_match' for e in effs)
    out['per_trajectory'] = {u.leaf.leaf_id: vals[u.leaf.leaf_id] for u in units}
    out['source_outcome'] = {u.leaf.leaf_id: u.leaf.outcome for u in units}
    return out


def _J(pairs: dict[float, list[tuple[float, float]]], side: int) -> float:
    return sum(p[side] for p in pairs[1.0]) / len(pairs[1.0]) * sum(p[side] for p in pairs[0.0]) / len(pairs[0.0])


def stratified_paired_bootstrap_J(parent: dict, cand: dict, acc: dict) -> dict:
    """Each replicate resamples the source-success and the source-failure (parent, candidate) pairs separately with
    replacement and recomputes J for both prompts; p_gain = P(dJ > 0). Accept iff both strata reach their minimum count,
    the observed dJ >= delta and p_gain >= p_min."""
    if set(parent['per_trajectory']) != set(cand['per_trajectory']) or parent['source_outcome'] != cand['source_outcome']:
        raise AssertionError('parent and candidate must be scored on the same trajectory mask (same valid-feedback units)')
    pairs = {o: [(parent['per_trajectory'][k], cand['per_trajectory'][k]) for k in sorted(parent['per_trajectory'])
                 if parent['source_outcome'][k] == o] for o in (1.0, 0.0)}
    n_s, n_f = len(pairs[1.0]), len(pairs[0.0])
    if not n_s or not n_f: return {'accepted': False, 'checks': {}, 'reason': 'missing outcome stratum', 'n_success': n_s, 'n_failure': n_f}
    observed = _J(pairs, 1) - _J(pairs, 0); rng = random.Random(acc['bootstrap_seed']); wins = 0
    for _ in range(acc['bootstrap_n']):
        rep = {o: rng.choices(pairs[o], k=len(pairs[o])) for o in (1.0, 0.0)}
        wins += _J(rep, 1) - _J(rep, 0) > 0
    p_gain = wins / acc['bootstrap_n']
    checks = {'minimum_success_trajectories': n_s >= acc['minimum_success_trajectories'],
              'minimum_failure_trajectories': n_f >= acc['minimum_failure_trajectories'], 'delta': observed >= acc['delta'],
              'bootstrap': p_gain >= acc['p_min']}
    return {'accepted': all(checks.values()), 'checks': checks, 'observed_dJ': observed, 'p_gain': p_gain, 'n_success': n_s, 'n_failure': n_f,
            'bootstrap_n': acc['bootstrap_n'], 'bootstrap_seed': acc['bootstrap_seed'],
            'reason': 'accepted' if all(checks.values()) else 'failed: ' + ', '.join(k for k, v in checks.items() if not v)}


ACCEPTANCE_RULES: dict = {'stratified_paired_bootstrap_J': stratified_paired_bootstrap_J}   # name -> fn(parent, cand, acceptance_cfg)


def j_acceptance(parent: dict, cand: dict, acc: dict) -> dict:
    """Parent and candidate scored on the same acceptance trajectories; dJ = J(candidate) - J(parent) is reported and the
    configured rule decides."""
    if set(parent['per_trajectory']) != set(cand['per_trajectory']) or parent['source_outcome'] != cand['source_outcome']:
        raise RuntimeError('parent and candidate were not scored on the same trajectories')
    rule = ACCEPTANCE_RULES.get(acc['rule'])
    if rule is None: raise ValueError(f"acceptance.rule {acc['rule']!r} is not defined")
    dJ = None if None in (parent['J'], cand['J']) else cand['J'] - parent['J']
    return {'dJ': dJ, **rule(parent, cand, acc)}


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
    """Per-target provenance already recorded in the verification: one entry per matched source response."""
    if e is None: return []
    return [{'response_node_id': t, 'agent_turn': next(r.turn for r in e.runs if r.node_id == t), 'level': e.level,
             'rerun_outcomes': [r.outcome for r in e.runs if r.node_id == t], 'per_node_R_O': e.per_node_R_O[t]} for t in e.target_node_ids]


def reflection_view(units: list[EvidenceUnit], results: dict[str, dict]) -> list[dict]:
    """What the prompt reflection sees: trajectory (without the Doctor FINAL turn), outcome, Doctor feedback, its one
    rubric with structural and gate failure reasons (rubric text withheld when the gate found answer content), and the
    per-target intervention reruns behind s(tau) (field R_O)."""
    rows = []
    for u in units:
        r = results[u.leaf.leaf_id]; c, e = r['rubric'], r['effectiveness']
        rows.append({'trajectory': u.leaf.leaf_id, 'outcome': u.tin.outcome, 'feedback': _t(u.tin.feedback, 800),
                     'dialogue': [{'speaker': d['speaker'], 'text': _t(d['text'], 400)} for d in to_dialogue(u.tin.dialogue) if d['stage'] != 'FINAL'],
                     'rubric': None if _answer_exposing(c.gate) else {k: (c.rubric or {}).get(k) for k in ('name', 'applicability_condition', 'required_behavior', 'scores')},
                     'valid': c.valid, 'problems': list(c.problems) + _gate_problems(c.gate),
                     'verification': e.status if e else 'not_verified', 'interventions': _interventions(e), 'R_O': e.R_O if e else None,
                     **_lineage(r)})
    return rows


def _lineage(r: dict) -> dict:
    """Same-reason reuse lineage for the reflection (only with rubric.reuse): assignment, rubric id, its members, rationale."""
    a = r.get('assignment')
    if a is None: return {}
    return {'assignment': a.assignment, 'canonical_rubric_id': a.canonical_rubric_id, 'members': r.get('members', []),
            'assignment_rationale': _t((a.decision or {}).get('rationale') or '', 300), 'assignment_note': _t(a.note, 200)}


def cached_backend(backend: SearchBackend, cache: CallCache) -> SearchBackend:
    """Reflection and refinement are API calls: cache them (role = cache kind, plus the model id) so a resumed event repeats
    no completed call."""
    fb = lambda parent, view, n: cache.call('reflect', {'parent': sha256_json(parent), 'view': sha256_json(view), 'n': n,
                                                        'model': getattr(backend.feedback, 'model', None)}, lambda: backend.feedback(parent, view, n))
    rf = lambda parent, diagnosis, direction, history, note: cache.call(
        'refine', {'parent': sha256_json(parent), 'diagnosis': diagnosis, 'direction': direction, 'history': sha256_json(history), 'note': note,
                   'model': getattr(backend.refine, 'model', None)},
        lambda: backend.refine(parent, diagnosis, direction, history, note))
    return SearchBackend(fb, rf, backend.generate, backend.gate, backend.score)


def split_units(units: list[EvidenceUnit], run_seed: int, fraction: float) -> tuple[list, list]:
    split = {u.leaf.leaf_id: split_of(run_seed, u.leaf.leaf_id, fraction) for u in units}
    return [u for u in units if split[u.leaf.leaf_id] == 'search'], [u for u in units if split[u.leaf.leaf_id] == 'acceptance']


def ready(units: list[EvidenceUnit], run_seed: int, fraction: float) -> bool:
    """J needs source-success and source-failure trajectories in both the search and the acceptance split."""
    return all(min(strata(part).values()) > 0 for part in split_units(units, run_seed, fraction))


def run_event(event: int, parent_id: str, parent_sha: str, units: list[EvidenceUnit], cases: dict, store: PromptStore, backend: SearchBackend,
              deps: ScoringDeps, cfg: dict, run_seed: int, case_cursor: int, history: list[dict]) -> dict:
    pu = cfg['prompt_update']
    if not ready(units, run_seed, pu['acceptance_fraction']):
        raise RuntimeError('prompt event needs source-success and source-failure trajectories in both splits')
    deps.strategies.need('prompt_unit_value')
    backend = cached_backend(backend, deps.cache)
    assert_not_protected([cases[u.case_id] for u in units], 'tree prompt-search evidence')
    search, accept_units = split_units(units, run_seed, pu['acceptance_fraction'])
    parent_text = store.get(parent_sha); allowed = store.texts(); case_list = [cases[c] for c in sorted({u.case_id for u in units})]
    quality = lambda res: directional_scores(search, res, deps.strategies)
    parent_search = score_prompt(parent_text, parent_sha, search, cases, deps)
    cands, frontier, reflections = [], [(parent_id, parent_text, parent_search)], []
    for depth in range(1, pu['refinement_depth'] + 1):
        level = []
        for (pid, ptext, presults), n in zip(frontier, _quota(pu['candidate_count'], len(frontier))):
            if n == 0: continue
            prop = _propose(ptext, reflection_view(search, presults), n, history, backend, allowed, case_list)
            reflections.append({'depth': depth, 'parent_prompt_id': pid, 'reflection': prop['reflection'], 'errors': prop['errors']})
            for p in prop['proposals']:
                c = {'parent_prompt_id': pid, 'depth': depth, 'direction': p['direction'], 'reflection': prop['reflection'], 'violations': p['violations']}
                if p['text'] is None:
                    cands.append({**c, 'prompt_id': None, 'sha': None, 'status': 'rejected', 'reason': 'prompt guard violation after retry'}); continue
                if deps.assign_fn is not None and MARKER not in p['text']:
                    cands.append({**c, 'prompt_id': None, 'sha': None, 'status': 'rejected', 'reason': f'missing {MARKER} section'}); continue
                sha = store.put(p['text']); allowed.append(p['text'])
                if sha == parent_sha or any(x['sha'] == sha for x in cands):
                    cands.append({**c, 'prompt_id': None, 'sha': sha, 'status': 'rejected', 'reason': 'duplicate prompt text'}); continue
                res = score_prompt(p['text'], sha, search, cases, deps)
                c.update(prompt_id=f'E{event}-d{depth}-{sha[:12]}', sha=sha, text=p['text'], results=res, search=quality(res), status='pruned', reason=None)
                cands.append(c); level.append(c)
        level.sort(key=lambda c: (-c['search']['J'], c['sha']))
        for rank, c in enumerate(level): c['rank'] = rank
        frontier = [(c['prompt_id'], c['text'], c['results']) for c in level[:pu['retain_top_k']]]
        if depth < pu['refinement_depth']:
            for c in level[:pu['retain_top_k']]: c['status'] = 'retained'
        if not frontier: break
    scored = sorted((c for c in cands if 'results' in c), key=lambda c: (-c['search']['J'], c['sha']))
    decision = {'accepted': False, 'reason': 'no scorable candidate', 'candidate_prompt_id': None}
    if scored:
        best = scored[0]
        parent_acc = score_prompt(parent_text, parent_sha, accept_units, cases, deps)
        best_acc = score_prompt(best['text'], best['sha'], accept_units, cases, deps)
        pa, ca = (directional_scores(accept_units, r, deps.strategies) for r in (parent_acc, best_acc))
        acc = {**j_acceptance(pa, ca, cfg['acceptance']), 'parent': pa, 'candidate': ca}
        best.update(status='accepted' if acc['accepted'] else 'rejected', acceptance=acc, reason=None if acc['accepted'] else acc.get('reason', 'rejected'))
        decision = {'accepted': bool(acc['accepted']), 'candidate_prompt_id': best['prompt_id'], 'candidate_sha256': best['sha'], 'acceptance': acc,
                    'reason': best['reason'] or 'accepted'}
    keys = (tuple(u.leaf.leaf_id for u in search), tuple(u.leaf.leaf_id for u in accept_units))
    versions = [PromptVersion(c['prompt_id'] or f'E{event}-d{c["depth"]}-rejected{i}', c['parent_prompt_id'], c['sha'] or '', event, c['depth'], c.get('rank'),
                              c['status'], 'refiner', c['reflection'], c['direction'], keys[0], keys[1],
                              {'search': c.get('search'), 'prompt_sha256': c['sha'], 'guard_violations': c['violations']}, c.get('acceptance'), c['reason'])
                for i, c in enumerate(cands)]
    return {'event': event, 'case_cursor': case_cursor, 'parent_prompt_id': parent_id, 'parent_sha256': parent_sha,
            'search_leaf_ids': list(keys[0]), 'acceptance_leaf_ids': list(keys[1]),
            'parent_search': quality(parent_search), 'reflections': reflections, 'versions': versions, 'decision': decision,
            'reruns_per_intervention': deps.reruns}
