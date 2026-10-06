"""Rubric-Generator prompt search on completed trajectories. A prompt p writes ONE rubric per stored trajectory tau_i; it is
matched, intervened on and rerun K times exactly as in training, giving s(tau_i). Prompt fitness S(p) = sum_i s(tau_i)
ranks the candidates; paired bootstrap acceptance on per-trajectory s(tau_i), the branching skeleton, prompt store,
lineage records and guards are those of src/online_rl/integrated."""
from __future__ import annotations
from dataclasses import dataclass
from src.online_rl.guards import assert_not_protected
from src.online_rl.integrated.prompt_search import PromptStore, SearchBackend, _propose, _quota, split_of
from src.online_rl.integrated.metrics import paired_bootstrap
from src.online_rl.integrated.records import PromptVersion, sha256_json
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.effectiveness import evaluate_rubric
from src.online_rl.tree.posthoc import generate_rubric
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


def score_prompt(text: str, sha: str, units: list[EvidenceUnit], cases: dict, d: ScoringDeps) -> dict[str, dict]:
    """Per trajectory: the ONE rubric this prompt writes and its verification (None if the rubric is invalid)."""
    out = {}
    for u in units:
        rub = generate_rubric(text, sha, u.tin, d.generate_fn, d.cache, d.gate_fn)
        _, eff = evaluate_rubric(cases[u.case_id], {n.node_id: n for n in u.path_nodes}, u.leaf, rub, scorer=d.scorer, writer=d.writer,
                                 continuation=d.continuation, reruns=d.reruns, run_seed=d.run_seed, policy_sha256=d.policy_sha256,
                                 strategies=d.strategies, cache=d.cache)
        out[u.leaf.leaf_id] = {'rubric': rub, 'effectiveness': eff}
    return out


def unit_values(results: dict[str, dict], strategies: Strategies) -> dict[str, float]:
    """leaf_id -> s(tau) of the trajectory's rubric under this prompt."""
    value = strategies.need('prompt_unit_value')
    return {leaf_id: v for leaf_id, r in results.items() if (v := value(r)) is not None}


def prompt_fitness(values: dict[str, float]) -> float:
    """S(p) = sum_i s(tau_i)."""
    return sum(values.values())


def paired_acceptance(parent: dict[str, float], cand: dict[str, float], acc: dict) -> dict:
    """Best-only acceptance: paired per-trajectory s(tau) differences with the bootstrap of src/online_rl/integrated/metrics."""
    keys = sorted(set(parent) & set(cand)); d = [cand[k] - parent[k] for k in keys]
    mean_d = sum(d) / len(d) if d else None; p_gain = paired_bootstrap(d, acc['bootstrap_n'], acc['bootstrap_seed'])
    checks = {'minimum_trajectories': len(d) >= acc['minimum_trajectories'], 'gain': mean_d is not None and mean_d >= acc['delta'],
              'bootstrap': p_gain >= acc['p_min']}
    return {'accepted': all(checks.values()), 'checks': checks, 'n': len(d), 'mean_paired_gain': mean_d, 'p_gain': p_gain,
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
                     'verification': e.status if e else 'not_verified', 'interventions': _interventions(e), 'R_O': e.R_O if e else None})
    return rows


def cached_backend(backend: SearchBackend, cache: CallCache) -> SearchBackend:
    """Reflection and refinement are Sol calls: cache them so a resumed event repeats no completed call."""
    fb = lambda parent, view, n: cache.call('reflect', {'parent': sha256_json(parent), 'view': sha256_json(view), 'n': n}, lambda: backend.feedback(parent, view, n))
    rf = lambda parent, diagnosis, direction, history, note: cache.call(
        'refine', {'parent': sha256_json(parent), 'diagnosis': diagnosis, 'direction': direction, 'history': sha256_json(history), 'note': note},
        lambda: backend.refine(parent, diagnosis, direction, history, note))
    return SearchBackend(fb, rf, backend.generate, backend.gate, backend.score)


def run_event(event: int, parent_id: str, parent_sha: str, units: list[EvidenceUnit], cases: dict, store: PromptStore, backend: SearchBackend,
              deps: ScoringDeps, cfg: dict, run_seed: int, case_cursor: int, history: list[dict]) -> dict:
    pu = cfg['prompt_update']; deps.strategies.need('prompt_unit_value')
    quality = lambda res: prompt_fitness(unit_values(res, deps.strategies))
    backend = cached_backend(backend, deps.cache)
    assert_not_protected([cases[u.case_id] for u in units], 'tree prompt-search evidence')
    split = {u.leaf.leaf_id: split_of(run_seed, u.leaf.leaf_id, pu['acceptance_fraction']) for u in units}
    search = [u for u in units if split[u.leaf.leaf_id] == 'search']; accept_units = [u for u in units if split[u.leaf.leaf_id] == 'acceptance']
    parent_text = store.get(parent_sha); allowed = store.texts(); case_list = [cases[c] for c in sorted({u.case_id for u in units})]
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
                sha = store.put(p['text']); allowed.append(p['text'])
                if sha == parent_sha or any(x['sha'] == sha for x in cands):
                    cands.append({**c, 'prompt_id': None, 'sha': sha, 'status': 'rejected', 'reason': 'duplicate prompt text'}); continue
                res = score_prompt(p['text'], sha, search, cases, deps)
                c.update(prompt_id=f'E{event}-d{depth}-{sha[:12]}', sha=sha, text=p['text'], results=res, search_score=quality(res), status='pruned', reason=None)
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
        acc = paired_acceptance(unit_values(parent_acc, deps.strategies), unit_values(best_acc, deps.strategies), cfg['acceptance'])
        best.update(status='accepted' if acc['accepted'] else 'rejected', acceptance=acc, reason=None if acc['accepted'] else acc.get('reason', 'rejected'))
        decision = {'accepted': bool(acc['accepted']), 'candidate_prompt_id': best['prompt_id'], 'candidate_sha256': best['sha'], 'acceptance': acc,
                    'reason': best['reason'] or 'accepted'}
    keys = (tuple(u.leaf.leaf_id for u in search), tuple(u.leaf.leaf_id for u in accept_units))
    versions = [PromptVersion(c['prompt_id'] or f'E{event}-d{c["depth"]}-rejected{i}', c['parent_prompt_id'], c['sha'] or '', event, c['depth'], c.get('rank'),
                              c['status'], 'refiner', c['reflection'], c['direction'], keys[0], keys[1],
                              {'search_score': c.get('search_score'), 'guard_violations': c['violations']}, c.get('acceptance'), c['reason'])
                for i, c in enumerate(cands)]
    return {'event': event, 'case_cursor': case_cursor, 'parent_prompt_id': parent_id, 'parent_sha256': parent_sha,
            'search_leaf_ids': list(keys[0]), 'acceptance_leaf_ids': list(keys[1]), 'parent_search_score': quality(parent_search),
            'reflections': reflections, 'versions': versions, 'decision': decision}
