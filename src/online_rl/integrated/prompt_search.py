"""GEPA-inspired reflective beam search over the Rubric-Generator prompt (top-k, bounded depth; no Pareto front or merge).

Candidates are scored on STORED training evidence: same states, same Agent replies, same R_O. Only rubric generation
and scoring change, so every comparison is paired and no Agent or Doctor call is made."""
from __future__ import annotations
import json, re, statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from src.online_rl.guards import assert_not_protected
from src.online_rl.integrated import metrics as M
from src.online_rl.integrated.continuation import unit_hash
from src.online_rl.integrated.records import PromptVersion, StateEvidence, sha256_text
from src.online_rl.integrated.rubric import VISIBLE, generate_rubric, score_responses
from src.prompt_opt.harness import norm
from src.prompt_opt.reflect import copy_violations, structural_problems


class PromptStore:
    """Content-addressed prompt texts; a sha never maps to different text."""

    def __init__(self, directory: Path): self.dir = Path(directory); self.dir.mkdir(parents=True, exist_ok=True)

    def put(self, text: str) -> str:
        sha = sha256_text(text); p = self.dir / f'{sha}.txt'
        if p.exists():
            if p.read_text() != text: raise RuntimeError(f'prompt store corrupted at {sha}')
        else:
            tmp = p.with_name(p.name + '.tmp'); tmp.write_text(text); tmp.replace(p)
        return sha

    def get(self, sha: str) -> str:
        text = (self.dir / f'{sha}.txt').read_text()
        if sha256_text(text) != sha: raise RuntimeError(f'prompt store text for {sha} does not match its hash')
        return text

    def texts(self) -> list[str]: return [self.get(p.stem) for p in sorted(self.dir.glob('*.txt'))]


@dataclass(frozen=True)
class SearchBackend:
    feedback: Callable          # (parent_text, view, n) -> {'diagnosis': str, 'directions': [str] * n}; ValueError if malformed
    refine: Callable            # (parent_text, diagnosis, direction, history, retry_note) -> new prompt text
    generate: Callable          # rubric generator, as in rubric.generate_rubric
    gate: Callable
    score: Callable


def split_of(run_seed: int, state_key: str, acceptance_fraction: float) -> str:
    """Deterministic and outcome-free; repeated realisations of one state always fall in the same split."""
    return 'acceptance' if unit_hash(run_seed, state_key, 'prompt_split') < acceptance_fraction else 'search'


def evidence_id(e: StateEvidence) -> str: return f'step{e.step:06d}'


def option_text_violations(text: str, allowed: list[str], options: list[str]) -> list[str]:
    """Exact option strings (>= 4 chars) newly introduced into the prompt; short drug names escape the 4-gram guard."""
    t, allowed_n, out = norm(text), [norm(a) for a in allowed], set()
    for o in options:
        n = norm(o)
        if len(n) < 4 or not any(c.isalpha() for c in n): continue
        pat = re.compile(r'(?<!\w)' + re.escape(n) + r'(?!\w)')
        if pat.search(t) and not any(pat.search(a) for a in allowed_n): out.add(o)
    return sorted(out)


def prompt_violations(text: str, allowed: list[str], cases: list[dict]) -> list[str]:
    case_texts = [c['question'] for c in cases] + [v for c in cases for v in c['options'].values()]
    options = [v for c in cases for v in c['options'].values()]
    copied = copy_violations(text, allowed, case_texts)
    return ([f'copied 4-grams: {copied[:10]}'] if copied else []) + structural_problems(text) + \
        [f'option text: {o}' for o in option_text_violations(text, allowed, options)]


def parse_reflection(raw: str, n: int) -> dict:
    m = re.search(r'\{.*\}', raw, re.S)
    if not m: raise ValueError('reflection is not a JSON object')
    obj = json.loads(m.group(0))
    dirs = obj.get('directions')
    if not isinstance(obj.get('diagnosis'), str) or not isinstance(dirs, list) or len(dirs) != n or not all(isinstance(d, str) and d.strip() for d in dirs):
        raise ValueError(f'reflection must hold a diagnosis and exactly {n} non-empty directions')
    return {'diagnosis': obj['diagnosis'], 'directions': dirs}


def _t(s: Optional[str], n: int) -> str:
    s = s or ''; return s if len(s) <= n else s[:n] + ' …'


def reflection_view(states: list[StateEvidence], results: dict[str, dict]) -> list[dict]:
    """What the feedback generator sees, worst-aligned first. Never the gold answer or option letter of the answer."""
    rows = []
    for e in sorted(states, key=lambda e: (results[evidence_id(e)]['Q'], evidence_id(e))):
        r = results[evidence_id(e)]; rub = r['rubric']
        rows.append({'state': evidence_id(e), 'agent_turn': e.turn, 'Q_cov': round(r['Q'], 4),
                     'dialogue_before_reply': [{'speaker': s, 'text': _t(t, 600)} for s, _, _, t in e.generator_input.history],
                     'rubric': {k: rub.rubric.get(k) for k in VISIBLE} if rub.rubric else None, 'rubric_failure': rub.failure,
                     'gate_reasons': (rub.gates or {}).get('reasons'), 'C': (rub.gates or {}).get('C'),
                     'replies': [{'reply': _t(b.response, 500), 'R_A': None if a is None else round(a, 3),
                                  'continuations_correct': None if b.R_O is None else f'{round(b.R_O * len(b.outcomes))}/{len(b.outcomes)}'}
                                 for b, a in zip(e.branches, r['R_A'])],
                     'diagnostics': {k: r['diagnostics'][k] for k in ('mean_R_A', 'std_R_A', 'corr', 'Q_conc', 'worst_endorsed')}})
    return rows


def score_prompt(text: str, prompt_id: str, sha: str, states: list[StateEvidence], cases: dict, backend: SearchBackend, cfg: dict,
                 step: int, stored: bool) -> dict[str, dict]:
    """Per state: Q_cov averaged over rubric samples, on the stored replies/outcomes. stored=True reuses the training rubric as sample 0."""
    out = {}
    eps_s, n_samples, attempts = cfg['metrics']['epsilon_score'], cfg['prompt_update']['rubric_samples_per_state'], cfg['rubric']['max_attempts']
    for e in states:
        replies = [b.response if b.R_O is not None else None for b in e.branches]; outcomes = [b.R_O for b in e.branches]
        samples = []
        for k in range(n_samples):
            if stored and k == 0:
                rub, R_A = e.rubric, [b.R_A for b in e.branches]; hard = [b.R_A_hard for b in e.branches]
            else:
                rub = generate_rubric(text, prompt_id, sha, e.generator_input, e.state_key, cases[e.case_id], backend.generate, backend.gate, attempts, step, sample=k)
                sc = score_responses(rub, e.generator_input, replies, backend.score)
                R_A = [None if s is None else s['R_A'] for s in sc]; hard = [None if s is None else s['R_A_hard'] for s in sc]
            samples.append((rub, R_A, hard, M.q_cov(R_A, outcomes, rub.valid, eps_s)))
        rub, R_A, hard, _ = samples[0]
        out[evidence_id(e)] = {'Q': st.mean(s[3] for s in samples), 'rubric': rub, 'R_A': R_A, 'gates': [s[0].gates for s in samples],
                               'diagnostics': M.state_diagnostics(R_A, hard, outcomes, eps_s), 'rubric_ids': [s[0].rubric_id for s in samples]}
    return out


def _summary(results: dict[str, dict]) -> dict:
    return {**M.prompt_summary([r['Q'] for r in results.values()], [bool(r['diagnostics']['flat']) for r in results.values()]),
            **M.gate_rates([g for r in results.values() for g in r['gates']])}


def _propose(parent_text: str, view: list[dict], n: int, history: list[dict], backend: SearchBackend, allowed: list[str], cases: list[dict]) -> dict:
    """One reflection, then n refinements; each refinement gets one retry if it violates a prompt guard."""
    reflection, errors = None, []
    for _ in range(2):
        try:
            reflection = backend.feedback(parent_text, view, n); break
        except ValueError as e:
            errors.append(str(e))
    if reflection is None: return {'reflection': None, 'errors': errors, 'proposals': []}
    proposals = []
    for direction in reflection['directions']:
        note, attempts = None, []
        while True:
            text = backend.refine(parent_text, reflection['diagnosis'], direction, history, note)
            viol = prompt_violations(text, allowed, cases); attempts.append(viol)
            if not viol or len(attempts) == 2: break
            note = 'Your previous draft was rejected: ' + '; '.join(viol[:10]) + '. Return a corrected complete prompt.'
        proposals.append({'direction': direction, 'text': None if attempts[-1] else text, 'violations': attempts})
    return {'reflection': reflection['diagnosis'], 'errors': errors, 'proposals': proposals}


def _quota(total: int, parents: int) -> list[int]:
    return [total // parents + (i < total % parents) for i in range(parents)]


def run_event(event: int, parent_id: str, parent_sha: str, window: list[StateEvidence], cases: dict, store: PromptStore,
              backend: SearchBackend, cfg: dict, run_seed: int, step: int, history: list[dict]) -> dict:
    """Bounded search: depth d expands the top-k of depth d-1 with candidate_count children in total; only the best
    search-split candidate is ever evaluated on the acceptance split."""
    pu = cfg['prompt_update']
    assert_not_protected([cases[e.case_id] for e in window], 'prompt-search evidence')
    if any(e.prompt_sha256 != parent_sha for e in window): raise RuntimeError('prompt-event window mixes generator prompts')
    split = {evidence_id(e): split_of(run_seed, e.state_key, pu['acceptance_fraction']) for e in window}
    search = [e for e in window if e.informative and split[evidence_id(e)] == 'search']
    accept_states = [e for e in window if e.informative and split[evidence_id(e)] == 'acceptance']
    parent_text = store.get(parent_sha); allowed = store.texts(); case_list = [cases[c] for c in sorted({e.case_id for e in window})]
    parent_search = score_prompt(parent_text, parent_id, parent_sha, search, cases, backend, cfg, step, stored=True)
    common = {'event': event, 'search_state_keys': tuple(e.state_key for e in search), 'acceptance_state_keys': tuple(e.state_key for e in accept_states)}
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
                if any(x['sha'] == sha for x in cands) or sha == parent_sha:
                    cands.append({**c, 'prompt_id': None, 'sha': sha, 'status': 'rejected', 'reason': 'duplicate prompt text'}); continue
                cid = f'E{event}-d{depth}-{sha[:12]}'
                res = score_prompt(p['text'], cid, sha, search, cases, backend, cfg, step, stored=False)
                c.update(prompt_id=cid, sha=sha, text=p['text'], results=res, search_Q=st.mean(r['Q'] for r in res.values()) if res else 0.0, status='pruned', reason=None)
                cands.append(c); level.append(c)
        level.sort(key=lambda c: (-c['search_Q'], c['sha']))
        for rank, c in enumerate(level): c['rank'] = rank
        frontier = [(c['prompt_id'], c['text'], c['results']) for c in level[:pu['retain_top_k']]]
        if depth < pu['refinement_depth']:
            for c in level[:pu['retain_top_k']]: c['status'] = 'retained'
        if not frontier: break
    scored = sorted((c for c in cands if 'results' in c), key=lambda c: (-c['search_Q'], c['sha']))
    decision = {'accepted': False, 'reason': 'no scorable candidate (reflection failure, guard violation or duplicate)', 'candidate_prompt_id': None}
    if scored:
        best = scored[0]
        parent_acc = score_prompt(parent_text, parent_id, parent_sha, accept_states, cases, backend, cfg, step, stored=True)
        best_acc = score_prompt(best['text'], best['prompt_id'], best['sha'], accept_states, cases, backend, cfg, step, stored=False)
        acc = M.accept({k: r['Q'] for k, r in parent_acc.items()}, {k: r['Q'] for k, r in best_acc.items()},
                       M.gate_rates([g for r in parent_acc.values() for g in r['gates']]), M.gate_rates([g for r in best_acc.values() for g in r['gates']]),
                       {**cfg['acceptance'], 'minimum_informative_states': pu['minimum_informative_states']})
        best.update(status='accepted' if acc['accepted'] else 'rejected', acceptance=acc,
                    reason=None if acc['accepted'] else 'failed: ' + ', '.join(k for k, v in acc['checks'].items() if not v),
                    acceptance_summary=_summary(best_acc))
        decision = {'accepted': acc['accepted'], 'candidate_prompt_id': best['prompt_id'], 'candidate_sha256': best['sha'], 'acceptance': acc,
                    'reason': best['reason'] or 'accepted', 'parent_acceptance_summary': _summary(parent_acc), 'candidate_acceptance_summary': _summary(best_acc)}
    versions = [PromptVersion(c['prompt_id'] or f'E{event}-d{c["depth"]}-rejected{i}', c['parent_prompt_id'], c['sha'] or '', event, c['depth'], c.get('rank'),
                              c['status'], 'refiner', c['reflection'], c['direction'], common['search_state_keys'], common['acceptance_state_keys'],
                              {'search': _summary(c['results']) if 'results' in c else None, 'acceptance': c.get('acceptance_summary'), 'guard_violations': c['violations']},
                              c.get('acceptance'), c['reason']) for i, c in enumerate(cands)]
    return {**common, 'step': step, 'parent_prompt_id': parent_id, 'parent_sha256': parent_sha,
            'search_evidence_ids': [evidence_id(e) for e in search], 'acceptance_evidence_ids': [evidence_id(e) for e in accept_states],
            'parent_search_summary': _summary(parent_search), 'reflections': reflections, 'versions': versions, 'decision': decision}


def sol_backend(key: str, feedback_system: str, refiner_system: str, generate: Callable, gate: Callable, score: Callable) -> SearchBackend:
    """Real reflection/refinement with frozen Sol (lazy: needs API credentials)."""
    from src.prompt_opt import harness as H

    def feedback(parent, view, n):
        user = (f'CURRENT RUBRIC-GENERATOR PROMPT:\n{parent}\n\nNUMBER OF DIRECTIONS: {n}\n\n'
                f'TRAINING EVIDENCE (JSON):\n{json.dumps(view, ensure_ascii=False)}')
        return parse_reflection(H.doctor_call(key, H.SOL, feedback_system, user, 3000, 0.4), n)

    def refine(parent, diagnosis, direction, history, note):
        user = (f'CURRENT PROMPT:\n{parent}\n\nDIAGNOSTIC REFLECTION:\n{diagnosis}\n\nDIRECTION TO IMPLEMENT:\n{direction}\n\n'
                f'EARLIER PROMPT-UPDATE DECISIONS (JSON):\n{json.dumps(history, ensure_ascii=False)}' + (f'\n\n{note}' if note else ''))
        return re.sub(r'^```(?:\w+)?\s*|\s*```$', '', H.doctor_call(key, H.SOL, refiner_system, user, 3000, 0.4).strip()).strip()
    return SearchBackend(feedback, refine, generate, gate, score)
