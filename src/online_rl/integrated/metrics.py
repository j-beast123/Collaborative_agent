"""Pure reward, rubric-quality and prompt-acceptance math."""
from __future__ import annotations
import math, random, statistics as st
from typing import Optional


def r_a(probabilities: dict) -> float:
    """Expected Jev adherence (p1 + 2 p2) / (2 (p0 + p1 + p2)) in [0, 1]."""
    p = {k: float(probabilities[k]) for k in ('0', '1', '2')}
    total = sum(p.values())
    if total <= 0: raise ValueError(f'adherence probabilities do not sum to a positive value: {probabilities}')
    return (p['1'] + 2 * p['2']) / (2 * total)


def hard_value(label) -> float:
    if int(label) not in (0, 1, 2): raise ValueError(f'rubric label must be 0, 1 or 2, got {label!r}')
    return int(label) / 2


def r_o(outcomes: list[Optional[float]]) -> Optional[float]:
    """Fraction of successful continuations; None when any continuation has no parseable final answer."""
    if not outcomes: raise ValueError('R_O needs at least one continuation')
    vals = [o for o in outcomes if o is not None]
    return sum(vals) / len(vals) if len(vals) == len(outcomes) else None


def r_e(R_O: Optional[float], V_h: Optional[float]) -> Optional[float]:
    return None if R_O is None or V_h is None else R_O - V_h


def _pvar(x: list[float]) -> float:
    return st.pvariance(x) if len(x) > 1 else 0.0


def informative(outcomes: list[Optional[float]], eps_O: float) -> bool:
    """Depends only on R_O, so no prompt or rubric can drop a state from the Q_cov denominator."""
    v = [x for x in outcomes if x is not None]
    return len(v) >= 2 and _pvar(v) > eps_O


def q_cov(scores: list[Optional[float]], outcomes: list[Optional[float]], rubric_valid: bool, eps_score: float) -> float:
    """Population Cov(R_A, R_O) over responses with an outcome; 0 for an invalid/masked or flat rubric."""
    with_outcome = [(s, v) for s, v in zip(scores, outcomes) if v is not None]
    if not rubric_valid or len(with_outcome) < 2: return 0.0
    pairs = [(s, v) for s, v in with_outcome if s is not None]
    if len(pairs) != len(with_outcome): raise RuntimeError('valid rubric with an unscored response')
    s, v = [p[0] for p in pairs], [p[1] for p in pairs]
    if math.sqrt(_pvar(s)) < eps_score: return 0.0
    ms, mv = st.mean(s), st.mean(v)
    return sum((a - ms) * (b - mv) for a, b in pairs) / len(pairs)


def reward(R_O: Optional[float], R_A: Optional[float], w_O: float, w_A: float, rubric_valid: bool) -> Optional[float]:
    """Use only active reward components; a missing rubric is not a behavioral zero.

    Mixed reward retains its existing outcome-only fallback for an inapplicable
    rubric. Rubric-only needs a valid score, but never needs a Doctor outcome;
    outcome-only never needs a rubric. R_E is not an additional reward term.
    """
    if w_O != 0 and R_O is None: return None
    outcome_part = w_O * R_O if w_O != 0 else 0.0
    if w_A == 0: return outcome_part
    if not rubric_valid: return outcome_part if w_O != 0 else None
    if R_A is None: raise RuntimeError('valid rubric with an unscored response')
    return outcome_part + w_A * R_A


def advantages(rewards: list[Optional[float]], mode: str) -> list[Optional[float]]:
    if mode not in ('mean_center', 'group_zscore'): raise ValueError(f'unknown advantage mode {mode}')
    v = [r for r in rewards if r is not None]
    if any(not math.isfinite(r) for r in v): raise ValueError('rewards must be finite numbers or None')
    if len(v) < 2: return [None] * len(rewards)
    m, sd = st.mean(v), math.sqrt(_pvar(v))
    if mode == 'mean_center': return [None if r is None else r - m for r in rewards]
    if mode == 'group_zscore': return [None if r is None else ((r - m) / (sd + 1e-8) if sd > 0 else 0.0) for r in rewards]
    raise ValueError(f'unknown advantage mode {mode}')


def state_diagnostics(scores: list[Optional[float]], hard: list[Optional[float]], outcomes: list[Optional[float]], eps_score: float) -> dict:
    """Diagnostics only: never used for training or prompt acceptance."""
    pairs = [(s, v) for s, v in zip(scores, outcomes) if s is not None and v is not None]
    s, v = [p[0] for p in pairs], [p[1] for p in pairs]
    out = {'mean_R_A': st.mean(s) if s else None, 'std_R_A': math.sqrt(_pvar(s)) if s else None,
           'hard_histogram': {k: sum(h == k / 2 for h in hard if h is not None) for k in (0, 1, 2)},
           'flat': bool(s) and math.sqrt(_pvar(s)) < eps_score, 'Q_conc': None, 'corr': None, 'worst_endorsed': None}
    if len(pairs) < 2 or _pvar(v) == 0: return out
    if _pvar(s) > 0: out['corr'] = st.correlation(s, v)
    num = den = 0.0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            dv, ds = v[i] - v[j], s[i] - s[j]
            if dv == 0: continue
            c = 0.5 if ds == 0 else float((ds > 0) == (dv > 0))
            num += abs(dv) * c; den += abs(dv)
    out['Q_conc'] = num / den
    out['worst_endorsed'] = any(s[i] >= max(s) for i in range(len(v)) if v[i] == min(v))
    return out


def effect_diagnostics(scores: list[Optional[float]], effects: list[Optional[float]], high_score: float, severe_effect: float) -> dict:
    pairs = [(s, e) for s, e in zip(scores, effects) if s is not None and e is not None]
    high = [e for s, e in pairs if s >= high_score]; severe = [s for s, e in pairs if e <= -severe_effect]
    return {'R_E': [e for e in effects], 'high_score_negative_effect_rate': sum(e < 0 for e in high) / len(high) if high else None,
            'severe_harm_endorsement_rate': sum(s >= high_score for s in severe) / len(severe) if severe else None}


def lower_tail(values: list[float], q: float = 0.1) -> Optional[float]:
    if not values: return None
    v = sorted(values); return v[min(len(v) - 1, int(q * len(v)))]


def gate_rates(gates: list[Optional[dict]]) -> dict:
    """Gate rates and mean C (over valid rubrics), one rubric per state."""
    g = [x or {'schema': False, 'passed': False} for x in gates]
    sv = [x for x in g if x.get('schema')]
    frac = lambda xs: sum(xs) / len(xs) if xs else None
    C = [x['C'] for x in g if x.get('passed') and x.get('C') is not None]
    return {'n': len(g), 'validity_rate': frac([bool(x.get('passed')) for x in g]),
            'leakage_rate': frac([bool(x.get('leakage_deterministic')) or bool(x.get('answer_presupposition')) for x in sv]),
            'new_premise_rate': frac([bool(x.get('new_domain_premise')) for x in sv]),
            'observable_rate': frac([bool(x.get('observable')) for x in sv]),
            'C_mean': st.mean(C) if C else None}


def paired_bootstrap(diffs: list[float], n: int, seed: int) -> float:
    """P(mean paired difference > 0) under a state bootstrap."""
    if not diffs: return 0.0
    rng = random.Random(seed)
    return sum(st.mean(rng.choices(diffs, k=len(diffs))) > 0 for _ in range(n)) / n


def accept(parent_q: dict[str, float], cand_q: dict[str, float], parent_gates: dict, cand_gates: dict, cfg: dict) -> dict:
    """Enough informative states, paired mean Q_cov gain with bootstrap confidence, and no essential-gate regression."""
    keys = sorted(set(parent_q) & set(cand_q))
    if set(parent_q) != set(cand_q): raise RuntimeError('parent and candidate were not scored on the same acceptance states')
    d = [cand_q[k] - parent_q[k] for k in keys]
    mean_d = st.mean(d) if d else None; p_gain = paired_bootstrap(d, cfg['bootstrap_n'], cfg['bootstrap_seed'])
    rate = lambda m, k: m[k] if m[k] is not None else 0.0
    checks = {'min_informative': len(keys) >= cfg['minimum_informative_states'],
              'Q_gain': mean_d is not None and mean_d >= cfg['delta_Q'],
              'bootstrap': p_gain >= cfg['p_min'],
              'validity': rate(cand_gates, 'validity_rate') >= rate(parent_gates, 'validity_rate') - cfg['max_validity_drop'],
              'leakage': rate(cand_gates, 'leakage_rate') <= rate(parent_gates, 'leakage_rate') + cfg['max_leakage_rise'],
              'new_premise': rate(cand_gates, 'new_premise_rate') <= rate(parent_gates, 'new_premise_rate') + cfg['max_new_premise_rise'],
              'observability': rate(cand_gates, 'observable_rate') >= rate(parent_gates, 'observable_rate') - cfg['max_observability_drop'],
              'C': cand_gates['C_mean'] is not None and (parent_gates['C_mean'] is None or cand_gates['C_mean'] >= parent_gates['C_mean'] - cfg['delta_C'])}
    return {'accepted': all(checks.values()), 'checks': checks, 'n_states': len(keys), 'mean_paired_Q_gain': mean_d, 'p_gain': p_gain,
            'parent_Q_mean': st.mean(parent_q.values()) if parent_q else None, 'candidate_Q_mean': st.mean(cand_q.values()) if cand_q else None}


def prompt_summary(q_values: list[float], flats: list[bool]) -> dict:
    return {'n': len(q_values), 'Q_cov_mean': st.mean(q_values) if q_values else None, 'Q_cov_median': st.median(q_values) if q_values else None,
            'Q_cov_p10': lower_tail(q_values), 'flat_rate': sum(flats) / len(flats) if flats else None}
