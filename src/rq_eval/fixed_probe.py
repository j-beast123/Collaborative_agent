"""Fixed-probe failure resolution: the SAME frozen probe histories are evaluated for an ordered list of checkpoints (Base, then saved steps)
(checkpoint reply + matched continuation -> success), and each probe carries a semantic theme (the theme of the contextual
rubric that the frozen probe-labelling prompt generates for that pre-action history, normalised with themes.py).
Transitions per theme between consecutive checkpoints reuse src/evaluation/rq3.probe_transitions."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Callable
from src.evaluation.rq3 import probe_transitions

def freeze_probes(probes: list[dict], path: Path) -> str:
    """probes: [{'probe_id','case_id','prefix','turn','theme','theme_source'}]; written once, hash returned; never rewritten."""
    data = json.dumps(sorted(probes, key=lambda p: p['probe_id']), ensure_ascii=False, sort_keys=True)
    if path.exists():
        if path.read_text() != data: raise RuntimeError('probe set already frozen with different content')
    else: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(data)
    return hashlib.sha256(data.encode()).hexdigest()


def evaluate_checkpoint(probes: list[dict], cases: dict, checkpoint: str, reply_fn: Callable, continue_fn: Callable, k: int, seed_fn: Callable,
                        order: tuple) -> list[dict]:
    """success = majority of K matched continuations correct (K odd). order: checkpoint labels, e.g. ('base', 'step_000050', ...)."""
    if checkpoint not in order: raise ValueError(f'checkpoint must be one of {order}')
    rows = []
    for p in probes:
        case = cases[p['case_id']]; reply = reply_fn(case, p['prefix'], p['turn'])
        v = [continue_fn(case, p['prefix'], p['turn'], reply, seed_fn(p['probe_id'], i))['correct'] for i in range(k)]
        rate = None if any(x is None for x in v) else sum(v) / k
        rows.append({'probe_id': p['probe_id'], 'case_id': p['case_id'], 'category_id': p['theme'], 'checkpoint': checkpoint, 'iteration': order.index(checkpoint),
                     'success_rate': rate, 'success': None if rate is None else rate > 0.5, 'reply': reply})
    return rows


def transitions_by_theme(rows: list[dict], order: tuple) -> list[dict]:
    """Fail->Success, Fail->Fail, Success->Fail (and Success->Success) per theme and consecutive checkpoints."""
    ok = [r for r in rows if r['success'] is not None]
    out = probe_transitions(ok)
    for r in out: r['from_checkpoint'] = order[r['from_iteration']]; r['to_checkpoint'] = order[r['to_iteration']]
    return out
