"""RQ2-B downstream effectiveness: SAME history, the frozen Base reply versus the trained model's reply, each continued with
the same matched continuation protocol (K continuations, paired seeds); case-level paired difference with a cluster
bootstrap CI. Training signals (R_O, Q_cov) are never used as the result."""
from __future__ import annotations
from typing import Callable
from src.evaluation.stats import paired_difference


def downstream_rows(histories: list[dict], base_reply: Callable, model_reply: Callable, continue_fn: Callable, k: int, seed_fn: Callable, system: str) -> list[dict]:
    """histories: [{'history_id','case','prefix','turn'}]. continue_fn(case, prefix, turn, fixed_reply, seed) -> {'correct': 0/1/None}."""
    rows = []
    for h in histories:
        rb, rm = base_reply(h), model_reply(h)
        sb = [continue_fn(h['case'], h['prefix'], h['turn'], rb, seed_fn(h['history_id'], i))['correct'] for i in range(k)]
        sm = [continue_fn(h['case'], h['prefix'], h['turn'], rm, seed_fn(h['history_id'], i))['correct'] for i in range(k)]   # paired seeds
        ok = all(v is not None for v in sb + sm)
        rows.append({'history_id': h['history_id'], 'case_id': h['case']['case_id'], 'system': system, 'k': k, 'base_reply': rb, 'model_reply': rm,
                     'base_success': sum(sb) / k if ok else None, 'model_success': sum(sm) / k if ok else None, 'complete': ok})
    return rows


def downstream_summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r['complete']]
    return {'histories': len(rows), 'complete': len(ok), 'paired_success_difference': paired_difference([(r['case_id'], r['base_success'], r['model_success']) for r in ok]),
            'base_mean': sum(r['base_success'] for r in ok) / len(ok) if ok else None, 'model_mean': sum(r['model_success'] for r in ok) / len(ok) if ok else None}
