"""RQ2-A behaviour appropriateness on SAME inputs for every system, judged with independent frozen criteria
(configs/online_rl/rq2_appropriateness_judge_v1.json) -- never a model's own training rubrics.
A1 collaboration-state appropriateness: fixed probe histories -> each system's reply -> judge.
A2 controlled evidence sensitivity: the same history with a controlled collaborator claim at different evidence strengths
   -> does the system's adoption of the claimed option respond to evidence strength (and only to it)?"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
from src.evaluation.stats import cluster_bootstrap_mean, paired_difference

JUDGE = json.loads((Path(__file__).resolve().parents[2] / 'configs/online_rl/rq2_appropriateness_judge_v1.json').read_text())
DIMENSIONS = tuple(JUDGE['questions'])


def judge_reply(decide: Callable, case: dict, prefix: list[dict], reply: str) -> dict:
    """decide(state_json, questions) -> {qid: {'value','probabilities'}} (a JevClient.decide-like callable)."""
    state = {'case_context': {'question': case['question'], 'options': case['options']}, 'dialogue_before_target_turn': prefix, 'target_turn': {'speaker': 'agent', 'text': reply}}
    ans = decide(json.dumps(state, ensure_ascii=False), JUDGE['questions'])
    out = {}
    for q in DIMENSIONS:
        p = {l: float(ans[q]['probabilities'][l]) for l in ('0', '1', '2')}; out[q] = (p['1'] + 2 * p['2']) / sum(p.values()) / 2
    out['overall'] = sum(out[q] for q in DIMENSIONS) / len(DIMENSIONS); return out


def state_appropriateness_rows(probes: list[dict], systems: dict[str, Callable], decide: Callable) -> list[dict]:
    """probes: [{'probe_id','case','prefix','turn'}]; systems: name -> respond(case, prefix, turn) -> reply. Same inputs for all."""
    rows = []
    for p in probes:
        for name, respond in systems.items():
            reply = respond(p['case'], p['prefix'], p['turn'])
            rows.append({'probe_id': p['probe_id'], 'case_id': p['case']['case_id'], 'system': name, 'reply': reply, **judge_reply(decide, p['case'], p['prefix'], reply),
                         'judge_version': JUDGE['version']})
    return rows


def summarize_appropriateness(rows: list[dict], reference: str = 'base') -> list[dict]:
    out = []; systems = sorted({r['system'] for r in rows})
    for s in systems:
        rs = [r for r in rows if r['system'] == s]
        entry = {'system': s, **{f'{d}_mean': cluster_bootstrap_mean([r[d] for r in rs], [r['case_id'] for r in rs]) for d in (*DIMENSIONS, 'overall')}}
        if s != reference:
            ref = {r['probe_id']: r for r in rows if r['system'] == reference}
            entry['overall_vs_reference'] = paired_difference([(r['case_id'], ref[r['probe_id']]['overall'], r['overall']) for r in rs if r['probe_id'] in ref])
        out.append(entry)
    return out


def stated_option(reply: str, options: dict) -> str | None:
    """The option the reply commits to, with the collaboration parser (src/collaboration/generator.option)."""
    from src.collaboration.generator import option
    o = option(reply)
    return o if o in options else None


CONDITIONS = ('correct_strong_valid', 'correct_weak_valid', 'incorrect_strong_looking_insufficient', 'incorrect_no_support')   # no incorrect + strong valid evidence


def evidence_sensitivity(items: list[dict], systems: dict[str, Callable]) -> list[dict]:
    """items (contract configs/online_rl/rq2_controlled_evidence_contract_v1.json):
    [{'item_id','case','prefix','turn','claim_option','claim_correct','evidence_condition' in CONDITIONS,'collaborator_message'}].
    The controlled collaborator message replaces the last Doctor turn of prefix; every system gets identical inputs."""
    rows = []
    for it in items:
        if it['evidence_condition'] not in CONDITIONS: raise ValueError(f"unknown evidence condition {it['evidence_condition']}")
        if it['claim_correct'] != it['evidence_condition'].startswith('correct'): raise ValueError('claim_correct does not match the evidence condition')
        pre = it['prefix'][:-1] + [{**it['prefix'][-1], 'text': it['collaborator_message']}]
        for name, respond in systems.items():
            reply = respond(it['case'], pre, it['turn']); so = stated_option(reply, it['case']['options'])
            rows.append({'item_id': it['item_id'], 'case_id': it['case']['case_id'], 'system': name, 'evidence_condition': it['evidence_condition'],
                         'claim_option': it['claim_option'], 'claim_correct': it['claim_correct'], 'stated_option': so, 'adopts_claim': so == it['claim_option'], 'reply': reply})
    return rows


def sensitivity_summary(rows: list[dict]) -> list[dict]:
    """Per system: adoption by condition and the contract's contrasts (paired by case), case-cluster bootstrap CIs."""
    out = []
    for s in sorted({r['system'] for r in rows}):
        rs = [r for r in rows if r['system'] == s]
        adoption = {c: cluster_bootstrap_mean([float(r['adopts_claim']) for r in rs if r['evidence_condition'] == c], [r['case_id'] for r in rs if r['evidence_condition'] == c]) for c in CONDITIONS}
        by = {}
        for r in rs: by.setdefault(r['case_id'], {})[r['evidence_condition']] = float(r['adopts_claim'])
        def contrast(a, b): return paired_difference([(c, v[b], v[a]) for c, v in by.items() if a in v and b in v])   # mean(a - b)
        out.append({'system': s, 'adoption_by_condition': adoption,
                    'valid_evidence_sensitivity': contrast('correct_strong_valid', 'correct_weak_valid'),
                    'misleading_minus_no_support': contrast('incorrect_strong_looking_insufficient', 'incorrect_no_support'),
                    'discrimination_actual_vs_apparent': contrast('correct_strong_valid', 'incorrect_strong_looking_insufficient')})
    return out
