"""Rubric gates with the new-domain-premise check, the contextual-appropriateness judge C and strategy labels; multi-candidate
generation with near-duplicate regeneration for offline prompt evaluation."""
from __future__ import annotations
import json
from src.prompt_opt import harness as H
from src.judges.rubric_scoring import validate_visible_rubric, applicability_questions, _batched_jev_decisions

ROOT = H.ROOT
V11 = json.loads((ROOT / 'prompts/prompt_opt/rubric_validator_questions_v1_1.json').read_text())
VIS = ('name', 'applicability_condition', 'required_behavior', 'scores')


def strata(successes: int, k: int = 3) -> str:
    return 'floor' if successes == 0 else 'ceiling' if successes == k else 'intermediate'


def generator_message(case, prefix, t, previous: list[dict]) -> str:
    msg = H.generator_user_message(case, prefix, t)
    if previous:
        prev = '\n'.join(f"- {p['name']}: {p['required_behavior']}" for p in previous)
        msg += ('\n\nRubrics already proposed for this same state:\n' + prev +
                '\nPropose a rubric whose required behavior is a materially different behavioral strategy from all of these, '
                'while still following every principle above (in particular, do not invent doubts or new premises merely to be different).')
    return msg


def _sol(key, prompt_P, msg):
    raw = H.doctor_call(key, H.SOL, prompt_P, msg, H.GEN_MAX_TOKENS, H.GEN_TEMPERATURE); r, err = H.parse_rubric(raw)
    return {'raw': raw, 'rubric': r, 'parse_error': err}


def _visible(r): return {k: r[k] for k in VIS if k in r}


def near_duplicate(client, case, prefix, new: dict, previous: list[dict]) -> bool:
    state = {'case_context': {'question': case['question'], 'options': case['options']}, 'dialogue_before_target_turn': prefix,
             'previous_candidates': [_visible(p) for p in previous], 'new_candidate': _visible(new)}
    ans, _ = _batched_jev_decisions(client, state, V11['near_duplicate'], purpose='prompt_opt_v11_near_duplicate', max_rubrics_per_call=None)
    return ans['near_duplicate']['value'] == 'Y'


def generate_candidates(key, client, prompt_P, case, prefix, t, n=4) -> list[dict]:
    """Sequential sampling with diversity history; a candidate judged a near-duplicate of an earlier one is regenerated once."""
    out, prev = [], []
    for j in range(n):
        c = _sol(key, prompt_P, generator_message(case, prefix, t, prev)); c['diversity_history_size'] = len(prev); c['regenerated'] = False
        if c['rubric'] is not None and prev and isinstance(c['rubric'].get('required_behavior'), str):
            dup = near_duplicate(client, case, prefix, c['rubric'], prev); c['near_duplicate_first_draw'] = dup
            if dup:
                c2 = _sol(key, prompt_P, generator_message(case, prefix, t, prev) + '\n\nYour previous draft repeated an earlier strategy; give a materially different one.')
                c2.update(diversity_history_size=len(prev), regenerated=True, first_draw=c)
                if c2['rubric'] is not None and isinstance(c2['rubric'].get('required_behavior'), str):
                    c2['near_duplicate_after_regeneration'] = near_duplicate(client, case, prefix, c2['rubric'], prev)
                c = c2
        out.append(c)
        if c['rubric'] is not None and isinstance(c['rubric'].get('required_behavior'), str) and isinstance(c['rubric'].get('name'), str): prev.append(c['rubric'])
    return out


def gates(client, case, prefix, cand, rid) -> dict:
    g = {'schema': None, 'leakage_deterministic': None, 'observable': None, 'answer_presupposition': None, 'new_domain_premise': None,
         'applicability': None, 'C': None, 'C_hard': None, 'strategy': None, 'reasons': []}
    r = cand['rubric']
    if r is None: g['schema'] = False; g['reasons'].append(f"schema: {cand['parse_error']}"); g['passed'] = False; return g
    r['rubric_id'] = rid
    probs = validate_visible_rubric(r) + ([] if isinstance(r.get('required_behavior'), str) and r['required_behavior'].strip() else ['missing required_behavior'])
    g['schema'] = not probs
    if probs: g['reasons'].append(f'schema: {probs}'); g['passed'] = False; return g
    leak = H.deterministic_leakage(r, case); g['leakage_deterministic'] = leak
    if leak: g['reasons'].append(f'deterministic leakage: {leak}')
    vc = {'question': case['question'], 'options': case['options']}
    state = {'case_context': vc, 'dialogue_before_target_turn': prefix, 'candidate_rubric': _visible(r)}
    ans, _ = _batched_jev_decisions(client, state, {**V11['questions'], **V11['strategy_label']}, purpose='prompt_opt_v11_validator', max_rubrics_per_call=None)
    g['observable'] = ans['observable']['value'] == 'Y'; g['answer_presupposition'] = ans['answer_presupposition']['value'] == 'Y'
    g['new_domain_premise'] = ans['new_domain_premise']['value'] == 'Y'; g['strategy'] = ans['strategy']['value']
    p = {l: float(ans['contextual_appropriateness']['probabilities'][l]) for l in ('0', '1', '2')}
    g['C'] = (p['1'] + 2 * p['2']) / sum(p.values()) / 2; g['C_hard'] = int(ans['contextual_appropriateness']['value']); g['C_probabilities'] = p
    g['judge_probabilities'] = {k: v.get('probabilities') for k, v in ans.items()}
    if not g['observable']: g['reasons'].append('unobservable (Jev)')
    if g['answer_presupposition']: g['reasons'].append('answer presupposition (Jev)')
    if g['new_domain_premise']: g['reasons'].append('new domain premise (Jev)')
    hist = {'trajectory_id': f"po1:{case['case_id']}", 'turn_id': len(prefix), 'dialogue_before_target_turn': prefix, 'case_context': vc}
    app, _ = _batched_jev_decisions(client, hist, applicability_questions([r]), purpose='prompt_opt_v11_applicability', max_rubrics_per_call=None)
    g['applicability'] = app[rid]['value'] == 'Y'
    if not g['applicability']: g['reasons'].append('non-applicable (Jev v5 stage 1)')
    g['passed'] = g['schema'] and not leak and g['observable'] and not g['answer_presupposition'] and not g['new_domain_premise'] and g['applicability']
    return g
