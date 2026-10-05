"""Rubric gates used by SHERPA (schema, leakage, observability, answer presupposition, new domain premise, generic
applicability, C), plus offline per-state evaluation of N candidate rubrics."""
from __future__ import annotations
import json, statistics as st
from src.prompt_opt import harness as H
from src.prompt_opt import harness_v11 as V
from src.judges.rubric_scoring import validate_visible_rubric, _batched_jev_decisions

V12 = json.loads((H.ROOT / 'prompts/prompt_opt/rubric_validator_questions_v1_2.json').read_text())
N, K = 4, 3


def gates(client, case, prefix, cand, rid) -> dict:
    g = {'schema': None, 'leakage_deterministic': None, 'observable': None, 'answer_presupposition': None, 'new_domain_premise': None,
         'generic_applicability': None, 'C': None, 'C_hard': None, 'strategy': None, 'reasons': [], 'passed': False}
    r = cand['rubric']
    if r is None: g['schema'] = False; g['reasons'].append(f"schema: {cand['parse_error']}"); return g
    r['rubric_id'] = rid
    probs = validate_visible_rubric(r) + ([] if isinstance(r.get('required_behavior'), str) and r['required_behavior'].strip() else ['missing required_behavior'])
    g['schema'] = not probs
    if probs: g['reasons'].append(f'schema: {probs}'); return g
    leak = H.deterministic_leakage(r, case); g['leakage_deterministic'] = leak
    if leak: g['reasons'].append(f'deterministic leakage: {leak}')
    state = {'case_context': {'question': case['question'], 'options': case['options']}, 'dialogue_before_target_turn': prefix, 'candidate_rubric': V._visible(r)}
    ans, _ = _batched_jev_decisions(client, state, {**V12['questions'], **V12['strategy_label']}, purpose='prompt_opt_v12_validator', max_rubrics_per_call=None)
    g['observable'] = ans['observable']['value'] == 'Y'; g['answer_presupposition'] = ans['answer_presupposition']['value'] == 'Y'
    g['new_domain_premise'] = ans['new_domain_premise']['value'] == 'Y'; g['generic_applicability'] = ans['generic_applicability']['value'] == 'Y'
    g['strategy'] = ans['strategy']['value']
    p = {l: float(ans['contextual_appropriateness']['probabilities'][l]) for l in ('0', '1', '2')}
    g['C'] = (p['1'] + 2 * p['2']) / sum(p.values()) / 2; g['C_hard'] = int(ans['contextual_appropriateness']['value']); g['C_probabilities'] = p
    for k, msg in (('observable', 'unobservable'), ('generic_applicability', 'non-applicable (generic judge)')):
        if not g[k]: g['reasons'].append(msg)
    for k, msg in (('answer_presupposition', 'answer presupposition'), ('new_domain_premise', 'new domain premise')):
        if g[k]: g['reasons'].append(msg)
    g['passed'] = g['schema'] and not leak and g['observable'] and not g['answer_presupposition'] and not g['new_domain_premise'] and g['generic_applicability']
    return g


def evaluate_state(env, client, prompt_text: str, prompt_id: str, case: dict, s: dict) -> dict:
    """All N candidates of one state under one prompt; A/B/E for every gate-passing candidate. s = natural-baseline strata record."""
    t = int(s['state'][1:]); prefix = s['prefix']; natural = s['natural_response']
    cands = V.generate_candidates(env.key, client, prompt_text, case, prefix, t, N); res = []
    for j, c in enumerate(cands):
        rid = f"{prompt_id}-{case['case_id']}-{s['state']}-c{j}"; g = gates(client, case, prefix, c, rid)
        rec = {'candidate_index': j, 'rubric_id': rid, 'rubric': c['rubric'], 'raw': c['raw'], 'gates': g, 'regenerated': c.get('regenerated'),
               'near_duplicate_first_draw': c.get('near_duplicate_first_draw'), 'near_duplicate_after_regeneration': c.get('near_duplicate_after_regeneration')}
        if g['passed']:
            guided = env.guided(case, prefix, t, c['rubric'], H.seed_of(f'po12-guided:{rid}'))
            ag = H.adherence(client, case, prefix, c['rubric'], guided, 'prompt_opt_v12_adherence_guided')
            an = H.adherence(client, case, prefix, c['rubric'], natural, 'prompt_opt_v12_adherence_natural')
            gm = [env.continue_from(case, prefix, t, guided, H.seed_of(f'po12-mc:{rid}:guided:{k}')) for k in range(K)]
            sg = [m['correct'] for m in gm]; sn = [m['correct'] for m in s['natural_mc']]
            rec.update(guided_response=guided, A=ag['adherence'], A_hard=ag['hard'], natural_adherence=an['adherence'], B=ag['adherence'] - an['adherence'],
                       E=(st.mean(sg) - st.mean(sn)) if all(v is not None for v in sg + sn) else None, guided_successes=sum(int(v or 0) for v in sg),
                       guided_mc=[{k: m[k] for k in ('final_option', 'correct', 'termination_reason', 'agent_turns')} for m in gm])
        res.append(rec)
    return {'prompt_id': prompt_id, 'case_id': case['case_id'], 'state': s['state'], 'stratum': s['stratum'], 'natural_successes': s['natural_successes'],
            'prefix': prefix, 'natural_response': natural, 'candidates': res, 'N': N, 'K': K}
