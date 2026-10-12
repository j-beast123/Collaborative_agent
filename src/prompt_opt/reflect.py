"""Feedback Generator and Prompt Refiner (frozen Sol) with the case-content copy guard."""
from __future__ import annotations
import json, re
from src.prompt_opt import harness as H
FG = (H.ROOT / 'prompts/prompt_opt/feedback_generator_v1.txt').read_text()
RF = (H.ROOT / 'prompts/prompt_opt/prompt_refiner_v1.txt').read_text()
SOL_REFLECT_TEMPERATURE = 0.4
_t = lambda s, n: s if len(s) <= n else s[:n] + ' …'


def minibatch_view(records: list[dict]) -> list[dict]:
    """What the Feedback Generator sees per state. The gold option is never shown; only whether continuations ended correct."""
    out = []
    for r in records:
        out.append({'state': r['state'], 'baseline_stratum': r['stratum'], 'natural_continuations_correct': f"{r['natural_successes']}/3",
                    'dialogue_before_reply': [{'speaker': d['speaker'], 'text': _t(d['text'], 700)} for d in r['prefix']],
                    'natural_reply': _t(r['natural_response'], 700),
                    'candidates': [{'rubric': {k: (c['rubric'] or {}).get(k) for k in ('name', 'applicability_condition', 'required_behavior', 'scores')} if c['rubric'] else c['raw'][:400],
                                    'passed_gates': c['gates']['passed'], 'gate_failures': c['gates']['reasons'], 'strategy': c['gates'].get('strategy'),
                                    'C': c['gates'].get('C'), 'A': c.get('A'), 'B_diagnostic': c.get('B'), 'E': c.get('E'),
                                    'guided_continuations_correct': (f"{c['guided_successes']}/3" if 'guided_successes' in c else None),
                                    'guided_reply': _t(c.get('guided_response') or '', 600)} for c in r['candidates']]})
    return out


def feedback(key: str, prompt_text: str, records: list[dict]) -> str:
    user = 'CURRENT RUBRIC-GENERATOR PROMPT:\n' + prompt_text + '\n\nMINIBATCH RESULTS (JSON):\n' + json.dumps(minibatch_view(records), ensure_ascii=False)
    return H.doctor_call(key, H.SOL, FG, user, 2500, SOL_REFLECT_TEMPERATURE)


def ngrams(text: str, n: int = 4) -> set:
    w = re.findall(r'[a-z0-9]+', text.lower()); return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def copy_violations(new_prompt: str, allowed_prompts: list[str], case_texts: list[str], n: int = 4) -> list[str]:
    """4-grams of the new prompt that occur in any Feedback case question/option text and in no earlier prompt."""
    allowed = set().union(*(ngrams(p, n) for p in allowed_prompts)); case = set().union(*(ngrams(t, n) for t in case_texts))
    return sorted((ngrams(new_prompt, n) & case) - allowed)


def structural_problems(p: str, *, require_observed_context: bool | None = None) -> list[str]:
    probs = []
    for k in ('"name"', '"required_behavior"', '"scores"'):
        if k not in p: probs.append(f'missing output key {k}')
    # Keep compatibility with older flat prompts, while tree prompt updates
    # explicitly require the observed-context schema from their parent prompt.
    observed = require_observed_context if require_observed_context is not None else '"user_side_context"' in p
    if observed:
        if '"applicability"' not in p:
            probs.append('missing output key "applicability"')
        for k in ('"user_side_context"', '"agent_side_state"'):
            if k not in p: probs.append(f'missing observed-context output key {k}')
    elif '"applicability_condition"' not in p:
        probs.append('missing output key "applicability_condition"')
    return probs


def refine(key: str, prompt_text: str, fb: str, history: list[dict], allowed_prompts: list[str], case_texts: list[str]) -> dict:
    hist = [{k: h.get(k) for k in ('iteration', 'decision', 'reason_summary', 'selection_metrics_summary')} for h in history]
    user = ('CURRENT PROMPT:\n' + prompt_text + '\n\nDIAGNOSTIC REFLECTION:\n' + fb + '\n\nREVISION HISTORY (JSON):\n' + json.dumps(hist, ensure_ascii=False))
    attempts = []
    for attempt in range(2):
        new = H.doctor_call(key, H.SOL, RF, user, 6000, SOL_REFLECT_TEMPERATURE).strip()
        new = re.sub(r'^```(?:\w+)?\s*|\s*```$', '', new).strip()
        observed = '"user_side_context"' in prompt_text and '"agent_side_state"' in prompt_text
        viol = copy_violations(new, allowed_prompts, case_texts)
        probs = structural_problems(new, require_observed_context=observed)
        attempts.append({'prompt': new, 'copy_violations': viol, 'structural_problems': probs})
        if not viol and not probs: return {'prompt': new, 'attempts': attempts, 'ok': True}
        user += ('\n\nYOUR PREVIOUS DRAFT WAS REJECTED. ' + (f'It copied case-specific wording: {viol[:20]}. ' if viol else '') +
                 (f'Problems: {probs}. ' if probs else '') + 'Return a corrected complete prompt.')
    return {'prompt': None, 'attempts': attempts, 'ok': False}
