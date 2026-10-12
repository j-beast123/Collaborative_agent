"""Rubric-Generator harness: collaboration environment, rubric generation and parsing, leakage checks, Jev adherence.

Environment = the MedMCQA collaboration of src/online_rl/environment.py (OpenAI gpt-6-luna Doctor
T 0.4, Qwen3-8B Agent T 1.0 / top_p 1.0 / model top_k, Doctor-first, max 4 rounds, early stopping, Doctor FINAL).
Rubric Generator = frozen Sol with prompt P; guided response = frozen base Qwen3-8B with the rubric appended through the
frozen guided-arm template; judges = frozen Jev v5 (applicability + behaviour) plus the frozen v1 validator questions.
"""
from __future__ import annotations
import hashlib, json, os, re, unicodedata
from argparse import Namespace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

from src.online_rl import environment as ST
from src.belief import build_initial_agent_response_prompt, build_plain_agent_response_prompt
from src.collaboration.generator import doctor_call, dialogue_text, case_text, final_decision_prompt, final_answer_option, stop_reason
from src.judges.jev import JevClient
from src.judges.rubric_scoring import validate_visible_rubric, behavior_questions, _batched_jev_decisions

SOL = 'openai/gpt-6.1-sol'; DOCTOR = 'gpt-6-luna'; ROUNDS = 4
JEV = dict(endpoint='https://openrouter.ai/api/alpha/decisions', model='typesafe/jev-1.13', api_key_env='OPENROUTER_API_KEY')
GEN_TEMPERATURE = 0.7; GEN_MAX_TOKENS = 1200
GUIDED_TEMPLATE = (ROOT / 'prompts/prompt_opt/guided_response_template_v1.txt').read_text()
VALIDATOR = json.loads((ROOT / 'prompts/prompt_opt/rubric_validator_questions_v1.json').read_text())
seed_of = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
norm = lambda x: ' '.join(unicodedata.normalize('NFKC', str(x)).split()).casefold()


def load_key() -> str:
    """OPENROUTER_API_KEY from the environment, else from the .env file at $SHERPA_ENV_FILE or the repository root."""
    if not os.environ.get('OPENROUTER_API_KEY'):
        for line in Path(os.environ.get('SHERPA_ENV_FILE', ROOT / '.env')).read_text().splitlines():
            if line.startswith('OPENROUTER_API_KEY='): os.environ['OPENROUTER_API_KEY'] = line.split('=', 1)[1].strip().strip('"\'')
    return os.environ['OPENROUTER_API_KEY']


class Env:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch; self.key = load_key()
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to('cuda:0').eval()
        self.agent_system = (ROOT / 'prompts/agent.txt').read_text(); self.doctor_system = (ROOT / 'prompts/user_simulator.txt').read_text()
        self.args = Namespace(rounds=ROUNDS, doctor_model=DOCTOR, max_tokens_doctor=512, max_completion_length=400,
                              generation_temperature=1.0, top_p=1.0, top_k=ST.resolve_generation_top_k(self.model, None))

        from src.online_rl.doctor import OpenAIDoctor
        self.doctor = OpenAIDoctor({'engine': 'openai_direct', 'model': DOCTOR, 'options': {'temperature': 0.4},
                                   'reasoning_effort': 'none'}, {'enabled': True, 'min_agent_turns': 2},
                                  self.doctor_system, self.args.max_tokens_doctor)

    def base(self, case: dict, seed: int) -> dict:
        self.torch.manual_seed(seed)
        return ST.generate_trajectory(self.model, self.tok, case, self.doctor_system, self.agent_system, self.key, self.args, doctor=self.doctor, seed=seed)

    def agent_user_prompt(self, case: dict, prefix: list[dict], t: int) -> str:
        if t == 1:
            return build_initial_agent_response_prompt(scenario=case.get('scenario', ''), question=case['question'], options=case['options'], clinician_message=prefix[-1]['text'])
        return build_plain_agent_response_prompt(scenario=case.get('scenario', ''), question=case['question'], options=case['options'],
                                                 dialogue_history=dialogue_text(prefix), current_round=t, max_rounds=self.args.rounds, is_final_round=t == self.args.rounds)

    def agent(self, user: str, seed: int) -> str:
        self.torch.manual_seed(seed); a = self.args
        return ST.generate_with_trace(self.model, self.tok, ST.chat_prompt(self.tok, self.agent_system, user), a.max_completion_length, a.generation_temperature, a.top_p, a.top_k)['completion_text']

    def guided(self, case: dict, prefix: list[dict], t: int, rubric: dict, seed: int) -> str:
        visible = {k: rubric[k] for k in ('name', 'applicability_condition', 'required_behavior', 'scores')}
        return self.agent(self.agent_user_prompt(case, prefix, t) + '\n\n' + GUIDED_TEMPLATE.format(rubric_json=json.dumps(visible, ensure_ascii=False)), seed)

    def continue_from(self, case: dict, prefix: list[dict], t: int, fixed_agent_text: str, seed: int, turn_cache=None) -> dict:
        """One intervention continuation: append the single newly generated, rubric-guided Agent reply at round t,
        then request exactly one Doctor FINAL. No discussion Doctor or extra Agent turns are generated.
        turn_cache(kind, identity, fn), when given, wraps the FINAL call so a resumed run reuses the exact observed answer."""
        return Env._continue_v2(self, case, prefix, t, fixed_agent_text, seed, turn_cache)

    def _continue_v2(self, case: dict, prefix: list[dict], t: int, fixed_agent_text: str, seed: int, turn_cache=None) -> dict:
        """Doctor-v2 one-step intervention: fixed guided Agent reply, followed immediately by the Doctor FINAL."""
        from src.online_rl.doctor import doctor_seed
        getattr(self, 'seed_all', self.torch.manual_seed)(seed); a = self.args; doc = self.doctor; cid = case['case_id']
        dialogue = [{k: v for k, v in x.items() if k != 'status'} for x in prefix]
        step = turn_cache or (lambda kind, identity, fn: fn())
        h = lambda x: hashlib.sha256(x.encode()).hexdigest()
        dialogue.append({'speaker': 'agent', 'stage': f'A{t}', 'round': t, 'text': fixed_agent_text})
        fs = doctor_seed(seed, cid, 'final', 0)
        final = step('doctor', {'model': doc.model, 'protocol': doc.protocol, 'stage': 'FINAL', 'seed': fs, 'input_sha256': h(dialogue_text(dialogue))},
                     lambda: doc.final(case, dialogue, fs))
        opt = final_answer_option(final)
        return {'final_option': opt, 'correct': ST.final_outcome(case, opt), 'termination_reason': 'intervention_agent_then_final', 'agent_turns': sum(x['speaker'] == 'agent' for x in dialogue),
                'dialogue': dialogue + [{'speaker': 'doctor', 'stage': 'FINAL', 'text': final, 'final': True}], 'doctor_status': []}


def generator_user_message(case: dict, prefix: list[dict], t: int) -> str:
    return (f"{case_text(case)}\n\nDialogue so far (through the physician's latest message):\n{dialogue_text(prefix)}\n\n"
            f"The Agent's next reply is discussion turn {t} of at most {ROUNDS}. Write the rubric for that reply.")


def parse_rubric(raw: str) -> tuple[dict | None, str | None]:
    s = raw.strip(); s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s)
    m = re.search(r'\{.*\}', s, re.S)
    if not m: return None, 'no_json_object'
    try: obj = json.loads(m.group(0))
    except json.JSONDecodeError as e: return None, f'json_error: {e}'
    if isinstance(obj.get('scores'), dict): obj['scores'] = {str(k): v for k, v in obj['scores'].items()}
    return obj, None


def generate_candidate(key: str, prompt_P: str, case: dict, prefix: list[dict], t: int) -> dict:
    raw = doctor_call(key, SOL, prompt_P, generator_user_message(case, prefix, t), GEN_MAX_TOKENS, GEN_TEMPERATURE)
    rubric, err = parse_rubric(raw)
    return {'raw': raw, 'rubric': rubric, 'parse_error': err}


OPTION_LETTER = re.compile(r'\b(?:option|choice|answer)\s*\(?[A-E]\)?\b|\(\s*[A-E]\s*\)', re.I)

def deterministic_leakage(rubric: dict, case: dict) -> list[str]:
    """Option/answer mentions in any rubric field. Uses the gold answer only inside this validator."""
    text = norm(' '.join([rubric.get('name', ''), rubric.get('applicability_condition', ''), rubric.get('required_behavior', '')] + list((rubric.get('scores') or {}).values())))
    found = []
    if OPTION_LETTER.search(text): found.append('mentions_option_letter')
    for k, v in case['options'].items():
        nv = norm(v)
        if len(nv) >= 8 or len(nv.split()) >= 2:
            if nv in text: found.append('gold_option_text' if k == case['ground_truth']['correct_option'] else f'option_text_{k}')
    return found


def jev() -> JevClient: return JevClient(**JEV)


def gates(client: JevClient, case: dict, prefix: list[dict], cand: dict, rid: str) -> dict:
    out = {'schema': None, 'leakage_deterministic': None, 'observable': None, 'answer_presupposition': None, 'applicability': None, 'reasons': []}
    r = cand['rubric']
    if r is None: out['schema'] = False; out['reasons'].append(f"schema: {cand['parse_error']}"); return out
    r['rubric_id'] = rid
    probs = validate_visible_rubric(r) + ([] if isinstance(r.get('required_behavior'), str) and r['required_behavior'].strip() else ['missing required_behavior'])
    out['schema'] = not probs
    if probs: out['reasons'].append(f'schema: {probs}'); return out
    leak = deterministic_leakage(r, case); out['leakage_deterministic'] = leak
    if leak: out['reasons'].append(f'deterministic leakage: {leak}')
    visible_case = {'question': case['question'], 'options': case['options']}
    state = {'case_context': visible_case, 'dialogue_before_target_turn': prefix,
             'candidate_rubric': {k: r[k] for k in ('name', 'applicability_condition', 'required_behavior', 'scores')}}
    ans, call = _batched_jev_decisions(client, state, VALIDATOR['questions'], purpose='prompt_opt_v1_validator', max_rubrics_per_call=None)
    out['observable'] = ans['observable']['value'] == 'Y'; out['answer_presupposition'] = ans['answer_presupposition']['value'] == 'Y'
    out['validator_probabilities'] = {k: v.get('probabilities') for k, v in ans.items()}
    if not out['observable']: out['reasons'].append('unobservable (Jev)')
    if out['answer_presupposition']: out['reasons'].append('answer presupposition (Jev)')
    hist = {'trajectory_id': f"po1:{case['case_id']}", 'turn_id': len(prefix), 'dialogue_before_target_turn': prefix, 'case_context': visible_case}
    from src.judges.rubric_scoring import applicability_questions
    app, _ = _batched_jev_decisions(client, hist, applicability_questions([r]), purpose='prompt_opt_v1_applicability', max_rubrics_per_call=None)
    out['applicability'] = app[rid]['value'] == 'Y'; out['applicability_probabilities'] = app[rid].get('probabilities')
    if not out['applicability']: out['reasons'].append('non-applicable (Jev v5 stage 1)')
    out['passed'] = out['schema'] and not leak and out['observable'] and not out['answer_presupposition'] and out['applicability']
    return out


def adherence(client: JevClient, case: dict, prefix: list[dict], rubric: dict, reply: str, purpose: str) -> dict:
    """Jev v5 stage-2 expected adherence (p1 + 2 p2) / (2 * total), identical formula to two_stage_jev_score."""
    target = {'trajectory_id': f"po1:{case['case_id']}", 'turn_id': len(prefix), 'dialogue_before_target_turn': prefix,
              'target_turn': {'speaker': 'agent', 'text': reply}, 'case_context': {'question': case['question'], 'options': case['options']}}
    ans, _ = _batched_jev_decisions(client, target, behavior_questions([rubric]), purpose=purpose, max_rubrics_per_call=None)
    a = ans[rubric['rubric_id']]; p = {l: float(a['probabilities'][l]) for l in ('0', '1', '2')}
    return {'hard': int(a['value']), 'probabilities': p, 'adherence': (p['1'] + 2 * p['2']) / sum(p.values()) / 2}
