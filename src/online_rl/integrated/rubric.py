"""Ephemeral state-local rubric: pre-action state elicitation, frozen generator input, generation with at most N
attempts, gates, scoring."""
from __future__ import annotations
import json
from typing import Callable, Optional
from src.online_rl.integrated.metrics import hard_value, r_a
from src.online_rl.integrated.records import EphemeralRubricRecord, GeneratorInput, sha256_json

HISTORY_KEYS = {'speaker', 'stage', 'text', 'round'}
VISIBLE = ('name', 'applicability_condition', 'required_behavior', 'scores')
DOCTOR_SIDE = ('latest_doctor_message', 'clinician_current_position', 'clinician_claims_so_far', 'clinician_evidence_so_far')
AGENT_SIDE = ('focal_claim', 'agent_current_position', 'agent_claims_so_far', 'agent_evidence_so_far', 'shared_evidence', 'resolved_points',
              'unresolved_points', 'unsupported_case_details', 'better_supported_position', 'better_supported_reason', 'evidence_sufficient',
              'evidence_sufficiency_reason')
STATE_OK = ('elicited', 'elicited_initial_turn')
STATE_FAILED = ('reconstruction_failure', 'judgment_failure')
NOT_ELICITED = {'status': 'not_elicited', 'doctor_side_context': None, 'agent_side_state': None}   # rubric disabled (outcome-only)
STATE_SCHEMA = {'version': 'medmcqa_pre_action_state_v1', 'doctor_side': DOCTOR_SIDE, 'agent_side': AGENT_SIDE,
                'source': 'reconstruct_call + local_epistemic_judgment (doctor_first; no Q1/Q2 at A1)',
                'consumers': {'rubric_generator': True, 'gates': False, 'contextual_appropriateness': False, 'adherence': False, 'agent_policy': False}}
STATE_SCHEMA_SHA256 = sha256_json(STATE_SCHEMA)


def split_state(reconstruction: dict, judgment: Optional[dict], prefix: list[dict], turn: int) -> dict:
    """Clinician reconstruction fields form the Doctor-side context; Agent, shared and role-mapped Q1/Q2 fields form the
    Agent-side state (Doctor-first; same reconstruction/judgment as src/collaboration/generator.belief_agent_prompt)."""
    from src.belief import map_judgment_to_roles
    if reconstruction.get('parse_status') != 'valid': return {'status': 'reconstruction_failure', 'doctor_side_context': None, 'agent_side_state': None}
    doctor = {'latest_doctor_message': prefix[-1]['text'], **{k: reconstruction[k] for k in DOCTOR_SIDE[1:]}}
    agent = {k: reconstruction.get(k) for k in AGENT_SIDE[:8]} | dict.fromkeys(AGENT_SIDE[8:])
    if turn == 1: return {'status': 'elicited_initial_turn', 'doctor_side_context': doctor, 'agent_side_state': agent}   # no Q1/Q2 before the Agent has a position
    if not judgment or judgment.get('parse_status') != 'valid': return {'status': 'judgment_failure', 'doctor_side_context': None, 'agent_side_state': None}
    m = map_judgment_to_roles(judgment, 'doctor_first')
    agent.update(better_supported_position=m['q1']['better_supported_position'], better_supported_reason=m['q1']['reason'],
                 evidence_sufficient=m['q2']['answer'] == 'Y', evidence_sufficiency_reason=m['q2']['reason'])
    return {'status': 'elicited', 'doctor_side_context': doctor, 'agent_side_state': agent}


def freeze_generator_input(case: dict, prefix: list[dict], turn: int, state: dict) -> GeneratorInput:
    """Generator boundary: visible case fields, the pre-action history and the pre-reply elicited state only."""
    for d in prefix:
        if set(d) - HISTORY_KEYS: raise ValueError(f'history entry carries non-dialogue fields {sorted(set(d) - HISTORY_KEYS)}')
        if d.get('stage') == 'FINAL' or d['speaker'] not in ('doctor', 'agent'): raise ValueError('history contains the Doctor FINAL or an unknown speaker')
    if not prefix or prefix[-1]['speaker'] != 'doctor': raise ValueError('pre-action history must end with the Doctor message')
    if sum(d['speaker'] == 'agent' for d in prefix) != turn - 1: raise ValueError(f'history is not the pre-action state of Agent turn {turn}')
    if set(state) != {'status', 'doctor_side_context', 'agent_side_state'} or state['status'] not in STATE_OK + STATE_FAILED + ('not_elicited',):
        raise ValueError(f'malformed pre-action state {sorted(state)}')
    if state['status'] in STATE_OK and (set(state['doctor_side_context']) != set(DOCTOR_SIDE) or set(state['agent_side_state']) != set(AGENT_SIDE)):
        raise ValueError('pre-action state carries unexpected fields')
    payload = {'case_id': case['case_id'], 'scenario': case.get('scenario') or '', 'question': case['question'],
               'options': [[str(k), str(v)] for k, v in sorted(case['options'].items())],
               'history': [[d['speaker'], d.get('stage') or '', int(d.get('round') or 0), d['text']] for d in prefix], 'turn': turn,
               'state_status': state['status'], 'doctor_side_context': state['doctor_side_context'], 'agent_side_state': state['agent_side_state']}
    return GeneratorInput(payload['case_id'], payload['scenario'], payload['question'], tuple(map(tuple, payload['options'])),
                          tuple(map(tuple, payload['history'])), turn, state['status'], state['doctor_side_context'], state['agent_side_state'],
                          sha256_json(payload))


def generate_rubric(prompt_text: str, prompt_id: str, prompt_sha256: str, gin: GeneratorInput, state_key: str, case: dict,
                    generate: Callable, gate: Callable, max_attempts: int, step: int, sample: int = 0) -> EphemeralRubricRecord:
    """generate(prompt_text, gin) sees only the frozen input; gate(case, gin, cand, rubric_id) may use the gold (validator only).
    Malformed output (no parse / schema failure) is regenerated; a well-formed rubric that fails a gate is not."""
    if max_attempts < 1: raise ValueError('max_attempts must be >= 1')
    gin.verify(); attempts = []
    if gin.state_status in STATE_FAILED + ('not_elicited',):
        reason = 'rubric_disabled' if gin.state_status == 'not_elicited' else 'state_elicitation_failed'
        return EphemeralRubricRecord(f'{prompt_sha256[:12]}-{gin.sha256[:12]}-s{step}-n{sample}-nostate', None, prompt_id, prompt_sha256, state_key,
                                     gin.sha256, None, None, False, reason, (), step)
    while True:
        rid = f'{prompt_sha256[:12]}-{gin.sha256[:12]}-s{step}-n{sample}-a{len(attempts)}'
        cand = generate(prompt_text, gin); gates = gate(case, gin, cand, rid)
        attempts.append({'attempt': len(attempts), 'rubric_id': rid, 'raw': cand.get('raw'), 'parse_error': cand.get('parse_error'), 'gates': gates})
        if gates.get('schema') or len(attempts) == max_attempts: break
    well_formed = bool(gates.get('schema')); valid = well_formed and bool(gates.get('passed'))
    rubric = cand['rubric'] if well_formed else None
    return EphemeralRubricRecord(rid, sha256_json({k: rubric[k] for k in VISIBLE}) if rubric else None, prompt_id, prompt_sha256, state_key, gin.sha256,
                                 rubric, gates, valid, None if valid else ('gate_failed' if well_formed else 'malformed_after_max_attempts'),
                                 tuple(attempts), step)


def score_responses(rubric: EphemeralRubricRecord, gin: GeneratorInput, responses: list[str | None], score: Callable) -> list[dict | None]:
    """score(case_view, gin, rubric, reply) -> {'probabilities': {'0','1','2'}, 'hard': 0|1|2}; None entries are skipped."""
    if not rubric.valid: return [None] * len(responses)
    out = []
    for reply in responses:
        if reply is None: out.append(None); continue
        s = score(gin.case_view(), gin, rubric.rubric, reply)
        out.append({'R_A': r_a(s['probabilities']), 'R_A_hard': hard_value(s['hard']), 'probabilities': s['probabilities']})
    return out


def generator_message(gin: GeneratorInput) -> str:
    """Generator message (case, dialogue, turn) plus the elicited pre-action state, when available."""
    from src.prompt_opt import harness as H
    msg = H.generator_user_message(gin.case_view(), gin.prefix(), gin.turn)
    if gin.state_status in STATE_OK:
        msg += ('\n\nStructured pre-action state (elicited before the Agent replies):\n'
                + json.dumps({'doctor_side_context': gin.doctor_side_context, 'agent_side_state': gin.agent_side_state}, ensure_ascii=False, indent=1))
    return msg


def real_state_elicitor(key: str, policy, reconstructor_model: str, reconstruction_max_tokens: int, judgment_system: str, judgment_max_tokens: int):
    """reconstruct_call (API), then the turn-local Q1/Q2 judgment from the current policy, greedy.
    The result is returned to the trainer only; it never enters an Agent response prompt."""
    from src.belief import build_local_judgment_prompt, parse_local_judgment
    from src.collaboration.generator import RECONSTRUCTION_FIELD_ORDER, agent_call, reconstruct_call

    def elicit(case_view: dict, prefix: list[dict], turn: int) -> dict:
        rec = reconstruct_call(key, reconstructor_model, case_view, prefix, reconstruction_max_tokens); judgment = None
        if turn > 1 and rec.get('parse_status') == 'valid':
            prompt = build_local_judgment_prompt(question=case_view['question'], options=case_view['options'], scenario=case_view.get('scenario', ''),
                                                 reconstruction={k: rec[k] for k in RECONSTRUCTION_FIELD_ORDER}, claim_order='doctor_first')
            policy.model.eval(); judgment = parse_local_judgment(agent_call(policy.model, policy.tok, judgment_system, prompt, judgment_max_tokens))
        return split_state(rec, judgment, prefix, turn)
    return elicit


def real_components(client, key: str):
    """Sol generator (with elicited state), v1.2 gates and Jev v5 adherence from src/prompt_opt (lazy: needs API credentials)."""
    from src.prompt_opt import harness as H
    from src.prompt_opt import harness_v12 as V12

    def generate(P, gin):
        raw = H.doctor_call(key, H.SOL, P, generator_message(gin), H.GEN_MAX_TOKENS, H.GEN_TEMPERATURE); rubric, err = H.parse_rubric(raw)
        return {'raw': raw, 'rubric': rubric, 'parse_error': err}
    gate = lambda case, gin, cand, rid: V12.gates(client, case, gin.prefix(), cand, rid)

    def score(case_view, gin, rubric, reply):
        a = H.adherence(client, case_view, gin.prefix(), rubric, reply, 'integrated_R_A')
        return {'probabilities': a['probabilities'], 'hard': a['hard']}
    return generate, gate, score
