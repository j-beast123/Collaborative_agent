"""Prospective need evidence, generated jointly with the public Doctor utterance.

Only `visible` enters dialogue; private_need is an immutable side record. Need
text is never fed into later Doctor turns, Agent turns or response scoring.
"""
import json
import re
import time
from pathlib import Path
from src.online_rl.integrated.records import sha256_json, sha256_text

PROTOCOL = 'doctor_public_and_prospective_need_v1'
ROOT = Path(__file__).resolve().parents[3]


def public_turns(dialogue):
    return {d['stage']: d['text'] for d in dialogue if d.get('stage') != 'FINAL'}


def validate_need(need, evidence):
    fields = {'status', 'current_uncertainty', 'information_needed', 'public_evidence'}
    if not isinstance(need, dict) or set(need) != fields:
        raise ValueError('Need fields must be status/current_uncertainty/information_needed/public_evidence')
    if need['status'] not in ('present', 'none', 'insufficient_public_evidence'):
        raise ValueError('Unknown need status')
    if any(not isinstance(need[k], str) for k in ('current_uncertainty', 'information_needed')):
        raise ValueError('Need descriptions must be strings')
    if not isinstance(need['public_evidence'], list): raise ValueError('Need evidence must be an array')
    for ref in need['public_evidence']:
        if not isinstance(ref, dict) or set(ref) != {'turn_id', 'quote'}:
            raise ValueError('Each evidence item needs turn_id and quote')
        if ref['turn_id'] not in evidence or not isinstance(ref['quote'], str) or not ref['quote'].strip() or ref['quote'] not in evidence[ref['turn_id']]:
            raise ValueError('Need evidence must quote an existing pre-response public utterance exactly')
    if need['status'] == 'present' and (not need['public_evidence'] or not need['current_uncertainty'].strip() or not need['information_needed'].strip()):
        raise ValueError('Present need requires uncertainty, information and public evidence')


class NeedFormatError(RuntimeError):
    def __init__(self, raw, errors):
        super().__init__('Doctor public/need JSON failed validation after one repair')
        self.audit = {'raw': raw, 'validation_errors': errors}


def attach(doctor, cfg):
    from src.online_rl.doctor import turn_prompt, split_status, doctor_seed
    settings = cfg['doctor_need']
    contract = (ROOT / settings['prompt_file']).read_text()
    if sha256_text(contract) != settings['prompt_sha256']: raise ValueError('Doctor need prompt changed')
    doctor.identity['private_need'] = dict(settings)
    doctor.protocol = sha256_json(doctor.identity)

    def turn(case, dialogue, round_, rounds, seed):
        # Never serialize incoming private/status fields. Preserve ordinary case,
        # Doctor discussion and hidden STATUS instructions in the request.
        visible_history = [{k: d[k] for k in ('speaker', 'stage', 'round', 'text') if k in d} for d in dialogue]
        user = turn_prompt(case, visible_history, round_, rounds, with_status=True)
        user += '\n\nPublic utterance IDs (for literal citations):\n' + json.dumps(public_turns(visible_history), ensure_ascii=False)
        user += f'\nYour next public utterance ID is D{round_}. Its private need is recorded before Agent reply {round_}.'
        system = doctor.system + '\n\n' + contract
        raw, errors, attempts = [], [], []
        start = time.monotonic()
        for attempt in range(2):
            actual_seed = seed if attempt == 0 else doctor_seed(seed, sha256_json(user), 'need_format_repair', 1)
            request = user if attempt == 0 else user + '\nJSON/evidence validation failed: ' + errors[-1] + '\nReturn the complete corrected object. Do not invent a need or a quotation.'
            body = doctor._chat(request, actual_seed, settings['combined_max_tokens'], system)
            raw.append(body)
            meta = dict(getattr(getattr(doctor, '_chat_metadata', None), 'last', {}) or {})
            attempts.append({'seed': actual_seed, 'attempt': attempt + 1, **meta})
            try:
                normalized = re.sub(r'^```(?:json)?\s*|\s*```$', '', body.strip())
                obj = json.loads(normalized)
                if not isinstance(obj, dict) or set(obj) != {'public_reply', 'private_need'} or not isinstance(obj['public_reply'], str):
                    raise ValueError('Expected exactly public_reply and private_need')
                visible, status, err = split_status(obj['public_reply'], case['options'])
                if not visible.strip(): raise ValueError('Public reply must be nonempty')
                if round_ == 1: status, err = None, None
                refs = {**public_turns(visible_history), f'D{round_}': visible}
                validate_need(obj['private_need'], refs)
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(str(exc)); continue
            need = {'doctor_turn_id': f'D{round_}', 'before_agent_turn': round_, **obj['private_need']}
            need['need_id'] = sha256_json({'case_id': case['case_id'], 'prefix': visible_history, 'public_reply': visible, 'need': need})
            return {'raw': body, 'visible': visible, 'status': status, 'status_error': err, 'seed': seed,
                    'private_need': need, 'need_audit': {'protocol': PROTOCOL, 'attempts': attempts,
                        'format_errors': errors, 'format_retries': attempt, 'missing': False,
                        'combined_generation_seconds': time.monotonic() - start,
                        'private_need_serialized_characters': len(json.dumps(obj['private_need'], ensure_ascii=False)),
                        'private_need_exact_generation_tokens': None,
                        'private_need_incremental_seconds': None,
                        'measurement_note': 'Joint generation: provider reports total public+need tokens/time only; no counterfactual public-only call.'}}
        raise NeedFormatError(raw, errors)

    doctor.turn = turn
    return doctor


def criterion_need_problem(tin, rubric):
    if not getattr(tin, 'private_needs', None): return None
    selected = ((rubric or {}).get('source_provenance', {}).get('analysis', {})).get('selected_agent_turn_id')
    rows = [n for n in tin.private_needs if n['before_agent_turn'] == selected]
    if len(rows) != 1 or rows[0]['status'] != 'present':
        return 'need_public_evidence_insufficient: selected response has no publicly grounded prospective need'
    from src.online_rl.tree.records import to_dialogue
    prefix = []
    for d in to_dialogue(tin.dialogue):
        if d['speaker'] == 'agent' and d['round'] == selected: break
        prefix.append(d)
    try: validate_need({k: rows[0][k] for k in ('status', 'current_uncertainty', 'information_needed', 'public_evidence')}, public_turns(prefix))
    except ValueError:
        return 'need_public_evidence_insufficient: evidence is not in the anchored public prefix'
    return None
