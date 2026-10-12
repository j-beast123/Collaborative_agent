"""Post-FINAL first-person reflection by the same pinned local Doctor.

Reflection is experience evidence for rubric construction, never an outcome
label or an extra training response. The normal Doctor shared queue transports
the call, including its per-call system override; no observer API is involved.
"""
from __future__ import annotations

import json
import re

from src.online_rl.doctor import doctor_seed
from src.online_rl.integrated.records import sha256_json, sha256_text

PROTOCOL = 'physician_post_final_reflection_v1'
TRANSCRIPT = 'visible_case_and_completed_dialogue_v1'
FORMAT_NORMALIZATION = 'single_outer_json_fence_v1'
PHASE = """[Post-decision reflection phase]
You are the same Doctor who participated in the supplied discussion and already
made the recorded FINAL decision. Speak about your own experience, in first
person, rather than as an outside observer. Preserve the clinician role and
information boundary above. For this call only, replace the discussion task,
sentence limit, decision/status output instructions, and request for further
discussion with the reflection task and exact JSON format below. Do not make a
new FINAL decision or append STATUS. The recorded decision is not a correctness
label; its correctness and the ground truth have not been supplied.
"""
OUTPUT = """Return exactly one JSON object with exactly these keys:
{"helpful": [{"agent_reply_id": 1, "experience": "I ..."}],
 "unresolved": [{"agent_reply_id": 1, "experience": "I ..."}]}
Each list may contain zero or one item. Each item identifies an actual numbered Agent reply and briefly explains what
in it helped you examine, maintain, or revise your judgment, or what concern
or question it left insufficiently addressed. Do not invent a helpful part or
an unresolved concern: use an empty list when none exists. Both lists may be
empty. Report your experience, not a rubric, behavior score, factual-correctness
grade, or an instruction for the Agent. Return no Markdown or additional text.
"""


def reflection_message(view: dict, dialogue: list[dict]) -> tuple[str, int]:
    """Whitelist visible case fields; reject a call before the Doctor FINAL."""
    from src.collaboration.generator import case_text
    from src.online_rl.tree.posthoc import labeled_dialogue
    if not dialogue or dialogue[-1].get('speaker') != 'doctor' or dialogue[-1].get('stage') != 'FINAL':
        raise ValueError('Doctor reflection requires a completed dialogue ending in Doctor FINAL')
    if any(turn.get('stage') == 'FINAL' for turn in dialogue[:-1]):
        raise ValueError('Doctor reflection requires exactly one final decision at the end')
    visible = {key: view.get(key) for key in ('scenario', 'question', 'options')}
    visible['scenario'] = visible['scenario'] or ''
    turns = [{'speaker': turn['speaker'], 'stage': turn.get('stage'), 'text': turn['text']} for turn in dialogue]
    if any(turn['speaker'] not in ('agent', 'doctor') or not isinstance(turn['text'], str) for turn in turns):
        raise ValueError('Doctor reflection requires visible Doctor/Agent dialogue text')
    transcript, count = labeled_dialogue(turns)
    if not count:
        raise ValueError('Doctor reflection requires at least one Agent reply')
    return (f'{case_text(visible)}\n\nYour completed discussion, including your recorded FINAL decision. '
            f'The Agent gave {count} replies, labelled Agent reply 1 to Agent reply {count}:\n{transcript}\n\n'
            'Reflect on your experience of the Agent\'s contributions in this completed discussion.'), count


class DoctorReflectionError(RuntimeError):
    """Operational failure with the actual attempted requests and responses."""
    def __init__(self, message: str, *, raw, evidence, settings):
        super().__init__(message)
        self.raw, self.evidence, self.settings = raw, evidence, settings
        self.audit = {'raw': raw, 'evidence': evidence, 'settings': settings}


def parse_reflection(raw: str, count: int) -> dict:
    # Some local models wrap valid JSON despite the no-Markdown instruction.
    # Remove only a single complete outer JSON fence, never surrounding prose,
    # multiple fenced blocks, missing fields, or values inside the JSON object.
    normalized = raw
    if isinstance(raw, str):
        fence = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*', raw, flags=re.S | re.I)
        if fence is not None and '```' not in fence.group('body'):
            normalized = fence.group('body')
    try:
        result = json.loads(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError('reflection must be one JSON object without Markdown') from exc
    if not isinstance(result, dict) or set(result) != {'helpful', 'unresolved'}:
        raise ValueError('reflection requires exactly helpful and unresolved arrays')
    for field in ('helpful', 'unresolved'):
        if not isinstance(result[field], list):
            raise ValueError(f'{field} must be an array')
        if len(result[field]) > 1:
            raise ValueError(f'{field} must contain zero or one experience')
        for item in result[field]:
            if not isinstance(item, dict) or set(item) != {'agent_reply_id', 'experience'}:
                raise ValueError(f'each {field} item requires exactly agent_reply_id and experience')
            reply = item['agent_reply_id']
            if isinstance(reply, bool) or not isinstance(reply, int) or not 1 <= reply <= count:
                raise ValueError(f'{field}.agent_reply_id must identify an existing Agent reply (1..{count})')
            if not isinstance(item['experience'], str) or not item['experience'].strip():
                raise ValueError(f'{field}.experience must be a non-empty string')
    return result


def local_doctor_feedback(doctor, system: str, max_tokens: int, run_seed: int = 0,
                          invalid_output_policy: str = 'raise'):
    """Return the posthoc feedback interface; its outcome argument is ignored.

    One deterministic, bounded format repair is allowed. Persistent invalid
    output raises an operational error, never an empty/negative experience or a
    zero reward. Seeds and exact model messages are retained in the audit value.
    """
    if not callable(getattr(doctor, '_chat', None)) or not getattr(doctor, 'protocol', None):
        raise ValueError('local Doctor reflection requires the pinned local Doctor instance')
    if not isinstance(system, str) or not system.strip():
        raise ValueError('Doctor reflection prompt must be non-empty')
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError('Doctor reflection max_tokens must be a positive integer')
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise ValueError('Doctor reflection run_seed must be an integer')
    if invalid_output_policy not in ('raise', 'exclude_trajectory'):
        raise ValueError('Unknown Doctor reflection invalid-output policy')
    combined = f'{doctor.system}\n\n{PHASE}\n{system.strip()}\n\n{OUTPUT}'
    settings = {'protocol': PROTOCOL, 'transcript': TRANSCRIPT, 'doctor_protocol': doctor.protocol,
                'model': doctor.model, 'system_sha256': sha256_text(combined),
                'max_tokens': max_tokens, 'run_seed': run_seed, 'format_repair_attempts': 1,
                'format_normalization': FORMAT_NORMALIZATION}

    def feedback(view, dialogue, outcome):
        from src.online_rl.tree.backends import with_role
        user, count = reflection_message(view, dialogue)
        identity = sha256_json({'case_id': view.get('case_id'), 'user': user, 'settings': settings})
        raw, evidence, errors = [], [], []
        for attempt in range(2):
            request = user if attempt == 0 else (user + '\n\nYour previous output did not satisfy the JSON contract: ' +
                errors[-1] + '. Rewrite the reflection as the required JSON object using only valid Agent reply IDs. '
                'Do not invent experience to fill empty lists.')
            seed = doctor_seed(run_seed, identity, PROTOCOL, attempt)
            messages = [{'role': 'user', 'content': f'{combined}\n\n{request}'}]
            body = with_role('doctor_reflection', lambda: doctor._chat(request, seed, max_tokens, combined))
            raw.append(body)
            evidence.append({'attempt': attempt + 1, 'seed': seed, 'model': doctor.model,
                             'doctor_protocol': doctor.protocol, 'messages': messages,
                             'messages_sha256': sha256_json(messages), 'max_tokens': max_tokens,
                             'response_sha256': sha256_text(body)})
            try:
                result = parse_reflection(body, count)
            except ValueError as exc:
                errors.append(str(exc))
                evidence[-1]['validation_error'] = str(exc)
                continue
            refs = [item['agent_reply_id'] for field in ('helpful', 'unresolved') for item in result[field]]
            return {'text': json.dumps(result, ensure_ascii=False), 'target_reply_id': refs[0] if refs else None,
                    'status': 'ok' if attempt == 0 else 'repaired', 'valid': True, 'kind': PROTOCOL,
                    'raw': raw, 'reflection': result, 'evidence': evidence, 'settings': dict(settings),
                    'protocol': sha256_json(settings), 'leakage': {'first': [], 'final': []}}
        if invalid_output_policy == 'exclude_trajectory':
            # Preserve both attempts and exclude only this trajectory from
            # rubric mapping; this is neither a score nor a task failure.
            failure = {'error': 'doctor_reflection_validation_failed',
                       'validation_error': errors[-1]}
            return {'text': json.dumps(failure, ensure_ascii=False),
                    'target_reply_id': None, 'status': 'invalid', 'valid': False,
                    'kind': PROTOCOL, 'raw': raw, 'reflection': None,
                    'evidence': evidence, 'settings': dict(settings),
                    'protocol': sha256_json(settings),
                    'leakage': {'first': [], 'final': []}}
        raise DoctorReflectionError('Doctor post-FINAL reflection failed local validation after one repair: ' + errors[-1],
                                    raw=raw, evidence=evidence, settings=dict(settings))

    feedback.model = doctor.model
    feedback.transcript = TRANSCRIPT
    feedback.settings = settings
    feedback.protocol = sha256_json(settings)
    return feedback
