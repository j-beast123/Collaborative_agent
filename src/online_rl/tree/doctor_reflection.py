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


def parse_reflection(raw: str, count: int, *, allow_multiple=False) -> dict:
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
        if not allow_multiple and len(result[field]) > 1:
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


SELECTION_REPAIR = """[Reflection format repair: selection only]
For THIS correction call, replace the reflection output format above with:
{"helpful_index": 0, "unresolved_index": null}
Return exactly these two keys. Select one zero-based index from each NONEMPTY
list of your previously reported experiences below. Use null ONLY for an empty
list. Choose the experience most directly supported by a specific recorded
Agent reply and your actual discussion; do not favor agreement, confidence,
correctness, or a desired final answer. Do not rewrite, combine or invent any
experience. These are indices in the supplied lists, NOT Agent reply IDs.
Return only this small JSON object, with no explanation, Markdown or STATUS.
"""


def selected_reflection(raw, candidates, count):
    # Same single-fence tolerance as the reflection contract; no arbitrary JSON
    # extraction and no first-item fallback if the Doctor chooses invalid IDs.
    fence = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*', raw, flags=re.S | re.I)
    if fence is not None and '```' not in fence.group('body'): raw = fence.group('body')
    try: choice = json.loads(raw)
    except (ValueError, TypeError) as exc: raise ValueError('Reflection selection must be one JSON object') from exc
    if not isinstance(choice, dict) or set(choice) != {'helpful_index', 'unresolved_index'}:
        raise ValueError('Reflection selection requires helpful_index and unresolved_index')
    selected = {}
    for field in ('helpful', 'unresolved'):
        index = choice[field + '_index']; items = candidates[field]
        if not items:
            if index is not None: raise ValueError(f'{field}_index must be null for an empty list')
            selected[field] = []
        else:
            if type(index) is not int or not 0 <= index < len(items):
                raise ValueError(f'{field}_index must be an integer in 0..{len(items)-1}; cannot discard a nonempty list')
            selected[field] = [items[index]]
    return parse_reflection(json.dumps(selected), count), choice


def local_doctor_feedback(doctor, system: str, max_tokens: int, run_seed: int = 0, *, prospective_needs=False):
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
    combined = f'{doctor.system}\n\n{PHASE}\n{system.strip()}\n\n{OUTPUT}'
    if prospective_needs:
        combined += NEED_OUTPUT
    settings = {'protocol': PROTOCOL, 'transcript': TRANSCRIPT, 'doctor_protocol': doctor.protocol,
                'model': doctor.model, 'system_sha256': sha256_text(combined),
                'max_tokens': max_tokens, 'run_seed': run_seed, 'format_repair_attempts': 1,
                'format_normalization': FORMAT_NORMALIZATION}
    if prospective_needs: settings['prospective_need_linkage'] = 'immutable_need_reply_links_v1'

    def feedback(view, dialogue, outcome, *, private_needs=()):
        from src.online_rl.tree.backends import with_role
        user, count = reflection_message(view, dialogue)
        if prospective_needs:
            if not private_needs: raise ValueError('Prospective need records are required for this reflection')
            user += '\n\nImmutable prospective needs (do not rewrite):\n' + json.dumps(private_needs, ensure_ascii=False)
            from src.online_rl.tree.doctor_need import public_turns
            user += '\n\nPublic utterance IDs for literal evidence:\n' + json.dumps(public_turns(dialogue), ensure_ascii=False)
        identity = sha256_json({'case_id': view.get('case_id'), 'user': user, 'settings': settings})
        raw, evidence, errors = [], [], []
        candidates = None; selection = None
        for attempt in range(2):
            call_system = combined
            if attempt == 0:
                request = user
            elif candidates is not None:
                call_system = combined + '\n\n' + SELECTION_REPAIR
                request = (user + '\n\nYour previous reflection had too many experiences. '
                    'Choose one index per nonempty list; the program will preserve the chosen text verbatim. '
                    'Previous experiences (array order is authoritative):\n' + json.dumps(candidates, ensure_ascii=False))
            else:
                request = (user + '\n\nYour previous output did not satisfy the JSON contract: ' +
                    errors[-1] + '. Rewrite the reflection as the required JSON object using only valid Agent reply IDs. '
                    'Each helpful/unresolved list must have at most ONE item. '
                    'Do not invent experience to fill empty lists.\nPrevious output:\n' + raw[-1])
            seed = doctor_seed(run_seed, identity, PROTOCOL, attempt)
            messages = [{'role': 'user', 'content': f'{call_system}\n\n{request}'}]
            body = with_role('doctor_reflection', lambda: doctor._chat(request, seed, max_tokens, call_system))
            raw.append(body)
            evidence.append({'attempt': attempt + 1, 'seed': seed, 'model': doctor.model,
                             'doctor_protocol': doctor.protocol, 'messages': messages,
                             'messages_sha256': sha256_json(messages), 'max_tokens': max_tokens,
                             'response_sha256': sha256_text(body),
                             **({'repair_mode': 'select_existing_experience_v1' if candidates is not None else 'rewrite_format_v1'} if attempt else {})})
            try:
                links = []
                if candidates is not None:
                    result, selection = selected_reflection(body, candidates, count)
                elif prospective_needs:
                    obj = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', body.strip()))
                    if set(obj) != {'helpful', 'unresolved', 'need_assessment'}: raise ValueError('Expected helpful, unresolved, need_assessment')
                    links = validate_need_links(obj['need_assessment'], private_needs, dialogue, count)
                    result = parse_reflection(json.dumps({k: obj[k] for k in ('helpful', 'unresolved')}), count)
                else:
                    result = parse_reflection(body, count)
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(str(exc))
                evidence[-1]['validation_error'] = str(exc)
                if attempt == 0 and not prospective_needs:
                    try:
                        multiple = parse_reflection(body, count, allow_multiple=True)
                        if any(len(multiple[k]) > 1 for k in ('helpful', 'unresolved')): candidates = multiple
                    except (ValueError, TypeError, KeyError): pass
                continue
            refs = [item['agent_reply_id'] for field in ('helpful', 'unresolved') for item in result[field]]
            return {'text': json.dumps(result, ensure_ascii=False), 'target_reply_id': refs[0] if refs else None,
                    'status': 'ok' if attempt == 0 else 'repaired', 'valid': True, 'kind': PROTOCOL,
                    'raw': raw, 'reflection': result, 'reflection_need_links': links, 'evidence': evidence, 'settings': dict(settings),
                    'protocol': sha256_json(settings), 'leakage': {'first': [], 'final': []},
                    **({'selection_repair': {'protocol': 'select_existing_experience_v1', 'choice': selection}} if selection is not None else {})}
        raise DoctorReflectionError('Doctor post-FINAL reflection failed local validation after one repair: ' + errors[-1],
                                    raw=raw, evidence=evidence, settings=dict(settings))

    feedback.model = doctor.model
    feedback.transcript = TRANSCRIPT
    feedback.settings = settings
    feedback.protocol = sha256_json(settings)
    feedback.accepts_private_needs = prospective_needs
    return feedback


NEED_OUTPUT = """
For this prospective-need protocol, ADD exactly one top-level field to the JSON contract above:
"need_assessment": [{"need_id": "the unchanged supplied id", "agent_reply_id": 1,
"effect": "helped|hindered|no_effect", "reason": "how the actual reply handled this previously recorded information need",
"public_evidence": [{"turn_id": "A1", "quote": "an exact public passage"}]}].
Include exactly one entry for each Agent reply's previously recorded need. If the prospective record reports no need or inadequate public evidence, acknowledge that limitation; do not infer a past need from the eventual result. No-effect is legitimate and does not require inventing an omission. Preserve need_id and the linked Agent reply number. Do not output rewritten need descriptions or new needs. Public evidence may describe subsequent uptake, but later evidence cannot establish a requirement that was absent before the response. Reasons must discuss information, reasoning and uncertainty, not whether the final answer was correct. Helpful/unresolved arrays remain separate reported experiences in the original format.
"""


def validate_need_links(links, needs, dialogue, count):
    from src.online_rl.tree.doctor_need import public_turns
    expected = {n['before_agent_turn']: n['need_id'] for n in needs if n['before_agent_turn'] <= count}
    if not isinstance(links, list) or len(links) != len(expected): raise ValueError('Need links must cover each actual Agent reply once')
    evidence = public_turns(dialogue); seen = set()
    for link in links:
        if not isinstance(link, dict) or set(link) != {'need_id', 'agent_reply_id', 'effect', 'reason', 'public_evidence'}:
            raise ValueError('Need linkage fields differ')
        turn = link['agent_reply_id']
        if isinstance(turn, bool) or turn in seen or expected.get(turn) != link['need_id']: raise ValueError('Need id/Agent reply mismatch')
        seen.add(turn)
        if link['effect'] not in ('helped', 'hindered', 'no_effect') or not isinstance(link['reason'], str) or not link['reason'].strip():
            raise ValueError('Need linkage requires a supported effect and reason')
        if not isinstance(link['public_evidence'], list): raise ValueError('Need linkage evidence must be an array')
        if link['effect'] != 'no_effect' and not link['public_evidence']: raise ValueError('Help/hindrance requires public evidence')
        for ref in link['public_evidence']:
            if set(ref) != {'turn_id', 'quote'} or ref['turn_id'] not in evidence or not ref['quote'] or ref['quote'] not in evidence[ref['turn_id']]:
                raise ValueError('Reflection linkage needs exact public quotes')
    return links
