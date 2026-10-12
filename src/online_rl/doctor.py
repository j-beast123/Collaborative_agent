"""OpenAI Doctor discussion, private state, adaptive halting and final decisions.
Seeds identify cached observations; OpenAI calls do not use a provider seed.
"""
from __future__ import annotations
import json, os, re, threading
from typing import Callable, Optional
_DOCTOR_API_CAPACITY = threading.BoundedSemaphore(50)
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.records import sha256_json, sha256_text

PROTOCOL_VERSION = 'doctor_v2_status_halting_v1'
STATUS_KEYS = ('current_option', 'open_issue', 'collaborator_objection', 'more_info_could_change_decision')
FLAGS = STATUS_KEYS[1:]
SEED_MOD = 2 ** 31 - 1

V2_TURN = ("Respond as the physician. This is discussion turn {turn} of at most {rounds}. Respond naturally to the collaborator's "
           "latest reasoning. If a substantive disagreement, uncertainty, or evidence gap remains, explain the single most important "
           "unresolved issue and, when useful, what information or reasoning would resolve it. If the discussion already provides "
           "sufficient support for a stable choice, briefly summarize the decisive reason instead of inventing another question, "
           "request, disagreement, or uncertainty merely to continue the conversation. Agreement alone is not sufficient if a "
           "substantive unresolved issue remains. A separate final-decision prompt will record the answer.")
V2_OPENING = ("Respond naturally as the physician. This is discussion turn 1 of at most {rounds}. State your current reasoning and "
              "what you would like the collaborator to weigh in on. A separate final-decision prompt will record the answer.")
FINAL_TURN = " This is the final allowed discussion turn, so present your best-supported reasoning."
STATUS_INSTRUCTION = (
    "\n\nAfter your reply, add one final line exactly in this form; it is removed before the collaborator sees your reply:\n"
    'STATUS: {{"current_option": "<{labels} or none>", "open_issue": <true|false>, "collaborator_objection": <true|false>, '
    '"more_info_could_change_decision": <true|false>}}\n'
    "- current_option: the option you currently favor, or none if you do not favor one yet.\n"
    "The fields describe the discussion AFTER the reply you just wrote.\n"
    "- open_issue: true only if a substantive unresolved issue remains that another exchange with this collaborator could "
    "meaningfully advance. An inherently ambiguous question, two options that both remain defensible, or an uncertainty you "
    "and the collaborator have already acknowledged and discussed is not by itself an open issue.\n"
    "- collaborator_objection: true only if a substantive objection, argument, evidence point or alternative from the "
    "collaborator remains unaddressed or unreconciled after your reply above; an objection your reply has already "
    "addressed does not count, even if you still disagree.\n"
    "- more_info_could_change_decision: true only if another exchange in this collaboration can realistically provide "
    "decision-relevant information or reasoning that could change your current choice. Do not mark it true because "
    "hypothetical external evidence might help, or because the same acknowledged uncertainty could be repeated.\n"
    "- If you are unsure about any field, use true.")
_STATUS_START = re.compile(r'(?im)^[ \t*_`>#-]*STATUS\s*[:：]')
_PRIVATE_START = re.compile(r'(?im)^[ \t*_`>#-]*PRIVATE_STATE\s*[:：]')
PRIVATE_STATE_PROTOCOL = 'hypothesis_confidence_v1'
JSON_TURN_PROTOCOL = 'private_turn_json_v2'
JSON_TURN_INSTRUCTION = '''\n\nFor this call, replace line-based STATUS and PRIVATE_STATE with the server-enforced
JSON object containing reply, status, and private_state. Put only your natural
visible discussion in reply (at most 60 words); never put status, private beliefs,
confidence, or analysis of sibling replies in that visible field.
private_state records your actual current belief AFTER that reply: current_option
is an option LABEL or none, hypothesis is one short sentence (at most 20 words),
and confidence is a number between 0 and 1. Do not infer it from a gold answer.
For the opening status is null. At later turns status describes the discussion
AFTER your reply: current_option is the label you favor or none; open_issue is
true for a substantive unresolved issue that this collaborator could advance;
collaborator_objection is true for a substantive unaddressed objection;
more_info_could_change_decision is true only when another exchange could
realistically change your choice. If unsure about any flag, use true.
Do not fabricate issues simply to prolong a resolved discussion. Do not append
STATUS or PRIVATE_STATE lines, a separate FINAL answer, or any extra fields.'''
PRIVATE_STATE_INSTRUCTION = '''\n\nAfter the visible reply and STATUS (if requested), append one last hidden line:
PRIVATE_STATE: {"current_option": "<option label or none>", "hypothesis": "<one short sentence describing your current hypothesis>", "confidence": <number between 0 and 1>}
Report your actual current belief. This field is recorded privately and removed before the colleague sees your reply.
Do not refer to this field in your visible reply.'''


def turn_prompt(case: dict, dialogue: list[dict], turn: int, rounds: int, with_status: bool) -> str:
    """v2 discussion prompt for Doctor turn `turn` (the grounding text is the v1 one); the status request from turn 2 on."""
    from src.collaboration.generator import case_text, dialogue_text
    body = (V2_OPENING.format(rounds=rounds) if turn == 1 else V2_TURN.format(turn=turn, rounds=rounds)) + (FINAL_TURN if turn == rounds else '')
    text = (f"{case_text(case)}\n\nGrounding constraint: Use only the question, answer choices, dialogue, and established domain "
            "knowledge. If no patient vignette or event description is provided, do not invent one.\n\n"
            f"Conversation so far:\n{dialogue_text(dialogue)}\n\n{body}")
    return text + (STATUS_INSTRUCTION.format(labels=', '.join(case['options'])) if with_status and turn >= 2 else '')


def split_status(raw: str, options) -> tuple[str, Optional[dict], Optional[str]]:
    """(visible reply, status or None, parse error or None). Everything from the first STATUS line on is removed from the
    visible reply, whether or not it parses. A status is valid only with exactly the four keys, strict booleans and an
    option label or 'none'."""
    private = _PRIVATE_START.search(raw or '')
    if private: raw = raw[:private.start()]
    m = _STATUS_START.search(raw or '')
    if not m: return (raw or '').strip(), None, 'no status line'
    visible, tail = raw[:m.start()].strip(), raw[m.end():]
    j = re.search(r'\{.*\}', tail, re.S)
    if not j: return visible, None, 'no JSON object'
    try: st = json.loads(j.group(0))
    except json.JSONDecodeError as e: return visible, None, f'not JSON: {e}'
    if not isinstance(st, dict) or set(st) != set(STATUS_KEYS): return visible, None, f'keys {sorted(st) if isinstance(st, dict) else type(st).__name__}'
    if any(not isinstance(st[k], bool) for k in FLAGS): return visible, None, 'non-boolean flag'
    opt = st['current_option']
    if not isinstance(opt, str): return visible, None, 'current_option is not a string'
    opt = opt.strip().upper() if opt.strip().lower() != 'none' else 'none'
    if opt != 'none' and opt not in {str(o).upper() for o in options}: return visible, None, f'current_option {opt!r} is not an option'
    return visible, {**st, 'current_option': opt}, None


def split_private_state(raw, options):
    """Strictly validate private belief evidence; never fabricate a missing state."""
    marker = _PRIVATE_START.search(raw or '')
    if marker is None: raise ValueError('missing PRIVATE_STATE')
    state = json.loads(raw[marker.end():].strip())
    if not isinstance(state, dict) or set(state) != {'current_option', 'hypothesis', 'confidence'}:
        raise ValueError('PRIVATE_STATE requires current_option, hypothesis and confidence')
    reported_option = state['current_option']
    if not isinstance(reported_option, str):
        raise ValueError('PRIVATE_STATE current_option is invalid')
    import unicodedata
    normalize = lambda text: ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    reported = normalize(reported_option)
    if reported == 'none':
        canonical_option = 'none'
    else:
        labels = [label for label in options if normalize(str(label)) == reported]
        if len(labels) == 1:
            canonical_option = labels[0]
        else:
            # Some partners report the exact option text instead of its label.
            # Resolve only a unique exact match; never infer from the hypothesis
            # or visible reply, and never replace the partner's chosen answer.
            matches = ([label for label, text in options.items()
                        if isinstance(text, str) and normalize(text) == reported]
                       if isinstance(options, dict) and reported else [])
            if len(matches) != 1:
                raise ValueError('PRIVATE_STATE current_option is invalid')
            canonical_option = matches[0]
    state = {**state, 'current_option': canonical_option}
    if not isinstance(state['hypothesis'], str) or not state['hypothesis'].strip():
        raise ValueError('PRIVATE_STATE hypothesis is empty')
    confidence = state['confidence']
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError('PRIVATE_STATE confidence must be a number in [0, 1]')
    return state


def minimum_agent_turns(doctor) -> int:
    """Minimum for every early-stop path; unwrap the role-tagging Doctor adapter."""
    return int(getattr(getattr(doctor, 'doctor', doctor), 'min_agent_turns', 1))


def halt_reason(*, agent_turns: int, status: Optional[dict], prev_status: Optional[dict], last_agent: str, options,
                min_agent_turns: int, stable_choice_no_objection_after_turns: Optional[int] = None) -> Optional[str]:
    """'doctor_halt' or None (continue). Called after a Doctor turn that follows `agent_turns` Agent replies."""
    from src.collaboration.generator import option
    if agent_turns < min_agent_turns or status is None or prev_status is None: return None
    cur = status['current_option']
    if cur == 'none' or cur not in {str(o).upper() for o in options}: return None
    if prev_status.get('current_option') != cur: return None                       # the decision must be stable
    said = option(last_agent or '')
    if said and said.upper() != cur: return None                                   # the Agent argues for another option
    if not any(status[k] for k in FLAGS): return 'doctor_halt'
    if (stable_choice_no_objection_after_turns is not None and
            agent_turns >= stable_choice_no_objection_after_turns and not status['collaborator_objection']):
        return 'doctor_halt_stable_choice'
    return None


def doctor_seed(base_seed: int, key: str, role: str, index: int) -> int:
    """Per-call Doctor seed from the existing rule derive_seed(run_seed, key, role, index), in the cache identity range."""
    return derive_seed(base_seed, key, role, index) % SEED_MOD


class DoctorUnavailable(RuntimeError):
    """The Doctor API failed or returned an invalid reply. The run stops
    and resumes from its caches."""


class OpenAIDoctor:
    """Doctor protocol v2 on OpenAI Responses. turn(...) -> {'raw', 'visible', 'status', 'status_error', 'seed'};
    final(...) -> text. .model / .protocol identify it in every cache key; .identity goes to the run manifest."""
    status_protocol = True

    def __init__(self, backend: dict, halting: Optional[dict], system: str, max_tokens: int):
        for k in ('engine', 'model', 'options'):
            if not backend.get(k): raise ValueError(f'rollout.doctor_backend.{k} is required')
        if backend['engine'] != 'openai_direct' or backend['model'] != 'gpt-6-luna':
            raise ValueError('Doctor requires OpenAI direct gpt-6-luna')
        self.backend, self.system, self.max_tokens = backend, system, max_tokens
        self.private_state_protocol = backend.get('private_state_protocol')
        if self.private_state_protocol not in (None, PRIVATE_STATE_PROTOCOL):
            raise ValueError('Unknown Doctor private_state_protocol')
        self.discussion_response_format = backend.get('discussion_response_format')
        if self.discussion_response_format not in (None, 'private_turn_json_v1', JSON_TURN_PROTOCOL):
            raise ValueError('Unknown Doctor discussion_response_format')
        if self.discussion_response_format and not self.private_state_protocol:
            raise ValueError('Structured discussion requires private_state_protocol')
        self.json_turn_instruction = JSON_TURN_INSTRUCTION
        if self.discussion_response_format == JSON_TURN_PROTOCOL:
            self.json_turn_instruction += '\nReport confidence using the allowed values in increments of 0.01.\n'
        self.turn_prompt_fn = turn_prompt
        profile = backend.get('discussion_prompt_profile')
        if profile is not None:
            from src.online_rl.clean import doctor_prompt as compact
            if profile != compact.PROFILE:
                raise ValueError(f'Unknown Doctor discussion prompt profile: {profile}')
            self.turn_prompt_fn = compact.turn_prompt
        self.halting = halting or {'enabled': False, 'min_agent_turns': 1}
        self.min_agent_turns = int(self.halting.get('min_agent_turns', 1))
        if self.min_agent_turns < 1:
            raise ValueError('min_agent_turns must be at least 1')
        self.require_agent_doctor_agreement = self.halting.get('require_agent_doctor_agreement', False)
        if type(self.require_agent_doctor_agreement) is not bool:
            raise ValueError('require_agent_doctor_agreement must be boolean')
        self.stable_choice_no_objection_after_turns = self.halting.get('stable_choice_no_objection_after_turns')
        late = self.stable_choice_no_objection_after_turns
        if late is not None and (isinstance(late, bool) or not isinstance(late, int) or late < self.min_agent_turns):
            raise ValueError('stable_choice_no_objection_after_turns must be an integer >= min_agent_turns')
        self.options = {**backend['options'], 'max_output_tokens': max_tokens}
        self.final_response = backend.get('final_response')
        if self.final_response is not None:
            if (self.final_response.get('mode') != 'label_only_v1' or
                not isinstance(self.final_response.get('max_tokens'), int) or
                isinstance(self.final_response['max_tokens'], bool) or self.final_response['max_tokens'] < 16):
                raise ValueError('Doctor final_response needs label_only_v1 and an integer max_tokens >= 16')
        # Semantic identity excludes runtime placement.
        self.identity = {'engine': 'openai_direct', 'api': 'openai_responses',
                         'model': backend['model'], 'options': self.options,
                         'reasoning_effort': backend.get('reasoning_effort', 'none'),
                         'seed_policy': 'cache identity only; no provider seed',
                         'protocol_version': PROTOCOL_VERSION, 'system_sha256': sha256_text(system),
                         'turn_prompt_sha256': sha256_text(V2_OPENING + V2_TURN + FINAL_TURN + STATUS_INSTRUCTION),
                         'halting': self.halting}
        if profile is not None:
            self.identity.update(discussion_prompt_profile=profile, turn_prompt_sha256=compact.prompt_sha256())
            from src.online_rl.clean import halting as resolved_halting
            self.identity['halting_rule_sha256'] = resolved_halting.protocol_sha256()
        if self.final_response is not None:
            from src.collaboration.generator import FINAL_LABEL_ONLY_INSTRUCTION
            self.identity['final_response'] = {**self.final_response,
                'instruction_sha256':sha256_text(FINAL_LABEL_ONLY_INSTRUCTION)}
        if self.private_state_protocol:
            self.identity['private_state'] = {'protocol': self.private_state_protocol,
                                             'instruction_sha256': sha256_text(PRIVATE_STATE_INSTRUCTION)}
        if self.discussion_response_format:
            self.identity['structured_turn'] = {'protocol': self.discussion_response_format,
                'instruction_sha256': sha256_text(self.json_turn_instruction), 'format_error_retries': 1}
            if self.discussion_response_format == JSON_TURN_PROTOCOL:
                self.identity['structured_turn']['confidence_resolution'] = 0.01
        self.model = backend['model']
        self.protocol = sha256_json(self.identity)

    def halts(self) -> bool: return bool(self.halting.get('enabled'))

    def _chat(self, user, seed, max_tokens=None, system_override=None, format_schema=None):
        from src.common.api_slots import lease
        from src.common.usage import event
        import time
        from src.online_rl.tree.openai_direct import responses_json
        key = os.environ.get('OPENAI_API_KEY', '')
        if not key:
            from scripts.online_rl.run_tree import load_key
            key = load_key({'api_provider': 'openai_direct'})
        settings = {'max_tokens': self.max_tokens if max_tokens is None else max_tokens,
                    'temperature': self.options.get('temperature'),
                    'reasoning_effort': self.backend.get('reasoning_effort', 'none'),
                    'response_format': 'json_schema_strict' if format_schema is not None else None}
        started = time.perf_counter()
        try:
            with _DOCTOR_API_CAPACITY, lease(self.model, slot_name='doctor'):
                raw, _ = responses_json(key, self.model, self.system if system_override is None else system_override,
                                        user, settings, format_schema, 'doctor_response', 'doctor')
            event('api_call_completed', model=self.model, role='doctor', seconds=time.perf_counter() - started)
        except Exception as exc:
            event('api_call_failed', model=self.model, role='doctor', error_type=type(exc).__name__,
                  seconds=time.perf_counter() - started)
            raise
        return raw

    def verify(self, require_loaded=False):
        return {'problems': [], 'engine': 'openai_direct', 'model': self.model}


    def turn(self, case: dict, dialogue: list[dict], round_: int, rounds: int, seed: int) -> dict:
        if self.discussion_response_format:
            return self._structured_turn(case, dialogue, round_, rounds, seed)
        prompt = self.turn_prompt_fn(case, dialogue, round_, rounds, with_status=True)
        if self.private_state_protocol: prompt += PRIVATE_STATE_INSTRUCTION
        raw = self._chat(prompt, seed)
        private_state = None
        if self.private_state_protocol:
            try: private_state = split_private_state(raw, case['options'])
            except (ValueError, TypeError) as exc:
                error = DoctorUnavailable(f'Doctor private state invalid: {exc}')
                error.audit = {'case_id': case.get('case_id'), 'round': round_, 'seed': seed,
                               'raw': raw, 'options': case['options'], 'validation_error': str(exc)}
                raise error from exc
        visible, status, err = split_status(raw, case['options']) if round_ >= 2 else (raw.strip(), None, None)
        if round_ == 1: visible = split_status(raw, case['options'])[0]            # never let a stray status line through
        return {'raw': raw, 'visible': visible, 'status': status, 'status_error': err, 'seed': seed, 'private_state': private_state}

    def _structured_turn(self, case, dialogue, round_, rounds, seed):
        import jsonschema
        labels = [str(label) for label in case['options']] + ['none']
        def obj(properties):
            return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
        choice = {'type': 'string', 'enum': labels}
        confidence_schema = {'type': 'number', 'minimum': 0, 'maximum': 1}
        if self.discussion_response_format == JSON_TURN_PROTOCOL:
            # Enum constrains confidence to the configured resolution; do not clamp or rescale reported belief.
            confidence_schema['enum'] = [index / 100 for index in range(101)]
        private_schema = obj({'current_option': choice, 'hypothesis': {'type': 'string', 'minLength': 1},
                              'confidence': confidence_schema})
        status_schema = ({'type': 'null'} if round_ == 1 else obj({
            'current_option': choice, **{flag: {'type': 'boolean'} for flag in FLAGS}}))
        schema = obj({'reply': {'type': 'string', 'minLength': 1}, 'status': status_schema, 'private_state': private_schema})
        prompt = self.turn_prompt_fn(case, dialogue, round_, rounds, with_status=False) + self.json_turn_instruction
        attempts = []
        for attempt in range(2):
            actual_seed = seed if attempt == 0 else doctor_seed(seed, case['case_id'], 'doctor_turn_format_retry', attempt)
            raw = self._chat(prompt, actual_seed, format_schema=schema)
            try:
                parsed = json.loads(raw)
                jsonschema.validate(parsed, schema)
                visible = parsed['reply'].strip()
                if not visible or _PRIVATE_START.search(visible) or _STATUS_START.search(visible):
                    raise ValueError('Hidden markers or empty text inside visible reply')
                # Validation uses the model's supplied fields only, never a
                # reconstructed or fabricated belief. Only reply enters dialogue.
                private = split_private_state('PRIVATE_STATE: ' + json.dumps(parsed['private_state']), case['options'])
                status = parsed['status']
                if status is not None:
                    _, status, error = split_status('STATUS: ' + json.dumps(status), case['options'])
                    if error: raise ValueError(error)
                attempts.append({'seed': actual_seed, 'raw': raw, 'valid': True})
                return {'raw': raw, 'visible': visible, 'status': status, 'status_error': None,
                        'seed': actual_seed, 'private_state': private, 'format_attempts': attempts}
            except (ValueError, TypeError, jsonschema.ValidationError) as exc:
                attempts.append({'seed': actual_seed, 'raw': raw, 'valid': False, 'error': str(exc)})
        error = DoctorUnavailable('Doctor structured turn invalid after one format retry')
        error.audit = {'case_id': case.get('case_id'), 'round': round_, 'seed': seed,
                       'options': case['options'], 'attempts': attempts}
        raise error

    def final(self, case: dict, dialogue: list[dict], seed: int) -> str:
        from src.collaboration.generator import final_decision_prompt
        if self.final_response is not None:
            return self._chat(final_decision_prompt(case, dialogue, label_only=True), seed,
                              self.final_response['max_tokens'])
        return self._chat(final_decision_prompt(case, dialogue), seed)

    def halt(self, *, agent_turns: int, status, prev_status, last_agent: str, options) -> Optional[str]:
        if not self.halts(): return None
        if self.require_agent_doctor_agreement:
            from src.collaboration.generator import option
            agent_choice = option(last_agent or '')
            valid_choices = {str(label).upper() for label in options}
            if (agent_choice not in valid_choices or not isinstance(status, dict)
                    or agent_choice != status.get('current_option')):
                return None
        if self.backend.get('discussion_prompt_profile') == 'structured_clinician_v1':
            from src.online_rl.clean.halting import halt_reason as resolved_halt
            return resolved_halt(agent_turns=agent_turns, status=status, prev_status=prev_status, options=options,
                                 min_agent_turns=self.min_agent_turns)
        return halt_reason(agent_turns=agent_turns, status=status, prev_status=prev_status, last_agent=last_agent, options=options,
                           min_agent_turns=self.min_agent_turns,
                           stable_choice_no_objection_after_turns=self.stable_choice_no_objection_after_turns)

def make_doctor(rollout: dict, system: str):
    """Build the OpenAI Doctor shared by rollout, reruns and evaluation."""
    return OpenAIDoctor(rollout['doctor_backend'], rollout.get('halting'), system, rollout['max_tokens_doctor'])
