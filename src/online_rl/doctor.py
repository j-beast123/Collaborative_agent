"""Local Doctor protocol v2: the Doctor served by Ollama (rollout.doctor_backend), the v2 discussion prompt, a hidden
per-turn status and adaptive halting (rollout.halting). One implementation for every Doctor-Agent conversation: the
training tree, intervention reruns, milestone / RQ evaluation episodes and the halting calibration.

- Visible dialogue: the v2 discussion prompt asks the Doctor to address the collaborator's latest reasoning and, when the
  discussion already supports a stable choice, to summarise the decisive reason instead of inventing another question.
- Hidden status: from Doctor turn 2 on, the Doctor appends one STATUS line (current option, open issue, unresolved
  collaborator objection, whether another exchange could change the choice). It is cut from the reply before the reply
  enters the dialogue, so the Agent, later Doctor turns, feedback, rubric generation, judges and evaluators never see it.
- Halting (halt_reason): at Doctor turn r >= min_agent_turns + 1 the branch ends ('doctor_halt') only if the status
  parses, no issue / objection / decision-changing exchange remains, the option is valid and unchanged since the previous
  Doctor turn, and the latest Agent reply does not name a different option. Anything else continues. T_max is unchanged;
  the normal FINAL prompt then records the answer. Halting sees only the visible dialogue state and the Doctor's status:
  never gold, outcome, rubrics, rewards, other branches or the arm.
- Optional stable_choice_no_objection_after_turns: from that Agent turn, stable valid choices with no collaborator
  objection or contrary Agent option may also halt while open_issue / more_info remain. Absent by default, recorded
  in the halting/protocol identity, and shared by training, reruns and evaluation.
- Seeds: every Doctor call gets derive_seed(...) of its call identity (doctor_seed); the seed is sent to Ollama and is
  part of the cache identity. Same seed rule for every arm. Results are reproducible only for the same Ollama version,
  model digest, options and GPU execution; the call cache remains the record."""
from __future__ import annotations
import json, os, queue, re, time, urllib.error, urllib.request
from typing import Callable, Optional
from src.common import usage
from src.common.retry import Retry
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
    """Per-call Doctor seed from the existing rule derive_seed(run_seed, key, role, index), in Ollama's int range."""
    return derive_seed(base_seed, key, role, index) % SEED_MOD


def http_json(url: str, payload: Optional[dict], timeout: float) -> dict:
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(), method='GET' if payload is None else 'POST',
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())


class DoctorUnavailable(RuntimeError):
    """The local Doctor endpoint failed after its retry budget, or a reply would exceed the pinned context. The run stops
    and resumes from its caches."""


class OllamaDoctor:
    """Doctor protocol v2 on a local Ollama server. turn(...) -> {'raw', 'visible', 'status', 'status_error', 'seed'};
    final(...) -> text. .model / .protocol identify it in every cache key; .identity goes to the run manifest."""
    status_protocol = True

    def __init__(self, backend: dict, halting: Optional[dict], system: str, max_tokens: int, http: Callable = http_json,
                 sleep: Optional[Callable] = None, timeout_s: float = 600.0):
        for k in ('engine', 'endpoint', 'model', 'model_digest', 'quantization', 'ollama_version', 'options'):
            if not backend.get(k): raise ValueError(f'rollout.doctor_backend.{k} is required')
        if backend['engine'] != 'ollama': raise ValueError(f"unsupported Doctor engine {backend['engine']!r}")
        self.backend, self.system, self.max_tokens, self.http, self.sleep, self.timeout_s = backend, system, max_tokens, http, sleep, timeout_s
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
        self.options = {**backend['options'], 'num_predict': max_tokens}
        self.final_response = backend.get('final_response')
        if self.final_response is not None:
            if (self.final_response.get('mode') != 'label_only_v1' or
                not isinstance(self.final_response.get('max_tokens'), int) or
                isinstance(self.final_response['max_tokens'], bool) or self.final_response['max_tokens'] < 16):
                raise ValueError('Doctor final_response needs label_only_v1 and an integer max_tokens >= 16')
        # semantic identity: no endpoint / GPU (runtime provenance only, `self.endpoints`); identical workers share one protocol
        self.identity = {'engine': 'ollama', 'api': 'ollama_native_chat', 'model': backend['model'],
                         'model_digest': backend['model_digest'], 'quantization': backend['quantization'],
                         'ollama_version': backend['ollama_version'], 'source_model': backend.get('source_model'), 'options': self.options,
                         'seed_policy': 'derive_seed(base, key, role, index) mod 2^31-1', 'protocol_version': PROTOCOL_VERSION,
                         'chat_format': 'single user turn: system + blank line + user (HF Gemma-3 template)',
                         'system_sha256': sha256_text(system), 'turn_prompt_sha256': sha256_text(V2_OPENING + V2_TURN + FINAL_TURN + STATUS_INSTRUCTION),
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
        self.model = f"ollama/{backend['model']}@{backend['model_digest'][:12]}"
        self.protocol = sha256_json(self.identity)
        env = os.environ.get('SHERPA_DOCTOR_ENDPOINTS')              # runtime pool of identical workers (comma-separated)
        self.endpoints = [e.strip() for e in env.split(',') if e.strip()] if env else list(backend.get('runtime_endpoints') or [backend['endpoint']])
        self._free: queue.Queue = queue.Queue()
        for e in self.endpoints: self._free.put(e)

    def halts(self) -> bool: return bool(self.halting.get('enabled'))

    def _chat(self, user: str, seed: int, max_tokens: Optional[int] = None, system_override: Optional[str] = None,
              format_schema: Optional[dict] = None) -> str:
        from src.common.live_gpu import activity
        if system_override is not None and (not isinstance(system_override, str) or not system_override.strip()):
            raise ValueError('Doctor system_override must be a non-empty string')
        system = self.system if system_override is None else system_override
        payload = {'model': self.backend['model'], 'stream': False, 'keep_alive': -1, 'options': {**self.options, 'seed': int(seed)},
                   # Gemma has no system role: the HF chat template folds the system prompt into the first user turn as
                   # system + blank line + user. Sending that single user message reproduces it (Ollama's own template would
                   # render a system message as a separate user turn).
                   'messages': [{'role': 'user', 'content': f'{system}\n\n{user}'}]}
        if max_tokens is not None:
            payload['options']['num_predict'] = max_tokens
        # Per-call only: a reflection's schema never persists into discussion or FINAL.
        if format_schema is not None:
            payload['format'] = format_schema
        retry = Retry(f"Ollama {self.backend['model']}", sleep=self.sleep)
        started = None; request_seconds = 0.0; retry_wait_seconds = 0.0
        request_started = None
        while True:
            try:
                with activity('doctor_queue_wait'):
                    endpoint = self._free.get()                              # one request per worker at a time
                try:
                    with activity('doctor_generate'):
                        if started is None: started=time.monotonic()
                        request_started=time.monotonic()
                        body = self.http(endpoint.rstrip('/') + '/api/chat', payload, self.timeout_s)
                        request_seconds += time.monotonic()-request_started
                        request_started=None
                finally: self._free.put(endpoint)
                break
            except urllib.error.HTTPError as e:
                if request_started is not None:
                    request_seconds += time.monotonic()-request_started;request_started=None
                text = e.read().decode('utf-8', errors='replace') if e.fp is not None else ''
                if e.code < 500:
                    raise DoctorUnavailable(f"local Doctor HTTP {e.code}: {text[:300]}") from e
                wait_started=time.monotonic();should_retry=retry.wait('transient', f'HTTP {e.code} {text[:300]}')
                retry_wait_seconds += time.monotonic()-wait_started
                if not should_retry: raise DoctorUnavailable(f"local Doctor HTTP {e.code}: {text[:300]}") from e
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                if request_started is not None:
                    request_seconds += time.monotonic()-request_started;request_started=None
                wait_started=time.monotonic();should_retry=retry.wait('transient', f'{type(e).__name__}: {str(e)[:200]}')
                retry_wait_seconds += time.monotonic()-wait_started
                if not should_retry:
                    raise DoctorUnavailable(f"local Doctor unreachable at {self.endpoints}: {e}") from e
        if body.get('error'): raise DoctorUnavailable(f"local Doctor error: {str(body['error'])[:300]}")
        n_in, n_out = body.get('prompt_eval_count') or 0, body.get('eval_count') or 0
        if n_in + n_out >= self.options['num_ctx']:
            raise DoctorUnavailable(f'Doctor context full ({n_in} + {n_out} >= num_ctx {self.options["num_ctx"]}): the reply may be truncated')
        text = ((body.get('message') or {}).get('content') or '').strip()
        if not text: raise DoctorUnavailable('local Doctor returned an empty reply')
        usage.record('collaboration', self.model, {'prompt_tokens': n_in, 'completion_tokens': n_out, 'cost': 0.0}, retry.attempts,
                     latency_seconds=time.monotonic()-(started or time.monotonic()), request_seconds=request_seconds,
                     retry_wait_seconds=retry_wait_seconds)
        retry.succeeded(); return text

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
            # Ollama grammar does not enforce numeric minimum/maximum. Enum
            # constrains decoding itself; do not clamp or rescale reported belief.
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

    def verify(self, require_loaded: bool = False) -> dict:
        """Every worker endpoint: server version and model digest must equal the pinned ones; with require_loaded, the model
        must be fully in VRAM."""
        problems, out = [], {'endpoints': {}}
        for base in (e.rstrip('/') for e in self.endpoints):
            ver = self.http(base + '/api/version', None, 30).get('version')
            m = {x['name']: x for x in self.http(base + '/api/tags', None, 30).get('models', [])}.get(self.backend['model'])
            p = []
            if ver != self.backend['ollama_version']: p.append(f"{base}: Ollama version {ver} != pinned {self.backend['ollama_version']}")
            if m is None: p.append(f"{base}: model {self.backend['model']} is not present")
            elif m.get('digest') != self.backend['model_digest']: p.append(f"{base}: model digest {m.get('digest')} != pinned {self.backend['model_digest']}")
            info = {'version': ver, 'digest': m and m.get('digest')}
            if require_loaded and not p:
                ps = {x['name']: x for x in self.http(base + '/api/ps', None, 30).get('models', [])}.get(self.backend['model'])
                if ps is None: p.append(f'{base}: model is not loaded')
                elif ps.get('size_vram') != ps.get('size'): p.append(f"{base}: model partly on CPU: size_vram {ps.get('size_vram')} of {ps.get('size')}")
                info['loaded'] = ps and {k: ps.get(k) for k in ('size', 'size_vram', 'context_length')}
            out['endpoints'][base] = info; problems += p
        first = next(iter(out['endpoints'].values()))
        return {**out, 'version': first['version'], 'digest': first['digest'], 'loaded': first.get('loaded'), 'problems': problems}


def make_doctor(rollout: dict, system: str):
    """The protocol-v2 Doctor for a config with rollout.doctor_backend, else None (legacy OpenRouter Doctor)."""
    b = rollout.get('doctor_backend')
    return OllamaDoctor(b, rollout.get('halting'), system, rollout['max_tokens_doctor']) if b else None


def wait_ready(doctor: OllamaDoctor, timeout_s: float = 120.0) -> dict:
    t0 = time.time()
    while True:
        try: return doctor.verify()
        except (urllib.error.URLError, OSError):
            if time.time() - t0 > timeout_s: raise
            time.sleep(2)
