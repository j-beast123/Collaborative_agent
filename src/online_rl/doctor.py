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


def minimum_agent_turns(doctor) -> int:
    """Minimum for every early-stop path; unwrap the role-tagging Doctor adapter."""
    return int(getattr(getattr(doctor, 'doctor', doctor), 'min_agent_turns', 1))


def halt_reason(*, agent_turns: int, status: Optional[dict], prev_status: Optional[dict], last_agent: str, options,
                min_agent_turns: int, stable_choice_no_objection_after_turns: Optional[int] = None,
                default_final_after_agent_turns: Optional[int] = None) -> Optional[str]:
    """'doctor_halt' or None (continue). Called after a Doctor turn that follows `agent_turns` Agent replies."""
    from src.collaboration.generator import option
    if agent_turns < min_agent_turns or status is None: return None
    cur = status['current_option']
    if cur == 'none' or cur not in {str(o).upper() for o in options}: return None
    said = option(last_agent or '')
    if (default_final_after_agent_turns is not None and
            agent_turns >= default_final_after_agent_turns):
        if not all(status[k] for k in FLAGS):
            return f'doctor_halt_default_after_a{default_final_after_agent_turns}'
        return None
    if prev_status is None:
        if agent_turns == 1 and not any(status[k] for k in FLAGS) and (not said or said.upper() == cur):
            return 'doctor_halt_a1_settled'
        return None
    if prev_status.get('current_option') != cur: return None                       # the decision must be stable
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
        self.halting = halting or {'enabled': False, 'min_agent_turns': 2}
        self.min_agent_turns = int(self.halting.get('min_agent_turns', 2))
        self.stable_choice_no_objection_after_turns = self.halting.get('stable_choice_no_objection_after_turns')
        self.default_final_after_agent_turns = self.halting.get('default_final_after_agent_turns')
        late = self.stable_choice_no_objection_after_turns
        if late is not None and (isinstance(late, bool) or not isinstance(late, int) or late < self.min_agent_turns):
            raise ValueError('stable_choice_no_objection_after_turns must be an integer >= min_agent_turns')
        default_final = self.default_final_after_agent_turns
        if default_final is not None and (isinstance(default_final, bool) or not isinstance(default_final, int) or default_final < 1):
            raise ValueError('default_final_after_agent_turns must be an integer >= 1')
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
        if self.final_response is not None:
            from src.collaboration.generator import FINAL_LABEL_ONLY_INSTRUCTION
            self.identity['final_response'] = {**self.final_response,
                'instruction_sha256':sha256_text(FINAL_LABEL_ONLY_INSTRUCTION)}
        self.model = f"ollama/{backend['model']}@{backend['model_digest'][:12]}"
        self.protocol = sha256_json(self.identity)
        env = os.environ.get('SHERPA_DOCTOR_ENDPOINTS')              # runtime pool of identical workers (comma-separated)
        self.endpoints = [e.strip() for e in env.split(',') if e.strip()] if env else [backend['endpoint']]
        self._free: queue.Queue = queue.Queue()
        for e in self.endpoints: self._free.put(e)

    def halts(self) -> bool: return bool(self.halting.get('enabled'))

    def _chat(self, user: str, seed: int, max_tokens: Optional[int] = None, system_override: Optional[str] = None) -> str:
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
        raw = self._chat(turn_prompt(case, dialogue, round_, rounds, with_status=True), seed)
        visible, status, err = split_status(raw, case['options']) if round_ >= 2 else (raw.strip(), None, None)
        if round_ == 1: visible = split_status(raw, case['options'])[0]            # never let a stray status line through
        return {'raw': raw, 'visible': visible, 'status': status, 'status_error': err, 'seed': seed}

    def final(self, case: dict, dialogue: list[dict], seed: int) -> str:
        from src.collaboration.generator import final_decision_prompt
        if self.final_response is not None:
            return self._chat(final_decision_prompt(case, dialogue, label_only=True), seed,
                              self.final_response['max_tokens'])
        return self._chat(final_decision_prompt(case, dialogue), seed)

    def halt(self, *, agent_turns: int, status, prev_status, last_agent: str, options) -> Optional[str]:
        if not self.halts(): return None
        return halt_reason(agent_turns=agent_turns, status=status, prev_status=prev_status, last_agent=last_agent, options=options,
                           min_agent_turns=self.min_agent_turns,
                           stable_choice_no_objection_after_turns=self.stable_choice_no_objection_after_turns,
                           default_final_after_agent_turns=self.default_final_after_agent_turns)

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
