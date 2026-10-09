"""Hard-label decisions from an OpenRouter chat model (Luna): the same typed choice questions as the Jev decisions API,
answered with exactly one allowed option per question through strict structured output, without probabilities.
Output alignment only: trivial representation differences are canonicalised; a malformed, incomplete or out-of-enum
answer gets ONE repair request to the same model with the same input plus a schema correction; a second failure is an
operational error. Nothing is mapped to a nearest value or filled in."""
from __future__ import annotations
import hashlib, json, re, threading, time, urllib.error, urllib.request
from typing import Callable, Optional
from src.common import usage
from src.common.retry import RETRYABLE_STATUS, Retry, rate_limited
from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_text

URL = 'https://openrouter.ai/api/v1/chat/completions'
REQUEST_CACHE = 'canonical_request_sha256_v1'   # judges.scorer_request_cache: identical full requests share one sampled answer


def request_key(payload: dict) -> str:
    """sha256 of the canonical full request (model, messages, schema, every parameter). Nothing outside the payload, such as a
    node id, enters the key, so two responses' requests share an answer only when the scorer sees exactly the same input."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def http_post(url: str, payload: dict, key: str, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST',
                                 headers={'Authorization': f'Bearer {key.strip()}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())


def canonical(answers, allowed: dict[str, list[str]]) -> tuple[Optional[dict], Optional[str]]:
    """{qid: allowed option} or an error; accepts str/int values and case differences of single-letter options only."""
    if not isinstance(answers, dict): return None, 'answers is not an object'
    if set(answers) != set(allowed): return None, f'question ids differ: missing {sorted(set(allowed) - set(answers))}, extra {sorted(set(answers) - set(allowed))}'
    out = {}
    for qid, v in answers.items():
        s = str(v).strip(); opts = {o.upper(): o for o in allowed[qid]}
        if s.upper() not in opts: return None, f'{qid}: {v!r} is not one of {allowed[qid]}'
        out[qid] = opts[s.upper()]
    return out, None


def parse(raw: str, allowed: dict[str, list[str]]) -> tuple[Optional[dict], Optional[str]]:
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', (raw or '').strip())
    try: obj = json.loads(text)
    except json.JSONDecodeError as e: return None, f'not JSON: {e}'
    if isinstance(obj, dict) and set(obj) == {'answers'}: obj = obj['answers']
    return canonical(obj, allowed)


class LunaDecisionClient:
    """decide(state, questions, purpose) -> ({qid: {'value', 'probabilities': None, 'confidence': None}}, record), like JevClient."""

    def __init__(self, *, model: str, key: str, system: str, max_tokens: int = 2000, post: Callable = http_post,
                 sleep: Callable = time.sleep, timeout_s: float = 180.0, request_cache: Optional[str] = None):
        self.model, self.key, self.system, self.max_tokens, self.post, self.sleep = model, key, system, max_tokens, post, sleep
        self.timeout_s, self.protocol_sha256, self.log = timeout_s, sha256_text(system), []
        if request_cache not in (None, REQUEST_CACHE): raise ValueError(f'request_cache must be None or {REQUEST_CACHE!r}')
        # In-process, single-flight: concurrent identical requests wait for the first answer. Off by default (frozen runs).
        self.request_cache, self._answers, self._inflight, self._cache_lock = request_cache, {}, {}, threading.Lock()
        self.cache_hits = 0
        self._last = threading.local()                             # (request_sha256, cache_hit) of this thread's last _chat
        if request_cache: self.protocol_sha256 = sha256_text(f'{system}|{request_cache}')

    def _chat(self, messages: list[dict], schema: dict, purpose: str) -> str:
        payload = {'model': self.model, 'messages': messages, 'max_tokens': self.max_tokens,
                   'response_format': {'type': 'json_schema', 'json_schema': {'name': 'decisions', 'strict': True, 'schema': schema}}}
        if not self.request_cache: return self._request(payload, purpose)
        key = request_key(payload); self._last.info = (key, True)
        with self._cache_lock:
            if key in self._answers: self.cache_hits += 1; return self._answers[key]
            event = self._inflight.get(key)
            owner = event is None
            if owner: event = self._inflight[key] = threading.Event()
        if not owner:
            event.wait()
            with self._cache_lock:
                if key in self._answers: self.cache_hits += 1; return self._answers[key]
            return self._chat(messages, schema, purpose)          # the first request failed: try again ourselves
        try:
            self._last.info = (key, False)
            text = self._request(payload, purpose)
            with self._cache_lock: self._answers[key] = text
            return text
        finally:
            with self._cache_lock: self._inflight.pop(key, None)
            event.set()

    def _request_info(self) -> dict:
        """Audit fields of this thread's last request: its canonical sha256 and whether a cached answer was reused."""
        if not self.request_cache: return {}
        key, hit = getattr(self._last, 'info', (None, None))
        return {'request_sha256': key, 'request_cache_hit': hit}

    def _request(self, payload: dict, purpose: str) -> str:
        retry = Retry(f'OpenRouter {self.model} ({purpose})', sleep=self.sleep)        # src/common/retry.py: bounded, larger for 429
        started = time.monotonic(); request_seconds = 0.0; retry_wait_seconds = 0.0
        request_started = None
        while True:
            try:
                request_started = time.monotonic()
                body = self.post(URL, payload, self.key, self.timeout_s)
                request_seconds += time.monotonic() - request_started
                request_started = None
                usage.record(purpose, body.get('model', self.model), body.get('usage'), retry.attempts,
                             latency_seconds=time.monotonic()-started, request_seconds=request_seconds,
                             retry_wait_seconds=retry_wait_seconds); retry.succeeded()
                return body['choices'][0]['message'].get('content') or ''
            except urllib.error.HTTPError as e:
                if request_started is not None:
                    request_seconds += time.monotonic()-request_started;request_started=None
                text = e.read().decode('utf-8', errors='replace') if e.fp is not None else ''
                if e.code not in RETRYABLE_STATUS: raise
                wait_started=time.monotonic()
                should_retry=retry.wait('rate_limit' if rate_limited(e.code, text) else 'transient', f'HTTP {e.code} {text[:300]}',
                                        e.headers.get('Retry-After') if e.headers else None)
                retry_wait_seconds += time.monotonic()-wait_started
                if not should_retry: raise
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if request_started is not None:
                    request_seconds += time.monotonic()-request_started;request_started=None
                wait_started=time.monotonic();should_retry=retry.wait('transient', f'{type(e).__name__}: {str(e)[:200]}')
                retry_wait_seconds += time.monotonic()-wait_started
                if not should_retry: raise

    def decide(self, state: str, questions: dict[str, dict], purpose: str = 'luna') -> tuple[dict, dict]:
        allowed = {qid: [str(o) for o in q['criteria']] for qid, q in questions.items()}
        schema = {'type': 'object', 'additionalProperties': False, 'required': ['answers'],
                  'properties': {'answers': {'type': 'object', 'additionalProperties': False, 'required': list(allowed),
                                             'properties': {qid: {'type': 'string', 'enum': opts} for qid, opts in allowed.items()}}}}
        user = json.dumps({'state': json.loads(state) if isinstance(state, str) else state,
                           'questions': {qid: {'instructions': q.get('instructions'), 'criteria': q.get('criteria')} for qid, q in questions.items()}},
                          ensure_ascii=False)
        messages = [{'role': 'system', 'content': self.system}, {'role': 'user', 'content': user}]
        raw = [self._chat(messages, schema, purpose)]; parsed, err = parse(raw[0], allowed); errors = [err] if err else []
        if err:
            fix = (f'Your previous output was not valid: {err}. Return only JSON of the form {{"answers": {{question_id: option}}}} with exactly '
                   f'these question ids, each answered with one of its allowed options: {json.dumps(allowed)}.')
            raw.append(self._chat(messages + [{'role': 'assistant', 'content': raw[0]}, {'role': 'user', 'content': fix}], schema, purpose))
            parsed, err = parse(raw[1], allowed)
            if err: errors.append(err)
        rec = {'purpose': purpose, 'model': self.model, 'protocol_sha256': self.protocol_sha256, 'raw': raw, 'parsed': parsed,
               'repaired': len(raw) > 1, 'repair_ok': len(raw) > 1 and err is None, 'errors': errors}
        self.log.append(rec)
        if err or parsed is None: raise IntegrationUnavailable(f'{self.model} returned no valid {purpose} decision after one repair: {err}')
        return {qid: {'value': v, 'probabilities': None, 'confidence': None} for qid, v in parsed.items()}, rec

    def judge(self, stage: str, state: dict, items: dict[str, dict], fields: dict[str, dict], purpose: str) -> tuple[dict, dict]:
        """Structured decisions: one object per item id with exactly `fields` ({name: JSON-schema type}). Values are
        type-checked (booleans strict, enums exact after trivial case canonicalisation); a malformed answer gets ONE repair
        request to the same model, a second failure raises IntegrationUnavailable (operational, never a score)."""
        if not items: return {}, {}
        schema = {'type': 'object', 'additionalProperties': False, 'required': ['answers'],
                  'properties': {'answers': {'type': 'object', 'additionalProperties': False, 'required': list(items),
                                             'properties': {i: {'type': 'object', 'additionalProperties': False, 'required': list(fields),
                                                                'properties': fields} for i in items}}}}
        user = json.dumps({'stage': stage, 'state': state, 'items': items}, ensure_ascii=False)
        messages = [{'role': 'system', 'content': self.system}, {'role': 'user', 'content': user}]
        raw = [self._chat(messages, schema, purpose)]; info = [self._request_info()]
        parsed, err = parse_objects(raw[0], list(items), fields); errors = [err] if err else []
        if err:
            fix = (f'Your previous output was not valid: {err}. Return only JSON {{"answers": {{item_id: object}}}} with exactly these item ids '
                   f'{list(items)} and, in each object, exactly these fields: {json.dumps(fields)}.')
            raw.append(self._chat(messages + [{'role': 'assistant', 'content': raw[0]}, {'role': 'user', 'content': fix}], schema, purpose))
            info.append(self._request_info())
            parsed, err = parse_objects(raw[1], list(items), fields)
            if err: errors.append(err)
        rec = {'purpose': purpose, 'stage': stage, 'model': self.model, 'protocol_sha256': self.protocol_sha256, 'raw': raw, 'parsed': parsed,
               'repaired': len(raw) > 1, 'repair_ok': len(raw) > 1 and err is None, 'errors': errors,
               **({'requests': info} if self.request_cache else {})}
        self.log.append(rec)
        if err or parsed is None: raise IntegrationUnavailable(f'{self.model} returned no valid {purpose} decision after one repair: {err}')
        return parsed, rec


def parse_objects(raw: str, ids: list[str], fields: dict[str, dict]) -> tuple[Optional[dict], Optional[str]]:
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', (raw or '').strip())
    try: obj = json.loads(text)
    except json.JSONDecodeError as e: return None, f'not JSON: {e}'
    if isinstance(obj, dict) and set(obj) == {'answers'}: obj = obj['answers']
    if not isinstance(obj, dict) or set(obj) != set(ids): return None, f'item ids differ: {sorted(obj) if isinstance(obj, dict) else type(obj).__name__}'
    out = {}
    for i in ids:
        v = obj[i]
        if not isinstance(v, dict) or set(v) != set(fields): return None, f'{i}: fields {sorted(v) if isinstance(v, dict) else type(v).__name__}'
        o = {}
        for f, spec in fields.items():
            x = v[f]
            if spec.get('type') == 'boolean':
                if not isinstance(x, bool): return None, f'{i}.{f}: {x!r} is not a boolean'
            elif 'enum' in spec:
                opts = {str(e).upper(): e for e in spec['enum']}
                if str(x).strip().upper() not in opts: return None, f'{i}.{f}: {x!r} is not one of {spec["enum"]}'
                x = opts[str(x).strip().upper()]
            elif spec.get('type') == 'string' and not isinstance(x, str): return None, f'{i}.{f}: not a string'
            o[f] = x
        out[i] = o
    return out, None
