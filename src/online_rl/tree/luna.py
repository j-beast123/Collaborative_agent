"""Hard-label decisions from an OpenRouter or direct OpenAI chat model (Luna): the same typed choice questions as the Jev decisions API,
answered with exactly one allowed option per question through strict structured output, without probabilities.
Output alignment only: trivial representation differences are canonicalised; a malformed, incomplete or out-of-enum
answer gets ONE repair request to the same model with the same input plus a schema correction; a second failure is an
operational error. Nothing is mapped to a nearest value or filled in."""
from __future__ import annotations
import json, re, time, urllib.error, urllib.request
from typing import Callable, Optional
from src.common import usage
from src.common.retry import RETRYABLE_STATUS, Retry, rate_limited
from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_json, sha256_text

URL = 'https://openrouter.ai/api/v1/chat/completions'
OPENAI_URL = 'https://api.openai.com/v1/chat/completions'
PROVIDERS = {'openrouter', 'openai_direct'}


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
                 sleep: Callable = time.sleep, timeout_s: float = 180.0, provider: str = 'openrouter'):
        if provider not in PROVIDERS: raise ValueError(f'Unknown Luna provider: {provider}')
        self.model, self.key, self.system, self.max_tokens, self.post, self.sleep = model, key, system, max_tokens, post, sleep
        self.provider = provider
        # The provider and wire contract are part of the judge/cache identity.
        # Keep the historical OpenRouter fingerprint byte-for-byte compatible.
        protocol = (sha256_text(system) if provider == 'openrouter' else
                    sha256_json({'system': system, 'provider': provider, 'api': 'chat_completions_v1'}))
        self.timeout_s, self.protocol_sha256, self.log = timeout_s, protocol, []

    def _chat(self, messages: list[dict], schema: dict, purpose: str, max_tokens: Optional[int] = None, *, return_body=False):
        direct = self.provider == 'openai_direct'
        payload = {'model': self.model, 'messages': messages,
                   'max_completion_tokens' if direct else 'max_tokens': max_tokens or self.max_tokens,
                   'response_format': {'type': 'json_schema', 'json_schema': {'name': 'decisions', 'strict': True, 'schema': schema}}}
        retry = Retry(f'{"OpenAI" if direct else "OpenRouter"} {self.model} ({purpose})', sleep=self.sleep)
        started = time.monotonic(); request_seconds = 0.0; retry_wait_seconds = 0.0
        request_started = None
        while True:
            try:
                request_started = time.monotonic()
                from src.common.api_slots import lease
                with lease(self.model):
                    body = self.post(OPENAI_URL if direct else URL, payload, self.key, self.timeout_s)
                request_seconds += time.monotonic() - request_started
                request_started = None
                usage.record(purpose, body.get('model', self.model), body.get('usage'), retry.attempts,
                             latency_seconds=time.monotonic()-started, request_seconds=request_seconds,
                             retry_wait_seconds=retry_wait_seconds); retry.succeeded()
                if direct:
                    try:
                        choice = body['choices'][0]
                        finish = choice['finish_reason']
                        message = choice['message']
                        if finish != 'stop' or message.get('refusal') or not message.get('content'):
                            raise ValueError(f'finish_reason={finish!r}, refusal={bool(message.get("refusal"))}, content={bool(message.get("content"))}')
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        error = IntegrationUnavailable(f'OpenAI {self.model} ({purpose}) returned an incomplete completion: {exc}')
                        error.audit = {'response': body, 'finish_reason': body.get('choices', [{}])[0].get('finish_reason') if isinstance(body.get('choices'), list) and body['choices'] else None,
                                       'usage': body.get('usage'), 'request_sha256': sha256_json(payload)}
                        raise error from exc
                return body if return_body else (body['choices'][0]['message'].get('content') or '')
            except urllib.error.HTTPError as e:
                if request_started is not None:
                    request_seconds += time.monotonic()-request_started;request_started=None
                text = e.read().decode('utf-8', errors='replace') if e.fp is not None else ''
                if self.key: text = text.replace(self.key, '[redacted]')
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

    def _answer(self, messages: list[dict], schema: dict, purpose: str, *, max_tokens: Optional[int] = None):
        """Return text and this exact response's metadata, safe under concurrent workers."""
        body = self._chat(messages, schema, purpose, max_tokens=max_tokens, return_body=True)
        choice = body['choices'][0]
        meta = {'finish_reason': choice.get('finish_reason'),
                'native_finish_reason': choice.get('native_finish_reason'),
                'usage': body.get('usage'), 'response_model': body.get('model', self.model),
                'response_id': body.get('id')}
        return choice['message'].get('content') or '', meta

    def decide(self, state: str, questions: dict[str, dict], purpose: str = 'luna') -> tuple[dict, dict]:
        allowed = {qid: [str(o) for o in q['criteria']] for qid, q in questions.items()}
        schema = {'type': 'object', 'additionalProperties': False, 'required': ['answers'],
                  'properties': {'answers': {'type': 'object', 'additionalProperties': False, 'required': list(allowed),
                                             'properties': {qid: {'type': 'string', 'enum': opts} for qid, opts in allowed.items()}}}}
        user = json.dumps({'state': json.loads(state) if isinstance(state, str) else state,
                           'questions': {qid: {'instructions': q.get('instructions'), 'criteria': q.get('criteria')} for qid, q in questions.items()}},
                          ensure_ascii=False)
        messages = [{'role': 'system', 'content': self.system}, {'role': 'user', 'content': user}]
        first, first_meta = self._answer(messages, schema, purpose)
        raw, completions = [first], [first_meta]
        parsed, err = parse(raw[0], allowed); errors = [err] if err else []
        rec = {'purpose': purpose, 'model': self.model, 'protocol_sha256': self.protocol_sha256, 'raw': raw, 'parsed': parsed,
               'repaired': len(raw) > 1, 'repair_ok': len(raw) > 1 and err is None, 'errors': errors}
        if self.provider == 'openai_direct': rec.update(provider=self.provider, completions=completions)
        self.log.append(rec)
        if err or parsed is None: raise IntegrationUnavailable(f'{self.model} returned no valid {purpose} decision; no retry: {err}')
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
        first, first_meta = self._answer(messages, schema, purpose)
        raw, completions = [first], [first_meta]
        parsed, err = parse_objects(raw[0], list(items), fields); errors = [err] if err else []
        rec = {'purpose': purpose, 'stage': stage, 'model': self.model, 'protocol_sha256': self.protocol_sha256, 'raw': raw, 'parsed': parsed,
               'repaired': len(raw) > 1, 'repair_ok': len(raw) > 1 and err is None, 'errors': errors}
        if self.provider == 'openai_direct': rec.update(provider=self.provider, completions=completions)
        self.log.append(rec)
        if err or parsed is None:
            error = IntegrationUnavailable(f'{self.model} returned no valid {purpose} decision; no retry: {err}')
            requests = [{'messages': messages, 'messages_sha256': sha256_json(messages)}]
            # Attach this call's evidence directly. Slicing the shared client log
            # can mix requests from concurrent rubric/member workers.
            error.audit = {'requests': requests, 'decisions': [rec], 'validation_errors': errors}
            raise error
        return parsed, rec

    def judge_batch(self, stage: str, shared_state: dict, responses: list[dict], items: dict[str, dict],
                    fields: dict[str, dict], purpose: str) -> tuple[dict, dict]:
        """Judge several sibling replies against one shared pre-reply state in one request.

        Each response carries its own exact item set. IDs and fields are validated
        independently so one reply cannot borrow another reply's score.
        """
        if not responses: return {}, {}
        response_ids = [row.get('response_id') for row in responses]
        if any(not isinstance(x, str) or not x for x in response_ids) or len(set(response_ids)) != len(response_ids):
            raise ValueError('Batch scorer requires unique non-empty response IDs')
        if not isinstance(items, dict) or not items:
            raise ValueError('Batch scorer needs a non-empty shared item set')
        if any(not isinstance(item, dict) or set(item.get('score_anchors') or {}) != {'0', '1', '2'}
               for item in items.values()):
            raise ValueError('Batch behavior items must have exactly the 0/1/2 score anchors')
        schema = {'type': 'object', 'additionalProperties': False, 'required': ['answers'],
                  'properties': {'answers': {'type': 'object', 'additionalProperties': False,
                      'required': response_ids, 'properties': {
                          response_id: {'type': 'object', 'additionalProperties': False,
                              'required': list(items), 'properties': {
                                  item_id: {'type': 'object', 'additionalProperties': False,
                                      'required': list(fields), 'properties': fields}
                                  for item_id in items}}
                          for response_id in response_ids}}}}
        packet = {'stage': stage, 'shared_pre_reply_state': shared_state,
                  'rubric_items': items,
                  'responses': [{'response_id': row['response_id'],
                                 'target_agent_reply': row['target_agent_reply']} for row in responses]}
        system = (self.system + '\nScore every response independently. The shared state is identical for all responses; '
                  'only the target reply changes. Never transfer evidence or a score between response IDs.')
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]
        # Stage-2 explanations add output tokens for every response/rubric pair.
        output_items = len(responses) * len(items)
        evidence_fields = {'minimum_contribution_quote', 'completion_quote', 'missing_or_conflicting'}
        per_item_tokens = 300 if evidence_fields <= set(fields) else 55
        token_budget = min(8192, max(self.max_tokens, 300 + per_item_tokens * output_items))
        def parse_batch(text):
            value = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', (text or '').strip()))
            if isinstance(value, dict) and set(value) == {'answers'}: value = value['answers']
            if not isinstance(value, dict) or set(value) != set(response_ids):
                raise ValueError('response IDs differ from requested batch')
            parsed = {}
            for response_id in response_ids:
                row = value[response_id]
                if not isinstance(row, dict) or set(row) != set(items):
                    raise ValueError(f'{response_id}: rubric IDs differ from requested batch')
                parsed[response_id] = {}
                for item_id, answer in row.items():
                    if not isinstance(answer, dict) or set(answer) != set(fields):
                        raise ValueError(f'{response_id}/{item_id}: fields differ from schema')
                    checked = {}
                    for name, spec in fields.items():
                        item = answer[name]
                        if spec.get('type') == 'boolean' and not isinstance(item, bool):
                            raise ValueError(f'{response_id}/{item_id}.{name}: expected boolean')
                        if 'enum' in spec:
                            allowed = {str(x).upper(): x for x in spec['enum']}
                            key = str(item).strip().upper()
                            if key not in allowed: raise ValueError(f'{response_id}/{item_id}.{name}: invalid enum')
                            item = allowed[key]
                        if spec.get('type') == 'string' and not isinstance(item, str):
                            raise ValueError(f'{response_id}/{item_id}.{name}: expected string')
                        checked[name] = item
                    parsed[response_id][item_id] = checked
            return parsed
        first, first_meta = self._answer(messages, schema, purpose, max_tokens=token_budget)
        raw, completions = [first], [first_meta]
        errors = []
        try: parsed = parse_batch(raw[0])
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(str(exc)); parsed = None
        if parsed is None:
            fix = ('Return exactly one JSON object shaped as {"answers": {response_id: {rubric_id: {field: value}}}}. '
                   'Include every requested response ID, rubric ID and field, with no extras. '
                   f'Previous output error: {errors[-1]}.')
            repaired, repaired_meta = self._answer(messages + [{'role': 'assistant', 'content': raw[0]},
                {'role': 'user', 'content': fix}], schema, purpose, max_tokens=token_budget)
            raw.append(repaired); completions.append(repaired_meta)
            try: parsed = parse_batch(raw[1])
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append(str(exc)); parsed = None
        rec = {'purpose': purpose, 'stage': stage, 'model': self.model,
               'protocol_sha256': self.protocol_sha256, 'response_ids': response_ids,
               'rubrics_per_response': {rid: len(items) for rid in response_ids},
               'max_tokens': token_budget, 'raw': raw, 'parsed': parsed,
               'repaired': len(raw) > 1, 'errors': errors}
        if self.provider == 'openai_direct': rec.update(provider=self.provider, completions=completions)
        self.log.append(rec)
        if parsed is None: raise IntegrationUnavailable(f'{self.model} returned invalid batch scoring output: {errors[-1]}')
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
                # Preserve literal string enums (e.g. evidence fields including
                # whitespace or case-distinct values) before choice canonicalisation.
                if not (isinstance(x, str) and x in spec['enum']):
                    opts = {str(e).upper(): e for e in spec['enum']}
                    if str(x).strip().upper() not in opts: return None, f'{i}.{f}: {x!r} is not one of {spec["enum"]}'
                    x = opts[str(x).strip().upper()]
            elif spec.get('type') == 'string' and not isinstance(x, str): return None, f'{i}.{f}: not a string'
            o[f] = x
        out[i] = o
    return out, None
