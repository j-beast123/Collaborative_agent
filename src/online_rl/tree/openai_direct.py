"""Direct OpenAI Responses transport for rubric generation and matching.

The caller owns the rubric contract and local validation. This transport records
the exact API payload and response without recording the Authorization header.
"""
from __future__ import annotations

import copy
import json
import re
import time
import urllib.error
import urllib.request

from src.common import retry as _retry
from src.common import usage as api_usage


RESPONSES_URL = 'https://api.openai.com/v1/responses'


def _provider_schema(schema: dict) -> dict:
    """Adapt the local schema to OpenAI's supported strict schema subset.

    The unmodified pinned schema remains the authority for local validation.
    In particular, array uniqueness is enforced locally; ``uniqueItems`` is
    not accepted by the provider's strict Structured Outputs schema.
    """
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: convert(item) for key, item in value.items()
                  if key not in ('$schema', '$id', 'title', 'uniqueItems')}
        if 'const' in result:
            result['enum'] = [result.pop('const')]
        return result
    return convert(copy.deepcopy(schema))


def _http_error_details(exc) -> dict:
    """Keep bounded error identifiers, never the potentially echoed message."""
    try:
        body = json.loads(exc.read(65536).decode('utf-8'))
        detail = body.get('error') if isinstance(body, dict) else None
        if not isinstance(detail, dict):
            return {}
        return {field: value for field in ('type', 'code', 'param')
                if isinstance(value := detail.get(field), str)
                and re.fullmatch(r'[A-Za-z0-9_.\[\]-]{1,120}', value)}
    except (ValueError, OSError, UnicodeError):
        return {}


def _output_text(body: dict) -> str:
    parts = []
    for item in body.get('output') or []:
        if item.get('type') != 'message':
            continue
        for content in item.get('content') or []:
            if content.get('type') == 'refusal':
                raise ValueError('OpenAI response refused the request')
            if content.get('type') == 'output_text':
                parts.append(content.get('text') or '')
    if not parts or not ''.join(parts).strip():
        raise ValueError('OpenAI response contained no output text')
    return ''.join(parts).strip()


def responses_json(key: str, model: str, system: str, user: str, settings: dict,
                   schema: dict | None, name: str, role: str) -> tuple[str, dict]:
    """Return raw output text and an auditable, secret-free request/response envelope.

    For reasoning effort other than ``none``, OpenAI does not accept temperature.
    The legacy setting is retained in the caller's cache identity; the effective
    request is recorded here. Any incomplete/refused response is operational.
    """
    from src.online_rl.tree.backends import RubricOutputIncomplete

    if not key or not key.strip():
        raise RuntimeError('OPENAI_API_KEY is missing for direct rubric API calls')
    if model not in ('gpt-6-luna', 'gpt-4o-mini'):
        raise ValueError('Direct API requires gpt-6-luna or gpt-4o-mini')
    effort = settings.get('reasoning_effort')
    non_reasoning = model == 'gpt-4o-mini'
    if non_reasoning and effort not in (None, 'none'):
        raise ValueError('gpt-4o-mini does not support reasoning_effort')
    if effort not in (None, 'none', 'low', 'medium', 'high', 'xhigh', 'max'):
        raise ValueError(f'Unsupported gpt-6-luna reasoning effort: {effort}')
    payload = {
        'model': model,
        'input': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        'max_output_tokens': settings['max_tokens'],
        'store': False,
    }
    if effort is not None and not non_reasoning:
        payload['reasoning'] = {'effort': effort}
    if (non_reasoning or effort == 'none') and settings.get('temperature') is not None:
        payload['temperature'] = settings['temperature']
    if settings.get('response_format') == 'json_schema_strict':
        payload['text'] = {'format': {'type': 'json_schema', 'name': name,
                                      'strict': True, 'schema': _provider_schema(schema)}}
    elif schema is not None:
        # The prompt itself asks for JSON. JSON mode preserves the legacy
        # locally-validated response shape when a provider schema is not pinned.
        payload['text'] = {'format': {'type': 'json_object'}}
    request = urllib.request.Request(
        RESPONSES_URL, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + key.strip(), 'Content-Type': 'application/json'},
        method='POST')
    retry = _retry.Retry(f'OpenAI {model} {role}')
    started = time.perf_counter()
    while True:
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode('utf-8'))
            if not isinstance(body, dict):
                raise ValueError('OpenAI returned a non-object response')
            usage = body.get('usage') or {}
            ledger_usage = {'prompt_tokens': usage.get('input_tokens'),
                            'completion_tokens': usage.get('output_tokens')}
            api_usage.record('collaboration', body.get('model', model), ledger_usage, retry.attempts,
                             latency_seconds=time.perf_counter() - started,
                             retry_wait_seconds=retry.total_wait_s)
            meta = {'api_provider': 'openai_direct', 'endpoint': RESPONSES_URL,
                    'request': payload, 'response': body,
                    'response_id': body.get('id'), 'model': body.get('model'),
                    'status': body.get('status'), 'usage': usage,
                    'finish_reason': ('stop' if body.get('status') == 'completed' else
                                      (body.get('incomplete_details') or {}).get('reason') or body.get('status')),
                    'native_finish_reason': (body.get('incomplete_details') or {}).get('reason')}
            if body.get('status') != 'completed':
                error = RubricOutputIncomplete(
                    f'{model} response status={body.get("status")}, reason={meta["finish_reason"]} before a complete {name}')
                error.audit = meta
                raise error
            try:
                raw = _output_text(body)
            except ValueError as exc:
                error = RubricOutputIncomplete(f'{model} did not return a complete {name}: {exc}')
                error.audit = meta
                raise error from exc
            retry.succeeded()
            return raw, meta
        except urllib.error.HTTPError as exc:
            # Error messages may echo inputs or credentials. Retain only
            # bounded identifiers; never the raw response or Authorization.
            if exc.code not in _retry.RETRYABLE_STATUS:
                details = _http_error_details(exc)
                suffix = ' ' + json.dumps(details, ensure_ascii=False) if details else ''
                error = RuntimeError(f'OpenAI {model} {role}: HTTP {exc.code}{suffix}')
                error.audit = {'api_provider': 'openai_direct', 'endpoint': RESPONSES_URL,
                               'request': payload, 'http_status': exc.code,
                               'error_details': details}
                raise error from exc
            kind = 'rate_limit' if exc.code == 429 else 'transient'
            if not retry.wait(kind, f'HTTP {exc.code}', exc.headers.get('Retry-After')):
                raise RuntimeError(f'OpenAI {model} {role}: HTTP {exc.code}') from exc
        except (urllib.error.URLError, OSError) as exc:
            if not retry.wait('transient', type(exc).__name__):
                raise RuntimeError(f'OpenAI {model} {role}: network error ({type(exc).__name__})') from exc
