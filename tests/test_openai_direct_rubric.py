"""Provider boundary checks use synthetic text and never spend API tokens."""
import json
import io
import urllib.error
from types import SimpleNamespace

import pytest

from src.online_rl.tree import backends as B
from src.online_rl.tree import openai_direct as O


class _HTTPResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self):
        return json.dumps(self.value).encode()


def _body(text='{"name":"N"}', status='completed'):
    return {'id': 'resp_test', 'model': 'gpt-6-luna', 'status': status,
            'incomplete_details': {'reason': 'max_output_tokens'} if status == 'incomplete' else None,
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}],
            'usage': {'input_tokens': 12, 'output_tokens': 8}}


def test_direct_request_preserves_envelope_and_omits_reasoning_temperature(monkeypatch):
    calls, ledger = [], []
    def urlopen(request, timeout):
        calls.append((request, timeout))
        return _HTTPResponse(_body())
    monkeypatch.setattr(O.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(O.api_usage, 'record', lambda *args: ledger.append(args))
    schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema',
              'type': 'object', 'additionalProperties': False, 'required': ['name'],
              'properties': {'name': {'type': 'string', 'const': 'N'}}}
    settings = {'max_tokens': 256, 'temperature': 0.7, 'reasoning_effort': 'low',
                'response_format': 'json_schema_strict'}
    raw, meta = O.responses_json('top-secret', 'gpt-6-luna', 'Return JSON', 'Synthetic',
                                  settings, schema, 'rubric', 'rubric_generator')
    request, timeout = calls[0]
    sent = json.loads(request.data)
    assert request.full_url == O.RESPONSES_URL and timeout == 180
    assert sent['model'] == 'gpt-6-luna' and sent['store'] is False
    assert sent['reasoning'] == {'effort': 'low'} and 'temperature' not in sent
    assert sent['max_output_tokens'] == 256
    assert sent['text']['format']['schema']['properties']['name']['enum'] == ['N']
    assert '$schema' not in sent['text']['format']['schema']
    assert raw == '{"name":"N"}'
    assert meta['request'] == sent and meta['response'] == _body()
    assert meta['finish_reason'] == 'stop' and meta['usage']['output_tokens'] == 8
    assert 'top-secret' not in json.dumps(meta)
    assert ledger[0][2]['prompt_tokens'] == 12 and ledger[0][2]['completion_tokens'] == 8


def test_incomplete_direct_response_is_operational_with_raw_audit(monkeypatch):
    body = _body(status='incomplete')
    monkeypatch.setattr(O.urllib.request, 'urlopen', lambda *_args, **_kw: _HTTPResponse(body))
    monkeypatch.setattr(O.api_usage, 'record', lambda *_args: None)
    with pytest.raises(B.RubricOutputIncomplete, match='max_output_tokens') as caught:
        O.responses_json('secret', 'gpt-6-luna', 'Return JSON', 'Synthetic',
                         {'max_tokens': 16, 'reasoning_effort': 'none'},
                         {'type': 'object'}, 'rubric', 'rubric_generator')
    assert caught.value.audit['response'] == body
    assert caught.value.audit['finish_reason'] == 'max_output_tokens'
    assert 'secret' not in json.dumps(caught.value.audit)


def test_generator_routes_direct_and_records_api_response(monkeypatch):
    calls = []
    rubric = {'name': 'Context', 'applicability_condition': 'Before reply',
              'required_behavior': 'Explain uncertainty',
              'scores': {'0': 'No explanation', '1': 'Partial', '2': 'Complete'}}
    def direct(*args):
        calls.append(args)
        return json.dumps(rubric), {'api_provider': 'openai_direct',
                                    'request': {'model': 'gpt-6-luna'},
                                    'response': _body(json.dumps(rubric)),
                                    'usage': {'input_tokens': 3, 'output_tokens': 5}}
    monkeypatch.setattr(O, 'responses_json', direct)
    monkeypatch.setattr(B, '_openrouter_json', lambda *_: pytest.fail('OpenRouter fallback'))
    monkeypatch.setattr('src.online_rl.tree.posthoc.trajectory_message', lambda _: 'Synthetic JSON input')
    settings = {'max_tokens': 100, 'temperature': 0.7, 'reasoning_effort': 'low',
                'response_format': 'json_schema_strict', 'api_provider': 'openai_direct'}
    result = B.api_rubric_generator('secret', 'gpt-6-luna', settings)('Prompt', SimpleNamespace())
    assert result['rubric'] == rubric
    assert result['api_attempts'][0]['response']['id'] == 'resp_test'
    assert calls[0][1] == 'gpt-6-luna' and calls[0][-1] == 'rubric_generator'


def test_generation_settings_provider_is_explicit_and_validated():
    assert B.generation_settings({'generation': {'api_provider': 'openai_direct'}})['api_provider'] == 'openai_direct'
    assert 'api_provider' not in B.generation_settings({'generation': {}})
    with pytest.raises(ValueError, match='provider'):
        B.generation_settings({'generation': {'api_provider': 'unknown'}})


def test_auth_failure_reports_status_without_secret_or_provider_fallback(monkeypatch):
    def unauthorized(*_args, **_kwargs):
        raise urllib.error.HTTPError(O.RESPONSES_URL, 401, 'unauthorized', {}, io.BytesIO(b'{"error":"bad secret"}'))
    monkeypatch.setattr(O.urllib.request, 'urlopen', unauthorized)
    monkeypatch.setattr(B, '_openrouter_json', lambda *_: pytest.fail('OpenRouter fallback'))
    with pytest.raises(RuntimeError, match='HTTP 401') as caught:
        B._generator_json('top-secret', 'gpt-6-luna', 'Return JSON', 'Synthetic JSON',
                          {'max_tokens': 32, 'api_provider': 'openai_direct'},
                          {'type': 'object'}, 'rubric', 'rubric_generator')
    assert 'top-secret' not in str(caught.value)
    assert 'top-secret' not in json.dumps(caught.value.audit)
