"""Rate limits abort prompt work without another transport attempt or sleep."""
import io
import urllib.error

import pytest

from src.common import retry
from src.collaboration import generator


@pytest.mark.parametrize('provider_error', [False, True])
def test_no_rate_limit_resend(monkeypatch, provider_error):
    monkeypatch.setenv('SHERPA_RATE_LIMIT_RETRIES', '0')
    calls = []
    def send(*args, **kwargs):
        calls.append(1)
        if provider_error:
            class Response(io.BytesIO):
                pass
            return Response(b'{"error":{"code":429,"message":"rate limit"}}')
        raise urllib.error.HTTPError('https://example.test', 429, 'rate limit', {}, io.BytesIO(b'{}'))
    monkeypatch.setattr(generator.urllib.request, 'urlopen', send)
    monkeypatch.setattr(retry.time, 'sleep', lambda _: pytest.fail('must not sleep'))
    with pytest.raises(retry.RetryExhausted):
        generator.doctor_call('fake', 'fake', 'system', 'user', 20)
    assert len(calls) == 1


def test_transient_retry_still_enabled(monkeypatch):
    monkeypatch.setenv('SHERPA_RATE_LIMIT_RETRIES', '0')
    waits = []
    state = retry.Retry('test', sleep=waits.append, log=lambda _: None)
    assert state.wait('transient', 'HTTP 503')
    assert len(waits) == 1 and state.attempts == 2


def test_openai_no_rate_limit_resend(monkeypatch):
    from src.online_rl.tree import openai_direct
    monkeypatch.setenv('SHERPA_RATE_LIMIT_RETRIES', '0')
    calls = []
    def send(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://example.test', 429, 'rate limit', {}, io.BytesIO(b'{}'))
    monkeypatch.setattr(openai_direct.urllib.request, 'urlopen', send)
    monkeypatch.setattr(retry.time, 'sleep', lambda _: pytest.fail('must not sleep'))
    with pytest.raises(retry.RetryExhausted):
        openai_direct.responses_json('fake', 'gpt-6-luna', 'system', 'user',
                                     {'max_tokens': 20}, None, 'test', 'doctor')
    assert len(calls) == 1


def test_execution_settings_preserve_identity_but_doctor_change_does_not(tmp_path):
    import copy
    from types import SimpleNamespace
    from src.online_rl.sherpa.prompt_search import base_identity
    data = tmp_path / 'data.jsonl'
    data.write_text('{}\n')
    cfg = {'policy': {'base_model': 'base'}, 'rollout': {'doctor_model': 'doctor1'},
           'schedule': {'seed': 0}, 'data': {'train': str(data)},
           'prompt_optimization': {'assistant_max_inflight': 16}}
    policy = SimpleNamespace(doctor=SimpleNamespace(protocol={'model': 'doctor1'}))
    before = base_identity(cfg, policy)
    changed = copy.deepcopy(cfg)
    changed['prompt_optimization'].update(assistant_max_inflight=128, rate_limit_retries=0)
    assert base_identity(changed, policy) == before
    changed['rollout']['doctor_model'] = 'doctor2'
    assert base_identity(changed, policy) != before
