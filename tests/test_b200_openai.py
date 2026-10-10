"""Direct API transport tests: fake HTTP only, no credentials/network/GPU."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import threading
import time
import urllib.error

import pytest
from src.common import api_limiter, retry, usage
from src.common.llm_provider import load_key, model_matches
from src.online_rl.tree.b200_runner import load_config, preflight
from src.online_rl.tree.backends import _openrouter_json, generation_settings, api_rubric_generator, with_role
from src.online_rl.tree.parallel_bank import api_consolidator
from src.online_rl.tree.luna import LunaDecisionClient
from test_parallel_bank import Response
from test_member_intervention_bank import sources
from test_hf_doctor import local_model


def test_openai_only_credentials_and_hf_preflight(local_model, monkeypatch):
    monkeypatch.setenv('SHERPA_DOCTOR_BACKEND', 'hf_local')
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-openai-key')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setattr('src.online_rl.tree.b200_runner.physical_devices', lambda: ['0', '1'])
    assert load_key('openai') == 'fake-openai-key'
    errors = preflight(load_config(), resources=True)
    assert not any('OPENAI_API_KEY' in e or 'OPENROUTER' in e or 'OLLAMA' in e for e in errors)
    monkeypatch.delenv('OPENAI_API_KEY')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'wrong-provider-secret')
    errors = preflight(load_config(), resources=True)
    assert 'OPENAI_API_KEY is missing in this terminal' in errors
    assert 'wrong-provider-secret' not in repr(errors)


def test_four_roles_direct_wire_and_provider_identity(monkeypatch):
    sent = []
    def urlopen(req, timeout):
        payload = json.loads(req.data)
        sent.append((req.full_url, usage.current_stage(), payload))
        stage = usage.current_stage()
        content = ({'clusters': [], 'excluded': []} if stage == 'rubric_consolidate' else
                   {'name': 'Explain', 'applicability_condition': 'any turn', 'required_behavior': 'explain',
                    'scores': {'0': 'none', '1': 'partial', '2': 'full'}} if stage == 'rubric_generator' else
                   {'answers': {'q': 'yes'}})
        return Response({'id': 'fake-request', 'model': 'gpt-6-luna-2026-10-01',
                         'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}],
                         'usage': {'completion_tokens': 12}})
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    cfg = load_config()
    settings = {k: v for k, v in generation_settings(cfg['rubric']).items()
                if k in ('max_tokens', 'temperature', 'reasoning_effort', 'strict_provider_parameters', 'api_provider')}
    gen = api_rubric_generator('fake', cfg['rubric']['generator_model'], settings)
    gen('fixed', next(iter(sources()[2].values())))
    api_consolidator('fake', cfg['rubric']['consolidate'])([])
    for role in ('gate', 'scorer'):
        client = LunaDecisionClient(model=cfg['judges'][role+'_model'], api_provider=cfg['judges']['api_provider'], key='fake', system='fixed')
        with_role('rubric_'+role, lambda: client.decide('{}', {'q': {'criteria': {'yes': 'yes', 'no': 'no'}}}))
        old = LunaDecisionClient(model=client.model, key='fake', system='fixed')
        assert client.protocol_sha256 != old.protocol_sha256
    assert len(sent) == 4
    assert {s for _, s, _ in sent} == {'rubric_generator', 'rubric_consolidate', 'rubric_gate', 'rubric_scorer'}
    for url, role, payload in sent:
        assert url == 'https://api.openai.com/v1/chat/completions'
        assert payload['model'] == 'gpt-6-luna'
        assert 'max_completion_tokens' in payload
        assert not {'max_tokens', 'provider', 'usage', 'reasoning', 'models', 'temperature'} & payload.keys()
        if role in ('rubric_generator', 'rubric_consolidate'):
            assert payload['reasoning_effort'] == 'none'
        else:
            assert 'reasoning_effort' not in payload  # preserve judge default
            assert payload['response_format']['json_schema']['strict'] is True
    assert gen.settings['api_provider'] == 'openai'


@pytest.mark.parametrize('actual', ['other-model', 'gpt-6-luna-pro', None])
def test_no_model_substitution(actual):
    client = LunaDecisionClient(model='gpt-6-luna', api_provider='openai', key='fake', system='fixed',
        post=lambda *a: {'model': actual, 'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]})
    with pytest.raises(RuntimeError, match='no substitution'):
        client._chat([], {}, 'gate')
    assert model_matches('gpt-6-luna', 'gpt-6-luna-2026-10-01', 'openai')
    assert not model_matches('gpt-6-luna', 'gpt-6-luna-2026-10-01', 'openrouter')


def test_direct_global_limit_retries_and_wait_outside_slot(monkeypatch):
    monkeypatch.setattr(api_limiter, '_limiter', None)
    limiter = api_limiter.configure(3)
    attempts = {}; lock = threading.Lock(); waits = []
    def urlopen(req, timeout):
        assert req.full_url == 'https://api.openai.com/v1/chat/completions'
        payload = json.loads(req.data); name = payload['messages'][1]['content']
        with lock:
            attempts[name] = attempts.get(name, 0) + 1; n = attempts[name]
        time.sleep(.01)
        if n == 1:
            raise urllib.error.HTTPError(req.full_url, 429, 'busy', {'Retry-After': '0.01'}, io.BytesIO(b'busy'))
        return Response({'model': 'gpt-6-luna', 'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]})
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    original = retry.Retry.__init__
    def init(self, *a, **kw):
        original(self, *a, **kw); self.sleep = waits.append; self.log = lambda _: None
    monkeypatch.setattr(retry.Retry, '__init__', init)
    roles = ['rubric_generator', 'rubric_consolidate', 'rubric_gate', 'rubric_scorer']
    def one(i):
        _openrouter_json('fake', 'gpt-6-luna', 'fixed', str(i), {'api_provider': 'openai',
            'max_tokens': 100, 'temperature': None, 'reasoning_effort': 'none'}, {}, 'test', roles[i % 4])
    with ThreadPoolExecutor(12) as pool: list(pool.map(one, range(12)))
    assert limiter.peak == 3 and limiter.active == 0 and limiter.attempts == 24
    assert waits == [.01] * 12
    # Single request makes slot release during retry directly observable.
    r = retry.Retry('fake')
    r.sleep = lambda delay: (_ for _ in ()).throw(AssertionError('held slot')) if limiter.active else None
    r.wait('rate_limit', 'busy', '0')
    assert load_config()['execution']['global_api_max_in_flight'] == 50
