import io
import json
import pytest
from src.collaboration import generator
from src.common import retry
from src.online_rl.tree.cache import CallCache


@pytest.mark.parametrize('code', [None, 429, 503, 400])
def test_choice_error_policy_and_persistent_cache(tmp_path, monkeypatch, code):
    monkeypatch.setenv('SHERPA_RATE_LIMIT_RETRIES', '0')
    calls, waits = [], []
    failed = {'id': 'request-test', 'choices': [{'finish_reason': 'error',
              'message': {'content': 'partial invalid answer'}}], 'usage': {'total_tokens': 0}}
    if code is not None:
        failed['choices'][0]['error'] = {'code': code, 'message': 'rate limit' if code == 429 else 'failure'}
    success = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'complete'}}]}
    def send(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(json.dumps(failed if len(calls) == 1 else success).encode())
    monkeypatch.setattr(generator.urllib.request, 'urlopen', send)
    monkeypatch.setattr(retry.time, 'sleep', waits.append)
    monkeypatch.setattr(generator.api_usage, 'record', lambda *a, **k: None)
    path = tmp_path / 'calls.jsonl'
    cache = CallCache(path)
    fn = lambda: generator.doctor_call('fake', 'model', 'system', 'user', 10)
    if code == 503:
        assert cache.call('rubric_set', {'case': 'one'}, fn) == 'complete'
        assert len(calls) == 2 and len(waits) == 1
    else:
        with pytest.raises(RuntimeError):
            cache.call('rubric_set', {'case': 'one'}, fn)
        assert len(calls) == 1 and not waits
        assert not cache.store
        if code != 429:  # the rate budget wrapper has its own diagnostic
            audit = json.loads(path.with_suffix('.errors.jsonl').read_text())['audit']
            assert audit['response_id'] == 'request-test'
        # Explicit restart retries only the failed, uncached call.
        resumed = CallCache(path)
        assert resumed.call('rubric_set', {'case': 'one'}, fn) == 'complete'
