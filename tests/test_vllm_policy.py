import io
import json
import threading
from types import SimpleNamespace

import pytest

from src.online_rl.vllm_policy import VLLMPolicy, server_command
from src.online_rl.sherpa.rollouts import base_cache_policy_hash
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.base_evaluation import original_base_policy


def test_legacy_cache_namespace_resumes_without_new_calls(tmp_path):
    path = tmp_path / 'calls.jsonl'
    identity = {'case': 'case', 'protocol': 'protocol', 'policy': 'legacy-disabled-adapter', 'path': '0'}
    cache = CallCache(path)
    cache.call('agent', identity, lambda: {'completion_text': 'already generated'})
    resumed = CallCache(path)
    policy_hash = base_cache_policy_hash(resumed, 'case', 'protocol', 'base')
    assert policy_hash == identity['policy']
    assert resumed.call('agent', {**identity, 'policy': policy_hash},
                        lambda: pytest.fail('must reuse cached response'))['completion_text'] == 'already generated'
    with pytest.raises(RuntimeError, match='mismatch'):
        base_cache_policy_hash(resumed, 'case', 'changed-doctor-protocol', 'base')


def test_base_engine_has_no_adapter_or_training_model():
    with original_base_policy(SimpleNamespace(original_base_only=True)):
        pass


def test_server_settings():
    cfg = json.load(open('configs/sherpa.json'))['prompt_optimization']
    command = server_command('python', '/weights', 12345, cfg)
    for flag, expected in [('--gpu-memory-utilization', '0.9'), ('--max-num-seqs', '128'),
                           ('--host', '127.0.0.1'), ('--tensor-parallel-size', '1'),
                           ('--generation-config', 'vllm')]:
        assert command[command.index(flag) + 1] == expected


def fake_policy():
    policy = VLLMPolicy.__new__(VLLMPolicy)
    policy.cfg = {'rollout': {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}}
    policy.tok = SimpleNamespace(encode=lambda *a, **k: [1, 2])
    policy.stop_token_ids = [151645, 151643]
    policy.limit = threading.BoundedSemaphore(128)
    policy.guard = threading.Lock()
    policy.loads = [0, 0]
    policy.urls = ['http://localhost:1', 'http://localhost:2']
    return policy


def test_payload_tokens_and_failure_releases_capacity(monkeypatch):
    import src.online_rl.vllm_policy as module
    policy = fake_policy()
    requests = []
    def send(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(b'{"choices":[{"text":"ok","token_ids":[8,9]}]}')
    monkeypatch.setattr(module.urllib.request, 'urlopen', send)
    out = policy._generate('prompt', 10, 123)
    assert out['completion_ids'] == [8, 9]
    assert out['batch_sampling_protocol'] == 'vllm_seeded_v1'
    assert requests[0]['prompt'] == [1, 2] and requests[0]['seed'] == 123
    assert requests[0]['top_k'] == -1
    assert requests[0]['stop_token_ids'] == [151645, 151643]
    assert policy.loads == [0, 0]
    def fail(*args, **kwargs):
        raise OSError('injected')
    monkeypatch.setattr(module.urllib.request, 'urlopen', fail)
    with pytest.raises(OSError):
        policy._generate('prompt', 10, 124)
    assert policy.loads == [0, 0]
    acquired = [policy.limit.acquire(blocking=False) for _ in range(128)]
    assert all(acquired)
    for _ in acquired:
        policy.limit.release()
