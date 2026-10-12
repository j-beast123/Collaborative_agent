"""Scheduling/adapter lifecycle tests with no API calls or model weights."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from src.online_rl import agent_batch
from src.online_rl.tree.base_evaluation import original_base_policy
from src.online_rl.sherpa import prompt_search


def test_two_batch_workers_share_queue_and_use_private_tokenizers(monkeypatch):
    barrier = threading.Barrier(2)
    calls = []
    class Model:
        def __init__(self, index): self.index = index
        def eval(self): pass
    models = [Model(0), Model(1)]
    policy = SimpleNamespace(model=models[0], models=models, tok=SimpleNamespace(padding_side='right'),
                             args=SimpleNamespace(generation_temperature=1., top_p=1., top_k=0),
                             _rlocks=[threading.Lock(), threading.Lock()])
    def generate(model, tokenizer, prompts, seeds, *args):
        calls.append((model.index, id(tokenizer)))
        barrier.wait(timeout=5)
        return [{'seed': seed} for seed in seeds]
    monkeypatch.setattr(agent_batch, 'generate_batch', generate)
    batcher = agent_batch.Batcher(policy, 1)
    try:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda seed: batcher.submit('prompt', 4, seed), [10, 20]))
        assert results == [{'seed': 10}, {'seed': 20}]
        assert {index for index, _ in calls} == {0, 1}
        assert len({tokenizer for _, tokenizer in calls}) == 2
    finally:
        batcher.close()
    assert not any(thread.is_alive() for thread in batcher.threads)


def test_original_base_disables_every_replica_and_restores_after_failure():
    class Model:
        def __init__(self):
            self.training = True
            self.config = SimpleNamespace(use_cache=False)
            self.disabled = False
        def eval(self): self.training = False
        def train(self, mode): self.training = mode
        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try: yield
            finally: self.disabled = False
    models = [Model(), Model()]
    torch = SimpleNamespace(get_rng_state=lambda: None, set_rng_state=lambda state: None,
                            cuda=SimpleNamespace(is_available=lambda: False))
    policy = SimpleNamespace(torch=torch, model=models[0], models=models, precommit=True)
    with pytest.raises(RuntimeError, match='injected'):
        with original_base_policy(policy):
            assert all(m.disabled and not m.training and m.config.use_cache for m in models)
            raise RuntimeError('injected')
    assert all(not m.disabled and m.training and not m.config.use_cache for m in models)
    assert policy.precommit is True


def test_assistant_limit_covers_all_gpu_queues_and_releases_on_error(monkeypatch):
    guard = threading.Lock()
    full = threading.Event()
    release = threading.Event()
    counts = {'current': 0, 'peak': 0}
    class CountedQueue(queue.Queue):
        def put(self, item, *args, **kwargs):
            if item is not None:
                with guard:
                    counts['current'] += 1
                    counts['peak'] = max(counts['peak'], counts['current'])
                    if counts['current'] == 4: full.set()
                def completed(future):
                    with guard: counts['current'] -= 1
                item[3].add_done_callback(completed)
            return super().put(item, *args, **kwargs)
    class Model:
        def eval(self): pass
    models = [Model(), Model()]
    policy = SimpleNamespace(model=models[0], models=models, tok=SimpleNamespace(),
                             args=SimpleNamespace(generation_temperature=1., top_p=1., top_k=0),
                             _rlocks=[threading.Lock(), threading.Lock()])
    monkeypatch.setattr(agent_batch.queue, 'Queue', CountedQueue)
    def generate(model, tokenizer, prompts, seeds, *args):
        assert release.wait(5)
        if 0 in seeds: raise RuntimeError('injected batch failure')
        return [{'seed': seed} for seed in seeds]
    monkeypatch.setattr(agent_batch, 'generate_batch', generate)
    batcher = agent_batch.Batcher(policy, 2, max_inflight=4)
    try:
        with ThreadPoolExecutor(16) as pool:
            futures = [pool.submit(batcher.submit, 'prompt', 4, seed) for seed in range(16)]
            try:
                assert full.wait(5)
                assert counts['current'] == 4
            finally:
                release.set()
            for future in futures:
                try: future.result(timeout=5)
                except RuntimeError as exc: assert 'injected' in str(exc)
        assert counts == {'current': 0, 'peak': 4}
        assert batcher.submit('prompt', 4, 99) == {'seed': 99}
    finally:
        release.set()
        batcher.close()


def test_base_cases_overlap_and_resume_without_regeneration(tmp_path, monkeypatch):
    cases = [{'case_id': str(i)} for i in range(120)]
    monkeypatch.setattr(prompt_search, 'base_identity', lambda *args: {'base': 'fixed'})
    monkeypatch.setattr(prompt_search, 'schedule', lambda *args: [cases[i:i+4] for i in range(0, 120, 4)])
    monkeypatch.setattr(prompt_search, 'original_base_policy', lambda policy: nullcontext())
    calls = []
    barrier = threading.Barrier(4)
    def collect(case, *args, **kwargs):
        calls.append(case['case_id'])
        barrier.wait(timeout=5)
        return None, [{'outcome': i % 2} for i in range(32)]
    monkeypatch.setattr(prompt_search, 'collect', collect)
    cfg = {'prompt_optimization': {'output_dir': 'search', 'rollout_case_workers': 4}}
    first = prompt_search.collect_base(cfg, None, cases, tmp_path, max_cases=4)
    assert first['completed_cases'] == 4 and not first['complete']
    result = prompt_search.collect_base(cfg, None, cases, tmp_path)
    assert result['completed_cases'] == 120 and result['complete']
    assert len(calls) == len(set(calls)) == 120
    assert json.loads((tmp_path / 'search/base_manifest.json').read_text()) == result
