"""The dedicated actor must dispatch Assistant4 and Doctor8 at the same time."""
from queue import Queue
from threading import Barrier, RLock
from types import SimpleNamespace
import shutil

from src.online_rl.tree import gpu_pool


def test_assistant_requests_are_not_blocked_by_eight_doctor_dispatches(monkeypatch, tmp_path):
    manager = SimpleNamespace(Queue=Queue, RLock=RLock, dict=dict)
    spec = gpu_pool.create_pool(manager, [0], agent_capacity=4, doctor_capacity=8)
    monkeypatch.setattr(gpu_pool, 'Listener', lambda *args, **kwargs: None)
    monkeypatch.setattr(gpu_pool.Actor, '_listen', lambda self: None)
    actor = gpu_pool.Actor(spec, 0, {}, tmp_path)
    try:
        assert spec['agent'].qsize() == 4 and spec['doctor'].qsize() == 8
        barrier = Barrier(12)
        futures = [actor.executor.submit(barrier.wait, 3) for _ in range(12)]
        assert sorted(f.result() for f in futures) == list(range(12))
    finally:
        actor.executor.shutdown(wait=True, cancel_futures=True)
        actor.thread.join()
        shutil.rmtree(spec['directory'])
