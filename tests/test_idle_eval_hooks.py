"""Exercise training priority through the real shared-actor RPC path."""
from concurrent.futures import ThreadPoolExecutor
import queue
import json
import shutil
import threading
import time
from types import SimpleNamespace

import pytest

from src.online_rl.tree import idle_eval_hooks as hooks
from src.online_rl.tree.gpu_pool import Actor, PoolClient, create_pool
from src.online_rl.tree.idle_eval_priority import PriorityCoordinator, EvaluationDeferred


def test_disabled_hooks_do_not_create_coordination_files(monkeypatch):
    monkeypatch.delenv('SHERPA_IDLE_EVAL_PRIORITY_DIR', raising=False)
    assert hooks.coordinator() is None
    hooks.register_worker(0)
    with hooks.training_guard(0, 'agent'):
        pass


def test_foreign_process_namespace_cannot_prune_registered_owners(tmp_path):
    priority = PriorityCoordinator(tmp_path)
    owner = priority.register_training(0)
    path = tmp_path/'priority.json'
    state = json.loads(path.read_text())
    state['process_view']['pid_namespace'] = 'different-process-namespace'
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match='namespace differs'):
        priority.status()
    assert path.read_bytes() == original


def test_actual_actor_waits_for_eval_then_prevents_eval_overtaking(tmp_path, monkeypatch):
    directory = tmp_path/'priority'
    monkeypatch.setenv('SHERPA_IDLE_EVAL_PRIORITY_DIR', str(directory))
    priority = PriorityCoordinator(directory, idle_grace_seconds=0)
    hooks.register_worker(0)
    resident = priority.evaluation_residency(0)
    call = priority.evaluation_call(0, resident)
    manager = SimpleNamespace(RLock=threading.RLock, dict=dict, Queue=queue.Queue)
    spec = create_pool(manager, [0], agent_capacity=1, doctor_capacity=1)
    handler_started, release_handler = threading.Event(), threading.Event()

    def generate(*_):
        handler_started.set()
        assert release_handler.wait(5)
        return 'trained reply'

    actor = Actor(spec, 0, {'agent': generate}, tmp_path/'audit')
    actor.expected = 'training-policy'
    try:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(PoolClient(spec, 'training-policy', 0).call, 'agent', 'prompt', 8, 1, 'agent')
            deadline = time.monotonic()+5
            while not priority.status()['gpus']['0']['training']:
                assert time.monotonic() < deadline
                time.sleep(.01)
            assert not handler_started.is_set()
            call.release()
            assert handler_started.wait(5)
            with pytest.raises(EvaluationDeferred):
                priority.evaluation_call(0, resident)
            assert not resident.revoked()
            release_handler.set()
            assert future.result(5) == 'trained reply'
        assert not priority.status()['gpus']['0']['training']
    finally:
        release_handler.set()
        call.release()
        resident.release()
        actor.close()
        shutil.rmtree(spec['directory'])
        hooks.close_registrations()


def test_80gb_resident_mode_pauses_calls_without_revoking_weights(tmp_path, monkeypatch):
    directory = tmp_path/'priority-resident'
    monkeypatch.setenv('SHERPA_IDLE_EVAL_PRIORITY_DIR', str(directory))
    monkeypatch.setenv('SHERPA_IDLE_EVAL_KEEP_RESIDENT_DURING_UPDATE', '1')
    hooks.register_worker(4)
    priority = PriorityCoordinator(directory, idle_grace_seconds=0)
    resident = priority.evaluation_residency(4)
    try:
        with hooks.training_guard(4, 'agent_update', release_memory=True):
            assert not resident.revoked()
            with pytest.raises(EvaluationDeferred):
                priority.evaluation_call(4, resident)
        assert not resident.revoked()
    finally:
        resident.release()
        hooks.close_registrations()
