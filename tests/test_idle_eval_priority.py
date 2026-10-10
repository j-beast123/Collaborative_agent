"""CPU-only proofs of priority, per-GPU drainage and crash-safe ownership."""
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import threading

import pytest

from src.online_rl.tree.idle_eval_priority import (
    EvaluationDeferred, PriorityCoordinator, PriorityTimeout, _process_identity,
)


def _contend_residency(root, results, release):
    coordinator = PriorityCoordinator(root, idle_grace_seconds=0)
    try:
        with coordinator.evaluation_residency(0):
            results.put(True)
            assert release.wait(20)
    except EvaluationDeferred:
        results.put(False)


def _wait_child(ready, release):
    ready.set()
    release.wait(20)


@pytest.fixture
def priority(tmp_path):
    coordinator = PriorityCoordinator(tmp_path, idle_grace_seconds=0,
                                      poll_seconds=0.005, training_timeout=2)
    with coordinator.register_training(0), coordinator.register_training(4):
        yield coordinator


def _wait_until(predicate, timeout=2):
    import time
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Condition did not become true')
        time.sleep(0.005)


def test_evaluation_requires_current_training_evidence(tmp_path):
    coordinator = PriorityCoordinator(tmp_path, idle_grace_seconds=0)
    with pytest.raises(EvaluationDeferred):
        coordinator.evaluation_residency(0)
    registration = coordinator.register_training(0)
    resident = coordinator.evaluation_residency(0)
    assert coordinator.ready([0]) and not coordinator.ready([0, 4])
    registration.release()
    assert not coordinator.ready([0])
    assert resident.revoked() and not resident.can_call()
    with pytest.raises(EvaluationDeferred):
        coordinator.evaluation_call(0, resident)
    resident.release()


def test_pending_training_drains_one_call_and_keeps_resident_weights(priority):
    resident = priority.evaluation_residency(0)
    call = priority.evaluation_call(0, resident)
    entered, finish = threading.Event(), threading.Event()

    def train():
        with priority.training_activity(0, 'agent'):
            entered.set()
            assert finish.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(train)
        try:
            _wait_until(call.revoked)
            assert not entered.is_set()
            assert not resident.revoked()
            assert not resident.can_call()
            with pytest.raises(EvaluationDeferred):
                priority.evaluation_call(0, resident)
            call.release()
            assert entered.wait(2)
            with pytest.raises(EvaluationDeferred):
                priority.evaluation_call(0, resident)
        finally:
            call.release()
            finish.set()
        future.result(timeout=2)
    assert resident.can_call() and not resident.revoked()
    with priority.evaluation_call(0, resident):
        pass
    resident.release()


def test_update_waits_for_gpu4_memory_without_blocking_gpu0(priority):
    resident0 = priority.evaluation_residency(0)
    resident4 = priority.evaluation_residency(4)
    entered, finish = threading.Event(), threading.Event()

    def update():
        with priority.training_update():
            entered.set()
            assert finish.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(update)
        try:
            _wait_until(resident4.revoked)
            assert not entered.is_set()
            assert resident0.can_call() and not resident0.revoked()
            with priority.evaluation_call(0, resident0):
                pass
            resident4.release()  # Caller has freed/synchronized actual model memory.
            assert entered.wait(2)
            with pytest.raises(EvaluationDeferred):
                priority.evaluation_residency(4)
        finally:
            resident4.release()
            finish.set()
        future.result(timeout=2)
    with priority.evaluation_residency(4):
        pass
    resident0.release()


def test_timeout_fails_closed_and_cleans_pending_intent(priority):
    with priority.evaluation_residency(4) as resident:
        with pytest.raises(PriorityTimeout):
            with priority.training_update(timeout=0):
                pytest.fail('Update ran while evaluation weights were resident')
        assert resident.revoked()
        assert priority.status()['gpus']['4']['training'] == {}
        with pytest.raises(EvaluationDeferred):
            priority.evaluation_call(4, resident)
    with priority.training_update(timeout=0):
        pass


def test_residency_and_call_capacity_and_gpu_binding(priority):
    with priority.evaluation_residency(0) as resident:
        with pytest.raises(EvaluationDeferred):
            priority.evaluation_residency(0)
        with pytest.raises(ValueError):
            priority.evaluation_call(4, resident)
        with priority.evaluation_call(0, resident):
            with pytest.raises(EvaluationDeferred):
                priority.evaluation_call(0, resident)
            with pytest.raises(RuntimeError, match='active'):
                resident.release()
        assert resident.can_call()


def test_idle_grace_restarts_after_last_training_call(tmp_path, monkeypatch):
    import src.online_rl.tree.idle_eval_priority as module
    now = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    coordinator = PriorityCoordinator(tmp_path, idle_grace_seconds=2)
    with coordinator.register_training(0):
        with pytest.raises(EvaluationDeferred):
            coordinator.evaluation_residency(0)
        now[0] = 102.0
        with coordinator.evaluation_residency(0) as resident:
            with coordinator.training_activity(0, 'doctor'):
                assert not resident.can_call()
            now[0] = 103.9
            assert not resident.can_call()
            now[0] = 104.0
            assert resident.can_call()


def test_training_calls_can_coexist(priority):
    with priority.training_activity(0, 'agent'):
        with priority.training_activity(0, 'doctor'):
            assert len(priority.status()['gpus']['0']['training']) == 2
        assert len(priority.status()['gpus']['0']['training']) == 1


def test_pid_reuse_does_not_preserve_training_registration(priority):
    with priority._state() as state:
        for record in state['gpus']['0']['registrations'].values():
            record['owner']['start_ticks'] = 'not-the-current-process'
    assert not priority.ready([0])
    with pytest.raises(EvaluationDeferred):
        priority.evaluation_residency(0)


def test_orphaned_doctor_keeps_memory_and_active_call_quarantined(priority):
    ctx = multiprocessing.get_context('spawn')
    ready, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_wait_child, args=(ready, release))
    child.start()
    assert ready.wait(5)
    resident = priority.evaluation_residency(4)
    call = priority.evaluation_call(4, resident)
    try:
        resident.add_process(child.pid)
        with priority._state() as state:
            entry = state['gpus']['4']
            entry['residents'][resident.token]['owner']['start_ticks'] = 'dead-worker'
            entry['calls'][call.token]['owner']['start_ticks'] = 'dead-worker'
        state = priority.status()['gpus']['4']
        assert resident.token in state['residents'] and call.token in state['calls']
        assert state['residents'][resident.token]['revoked']
        with pytest.raises(PriorityTimeout):
            with priority.training_activity(4, 'agent', timeout=0):
                pytest.fail('Training overlapped an orphaned Doctor call')
        with pytest.raises(PriorityTimeout):
            with priority.training_update(timeout=0):
                pytest.fail('Update ran while orphaned Doctor retained GPU memory')
    finally:
        release.set()
        child.join(timeout=5)
    assert child.exitcode == 0
    assert not priority.status()['gpus']['4']['residents']
    assert not priority.status()['gpus']['4']['calls']
    with priority.training_update(timeout=0):
        pass


def test_atomic_admission_across_processes(priority):
    ctx = multiprocessing.get_context('spawn')
    results, release = ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=_contend_residency,
                           args=(str(priority.root), results, release)) for _ in range(4)]
    for worker in workers:
        worker.start()
    try:
        admitted = [results.get(timeout=10) for _ in workers]
        assert sum(admitted) == 1
        assert len(priority.status()['gpus']['0']['residents']) == 1
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=5)
    assert all(worker.exitcode == 0 for worker in workers)
    assert not priority.status()['gpus']['0']['residents']


def test_corrupt_state_fails_closed_without_repair(tmp_path):
    coordinator = PriorityCoordinator(tmp_path)
    path = tmp_path / 'priority.json'
    path.write_text('{not-json')
    with pytest.raises(json.JSONDecodeError):
        coordinator.evaluation_residency(0)
    assert path.read_text() == '{not-json'


def test_current_process_identity_is_stable():
    identity = _process_identity()
    assert identity and _process_identity(identity['pid']) == identity
