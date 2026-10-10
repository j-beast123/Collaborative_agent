"""Optional execution-only hooks for an independent low-priority evaluator.

Disabled unless the launcher supplies SHERPA_IDLE_EVAL_PRIORITY_DIR. No prompt,
reward, sampling seed, adapter or research configuration is changed here.
"""
import atexit
from contextlib import nullcontext
import os
import threading

_lock = threading.RLock()
_coordinators = {}
_registrations = {}


def coordinator():
    directory = os.environ.get('SHERPA_IDLE_EVAL_PRIORITY_DIR')
    if not directory:
        return None
    with _lock:
        if directory not in _coordinators:
            from src.online_rl.tree.idle_eval_priority import PriorityCoordinator
            _coordinators[directory] = PriorityCoordinator(directory)
        return _coordinators[directory]


def register_worker(gpu):
    coord = coordinator()
    if coord is None:
        return
    key = (os.environ['SHERPA_IDLE_EVAL_PRIORITY_DIR'], gpu)
    with _lock:
        if key not in _registrations:
            _registrations[key] = coord.register_training(gpu)


def training_guard(gpu, kind, *, release_memory=False):
    coord = coordinator()
    if coord is None:
        return nullcontext()
    return (coord.training_update(gpu=gpu) if release_memory else
            coord.training_activity(gpu, kind))


def close_registrations():
    with _lock:
        for registration in _registrations.values():
            registration.release()
        _registrations.clear()


atexit.register(close_registrations)
