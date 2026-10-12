"""Execution-only API capacity shared by all coordinators in one training run."""
from contextlib import contextmanager
import threading

_slots = {}


def create(limits, manager=None):
    if not isinstance(limits, dict) or set(limits) - {'gemini', 'luna', 'doctor'}:
        raise ValueError('API limits must map gemini/luna/doctor to positive integer capacities')
    for name, capacity in limits.items():
        if type(capacity) is not int or capacity < 1:
            raise ValueError(f'Invalid API capacity for {name}')
    factory = manager.BoundedSemaphore if manager is not None else threading.BoundedSemaphore
    return {name: factory(capacity) for name, capacity in limits.items()}


def install(slots):
    global _slots
    _slots = slots


def family(model):
    name = model.lower().split('/')[-1]
    if name.startswith('gemini'): return 'gemini'
    if name == 'gpt-6-luna': return 'luna'
    return None


@contextmanager
def lease(model, *, slot_name=None):
    name = slot_name or family(model)
    slot = _slots.get(name)
    if slot is None:
        yield
        return
    from src.common.live_gpu import activity
    with activity(f'api.{name}.slot_wait'):
        slot.acquire()
    try:
        with activity(f'api.{name}.request'):
            yield
    finally:
        slot.release()
