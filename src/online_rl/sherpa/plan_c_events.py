"""Serialize Plan C progress writes from concurrent collection/reward workers."""
import threading
from src.common.progress import emit as _emit

_lock = threading.Lock()


def emit(event):
    with _lock:
        _emit(event)
