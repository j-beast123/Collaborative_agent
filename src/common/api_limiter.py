"""One process-wide limiter for physical HTTP attempts, not logical calls.

B200 executes all case/role API work in this process. Retry sleeps happen outside
request_json, so waiting retries never reserve slots. No per-case limiter exists.
"""
from contextlib import contextmanager
import threading
import time


class Limiter:
    def __init__(self, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError('API in-flight limit must be a positive integer')
        self.limit = limit
        self.semaphore = threading.BoundedSemaphore(limit)
        self.lock = threading.Lock()
        self.active = self.peak = self.attempts = 0
        self.wait_seconds = 0.0

    @contextmanager
    def slot(self):
        start = time.monotonic()
        self.semaphore.acquire()
        wait = time.monotonic() - start
        with self.lock:
            self.active += 1
            self.attempts += 1
            self.peak = max(self.peak, self.active)
            self.wait_seconds += wait
            row = {'limiter_wait_seconds': wait, 'in_flight_at_entry': self.active,
                   'global_peak_in_flight': self.peak, 'global_limit': self.limit}
        try:
            yield row
        finally:
            with self.lock: self.active -= 1
            self.semaphore.release()


_guard = threading.Lock()
_limiter = None


def configure(limit):
    global _limiter
    with _guard:
        if _limiter is not None and _limiter.limit != limit:
            raise ValueError('Cannot change the global API limit within a running process')
        if _limiter is None: _limiter = Limiter(limit)
    return _limiter


@contextmanager
def request_slot():
    if _limiter is None:
        yield {}
    else:
        with _limiter.slot() as row: yield row
