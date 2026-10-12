"""Bounded retry policy for OpenRouter chat calls. Operational only: it decides whether and how long to wait before the
SAME request is sent again to the SAME model; it never changes the request, the model or the provider.

- Rate limiting / provider overload (HTTP 429, or an error body naming engine_overloaded / rate limit) gets a larger
  budget: up to RATE_LIMIT_RETRIES retries and RATE_LIMIT_WAIT_BUDGET_S seconds of total waiting per call.
- Other transient failures (HTTP 5xx, timeouts, dropped connections, a 200 body that is not a completion) keep the
  previous budget of TRANSIENT_RETRIES retries.
- Anything else (e.g. HTTP 400/401/402/403/404) is not retried.
- The wait is the server's Retry-After when given (capped at RETRY_AFTER_MAX_S), else exponential backoff capped at
  MAX_BACKOFF_S with jitter in [0.5, 1.0) of the step. The jitter uses its own RNG, never the global one.
- Every retry and the final outcome are logged (attempt, kind, reason, wait, total wait); exhausting the rate-limit
  budget raises RetryExhausted with that summary."""
from __future__ import annotations
import email.utils, os, random, time
from typing import Callable, Optional

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
TRANSIENT_RETRIES = 6                    # = the previous 7 attempts
RATE_LIMIT_RETRIES = 11                  # 12 attempts
RATE_LIMIT_WAIT_BUDGET_S = 1200.0        # total wait for rate-limit retries of one call
MAX_BACKOFF_S = 120.0
RETRY_AFTER_MAX_S = 300.0
_JITTER = random.Random()                # private: retries never consume the global RNG


class RetryExhausted(RuntimeError):
    """The rate-limit retry budget of one call ran out; the run stops here and resumes from its caches."""


def rate_limited(status: Optional[int], text: str = '') -> bool:
    t = (text or '').lower()
    return status == 429 or 'engine_overloaded' in t or 'rate limit' in t or 'rate-limit' in t or 'too many requests' in t


def retry_after_seconds(value) -> Optional[float]:
    """Retry-After as delta-seconds or an HTTP date; None if absent or unreadable."""
    if value in (None, ''): return None
    try: return max(0.0, float(value))
    except (TypeError, ValueError): pass
    try: return max(0.0, email.utils.parsedate_to_datetime(str(value)).timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError): return None


def backoff_s(retry: int, rng: random.Random = _JITTER) -> float:
    """Wait before retry number `retry` (0-based): min(MAX_BACKOFF_S, 2**retry) scaled by a jitter in [0.5, 1.0)."""
    return min(MAX_BACKOFF_S, 2.0 ** retry) * (0.5 + 0.5 * rng.random())


class Retry:
    """State of one call's retries. wait(kind, reason, retry_after) sleeps or, when the budget of that kind is spent,
    returns False (transient) or raises RetryExhausted (rate_limit)."""

    def __init__(self, label: str, sleep: Optional[Callable[[float], None]] = None, log: Optional[Callable[[str], None]] = None,
                 rng: random.Random = _JITTER):
        self.label, self.sleep, self.log, self.rng = label, sleep, log, rng
        self.rate_limit_retries = int(os.environ.get('SHERPA_RATE_LIMIT_RETRIES', RATE_LIMIT_RETRIES))
        if self.rate_limit_retries < 0:
            raise ValueError('SHERPA_RATE_LIMIT_RETRIES must be nonnegative')
        self.retries = {'rate_limit': 0, 'transient': 0}; self.total_wait_s = 0.0

    def _say(self, msg: str) -> None:
        if self.log: self.log(f'[retry] {self.label}: {msg}')
        else: print(f'[retry] {self.label}: {msg}', flush=True)

    def wait(self, kind: str, reason: str, retry_after=None) -> bool:
        from src.common.usage import event
        event('api_attempt_failed', label=self.label, kind=kind, attempts=self.attempts)
        n = self.retries[kind]; ra = retry_after_seconds(retry_after)
        delay = min(RETRY_AFTER_MAX_S, ra) if ra is not None else backoff_s(n, self.rng)
        if kind == 'rate_limit':
            spent = n >= self.rate_limit_retries or self.total_wait_s + delay > RATE_LIMIT_WAIT_BUDGET_S
            if spent:
                msg = (f'FAILED after {sum(self.retries.values()) + 1} attempts: rate-limit budget exhausted ({n} rate-limit retries, '
                       f'{self.retries["transient"]} transient, waited {self.total_wait_s:.0f}s of {RATE_LIMIT_WAIT_BUDGET_S:.0f}s); last: {reason}')
                self._say(msg); raise RetryExhausted(f'{self.label}: {msg}')
        elif n >= TRANSIENT_RETRIES:
            self._say(f'FAILED after {sum(self.retries.values()) + 1} attempts ({n} transient retries, waited {self.total_wait_s:.0f}s); last: {reason}')
            return False
        self.retries[kind] = n + 1; self.total_wait_s += delay
        event('api_retry', label=self.label, kind=kind, delay_seconds=delay, attempts=self.attempts)
        self._say(f'{kind} retry {n + 1}/{self.rate_limit_retries if kind == "rate_limit" else TRANSIENT_RETRIES}: {reason}; waiting {delay:.1f}s'
                  f' ({"Retry-After " + str(retry_after) if ra is not None else "backoff"}; total wait {self.total_wait_s:.0f}s)')
        (self.sleep or time.sleep)(delay); return True

    def succeeded(self) -> None:
        if any(self.retries.values()):
            self._say(f'succeeded after {self.retries["rate_limit"]} rate-limit and {self.retries["transient"]} transient retries '
                      f'(waited {self.total_wait_s:.0f}s)')

    @property
    def attempts(self) -> int: return sum(self.retries.values()) + 1
