"""Immutable call cache for every expensive tree operation (Doctor, Agent, FINAL, feedback, rubric generation, rubric
scoring, behavior text, reruns). The Doctor API is not seedable, so a stored result is evidence and is never replaced."""
from __future__ import annotations
import json, os, threading
from pathlib import Path
from typing import Callable, Optional
from src.online_rl.integrated.records import sha256_json


class CallCache:
    """call() is single-flight per key: concurrent requests for the same key run fn once (the leader) and the others wait
    for its stored result; different keys run in parallel. A failed leader stores nothing and wakes its waiters, which
    retry exactly as a fresh request would (one of them becomes the next leader)."""
    def __init__(self, path: Optional[Path], *, enabled=None):
        self.enabled = (os.environ.get('SHERPA_CALL_CACHE', '1') != '0') if enabled is None else enabled
        self.path = Path(path) if path else None; self.store: dict[str, dict] = {}; self._lock = threading.Lock()
        self.hits: dict[str, int] = {}; self.calls: dict[str, int] = {}; self._inflight: dict[str, threading.Event] = {}
        if self.enabled and self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip(): continue
                r = json.loads(line)
                if sha256_json([r['kind'], r['identity']]) != r['key']: raise RuntimeError(f"corrupted cache record {r['key']}")
                self.store[r['key']] = r

    def call(self, kind: str, identity: dict, fn: Callable):
        key = sha256_json([kind, identity])
        done = None
        while self.enabled:
            with self._lock:
                if key in self.store:
                    self.hits[kind] = self.hits.get(kind, 0) + 1; return self.store[key]['value']
                done = self._inflight.get(key)
                if done is None: done = self._inflight[key] = threading.Event(); break      # this thread leads
            done.wait()                                            # the leader stored a value (-> hit) or failed (-> retry)
        try:
            from src.common.live_gpu import activity
            with activity('cache.' + kind):
                value = fn()
            rec = {'key': key, 'kind': kind, 'identity': identity, 'value': value}
            json.dumps(rec)                                        # results must be JSON evidence
            with self._lock:
                if self.enabled and key in self.store: return self.store[key]['value']
                if self.enabled: self.store[key] = rec
                self.calls[kind] = self.calls.get(kind, 0) + 1
                if self.enabled and self.path:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open('a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            return value
        except Exception as exc:
            # Failed evidence is diagnostic only: never insert it into the
            # immutable success cache or convert it to NA/a zero reward.
            if self.path and getattr(exc, 'audit', None) is not None:
                try:
                    from datetime import datetime, timezone
                    failure = {'time': datetime.now(timezone.utc).isoformat(),
                               'key': key, 'kind': kind, 'identity': identity,
                               'exception_type': type(exc).__name__, 'error': str(exc),
                               'audit': exc.audit}
                    line = json.dumps(failure, ensure_ascii=False) + '\n'
                    with self._lock:
                        self.path.parent.mkdir(parents=True, exist_ok=True)
                        with self.path.with_suffix('.errors.jsonl').open('a') as f:
                            f.write(line)
                except Exception as audit_error:
                    exc.add_note(f'Failed to persist error audit: {type(audit_error).__name__}: {audit_error}')
            raise
        finally:
            if done is not None:
                with self._lock: del self._inflight[key]
                done.set()
