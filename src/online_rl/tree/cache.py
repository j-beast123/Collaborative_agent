"""Immutable call cache for every expensive tree operation (Doctor, Agent, FINAL, feedback, rubric generation, rubric
scoring, behavior text, reruns). The Doctor API is not seedable, so a stored result is evidence and is never replaced."""
from __future__ import annotations
import json, threading
from pathlib import Path
from typing import Callable, Optional
from src.online_rl.integrated.records import sha256_json


class CallCache:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None; self.store: dict[str, dict] = {}; self._lock = threading.Lock()
        self.hits: dict[str, int] = {}; self.calls: dict[str, int] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip(): continue
                r = json.loads(line)
                if sha256_json([r['kind'], r['identity']]) != r['key']: raise RuntimeError(f"corrupted cache record {r['key']}")
                self.store[r['key']] = r

    def call(self, kind: str, identity: dict, fn: Callable):
        key = sha256_json([kind, identity])
        with self._lock:
            if key in self.store:
                self.hits[kind] = self.hits.get(kind, 0) + 1; return self.store[key]['value']
        try:
            value = fn()
        except Exception as exc:
            from src.collaboration.generator import DoctorProtocolError
            if isinstance(exc, DoctorProtocolError) and self.path:
                failure = {'kind': kind, 'identity': identity, 'status': 'technical_incomplete',
                           'error_type': 'doctor_protocol_error', 'error': str(exc)}
                with self._lock:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.with_name(self.path.stem + '_protocol_errors.jsonl').open('a') as f:
                        f.write(json.dumps(failure, ensure_ascii=False) + '\n')
            raise
        rec = {'key': key, 'kind': kind, 'identity': identity, 'value': value}
        json.dumps(rec)                                            # results must be JSON evidence
        with self._lock:
            if key in self.store: return self.store[key]['value']
            self.store[key] = rec; self.calls[kind] = self.calls.get(kind, 0) + 1
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open('a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return value
