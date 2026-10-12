"""Cooperative, process-safe admission for optional idle-GPU evaluation.

Training publishes its intent *before* waiting for an already admitted evaluation
call. Resident evaluation weights may coexist with training inference, but an
update revokes residency and waits for memory to be released on that GPU only.
There is no CUDA dependency and no process signalling in this module.

All GPU work, including evaluation model loading, must hold a call lease. CUDA
work must be synchronized before releasing that lease. A residency lease must
outlive its model allocations and any Doctor process (register it with
``add_process``); explicit release acknowledges that those allocations are gone.
No heartbeat timeout is treated as proof that GPU memory has been released.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid


class EvaluationDeferred(RuntimeError):
    """Training has priority, or no live cooperative trainer is registered."""


class PriorityTimeout(TimeoutError):
    """Evaluation failed to drain; training must not enter its GPU operation."""


def _process_identity(pid=None):
    """Include start ticks and boot ID so PID reuse cannot revive an old lease."""
    pid = os.getpid() if pid is None else int(pid)
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError:
        return None


def _alive(identity):
    try:
        return _process_identity(identity['pid']) == identity
    except (OSError, IndexError, KeyError, ValueError):
        # An unreadable proc filesystem is not evidence that GPU memory is gone.
        return True


class Lease:
    """A context-capable registration, residency, or individual GPU-call lease."""
    def __init__(self, coordinator, gpu, bucket, token):
        self.coordinator, self.gpu = coordinator, str(gpu)
        self.bucket, self.token, self.released = bucket, token, False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def release(self):
        if self.released:
            return
        with self.coordinator._state() as state:
            gpu = self.coordinator._gpu(state, self.gpu)
            if self.bucket == 'residents' and any(
                    call.get('residency') == self.token for call in gpu['calls'].values()):
                raise RuntimeError('Cannot release residency while an evaluation call is active')
            gpu[self.bucket].pop(self.token, None)
            if self.bucket == 'training':
                gpu['last_training_end'] = time.monotonic()
        self.released = True

    def revoked(self):
        if self.released:
            return True
        with self.coordinator._state() as state:
            gpu = self.coordinator._gpu(state, self.gpu)
            record = gpu[self.bucket].get(self.token)
            return (record is None or bool(record.get('revoked')) or
                    not self.coordinator._has_training(gpu))

    def can_call(self):
        if self.bucket != 'residents' or self.released:
            return False
        with self.coordinator._state() as state:
            gpu = self.coordinator._gpu(state, self.gpu)
            resident = gpu['residents'].get(self.token)
            return bool(resident and not resident.get('revoked') and
                        self.coordinator._admissible(gpu))

    def add_process(self, pid):
        """Track a child with GPU allocations before letting it perform GPU work.

        A surviving child keeps an orphaned residency/call quarantined. The
        caller must arrange parent-death cleanup for the spawn-to-registration
        window; this module never assumes a detached child dies with its parent.
        """
        if self.bucket != 'residents' or self.released:
            raise RuntimeError('A live residency lease is required')
        identity = _process_identity(pid)
        if identity is None:
            raise RuntimeError('Cannot register an exited evaluation process')
        with self.coordinator._state() as state:
            record = self.coordinator._gpu(state, self.gpu)['residents'].get(self.token)
            if record is None:
                raise RuntimeError('Evaluation residency is no longer registered')
            if identity not in record['processes']:
                record['processes'].append(identity)


class PriorityCoordinator:
    """Shared directory interface; absence of live trainer evidence fails closed."""
    def __init__(self, root, *, idle_grace_seconds=2.0, poll_seconds=0.05,
                 training_timeout=120.0, require_training=True):
        for name, value, minimum in (
                ('idle_grace_seconds', idle_grace_seconds, 0),
                ('poll_seconds', poll_seconds, 0),
                ('training_timeout', training_timeout, 0)):
            if not math.isfinite(value) or value < minimum or (name == 'poll_seconds' and value == 0):
                raise ValueError(f'Invalid {name}')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.idle_grace_seconds, self.poll_seconds = idle_grace_seconds, poll_seconds
        self.training_timeout, self.require_training = training_timeout, require_training

    @staticmethod
    def _gpu(state, gpu):
        return state['gpus'].setdefault(str(gpu), {
            'registrations': {}, 'training': {}, 'residents': {}, 'calls': {},
            'last_training_end': 0.0})

    @staticmethod
    def _prune(state):
        for gpu in state['gpus'].values():
            for name in ('registrations', 'training'):
                for token, record in list(gpu[name].items()):
                    if not _alive(record['owner']):
                        del gpu[name][token]
                        gpu['last_training_end'] = time.monotonic()
            for token, record in list(gpu['residents'].items()):
                if not _alive(record['owner']):
                    record['revoked'] = True
                    if not any(_alive(identity) for identity in record['processes']):
                        del gpu['residents'][token]
            for token, record in list(gpu['calls'].items()):
                if not _alive(record['owner']):
                    resident = gpu['residents'].get(record['residency'])
                    if resident and any(_alive(identity) for identity in resident['processes']):
                        record['revoked'] = True
                    else:
                        del gpu['calls'][token]

    @contextmanager
    def _state(self):
        # Open on every transaction: independently opened flock descriptions
        # serialize threads as well as processes, including after fork.
        with (self.root / 'priority.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'priority.json'
            try:
                state = json.loads(path.read_text()) if path.exists() else {'version': 1, 'gpus': {}}
                if not isinstance(state, dict) or state.get('version') != 1 or not isinstance(state.get('gpus'), dict):
                    raise RuntimeError('Invalid idle evaluation priority state')
                # A sandbox with a different /proc view must never prune host
                # owners simply because their PIDs are invisible there.
                view = {'pid_namespace': os.readlink('/proc/self/ns/pid'),
                        'init_process': _process_identity(1)}
                if state.get('process_view', view) != view:
                    raise RuntimeError('Priority coordinator process namespace differs; use the host runtime')
                before = json.dumps(state, sort_keys=True)
                state['process_view'] = view
                self._prune(state)
                try:
                    yield state
                finally:
                    encoded = json.dumps(state, sort_keys=True)
                    if encoded != before or not path.exists():
                        descriptor, temporary = tempfile.mkstemp(prefix='.priority-', dir=self.root)
                        try:
                            with os.fdopen(descriptor, 'w') as handle:
                                handle.write(encoded + '\n')
                            os.replace(temporary, path)
                        finally:
                            if os.path.exists(temporary):
                                os.unlink(temporary)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _has_training(self, gpu):
        return not self.require_training or bool(gpu['registrations'])

    def _admissible(self, gpu):
        return (self._has_training(gpu) and not gpu['training'] and
                time.monotonic() - gpu['last_training_end'] >= self.idle_grace_seconds)

    def _new(self, gpu, bucket, **fields):
        token = uuid.uuid4().hex
        owner = _process_identity()
        if owner is None:
            raise RuntimeError('Cannot identify current process')
        return Lease(self, gpu, bucket, token), {'owner': owner, 'created_at': time.time(), **fields}

    def register_training(self, gpu):
        """Retain for the lifetime of the instrumented training GPU owner."""
        lease, record = self._new(gpu, 'registrations')
        with self._state() as state:
            entry = self._gpu(state, gpu)
            entry['registrations'][lease.token] = record
            entry['last_training_end'] = time.monotonic()
        return lease

    def evaluation_residency(self, gpu):
        lease, record = self._new(gpu, 'residents', revoked=False, processes=[])
        with self._state() as state:
            entry = self._gpu(state, gpu)
            if not self._admissible(entry) or entry['residents']:
                raise EvaluationDeferred(f'GPU {gpu} is unavailable for evaluation residency')
            entry['residents'][lease.token] = record
        return lease

    def evaluation_call(self, gpu, residency):
        if (residency.coordinator.root.resolve() != self.root.resolve() or
                residency.gpu != str(gpu) or residency.bucket != 'residents' or residency.released):
            raise ValueError('Evaluation call requires its GPU residency lease')
        lease, record = self._new(gpu, 'calls', residency=residency.token, revoked=False)
        with self._state() as state:
            entry = self._gpu(state, gpu)
            resident = entry['residents'].get(residency.token)
            if (not resident or resident.get('revoked') or not self._admissible(entry)
                    or entry['calls']):
                raise EvaluationDeferred(f'GPU {gpu} is unavailable for an evaluation call')
            entry['calls'][lease.token] = record
        return lease

    @contextmanager
    def _training(self, gpu, kind, clear_residency, timeout):
        timeout = self.training_timeout if timeout is None else timeout
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError('Invalid training priority timeout')
        lease, record = self._new(gpu, 'training', kind=kind, clear_residency=clear_residency)
        with self._state() as state:
            entry = self._gpu(state, gpu)
            entry['training'][lease.token] = record
            for call in entry['calls'].values():
                call['revoked'] = True
            if clear_residency:
                for resident in entry['residents'].values():
                    resident['revoked'] = True
        deadline = time.monotonic() + timeout
        try:
            while True:
                with self._state() as state:
                    entry = self._gpu(state, gpu)
                    ready = not entry['calls'] and (not clear_residency or not entry['residents'])
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise PriorityTimeout(f'Evaluation did not release GPU {gpu} for training {kind}')
                time.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))
            yield lease
        finally:
            lease.release()

    def training_activity(self, gpu, kind, *, timeout=None):
        return self._training(gpu, kind, False, timeout)

    def training_update(self, gpu=4, *, timeout=None):
        return self._training(gpu, 'update', True, timeout)

    def status(self):
        """Return a detached, pruned snapshot; never repairs malformed state."""
        with self._state() as state:
            return json.loads(json.dumps(state))

    def ready(self, gpus):
        """All requested GPUs have current, live cooperative training owners."""
        with self._state() as state:
            return all(bool(self._gpu(state, gpu)['registrations']) for gpu in gpus)
