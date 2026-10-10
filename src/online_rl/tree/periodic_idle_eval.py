"""CPU-only, durable capture queue for an independently scheduled evaluator.

The training directory is read-only. A checkpoint is eligible only after its
receipt agrees with the committed run state. Missing historical checkpoints are
reported explicitly; a later policy never stands in for a missed target.
"""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import threading
import time

from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.idle_eval_snapshot import _check_receipt, _check_source, snapshot

CONTRACT = 'periodic_idle_checkpoint_eval_v1'


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f'Expected JSON object: {path}')
    return value


def committed(source, manifest_sha):
    """Read an internally consistent commitment, retrying a cleanup race."""
    source = Path(source).resolve()
    for _ in range(3):
        manifest = read(source / 'run_manifest.json')
        state = read(source / 'run_state.json')
        _check_source(manifest, state, manifest_sha)
        checkpoint = Path(state['checkpoint'])
        if checkpoint.is_symlink() or checkpoint.resolve().parent != source / 'checkpoints':
            raise RuntimeError('Committed checkpoint is outside the training run')
        try:
            receipt = read(checkpoint / 'COMPLETE.json')
        except FileNotFoundError:
            if read(source / 'run_state.json') != state:
                continue
            raise
        _check_receipt(receipt, state, state['case_cursor'])
        steps = receipt.get('update', {}).get('optimizer_steps', 0)
        if type(steps) is not int or steps < 0:
            raise RuntimeError('Invalid committed optimizer step count')
        if read(source / 'run_state.json') != state:
            continue
        return manifest, state, receipt, steps
    return None


class CaptureQueue:
    """Persist snapshots separately from GPU scheduling and worker lifetimes."""
    def __init__(self, output, plan):
        self.root, self.plan = Path(output).resolve(), plan
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._verified_snapshots = set()
        with self.transaction() as state:
            if not state:
                state.update(contract=CONTRACT, plan=plan, jobs=[], missed_targets=[],
                             next_target_update=plan['start_update'], latest_committed=None)
            elif state.get('contract') != CONTRACT or state.get('plan') != plan:
                raise RuntimeError('Periodic queue belongs to a different pinned plan')

    @contextmanager
    def transaction(self):
        with self._thread_lock, (self.root / 'queue.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            path = self.root / 'queue.json'
            state = read(path) if path.exists() else {}
            before = sha256_json(state)
            try:
                yield state
            except BaseException:
                raise
            else:
                if sha256_json(state) != before or not path.exists():
                    atomic_write_json(path, state)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def status(self):
        with self.transaction() as state:
            return json.loads(json.dumps(state))

    def _register(self, state, destination, *, reason):
        proof = read(destination / 'SNAPSHOT.json')
        cursor = proof['completed_cases']
        if str(destination) not in self._verified_snapshots:
            snapshot(self.plan['source_run'], cursor, destination,
                     expected_manifest_sha256=self.plan['source_manifest_sha256'])
            self._verified_snapshots.add(str(destination))
        receipt = read(destination / 'COMPLETE.json')
        steps = receipt.get('update', {}).get('optimizer_steps', 0)
        if type(steps) is not int or steps < 0:
            raise RuntimeError('Snapshot has invalid optimizer step count')
        previous = next((job for job in state['jobs'] if job['optimizer_steps'] == steps), None)
        if previous:
            if previous['policy_sha256'] != proof['policy_sha256']:
                raise RuntimeError('One optimizer step has two different committed policies')
            return previous
        job_root = self.root / 'jobs' / f'update_{steps:06d}'
        row = {'optimizer_steps': steps, 'completed_cases': cursor,
               'policy_sha256': proof['policy_sha256'], 'snapshot': str(destination),
               'snapshot_sha256': sha256_json(proof), 'evaluation': str(job_root / 'evaluation'),
               'status': 'pending', 'reason': reason, 'captured_at': time.time()}
        state['jobs'].append(row)
        state['jobs'].sort(key=lambda job: job['optimizer_steps'])
        return row

    def capture_once(self):
        """Capture a due policy promptly; safe to call while a GPU job runs."""
        observed = committed(self.plan['source_run'], self.plan['source_manifest_sha256'])
        if observed is None:
            return self.status()
        manifest, source_state, receipt, steps = observed
        cursor = source_state['case_cursor']
        with self.transaction() as state:
            previous = state['latest_committed']
            if previous and (steps < previous['optimizer_steps'] or cursor < previous['completed_cases']):
                raise RuntimeError('Committed training progress moved backwards')
            if previous and steps == previous['optimizer_steps'] and source_state['policy_sha256'] != previous['policy_sha256']:
                raise RuntimeError('Committed policy changed without an optimizer update')
            if previous is None or steps != previous['optimizer_steps'] or cursor != previous['completed_cases']:
                state['latest_committed'] = {'optimizer_steps': steps, 'completed_cases': cursor,
                                            'policy_sha256': source_state['policy_sha256'], 'observed_at': time.time()}
            # Recover snapshots published just before a supervisor crash, before
            # deciding a target was missed. Existing snapshots survive cleanup.
            initial = Path(self.plan['initial_snapshot'])
            if initial.exists():
                initial_proof = read(initial / 'SNAPSHOT.json')
                initial_receipt = read(initial / 'COMPLETE.json')
                if (initial_proof['completed_cases'] != self.plan['start_cases'] or
                        initial_receipt.get('update', {}).get('optimizer_steps', 0) != self.plan['start_update']):
                    raise RuntimeError('Initial snapshot differs from the scheduling anchor')
                self._register(state, initial, reason='initial_latest_commit')
            for candidate in sorted((self.root / 'jobs').glob('update_*/snapshot')):
                proof = read(candidate / 'SNAPSHOT.json')
                step = read(candidate / 'COMPLETE.json').get('update', {}).get('optimizer_steps', 0)
                if candidate.parent.name != f'update_{step:06d}':
                    raise RuntimeError('Recovered snapshot path/update differs')
                if step < self.plan['start_update']:
                    raise RuntimeError('Recovered snapshot precedes scheduling anchor')
                if ((step - self.plan['start_update']) % self.plan['every_updates'] and
                        not (self.plan['include_terminal'] and proof['completed_cases'] == self.plan['target_cases'])):
                    raise RuntimeError('Recovered snapshot is outside the declared schedule')
                self._register(state, candidate, reason='recovered_snapshot')
            target = state['next_target_update']
            existing = {job['optimizer_steps'] for job in state['jobs']}
            while target <= steps:
                if target in existing:
                    target += self.plan['every_updates']
                    continue
                if target < steps:
                    state['missed_targets'].append({'optimizer_steps': target,
                        'observed_update': steps, 'observed_cases': cursor,
                        'reason': 'target_passed_before_immutable_capture', 'recorded_at': time.time()})
                    target += self.plan['every_updates']
                    continue
                destination = initial if target == self.plan['start_update'] else self.root / 'jobs' / f'update_{target:06d}' / 'snapshot'
                if target == self.plan['start_update'] and cursor != self.plan['start_cases']:
                    raise RuntimeError('Initial commitment changed before initial snapshot capture')
                try:
                    result = snapshot(self.plan['source_run'], cursor, destination,
                                      expected_manifest_sha256=self.plan['source_manifest_sha256'])
                except (RuntimeError, FileNotFoundError):
                    # Only a real commitment advance licenses retrying next poll.
                    now = read(Path(self.plan['source_run']) / 'run_state.json')
                    if now['case_cursor'] > cursor:
                        break
                    raise
                if result is None:
                    break
                row = self._register(state, destination, reason='initial_latest_commit' if target == self.plan['start_update'] else 'periodic_update')
                if row['optimizer_steps'] != target:
                    raise RuntimeError('Captured policy differs from scheduled optimizer step')
                target += self.plan['every_updates']
            state['next_target_update'] = target
            if (self.plan['include_terminal'] and cursor == self.plan['target_cases'] and
                    not any(job['optimizer_steps'] == steps for job in state['jobs'])):
                destination = self.root / 'jobs' / f'update_{steps:06d}' / 'snapshot'
                if snapshot(self.plan['source_run'], cursor, destination,
                            expected_manifest_sha256=self.plan['source_manifest_sha256']) is not None:
                    self._register(state, destination, reason='terminal_checkpoint')
        return self.status()

    def mark(self, steps, status, **fields):
        if status not in ('pending', 'running', 'complete'):
            raise ValueError('Invalid periodic job status')
        with self.transaction() as state:
            job = next(job for job in state['jobs'] if job['optimizer_steps'] == steps)
            if job['status'] == 'complete' and status != 'complete':
                raise RuntimeError('Completed evaluation cannot be reopened')
            if job.get('job_sha256') and fields.get('job_sha256', job['job_sha256']) != job['job_sha256']:
                raise RuntimeError('Immutable evaluation job changed')
            job.update(status=status, **fields)

    def collect(self, stop, errors, *, poll_seconds=0.5):
        """Dedicated capture thread; GPU evaluation never blocks this loop."""
        try:
            while not stop.is_set():
                self.capture_once()
                stop.wait(poll_seconds)
        except BaseException as exc:
            errors.append(exc)
            stop.set()
