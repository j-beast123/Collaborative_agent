"""Capture every two actual optimizer updates and evaluate a fixed subset idle."""
import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.idle_eval_snapshot import _check_source
from src.online_rl.tree.periodic_idle_eval import CONTRACT, CaptureQueue, committed, read
from scripts.online_rl.run_idle_checkpoint_eval import terminal_training_receipt


def opening_inventory(selection_path, source):
    """Pin exact baseline D1 evidence; an opening is never freshly generated."""
    from src.online_rl.tree.balanced_eval_subset import validate_selection
    from src.online_rl.cases import normalize
    selection_path, source = Path(selection_path).resolve(), Path(source).resolve()
    selected = validate_selection(selection_path)
    baseline = read(selection_path.parent / 'baseline_manifest.json')
    inventory = {}
    for item in selected['cases']:
        result = read(selection_path.parent / item['result_path'])
        case = normalize(result['original_case'])
        identity = {'case': {key: case.get(key) for key in ('case_id', 'scenario', 'question', 'options')},
                    'protocol': baseline['opening_identity'], 'rounds': baseline['rollout']['rounds']}
        opening_sha = sha256_json(identity)
        name = opening_sha + '.jsonl'
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Selected baseline opening is missing or symlinked')
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if len(rows) != 1:
            raise RuntimeError('Baseline opening cache must contain exactly one immutable D1')
        row = rows[0]
        expected_seed = int(opening_sha[:8], 16) % (2**31 - 1)
        opening = next((turn for turn in result['dialogue'] if turn.get('stage') == 'D1' and turn.get('speaker') == 'doctor'), None)
        if (row.get('kind') != 'shared_D1' or row.get('identity') != identity or
                row.get('key') != sha256_json(['shared_D1', identity]) or
                row.get('value', {}).get('seed') != expected_seed or
                result.get('opening_identity_sha256') != opening_sha or
                result.get('opening_seed') != expected_seed or opening is None or
                row.get('value', {}).get('visible') != opening.get('text')):
            raise RuntimeError('Baseline D1 differs from selected result evidence')
        inventory[name] = {'file_sha256': hashlib.sha256(payload).hexdigest(),
                           'case_id': case['case_id'], 'opening_identity_sha256': opening_sha}
    return inventory


def install_openings(plan, evaluation):
    """Copy D1 only, validating existing files rather than overwriting them."""
    job_root = Path(evaluation).parent
    destination = job_root / 'openings'
    destination.mkdir(parents=True, exist_ok=True)
    for name, proof in plan['opening_files'].items():
        source = Path(plan['opening_cache_source']) / name
        if source.is_symlink():
            raise RuntimeError('Baseline opening source became a symlink')
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != proof['file_sha256']:
            raise RuntimeError('Pinned baseline D1 cache changed')
        target = destination / name
        if target.exists():
            if target.is_symlink() or target.read_bytes() != payload:
                raise RuntimeError('Job D1 cache differs from pinned baseline')
            continue
        descriptor, temporary = tempfile.mkstemp(prefix='.opening-', dir=destination)
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    proof = {'contract': 'periodic_fixed_baseline_openings_v1',
             'source': plan['opening_cache_source'], 'selection_file_sha256': plan['selection_file_sha256'],
             'files': plan['opening_files']}
    proof['sha256'] = sha256_json(proof)
    path = job_root / 'opening_cache_manifest.json'
    if path.exists() and read(path) != proof:
        raise RuntimeError('Pinned job opening manifest changed')
    atomic_write_json(path, proof)
    return proof['sha256']


def preflight(args):
    source, output = Path(args.source_run).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Periodic evaluation output must be separate from training')
    if (args.start_update < 0 or args.start_cases < 1 or args.every_updates < 1 or
            not args.gpus or len(set(args.gpus)) != len(args.gpus) or
            any(gpu not in range(5) for gpu in args.gpus)):
        raise ValueError('Invalid periodic schedule or GPUs')
    selection = Path(args.selection).resolve()
    openings = opening_inventory(selection, args.opening_cache_source)
    manifest = read(source / 'run_manifest.json')
    manifest_sha = sha256_json(manifest)
    observed = committed(source, manifest_sha)
    if observed is None:
        raise RuntimeError('Training state changed repeatedly during preflight; retry')
    _, state, _, steps = observed
    if steps < args.start_update or state['case_cursor'] < args.start_cases:
        raise RuntimeError('Training has not reached the declared start commitment')
    initial = (Path(args.initial_snapshot).resolve() if args.initial_snapshot else
               output / 'jobs' / f'update_{args.start_update:06d}' / 'snapshot')
    if initial.exists():
        from src.online_rl.tree.idle_eval_worker import _inputs
        _, proof, receipt, _, _ = _inputs(args.config, initial)
        if (proof['source_run'] != str(source) or proof['source_manifest_sha256'] != manifest_sha or
                proof['completed_cases'] != args.start_cases or
                receipt.get('update', {}).get('optimizer_steps', 0) != args.start_update):
            raise RuntimeError('Initial snapshot differs from source/schedule')
    elif steps != args.start_update or state['case_cursor'] != args.start_cases:
        raise RuntimeError('Initial checkpoint already passed without a pinned snapshot')
    else:
        from src.online_rl.tree.experiment import load_config
        cfg = load_config(args.config)
        _check_source({'config': cfg, 'protocol_sha256': manifest['protocol_sha256']}, state, None)
    return {'contract': CONTRACT, 'source_run': str(source),
            'source_manifest_sha256': manifest_sha, 'config': str(Path(args.config).resolve()),
            'selection': str(selection), 'selection_file_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
            'opening_cache_source': str(Path(args.opening_cache_source).resolve()), 'opening_files': openings,
            'start_update': args.start_update, 'start_cases': args.start_cases,
            'every_updates': args.every_updates, 'initial_snapshot': str(initial),
            'include_terminal': not args.no_terminal, 'target_cases': manifest['config']['schedule']['cases'],
            'gpus': args.gpus, 'priority_dir': str(Path(args.priority_dir).resolve()),
            'doctor_binary': str(Path(args.doctor_binary).resolve()), 'doctor_base_port': args.doctor_base_port}


def runner_active(source):
    try:
        handle = (Path(source) / 'runner.lock').open('rb')
    except FileNotFoundError:
        return False
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def bucket_gpus(priority, bucket):
    return sorted(int(g) for g, value in priority['gpus'].items() if value.get(bucket))


def publish_status(output, queue, priority, *, current=None, summary=None, children=None,
                   require_training=True, stage=None, terminal=None):
    latest = queue.get('latest_committed') or {}
    last_completed = next((j for j in reversed(queue['jobs']) if j['status'] == 'complete'), None)
    displayed = current or last_completed or {}
    if summary is None and last_completed is not None:
        summary = last_completed.get('summary')
    pending = [j['optimizer_steps'] for j in queue['jobs'] if j['status'] != 'complete' and
               (current is None or j['optimizer_steps'] != current['optimizer_steps'])]
    active = bucket_gpus(priority, 'calls')
    if stage is None:
        stage = 'running' if active else ('waiting_for_idle_gpu' if current else 'waiting_checkpoint')
    row = {**(summary or {}), 'contract': CONTRACT, 'stage': stage, 'updated_at': time.time(),
           'pid': os.getpid(), 'latest_committed_update': latest.get('optimizer_steps'),
           'latest_committed_cases': latest.get('completed_cases'),
           'evaluating_update': displayed.get('optimizer_steps'), 'evaluation_active': current is not None,
           'checkpoint_cases': displayed.get('completed_cases', (summary or {}).get('completed_cases')),
           'evaluation': displayed.get('evaluation'), 'current_job': current,
           'last_completed_job': last_completed,
           'opening_cache_sha256': displayed.get('opening_cache_sha256'),
           'dataset_cases': (summary or {}).get('dataset_cases', 200),
           'evaluated': (summary or {}).get('evaluated', 0),
           'queued_updates': pending, 'queue_depth': len(pending),
           'lag_updates': max(0, latest.get('optimizer_steps', 0) - displayed['optimizer_steps']) if displayed else 0,
           'missed_targets': queue['missed_targets'], 'require_training': require_training,
           'active_gpus': active, 'resident_gpus': bucket_gpus(priority, 'residents'),
           'training_registered_gpus': bucket_gpus(priority, 'registrations'),
           'training_busy_gpus': bucket_gpus(priority, 'training'),
           'worker_pids': {str(g): p.pid for g, p in (children or {}).items() if p.is_alive()},
           'terminal_training_receipt': terminal}
    atomic_write_json(Path(output) / 'status.json', row)
    return row


def supervise(args, plan):
    from src.online_rl.tree.idle_eval_priority import PriorityCoordinator
    from src.online_rl.tree.idle_eval_worker import ensure_job, run_worker, summarize
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = output / 'plan.json'
        if plan_path.exists() and read(plan_path) != plan:
            raise RuntimeError('Periodic supervisor plan changed')
        atomic_write_json(plan_path, plan)
        priority = PriorityCoordinator(plan['priority_dir'], require_training=True)
        initial_priority = priority.status()
        if bucket_gpus(initial_priority, 'residents') or bucket_gpus(initial_priority, 'calls'):
            raise RuntimeError('Previous evaluator owners remain; clean its service cgroup first')
        queue = CaptureQueue(output, plan)
        queue.capture_once()
        capture_stop, errors = threading.Event(), []
        collector = threading.Thread(target=queue.collect, args=(capture_stop, errors),
                                     name='periodic-checkpoint-capture', daemon=True)
        collector.start()
        stopping = False
        def stop(*_):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        children, retry_at = {}, {}
        current = job = terminal = None
        draining = False
        drain_started = None
        summary = None
        context = multiprocessing.get_context('spawn')
        try:
            while not stopping:
                if errors:
                    raise RuntimeError(f'Checkpoint capture failed: {errors[0]}') from errors[0]
                if hashlib.sha256(Path(plan['selection']).read_bytes()).hexdigest() != plan['selection_file_sha256']:
                    raise RuntimeError('Frozen evaluation selection changed')
                current_priority = priority.status()
                if terminal is None:
                    terminal = terminal_training_receipt(plan['source_run'], plan['source_manifest_sha256'],
                                                         plan['target_cases'], current_priority)
                    if terminal:
                        atomic_write_json(output / 'training_complete.json', terminal)
                        queue.capture_once()
                require_training = terminal is None
                live = runner_active(plan['source_run'])
                for gpu, process in list(children.items()):
                    if process.is_alive():
                        continue
                    process.join()
                    del children[gpu]
                    if process.exitcode != 0:
                        raise RuntimeError(f'Periodic evaluator GPU{gpu} exited {process.exitcode}; refusing respawn')
                    retry_at[gpu] = time.monotonic() + 3
                current_priority = priority.status()
                if current is not None:
                    summary = summarize(current['evaluation'], job)
                    if summary['status'] == 'complete' and not draining:
                        draining, drain_started = True, time.monotonic()
                        atomic_write_json(Path(current['evaluation']) / 'STOP.json', {'requested_at': time.time(), 'reason': 'job_complete'})
                    if draining:
                        if not children and not bucket_gpus(current_priority, 'residents') and not bucket_gpus(current_priority, 'calls'):
                            queue.mark(current['optimizer_steps'], 'complete', summary=summary, finished_at=time.time())
                            current = job = None
                            draining = False
                            retry_at = {}
                        elif time.monotonic() - drain_started > 120:
                            raise RuntimeError('Completed job GPU/Doctor cleanup did not finish; refusing next job')
                state = queue.status()
                if current is None:
                    current = next((j for j in state['jobs'] if j['status'] != 'complete'), None)
                    if current:
                        if bucket_gpus(current_priority, 'residents') or bucket_gpus(current_priority, 'calls'):
                            raise RuntimeError('Cannot begin a new job while evaluation GPU owners remain')
                        job = ensure_job(plan['config'], current['snapshot'], current['evaluation'], selection=plan['selection'])
                        if job['dataset_cases'] != 200:
                            raise RuntimeError('Periodic job must contain the pinned 200 cases')
                        opening_sha = install_openings(plan, current['evaluation'])
                        current = {**current, 'job_sha256': job['job_sha256'], 'opening_cache_sha256': opening_sha}
                        queue.mark(current['optimizer_steps'], 'running', job_sha256=job['job_sha256'],
                                   opening_cache_sha256=opening_sha)
                        (Path(current['evaluation']) / 'STOP.json').unlink(missing_ok=True)
                        summary = summarize(current['evaluation'], job)
                    elif terminal:
                        publish_status(output, state, current_priority, summary=summary,
                                       require_training=False, stage='complete', terminal=terminal)
                        return
                registered = bucket_gpus(current_priority, 'registrations')
                ready = not require_training or (live and all(g in registered for g in plan['gpus']))
                if current and require_training and not live:
                    # A stopped/failed trainer is not a terminal-completion proof.
                    for process in children.values():
                        if process.is_alive():
                            process.terminate()
                if current and ready and not draining:
                    for gpu in plan['gpus']:
                        if gpu in children or time.monotonic() < retry_at.get(gpu, 0):
                            continue
                        process = context.Process(target=run_worker, args=(plan['config'], current['snapshot'],
                            current['evaluation'], gpu, plan['priority_dir'], plan['doctor_binary'],
                            plan['doctor_base_port'] + gpu), kwargs={'require_training': require_training,
                            'selection': plan['selection']}, name=f'periodic-eval-gpu{gpu}')
                        process.start()
                        children[gpu] = process
                stage = 'draining_workers' if draining else ('waiting_training_priority_hooks' if not ready else None)
                publish_status(output, queue.status(), priority.status(), current=current, summary=summary,
                               children=children, require_training=require_training, stage=stage, terminal=terminal)
                time.sleep(.5)
        except BaseException as exc:
            atomic_write_json(output / 'error.json', {'time': time.time(), 'error': str(exc), 'type': type(exc).__name__})
            raise
        finally:
            capture_stop.set()
            collector.join(timeout=10)
            if current:
                atomic_write_json(Path(current['evaluation']) / 'STOP.json', {'requested_at': time.time(), 'reason': 'supervisor_stopping'})
            deadline = time.monotonic() + 70
            while any(p.is_alive() for p in children.values()) and time.monotonic() < deadline:
                for process in children.values():
                    process.join(timeout=.1)
            for process in children.values():
                if process.is_alive():
                    process.terminate()
            for process in children.values():
                process.join(timeout=5)
            remaining = priority.status()
            if stopping:
                publish_status(output, queue.status(), remaining, current=current, summary=summary,
                               children=children, require_training=terminal is None, stage='stopped', terminal=terminal)
            if any(p.is_alive() for p in children.values()) or bucket_gpus(remaining, 'residents') or bucket_gpus(remaining, 'calls'):
                raise RuntimeError('Evaluator cleanup remains; owning service must clean its complete cgroup')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'source-run', 'output', 'selection', 'priority-dir', 'doctor-binary', 'opening-cache-source'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--start-update', type=int, required=True)
    parser.add_argument('--start-cases', type=int, required=True)
    parser.add_argument('--every-updates', type=int, default=2)
    parser.add_argument('--initial-snapshot')
    parser.add_argument('--gpus', type=lambda value: [int(x) for x in value.split(',')], default=list(range(5)))
    parser.add_argument('--doctor-base-port', type=int, default=11785)
    parser.add_argument('--no-terminal', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    plan = preflight(args)
    if args.check:
        print(json.dumps({'status': 'ready', 'plan': plan}), flush=True)
        return
    supervise(args, plan)


if __name__ == '__main__':
    main()
