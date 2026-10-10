"""Independent, training-priority evaluation of one immutable checkpoint.

Captures C20 without changing the live training run, then runs one additional
Agent/Doctor pair on each requested GPU only through cooperative admission.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.idle_eval_snapshot import _check_receipt, _check_source, snapshot

TRAINING_CONTINUATION = 'idle_eval_training_continuation_v1'


def resolve_training_control(source, manifest_sha, target_cases, pointer_path):
    """Follow an explicit supervision handoff for terminal control only.

    The C20 evaluation source, model, job and cooperative priority directory are
    untouched. A pointer is accepted only with a pinned destination manifest and
    a complete handoff proof back to the original, pinned source run.
    """
    from src.online_rl.tree.supervision_handoff import CONTRACT, SPEC_KEYS
    source, pointer_path = Path(source).resolve(), Path(pointer_path)
    if pointer_path.is_symlink():
        raise RuntimeError('Training continuation pointer must not be a symlink')
    if not pointer_path.exists():
        return source, manifest_sha, target_cases

    def read(path):
        if path.is_symlink():
            raise RuntimeError('Training continuation identity file must not be a symlink')
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise RuntimeError('Training continuation requires JSON identity objects')
        return value

    pointer = read(pointer_path)
    if (set(pointer) != {'contract', 'training_run', 'training_manifest_sha256'} or
            pointer.get('contract') != TRAINING_CONTINUATION or
            not isinstance(pointer.get('training_run'), str) or not pointer['training_run'] or
            not isinstance(pointer.get('training_manifest_sha256'), str) or
            not pointer['training_manifest_sha256']):
        raise RuntimeError('Invalid training continuation pointer contract')
    destination = Path(pointer['training_run'])
    if not destination.is_absolute():
        raise RuntimeError('Training continuation run must use an absolute path')
    destination = destination.resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise RuntimeError('Training continuation must use a separate run directory')
    old_manifest, old_state = read(source/'run_manifest.json'), read(source/'run_state.json')
    _check_source(old_manifest, old_state, manifest_sha)
    manifest, state = read(destination/'run_manifest.json'), read(destination/'run_state.json')
    next_sha = pointer['training_manifest_sha256']
    _check_source(manifest, state, next_sha)
    if (type(target_cases) is not int or target_cases < 1 or
            old_manifest['config']['schedule']['cases'] != target_cases or
            manifest['config']['schedule']['cases'] != target_cases):
        raise RuntimeError('Training continuation changed the declared final target')
    proof = manifest.get('supervision_handoff')
    spec = manifest['config'].get('supervision_handoff')
    if (not isinstance(proof, dict) or not isinstance(spec, dict) or set(spec) != SPEC_KEYS or
            proof.get('contract') != CONTRACT or spec.get('contract') != CONTRACT or
            any(proof.get(key) != value for key, value in spec.items())):
        raise RuntimeError('Training continuation has no matching supervision handoff')
    if (Path(proof['source_run']).resolve() != source or proof['source_manifest_sha256'] != manifest_sha or
            proof['source_state_sha256'] != sha256_json(old_state) or
            proof.get('completed_cases') != old_state['case_cursor'] or
            proof.get('source_policy_sha256') != old_state['policy_sha256'] or
            proof.get('source_protocol_sha256') != old_state['protocol_sha256'] or
            proof.get('destination_protocol_sha256') != state['protocol_sha256'] or
            proof.get('destination_config_sha256') != state['config_sha256']):
        raise RuntimeError('Training continuation handoff source/destination identity differs')
    cursor = proof['completed_cases']
    if not 0 < cursor < target_cases or state['case_cursor'] < cursor:
        raise RuntimeError('Training continuation handoff cursor is invalid')
    proof_sha = sha256_json(proof)
    if (state.get('supervision_handoff_sha256') != proof_sha or
            read(destination/'supervision_handoff.json') != proof):
        raise RuntimeError('Training continuation handoff proof changed')
    source_checkpoint = Path(old_state['checkpoint']).resolve()
    if (Path(proof['source_checkpoint']).resolve() != source_checkpoint or
            source_checkpoint.parent != source/'checkpoints'):
        raise RuntimeError('Training continuation source checkpoint differs')
    source_receipt = read(source_checkpoint/'COMPLETE.json')
    _check_receipt(source_receipt, old_state, cursor)
    if proof.get('source_receipt') != source_receipt:
        raise RuntimeError('Training continuation source receipt differs')
    receipt = read(destination/'checkpoints'/f'handoff_{cursor:06d}'/'COMPLETE.json')
    expected = {'completed_cases': cursor, 'policy_sha256': old_state['policy_sha256'],
                'protocol_sha256': state['protocol_sha256'], 'supervision_handoff_sha256': proof_sha,
                'source_policy_sha256': old_state['policy_sha256']}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Training continuation imported checkpoint identity differs')
    return destination, next_sha, target_cases


def terminal_training_receipt(source, manifest_sha, target_cases, priority_state):
    """Prove normal training completion before allowing evaluation to continue.

    ``priority_state`` must be a freshly pruned coordinator snapshot obtained in
    the supervisor's host process. This helper never opens the coordinator.
    A missing trainer before the exact final cursor does not grant permission.
    """
    source = Path(source).resolve()
    if type(target_cases) is not int or target_cases < 1:
        raise ValueError('Invalid final training cursor')
    if (priority_state.get('version') != 1 or
            not isinstance(priority_state.get('gpus'), dict)):
        raise RuntimeError('Invalid training priority snapshot')
    if any(gpu.get('registrations') or gpu.get('training')
           for gpu in priority_state['gpus'].values()):
        return None
    # The training runner owns this lock throughout its final cleanup. Open the
    # existing lock read-only; absence is not evidence of an exited trainer.
    try:
        handle = (source / 'runner.lock').open('rb')
    except FileNotFoundError:
        return None
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            manifest = json.loads((source / 'run_manifest.json').read_text())
            state = json.loads((source / 'run_state.json').read_text())
            _check_source(manifest, state, manifest_sha)
            if manifest['config']['schedule']['cases'] != target_cases:
                raise RuntimeError('Final training target differs from the pinned plan')
            if state['case_cursor'] < target_cases:
                return None
            if state['case_cursor'] != target_cases:
                raise RuntimeError('Source passed the configured final training cursor')
            checkpoint = source / 'checkpoints' / f'checkpoint_{target_cases:06d}'
            if (checkpoint.is_symlink() or (checkpoint / 'adapter').is_symlink() or
                    Path(state['checkpoint']).resolve() != checkpoint):
                raise RuntimeError('Final training checkpoint differs from the committed state')
            receipt_path = checkpoint / 'COMPLETE.json'
            for path in (receipt_path, checkpoint / 'adapter/adapter_model.safetensors',
                         checkpoint / 'adapter/adapter_config.json'):
                if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError('Final training checkpoint payload is missing or invalid')
            receipt = json.loads(receipt_path.read_text())
            _check_receipt(receipt, state, target_cases)
            return {'source_run': str(source), 'completed_cases': target_cases,
                    'source_manifest_sha256': manifest_sha,
                    'source_state_sha256': sha256_json(state),
                    'checkpoint_receipt_sha256': sha256_json(receipt),
                    'policy_sha256': state['policy_sha256'],
                    'protocol_sha256': state['protocol_sha256']}
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def evaluation_progress(eval_root, job):
    """Use file counts for progress, validated results for completion."""
    from src.online_rl.tree.idle_eval_worker import summarize
    count = len(list((Path(eval_root) / 'cases').glob('case_*.json')))
    if count >= job['dataset_cases']:
        summary = summarize(eval_root, job)
        return summary['evaluated'], summary['status'] == 'complete'
    return count, False


def supervise(args):
    source, out = Path(args.source_run).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.capture_only:
        # The owning systemd service cleans its old cgroup before a restart.
        (out/'evaluation/STOP.json').unlink(missing_ok=True)
    config = str(Path(args.config).resolve())
    manifest = json.loads((source/'run_manifest.json').read_text())
    manifest_sha = sha256_json(manifest)
    plan = {'contract': 'idle_checkpoint_evaluation_v1', 'source_run': str(source),
            'source_manifest_sha256': manifest_sha, 'checkpoint_cases': args.checkpoint_cases,
            'config': config, 'gpus': args.gpus, 'priority_dir': str(Path(args.priority_dir).resolve()),
            'doctor_base_port': args.doctor_base_port, 'doctor_binary': str(Path(args.doctor_binary).resolve()),
            'dataset_cases': 622, 'snapshot': str(out/'snapshot'), 'evaluation': str(out/'evaluation')}
    plan_path = out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Existing idle evaluation plan differs')
    atomic_write_json(plan_path, plan)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    processes = {}
    retries = {}
    state_path = out/('capture_status.json' if args.capture_only else 'status.json')

    def status(stage, **extra):
        row = {'stage': stage, 'updated_at': time.time(), 'pid': os.getpid(),
               'checkpoint_cases': args.checkpoint_cases,
               'worker_pids': {str(g): p.pid for g,p in processes.items() if p.is_alive()}, **extra}
        atomic_write_json(state_path, row)

    try:
        status('waiting_checkpoint')
        while not stopping:
            captured = snapshot(source, args.checkpoint_cases, out/'snapshot',
                                expected_manifest_sha256=manifest_sha)
            if captured is not None:
                break
            time.sleep(1)
        if stopping:
            return
        print(json.dumps({'stage': 'checkpoint_captured', 'checkpoint_cases': args.checkpoint_cases,
                          'snapshot': str(captured)}), flush=True)
        if args.capture_only:
            status('captured')
            return
        from src.online_rl.tree.idle_eval_priority import PriorityCoordinator
        from src.online_rl.tree.idle_eval_worker import ensure_job, run_worker
        job = ensure_job(config, captured, out/'evaluation')
        if job['dataset_cases'] != plan['dataset_cases']:
            raise RuntimeError('Evaluation dataset count differs from the pinned plan')
        priority = PriorityCoordinator(args.priority_dir)
        context = multiprocessing.get_context('spawn')
        training_complete = None
        continuation_control = None
        while not stopping:
            for gpu, process in list(processes.items()):
                if process.is_alive():
                    continue
                process.join()
                del processes[gpu]
                if process.exitcode != 0:
                    # Do not respawn over a possibly surviving Doctor or GPU
                    # context. Service teardown clears this entire eval cgroup.
                    raise RuntimeError(f'Evaluation GPU{gpu} worker exited {process.exitcode}')
                retries[gpu] = time.monotonic() + 3
            evaluated, complete = evaluation_progress(out/'evaluation', job)
            if complete:
                status('complete', evaluated=evaluated)
                break
            control = resolve_training_control(source, manifest_sha, manifest['config']['schedule']['cases'],
                                               out/'training_continuation.json')
            if continuation_control is not None and control != continuation_control:
                raise RuntimeError('Pinned training continuation was removed or changed')
            if control[0] != source:
                continuation_control = control
            if training_complete is None:
                training_complete = terminal_training_receipt(
                    *control, priority.status())
                if training_complete is not None:
                    atomic_write_json(out/'training_complete.json', training_complete)
            # After proven terminal completion only the registration requirement
            # is waived. Workers still honor all activity and update leases.
            require_training = training_complete is None
            ready = not require_training or priority.ready(args.gpus)
            if ready:
                for gpu in args.gpus:
                    if gpu in processes or time.monotonic() < retries.get(gpu, 0):
                        continue
                    process = context.Process(target=run_worker,
                        args=(config, str(captured), str(out/'evaluation'), gpu,
                              args.priority_dir, args.doctor_binary, args.doctor_base_port+gpu),
                        kwargs={'require_training': require_training},
                        name=f'idle-eval-gpu{gpu}')
                    process.start()
                    processes[gpu] = process
            status('running' if ready else 'waiting_training_priority_hooks', evaluated=evaluated,
                   require_training=require_training, terminal_control_run=str(control[0]))
            time.sleep(1)
    finally:
        # Set a durable request checked between model calls. The service owns
        # all child processes and has KillMode=control-group as a final cleanup.
        if processes:
            atomic_write_json(out/'evaluation/STOP.json', {'requested_at': time.time()})
        for process in processes.values():
            process.join(timeout=5)
        for process in processes.values():
            if process.is_alive():
                process.terminate()
        for process in processes.values():
            process.join(timeout=5)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint-cases', type=int, default=20)
    parser.add_argument('--priority-dir', required=True)
    parser.add_argument('--gpus', type=lambda s: [int(x) for x in s.split(',')], default=[0,1,2,3,4])
    parser.add_argument('--doctor-binary', required=True)
    parser.add_argument('--doctor-base-port', type=int, default=11785)
    parser.add_argument('--capture-only', action='store_true')
    args = parser.parse_args()
    supervise(args)


if __name__ == '__main__':
    main()
