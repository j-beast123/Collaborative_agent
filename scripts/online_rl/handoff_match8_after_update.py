#!/usr/bin/env python3
"""At the next committed batch update, restart the five-GPU runner with matcher-8.

This watcher never interrupts rollout or an optimizer step. The resumed runner
validates the committed checkpoint, records the one-time protocol transition,
and archives any uncommitted next-batch artifacts before replaying them.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT/'outputs/diagnostics/doctor_parallel_20261008/handoff_status.json'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    os.replace(tmp, path)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmdline(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except OSError:
        return ''


def log(message, path):
    line = time.strftime('%Y-%m-%dT%H:%M:%S%z')+' '+message
    with path.open('a') as f:
        f.write(line+'\n')
        f.flush()
    print(line, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runner-pid', type=int, required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--target-config', default='configs/online_rl/tree_v21_five_gpu_shared5_random10_promptv7_match8.json')
    p.add_argument('--target-case-cursor', type=int, default=10)
    p.add_argument('--poll-seconds', type=float, default=0.25)
    p.add_argument('--python', default='/home/kai/.venvs/gpu1_probe/bin/python')
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    state_path = run_dir/'run_state.json'
    log_path = run_dir/'logs/matcher8_transition.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    old_cmd = cmdline(args.runner_pid)
    if 'scripts/online_rl/run_five_gpu.py' not in old_cmd or '--launch' not in old_cmd:
        raise SystemExit(f'Refusing: PID {args.runner_pid} is not the expected training runner: {old_cmd}')
    last_cursor = -1
    while True:
        if not alive(args.runner_pid):
            raise SystemExit(f'Runner PID {args.runner_pid} exited before target update; no restart launched.')
        state = read_json(state_path)
        if state:
            cursor = int(state.get('case_cursor', -1))
            last_cursor = max(last_cursor, cursor)
            if cursor >= args.target_case_cursor:
                checkpoint = Path(state.get('checkpoint', ''))
                receipt = read_json(checkpoint/'COMPLETE.json')
                if (int(state.get('batch_cursor', 0)) * 5 != cursor or not receipt
                        or receipt.get('completed_cases') != cursor
                        or receipt.get('policy_sha256') != state.get('policy_sha256')):
                    time.sleep(args.poll_seconds)
                    continue
                log(f'Committed update boundary reached: cases={cursor}, batch={state["batch_cursor"]}; interrupting old runner.', log_path)
                os.kill(args.runner_pid, signal.SIGINT)
                deadline = time.monotonic()+600
                while alive(args.runner_pid) and time.monotonic() < deadline:
                    time.sleep(0.5)
                if alive(args.runner_pid):
                    raise SystemExit('Old runner did not shut down cleanly within 10 minutes; refusing to start a second runner.')
                log('Old runner stopped; launching matcher-8 resume from the committed checkpoint.', log_path)
                cfg_path = ROOT/args.target_config
                cfg = json.loads(cfg_path.read_text())
                if Path(cfg['output_dir']).resolve() != run_dir.resolve():
                    raise SystemExit('Target config output_dir does not point to the existing run directory.')
                with log_path.open('a') as output:
                    proc = subprocess.Popen([args.python, 'scripts/online_rl/run_five_gpu.py',
                                             '--config', args.target_config, '--launch'],
                                            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                            start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED':'1'})
                handoff = read_json(HANDOFF) or {}
                handoff.update(stage='training_running', pid=proc.pid, config=args.target_config,
                               run_dir=str(run_dir), resumed_at=time.time(), resumed_from_checkpoint_cases=cursor,
                               matcher_batch_transition={'from':1,'to':8,'after_completed_cases':cursor,
                                                         'log':str(log_path)})
                atomic_json(HANDOFF, handoff)
                time.sleep(15)
                if proc.poll() is not None:
                    raise SystemExit(f'Matcher-8 runner exited during startup (code {proc.returncode}); see {log_path}.')
                log(f'Matcher-8 runner launched successfully: PID {proc.pid}.', log_path)
                return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
