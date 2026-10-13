"""Resume an owned, pre-update shared-pool run with wider submission windows.

Completed immutable call caches and checkpoints are retained. Credentials travel
only through memory/process environments. No Doctor server is stopped.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.after_batch_doctor_bench import info, same, children
from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.effectiveness import RERUN_CONTINUATION_PROTOCOL


def frozen_batch(cfg, run):
    state = json.loads((run/'run_state.json').read_text())
    if state['config_sha256'] != sha256_json(cfg):
        raise RuntimeError('Run configuration differs; restart refused')
    receipt = json.loads((Path(state['checkpoint'])/'COMPLETE.json').read_text())
    if (receipt['completed_cases'], receipt['policy_sha256']) != (state['case_cursor'], state['policy_sha256']):
        raise RuntimeError('Checkpoint does not match committed state')
    batch = run/'batches'/f'batch_{state["batch_cursor"]+1:06d}'
    manifest = json.loads((batch/'manifest.json').read_text())
    if manifest['source_policy_sha256'] != state['policy_sha256']:
        raise RuntimeError('Pending batch has a different policy')
    cases = [batch/f'case_{a["index"]:06d}' for a in manifest['assignments']]
    if any((c/name).exists() for c in cases for name in ('P1.json', 'union.json')):
        raise RuntimeError('Only an unfinished P0 batch can be restarted here')
    if all((c/'P0.json').exists() for c in cases):
        raise RuntimeError('P0 barrier already passed; restart refused')
    stage = None
    for line in (run/'train.log').open():
        try: row = json.loads(line)
        except ValueError: continue
        if isinstance(row, dict) and row.get('stage'): stage = row['stage']
    if stage != 'prompt_P0':
        raise RuntimeError('Coordinator is not in the frozen P0 phase')
    return state, cases


def descendants(pid):
    owned = children(pid)
    for child, _ in list(owned): owned.extend(descendants(child))
    return owned


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--config', default='configs/online_rl/tree_v21_five_gpu_shared5.json')
    p.add_argument('--one-step-rerun-restart', action='store_true',
                   help='Preserve rollout caches, archive stale P0 results, and recompute P0 under the one-step rerun protocol')
    p.add_argument('--launch', action='store_true')
    args = p.parse_args()
    cfg = json.loads((ROOT/args.config).read_text())
    if cfg['five_gpu'].get('execution_mode') != 'shared_ready_tasks_v1':
        raise RuntimeError('Only shared five-GPU runs are supported')
    run = (ROOT/cfg['output_dir']).resolve()
    guard = (run/'pool_restart.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original = info(args.pid)
    if (b'scripts/online_rl/run_five_gpu.py' not in original['cmd'] or
            os.readlink(f'/proc/{args.pid}/cwd') != str(ROOT)):
        raise RuntimeError('PID is not the repository training coordinator')
    state, cases = frozen_batch(cfg, run)
    print(json.dumps({'stage':'restart_preflight','completed_cases':state['case_cursor'],
                      'preserve_completed_caches':True,'native_agent_batch':4,'native_doctor_requests':1,
                      'case_agent_workers':20,'case_doctor_workers':5,'launch':args.launch}), flush=True)
    if not args.launch: return
    env = {}
    for entry in Path(f'/proc/{args.pid}/environ').read_bytes().split(b'\0'):
        if b'=' in entry:
            key, value = entry.split(b'=', 1); env[os.fsdecode(key)] = os.fsdecode(value)
    if not env.get('OPENROUTER_API_KEY'): raise RuntimeError('No in-memory API credential')
    env.update(SHERPA_FIVE_GPU_CONFIG=args.config, SHERPA_AGENT_BATCH_SIZE='4', SHERPA_DOCTOR_CONCURRENCY='1')
    status_path = ROOT/'outputs/diagnostics/doctor_parallel_20261008/handoff_status.json'
    status = json.loads(status_path.read_text())
    if status.get('pid') != args.pid or status.get('config') != args.config:
        raise RuntimeError('Active controller identity differs')
    audit_path = run/'pool_request_fanout_restart.json'
    audit = {'old_pid':args.pid,'requested_at':time.time(),'checkpoint':state['checkpoint'],
             'case_cursor':state['case_cursor'],'config_sha256':state['config_sha256'],
             'credential_storage':'process_environment_only','phase':'P0',
             'case_agent_workers':20,'case_doctor_workers':5}
    if args.one_step_rerun_restart:
        manifests=[json.loads((case/'run_manifest.json').read_text()) for case in cases]
        protocols={m.get('rerun_continuation_protocol') for m in manifests}
        if protocols != {None}:
            raise RuntimeError('Pending P0 case manifests are not uniformly on the old rerun protocol')
        if any((case/'P0_one_step_legacy.json').exists() for case in cases):
            raise RuntimeError('Legacy P0 archive already exists; refusing to overwrite')
        audit['rerun_transition']={'from':'multi_turn_doctor_agent_continuation','to':RERUN_CONTINUATION_PROTOCOL,
                                   'rollout_call_cache_reused':True,'old_p0_results_archived':[]}
    stopped = False
    os.kill(args.pid, signal.SIGSTOP)
    try:
        if frozen_batch(cfg, run)[0] != state:
            raise RuntimeError('Committed state changed before suspension')
        owned = descendants(args.pid)
        doctor_pids = {int(path.read_text()) for path in (run/'doctor_servers').glob('gpu*.pid')}
        if doctor_pids & {pid for pid, _ in owned}:
            raise RuntimeError('Doctor server appears in coordinator process tree')
        for pid, start in owned:
            if same(pid, start): os.kill(pid, signal.SIGTERM)
        os.kill(args.pid, signal.SIGTERM); os.kill(args.pid, signal.SIGCONT)
        stopped = True
        deadline = time.monotonic()+30
        while same(args.pid, original['start']) or any(same(pid, start) for pid, start in owned):
            if time.monotonic() >= deadline: raise RuntimeError('Old training processes did not exit')
            time.sleep(.25)
        # Refuse to continue on damaged evidence rather than silently dropping it.
        from src.online_rl.tree.cache import CallCache
        audit['completed_cache_rows'] = {c.name:len(CallCache(c/'calls.jsonl').store) for c in cases}
        if json.loads((run/'run_state.json').read_text()) != state:
            raise RuntimeError('Committed state changed during stop')
        if args.one_step_rerun_restart:
            for case in cases:
                old=case/'P0.json'
                if old.exists():
                    archived=case/'P0_one_step_legacy.json'
                    old.replace(archived)
                    audit['rerun_transition']['old_p0_results_archived'].append(str(archived.relative_to(run)))
                manifest_path=case/'run_manifest.json'
                manifest=json.loads(manifest_path.read_text())
                manifest['rerun_continuation_protocol']=RERUN_CONTINUATION_PROTOCOL
                atomic_write_json(manifest_path,manifest)
            manifest_path=run/'run_manifest.json'
            manifest=json.loads(manifest_path.read_text())
            manifest['rerun_continuation_protocol']=RERUN_CONTINUATION_PROTOCOL
            atomic_write_json(manifest_path,manifest)
        atomic_write_json(audit_path, {**audit,'stage':'stopped'})
        with (run/'train.log').open('a') as log:
            proc = subprocess.Popen(['bash','scripts/online_rl/launch_five_gpu.sh'],cwd=ROOT,env=env,
                                    stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        status.update(stage='pool_resume_launched',pid=proc.pid,restarted_at=time.time(),
                      case_agent_workers=20,case_doctor_workers=5)
        atomic_write_json(status_path,status)
        atomic_write_json(audit_path,{**audit,'stage':'resume_launched','new_pid':proc.pid})
        print(json.dumps({'stage':'resume_launched','pid':proc.pid,'run_dir':str(run),
                          'completed_cache_rows':audit['completed_cache_rows']}),flush=True)
        time.sleep(3)
        if proc.poll() is not None: raise RuntimeError('Resumed launcher exited; inspect train.log')
    finally:
        if not stopped and same(args.pid, original['start']): os.kill(args.pid, signal.SIGCONT)


if __name__ == '__main__': main()
