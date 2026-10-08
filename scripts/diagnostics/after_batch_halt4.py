"""Owned run: wait for a durable batch, migrate the halt rule, resume automatically.

Credentials remain in memory. Doctor servers are reused. Only the identified
training coordinator and its descendants are stopped after checkpoint validation.
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.diagnostics.after_batch_doctor_bench import info,same
from scripts.diagnostics.restart_shared_pool import descendants
from scripts.diagnostics.transition_halt4 import transition, validate_config_change
from src.online_rl.integrated.records import atomic_write_json,sha256_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--old-config',default='configs/online_rl/tree_v21_five_gpu_shared5.json')
    p.add_argument('--new-config',default='configs/online_rl/tree_v21_five_gpu_shared5_halt4stable.json')
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--launch',action='store_true')
    action.add_argument('--dry-run',action='store_true',help='Read-only validation; never stop, launch or write status')
    args=p.parse_args()
    old=json.loads((ROOT/args.old_config).read_text());new=json.loads((ROOT/args.new_config).read_text())
    validate_config_change(old,new)
    from scripts.online_rl.run_five_gpu import preflight
    for cfg in (old,new):
        errors=preflight(cfg)
        if errors:raise RuntimeError('Preflight failed: '+str(errors))
    original=info(args.pid)
    if (b'scripts/online_rl/run_five_gpu.py' not in original['cmd'] or
            os.readlink(f'/proc/{args.pid}/cwd')!=str(ROOT)):
        raise RuntimeError('Target is not the repository training coordinator')
    source=(ROOT/old['output_dir']).resolve()
    state=json.loads((source/'run_state.json').read_text());target_batch=state['batch_cursor']+1
    if state['config_sha256']!=sha256_json(old):raise RuntimeError('Source config differs')
    if (ROOT/new['output_dir']).exists():raise RuntimeError('Destination run already exists')
    print(json.dumps({'stage':'transition_preflight','pid':args.pid,'target_batch':target_batch,
                      'launch':args.launch,'new_config':args.new_config}),flush=True)
    if not args.launch:return
    guard=(source/'halt4_handoff.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env={}
    for entry in Path(f'/proc/{args.pid}/environ').read_bytes().split(b'\0'):
        if b'=' in entry:
            key,value=entry.split(b'=',1);env[os.fsdecode(key)]=os.fsdecode(value)
    if not env.get('OPENROUTER_API_KEY'):raise RuntimeError('No process-environment API credential')
    status_path=ROOT/'outputs/diagnostics/doctor_parallel_20261008/handoff_status.json'
    status=json.loads(status_path.read_text())
    if status.get('pid')!=args.pid or status.get('config')!=args.old_config:
        raise RuntimeError('Controller identity differs')
    status.update(stage='waiting_for_batch_halt4',transition_controller_pid=os.getpid(),
                  target_batch=target_batch,pending_config=args.new_config,
                  pending_rubric_schema=new['rubric'].get('schema'),pending_candidate_seed=new['prompt_update'].get('candidate_seed'),
                  pending_match_batch_size=new['rubric'].get('match_batch_size',1),
                  pending_repeat_agent_reply_after_turns=new['rollout']['halting'].get('repeat_agent_reply_after_turns'))
    atomic_write_json(status_path,status)
    print(json.dumps({'stage':'waiting_for_batch_halt4','target_batch':target_batch}),flush=True)
    while True:
        if not same(args.pid,original['start']):
            status.update(stage='halt4_wait_failed',transition_error='TrainingExitedBeforeTransition')
            atomic_write_json(status_path,status)
            raise RuntimeError('Training exited before transition')
        state=json.loads((source/'run_state.json').read_text())
        if state['batch_cursor']>=target_batch:break
        time.sleep(.5)
    stopped=False;launched=False;run=source;config=args.old_config
    os.kill(args.pid,signal.SIGSTOP)
    try:
        state=json.loads((source/'run_state.json').read_text())
        proof=json.loads((Path(state['checkpoint'])/'COMPLETE.json').read_text())
        if (proof['completed_cases'],proof['policy_sha256'])!=(state['case_cursor'],state['policy_sha256']):
            raise RuntimeError('Checkpoint receipt differs from committed state')
        if state['case_cursor']%5:raise RuntimeError('Not a five-case batch boundary')
        if any(not (source/'cases'/f'case_{i:06d}.json').exists() for i in range(state['case_cursor'])):
            raise RuntimeError('Committed evidence missing')
        owned=descendants(args.pid)
        doctor_pids={int(path.read_text()) for path in (source/'doctor_servers').glob('gpu*.pid')}
        if doctor_pids & {pid for pid,_ in owned}:raise RuntimeError('Doctor server in training process tree')
        status.update(stage='transitioning_halt4',completed_cases=state['case_cursor'])
        atomic_write_json(status_path,status)
        for pid,start in owned:
            if same(pid,start):os.kill(pid,signal.SIGTERM)
        os.kill(args.pid,signal.SIGTERM);os.kill(args.pid,signal.SIGCONT);stopped=True
        deadline=time.monotonic()+30
        while same(args.pid,original['start']) or any(same(pid,start) for pid,start in owned):
            if time.monotonic()>=deadline:raise RuntimeError('Owned training processes did not exit')
            time.sleep(.25)
        run=transition(args.old_config,args.new_config);config=args.new_config
        env.update(SHERPA_FIVE_GPU_CONFIG=config,SHERPA_AGENT_BATCH_SIZE='4',SHERPA_DOCTOR_CONCURRENCY='1')
        with (run/'train.log').open('a') as log:
            proc=subprocess.Popen(['bash','scripts/online_rl/launch_five_gpu.sh'],cwd=ROOT,env=env,
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        launched=True
        status.update(stage='training_running',pid=proc.pid,config=config,run_dir=str(run),
                      stable_choice_no_objection_after_turns=4,case_agent_workers=20,case_doctor_workers=5,
                      restarted_at=time.time(),transition_completed_cases=state['case_cursor'])
        status.update(rubric_schema=new['rubric'].get('schema'),candidate_seed=new['prompt_update'].get('candidate_seed'),
                      repeat_agent_reply_after_turns=new['rollout']['halting'].get('repeat_agent_reply_after_turns'))
        status['match_batch_size']=new['rubric'].get('match_batch_size',1)
        status.pop('pending_config',None)
        status.pop('pending_rubric_schema',None);status.pop('pending_candidate_seed',None)
        status.pop('pending_match_batch_size',None);status.pop('pending_repeat_agent_reply_after_turns',None)
        atomic_write_json(status_path,status)
        print(json.dumps({'stage':'halt4_resume_launched','pid':proc.pid,'completed_cases':state['case_cursor'],
                          'run_dir':str(run)}),flush=True)
        time.sleep(3)
        if proc.poll() is not None:raise RuntimeError('Resumed launcher exited; inspect train.log')
    except BaseException as exc:
        status.update(transition_error=type(exc).__name__)
        if not stopped:
            if same(args.pid,original['start']):os.kill(args.pid,signal.SIGCONT)
            status['stage']='training_running'
        elif not launched:
            # A failed migration resumes the original immutable experiment.
            env['SHERPA_FIVE_GPU_CONFIG']=args.old_config
            with (source/'train.log').open('a') as log:
                proc=subprocess.Popen(['bash','scripts/online_rl/launch_five_gpu.sh'],cwd=ROOT,env=env,
                                      stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            status.update(stage='training_running',pid=proc.pid,config=args.old_config,run_dir=str(source))
        else:status['stage']='halt4_resume_failed'
        atomic_write_json(status_path,status)
        raise


if __name__=='__main__':main()
