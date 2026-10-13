"""User-authorized one-shot handoff: durable batch -> benchmark -> resume.
Credentials remain in memory; only verified run-owned processes are signalled.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def info(pid):
    root=Path('/proc')/str(pid)
    cmd=root.joinpath('cmdline').read_bytes().split(b'\0')
    stat=root.joinpath('stat').read_text().rsplit(')',1)[1].split()
    return {'cmd':cmd,'ppid':int(stat[1]),'start':stat[19],'state':stat[0]}


def same(pid,start):
    try:r=info(pid);return r['start']==start and r['state']!='Z'
    except OSError:return False


def children(pid):
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:r=info(int(path.name))
        except (OSError,ValueError):continue
        if r['ppid']==pid:result.append((int(path.name),r['start']))
    return result


def select(out):
    from src.online_rl.doctor import split_status
    from src.collaboration.generator import final_answer_option as final_option
    from src.online_rl.cases import load_cases
    prepared=json.loads((out/'requests.json').read_text());cfg=prepared['config']
    cases={c['case_id']:c for c in load_cases(ROOT/cfg['data']['train'])}
    results=json.loads((out/'results.json').read_text());decision={'parallel':1,'reason':'no verified improvement','arms':[]}
    baseline=statistics.median(r['wall_seconds'] for r in results if r['parallel']==1)
    for p in (4,8):
        times=[r['wall_seconds'] for r in results if r['parallel']==p]
        valid=len(times)==3;matches=total=0;final_matches=final_total=0;parse_errors=base_errors=0
        for repeat in range(3):
            base=json.loads((out/f'responses_r{repeat}_p1.json').read_text())
            trial=json.loads((out/f'responses_r{repeat}_p{p}.json').read_text())
            for a,b in zip(base,trial):
                if not b['response'].strip():valid=False
                if a['kind']=='final':
                    final_total+=1;aa=final_option(a['response']);bb=final_option(b['response'])
                    final_matches+=bool(aa) and aa==bb and bb in cases[b['case_id']]['options']
                else:
                    _,sa,ea=split_status(a['response'],cases[a['case_id']]['options'])
                    _,sb,eb=split_status(b['response'],cases[b['case_id']]['options'])
                    base_errors+=ea is not None;parse_errors+=eb is not None
                    total+=1;matches+=sa==sb
        seconds=statistics.median(times);speedup=baseline/seconds
        valid=valid and parse_errors<=base_errors and final_matches==final_total and (not total or matches/total>=.95) and speedup>=1.1
        arm={'parallel':p,'speedup':speedup,'status_agreement':matches/max(1,total),'final_matches':final_matches,'final_total':final_total,
             'status_errors':parse_errors,'baseline_status_errors':base_errors,'eligible':valid,'median_seconds':seconds}
        decision['arms'].append(arm)
    eligible=[r for r in decision['arms'] if r['eligible']]
    if eligible:
        best=min(eligible,key=lambda r:r['median_seconds']);decision.update(parallel=best['parallel'],reason='>=10% faster with response checks passed')
    (out/'decision.json').write_text(json.dumps(decision,indent=2));return decision


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--target-batch',type=int)
    parser.add_argument('--config',default='configs/online_rl/tree_v21_five_gpu.json');args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());run=ROOT/cfg['output_dir'];out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=info(args.pid)
    if b'scripts/online_rl/run_five_gpu.py' not in original['cmd'] or str(ROOT)!=os.readlink(f'/proc/{args.pid}/cwd'):
        raise RuntimeError('Target is not this repository training coordinator')
    env={}
    for entry in Path(f'/proc/{args.pid}/environ').read_bytes().split(b'\0'):
        if b'=' in entry:
            key,value=entry.split(b'=',1);env[os.fsdecode(key)]=os.fsdecode(value)
    env['SHERPA_FIVE_GPU_CONFIG']=args.config
    initial=json.loads((run/'run_state.json').read_text())['batch_cursor'];target=args.target_batch or initial+1
    def status(stage,**fields):
        row={'stage':stage,'time':time.time(),'target_batch':target,**fields}
        (out/'handoff_status.json').write_text(json.dumps(row));print(json.dumps(row),flush=True)
    status('waiting_for_durable_batch',pid=args.pid)
    while True:
        if not same(args.pid,original['start']):raise RuntimeError('Training exited before handoff; no process was stopped')
        state=json.loads((run/'run_state.json').read_text())
        if state['batch_cursor']>=target:break
        time.sleep(.5)
    # Freeze scheduling before checking the committed receipt; any next-batch cache is retained.
    os.kill(args.pid,signal.SIGSTOP)
    owned=children(args.pid);stopped=False
    try:
        state=json.loads((run/'run_state.json').read_text());receipt=json.loads((Path(state['checkpoint'])/'COMPLETE.json').read_text())
        if receipt['completed_cases']!=state['case_cursor'] or receipt['policy_sha256']!=state['policy_sha256']:
            raise RuntimeError('Checkpoint does not match durable run state')
        if any(not (run/'cases'/f'case_{i:06d}.json').exists() for i in range(state['case_cursor'])):
            raise RuntimeError('Committed case record missing')
        status('stopping_committed_run',completed_cases=state['case_cursor'],checkpoint=state['checkpoint'])
        for pid,start in owned:
            if same(pid,start):os.kill(pid,signal.SIGTERM)
        os.kill(args.pid,signal.SIGTERM);os.kill(args.pid,signal.SIGCONT);stopped=True
        for _ in range(120):
            if not same(args.pid,original['start']) and all(not same(pid,start) for pid,start in owned):break
            time.sleep(.5)
        else:raise RuntimeError('Training processes did not exit; benchmark not started')
        # Stop only the five Doctor servers registered to this run, after checking GPU/URL ownership.
        for gpu,url in enumerate(cfg['five_gpu']['doctor_urls']):
            pidfile=run/'doctor_servers'/f'gpu{gpu}.pid'
            pid=int(pidfile.read_text());r=info(pid)
            server_env=Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            if f'CUDA_VISIBLE_DEVICES={gpu}'.encode() not in server_env or f'OLLAMA_HOST={url.removeprefix("http://")}'.encode() not in server_env:
                raise RuntimeError('Doctor server ownership mismatch')
            from scripts.diagnostics.doctor_parallel_bench import request
            request(url+'/api/generate',{'model':cfg['rollout']['doctor_backend']['model'],'keep_alive':0})
            os.kill(pid,signal.SIGTERM)
            for _ in range(60):
                if not same(pid,r['start']):break
                time.sleep(.5)
            else:raise RuntimeError('Owned Doctor server did not exit')
        time.sleep(3);status('benchmark_running',gpu=1)
        subprocess.run([sys.executable,'scripts/diagnostics/doctor_parallel_bench.py','--out',args.out,'--gpu','1','--repeats','3','--with-agent'],cwd=ROOT,env=env,check=True)
        decision=select(out);env['SHERPA_DOCTOR_CONCURRENCY']=str(decision['parallel'])
        status('benchmark_complete',decision=decision)
    except BaseException as e:
        status('handoff_error',error=type(e).__name__)
        if not stopped:
            if same(args.pid,original['start']):os.kill(args.pid,signal.SIGCONT)
            raise
        env['SHERPA_DOCTOR_CONCURRENCY']='1'
    finally:
        if stopped:
            restart=out/'restart_request.json'
            if restart.exists():
                new_config=json.loads(restart.read_text())['config']
                try:
                    from scripts.diagnostics.transition_min_turns import transition
                    run=transition(args.config,new_config)
                    env['SHERPA_FIVE_GPU_CONFIG']=new_config
                    status('protocol_transition_ready',config=new_config,run_dir=str(run))
                except Exception as e:
                    status('protocol_transition_failed',error=type(e).__name__)
                    # A failed transition never prevents resuming the original run.
            # Resume at the saved case/prompt/optimizer state. Never print or persist the inherited environment.
            with (run/'train.log').open('a') as log:
                proc=subprocess.Popen(['bash','scripts/online_rl/launch_five_gpu.sh'],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
            status('resume_launched',pid=proc.pid,doctor_concurrency=int(env.get('SHERPA_DOCTOR_CONCURRENCY','1')))
            time.sleep(5)
            if proc.poll() is not None:status('resume_failed',exit_code=proc.returncode)

if __name__=='__main__':main()
