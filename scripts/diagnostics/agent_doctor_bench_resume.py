"""Run the authorized joint 1/4/8 comparison, then resume 500 cases under minimum 2."""
import json
import os
from pathlib import Path
import statistics
import socket
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def choose(out):
    from scripts.diagnostics.after_batch_doctor_bench import select
    from src.collaboration.generator import option
    doctor=select(out);results=json.loads((out/'results.json').read_text())
    baseline=statistics.median(r['combined_workload_seconds'] for r in results if r['parallel']==1)
    arms=[]
    for d in doctor['arms']:
        p=d['parallel'];matches=total=0;empty=0
        for repeat in range(3):
            a=json.loads((out/f'agent_responses_r{repeat}_p1.json').read_text())
            b=json.loads((out/f'agent_responses_r{repeat}_p{p}.json').read_text())
            for x,y in zip(a,b):
                total+=2;matches+=option(x['private_assessment'])==option(y['private_assessment'])
                matches+=option(x['completion_text'])==option(y['completion_text'])
                empty+=not y['completion_text'].strip()
        seconds=statistics.median(r['combined_workload_seconds'] for r in results if r['parallel']==p)
        speedup=baseline/seconds
        eligible=(speedup>=1.1 and d['status_errors']<=d['baseline_status_errors'] and d['status_agreement']>=.95
                  and d['final_matches']==d['final_total'] and matches/max(1,total)>=.95 and not empty
                  and all(not r['empty_responses'] for r in results if r['parallel']==p))
        arms.append({'parallel':p,'combined_speedup':speedup,'agent_option_agreement':matches/max(1,total),
                     'eligible':eligible,'combined_median_seconds':seconds,'doctor':d})
    good=[r for r in arms if r['eligible']];best=min(good,key=lambda r:r['combined_median_seconds']) if good else None
    decision={'parallel':best['parallel'] if best else 1,'arms':arms,'scope':'64 Agent actions + 64 Doctor requests, separate stages; not full training-batch timing'}
    (out/'joint_decision.json').write_text(json.dumps(decision,indent=2));return decision


def main():
    out=ROOT/'outputs/diagnostics/doctor_parallel_20261008'
    private=out/'restart_environment.private'
    env={}
    for item in private.read_bytes().split(b'\0'):
        if b'=' in item:
            k,v=item.split(b'=',1);env[os.fsdecode(k)]=os.fsdecode(v)
    def status(stage,**kw):
        row={'stage':stage,'time':time.time(),**kw};(out/'handoff_status.json').write_text(json.dumps(row));print(json.dumps(row),flush=True)
    cfg='configs/online_rl/tree_v21_five_gpu_min2_parallel.json';parallel=1;success=False
    status('joint_benchmark_running',gpu=1,agent_actions=64,doctor_requests=64,repeats=3)
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        status('benchmark_port_selected',port=port)
        subprocess.run([sys.executable,'scripts/diagnostics/doctor_parallel_bench.py','--out',str(out),'--gpu','1','--port',str(port),'--repeats','3','--both'],cwd=ROOT,env=env,check=True)
        decision=choose(out);parallel=decision['parallel'];success=True
        status('joint_benchmark_complete',decision=decision)
    except Exception as e:
        status('joint_benchmark_failed',error=type(e).__name__)
        cfg='configs/online_rl/tree_v21_five_gpu_min2.json'
    env['SHERPA_FIVE_GPU_CONFIG']=cfg;env['SHERPA_DOCTOR_CONCURRENCY']=str(parallel);env['SHERPA_AGENT_BATCH_SIZE']=str(parallel)
    data=json.loads((ROOT/cfg).read_text());run=ROOT/data['output_dir'];run.mkdir(parents=True,exist_ok=True)
    (run/'restart_from_previous.json').write_text(json.dumps({'source_run':'outputs/sherpa_tree_v21_five_gpu/run500_ckpt24_20261008',
        'source_committed_cases':0,'discard_incomplete_old_protocol_trees':True,'min_agent_turns':2,
        'agent_batch_size':parallel,'doctor_concurrency':parallel,'joint_benchmark_succeeded':success}))
    status('training_resume_launch',config=cfg,parallel=parallel)
    with (run/'train.log').open('a') as log:
        proc=subprocess.Popen(['bash','scripts/online_rl/launch_five_gpu.sh'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    private.unlink()
    status('resume_launched',pid=proc.pid,config=cfg,run_dir=str(run),parallel=parallel)
    time.sleep(10)
    if proc.poll() is not None:status('resume_failed',exit_code=proc.returncode)

if __name__=='__main__':main()
