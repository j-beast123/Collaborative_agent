"""GPU4 evaluator consumes ready snapshots, without touching the training model."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import atomic_write_json
from src.online_rl.tree.eval_queue import pending_jobs,register_ready


def evaluate_job(out,path,job):
    from src.online_rl.runtime import SharedPolicy
    from src.online_rl.integrated.continuation import adapter_sha256
    from src.online_rl.cases import load_cases
    from src.online_rl.tree.backends import ApiDoctor
    from src.online_rl.tree.ollama_doctor import OllamaDoctorCall
    from scripts.evaluation.run_casimedicos_base import run_case
    settings=job['settings'];data=ROOT/settings['data_file']
    if hashlib.sha256(data.read_bytes()).hexdigest()!=settings['data_sha256']:raise RuntimeError('Evaluation dataset changed')
    cases=load_cases(data);dest=out/'evaluation/results'/path.stem;dest.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((Path(job['checkpoint'])/'run_manifest.json').read_text())['config']
    # Independent model object/adapter; checkpoint remains fixed for this whole job.
    cfg['policy']['init_adapter']=str(Path(job['model_checkpoint'])/'adapter')
    policy=SharedPolicy(cfg,'',device=settings['device'],trainable=False)
    if adapter_sha256(policy.model)!=job['policy_sha256']:raise RuntimeError('Evaluation policy does not match checkpoint')
    policy.agent_system=job['prompts']['agent'];policy.doctor_system=job['prompts']['doctor']
    call=OllamaDoctorCall(settings['doctor_backend'])
    doctor=ApiDoctor('',cfg['rollout']['doctor_model'],policy.doctor_system,cfg['rollout']['max_tokens_doctor'],call=call)
    for i,case in enumerate(cases):
        target=dest/'cases'/f'case_{i:06d}.json';partial=dest/'partial'/f'case_{i:06d}.json'
        partial.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            r=json.loads(target.read_text())
            if r['case_id']!=case['case_id'] or r['job_sha256']!=job['job_sha256']:raise RuntimeError('Saved evaluation case identity differs')
            continue
        result=run_case(case,doctor,policy,settings['rounds'],settings['seed'],partial)
        result.update(job_sha256=job['job_sha256'],policy_sha256=job['policy_sha256'],checkpoint_cases=job['completed_cases'])
        atomic_write_json(target,result)
        atomic_write_json(dest/'progress.json',{'job_sha256':job['job_sha256'],'completed':i+1,'total':len(cases),'checkpoint_cases':job['completed_cases']})
        print(json.dumps({'checkpoint_cases':job['completed_cases'],'completed':i+1,'total':len(cases)}),flush=True)
    splits={}
    for p in sorted((dest/'cases').glob('case_*.json')):
        r=json.loads(p.read_text())
        for name in ('all',r['source_split']):
            s=splits.setdefault(name,{'saved':0,'scorable':0,'correct':0,'unparseable_final':0})
            s['saved']+=1
            if r['outcome'] is None:s['unparseable_final']+=1
            else:s['scorable']+=1;s['correct']+=int(r['outcome'])
    for s in splits.values():s['accuracy']=s['correct']/s['scorable'] if s['scorable'] else None
    summary={'job_sha256':job['job_sha256'],'checkpoint_cases':job['completed_cases'],'policy_sha256':job['policy_sha256'],
             'dataset_sha256':settings['data_sha256'],'splits':splits,'status':'complete'}
    atomic_write_json(dest/'summary.json',summary)
    del policy,doctor
    import gc,torch
    gc.collect();torch.cuda.empty_cache()
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--once',action='store_true');p.add_argument('--config',default='configs/evaluation/tree_checkpoint_queue.json');a=p.parse_args();out=Path(a.run_dir).resolve()
    settings=json.loads((ROOT/a.config).read_text())
    if not settings['enabled']:return
    lock=out/'evaluation/worker.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_write_json(out/'evaluation/worker_status.json',{'pid':os.getpid(),'stage':'waiting'})
        while True:
            register_ready(out,settings,ROOT)
            pending=list(pending_jobs(out))
            if not pending:
                if a.once:return
                time.sleep(5);continue
            for path,job in pending:
                atomic_write_json(out/'evaluation/worker_status.json',{'pid':os.getpid(),'stage':'evaluating','checkpoint_cases':job['completed_cases'],'job_sha256':job['job_sha256']})
                try:evaluate_job(out,path,job)
                except Exception as error:
                    # Leave job queued. Partial cases are resumable; no false outcome.
                    atomic_write_json(out/'evaluation/last_error.json',{'checkpoint_cases':job['completed_cases'],'error':str(error)})
                    print(f'Evaluation deferred: {type(error).__name__}: {error}',flush=True)
                    import gc,torch
                    gc.collect();torch.cuda.empty_cache()
                    if a.once:raise
                    time.sleep(10);break
                atomic_write_json(out/'evaluation/worker_status.json',{'pid':os.getpid(),'stage':'waiting'})
            if a.once:return

if __name__=='__main__':main()
