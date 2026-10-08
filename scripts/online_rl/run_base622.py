"""GPU4 original Qwen base dialogue evaluation, without any rubric APIs or learning."""
import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/online_rl/tree_v21_base622.json')
    parser.add_argument('--gpu',type=int,help='Physical GPU used for the BASE Agent and Doctor (default from config)')
    parser.add_argument('--max-cases',type=int);args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    gpu=args.gpu if args.gpu is not None else cfg.get('five_gpu',{}).get('base_evaluation',{}).get('gpu',4)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise RuntimeError(f'BASE-only runner requires physical CUDA_VISIBLE_DEVICES={gpu}')
    os.environ['SHERPA_AGENT_REPLICAS']='1';os.environ.pop('SHERPA_DOCTOR_ENDPOINTS',None)
    cfg['rollout']['doctor_backend']['endpoint']=cfg['five_gpu']['doctor_urls'][gpu]
    from src.online_rl.cases import load_cases,file_sha256
    if file_sha256(ROOT/cfg['five_gpu']['eval_file'])!=cfg['five_gpu']['eval_sha256']:raise RuntimeError('Evaluation data hash changed')
    cases=load_cases(ROOT/cfg['five_gpu']['eval_file'])
    if len(cases)!=622 or len({c['case_id'] for c in cases})!=622:raise RuntimeError('BASE requires every one of the 622 unique cases')
    out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    with Path(f'/tmp/sherpa_tree_physical_gpu{gpu}.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Another evaluation/training runner owns GPU{gpu}')
        from src.online_rl.runtime import SharedPolicy
        from src.online_rl.tree.base_evaluation import BaseEvaluation
        policy=SharedPolicy(cfg,key='',device='cuda:0')
        problems=policy.doctor.verify()['problems']
        if problems:raise RuntimeError('Doctor identity differs: '+'; '.join(problems))
        evaluation=BaseEvaluation(out/'evaluation/BASE_raw_no_precommit',cases,policy,cfg,gpu=gpu)
        atomic=__import__('src.online_rl.integrated.records',fromlist=['atomic_write_json']).atomic_write_json
        atomic(out/'base_runner.json',{'pid':os.getpid(),'physical_gpu':gpu,'config':args.config,
                                     'mode':'original_base_no_precommit_v1'})
        done=0
        while args.max_cases is None or done<args.max_cases:
            if (out/'STOP_BASE_AFTER_CASE').exists():break
            if not evaluation.step():break
            done+=1

if __name__=='__main__':main()
