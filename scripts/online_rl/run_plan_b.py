"""Prepare Plan B data; GPU filtering starts only with --launch and free GPU locks."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.cases import load_cases
from src.online_rl.integrated.records import atomic_write_json, read_json
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.sherpa.plan_b import prepare_pool, screening_identity, finalize_dataset


def screen_case(case, task, cfg, policy, home, identity):
    from src.online_rl.sherpa.rollouts import collect
    from src.online_rl.tree.cache import CallCache
    folder = home / 'screen' / case['case_id']; folder.mkdir(parents=True, exist_ok=True)
    saved = folder / 'outcomes.json'
    if saved.exists():
        row = read_json(saved)
        if row['identity'] != identity: raise RuntimeError('Screening settings changed; use a fresh output directory')
        return row
    cache = CallCache(folder / 'calls.jsonl')
    seed = derive_seed(task['seed'], case['case_id'], 'plan_b_screen8', 0)
    _, trajectories = collect(case, cfg, policy, cache, 0, behavior=False, branching=8, seed_override=seed)
    outcomes = [row['outcome'] for row in trajectories]
    if len(outcomes) != 8 or any(value not in (0,1) for value in outcomes):
        raise RuntimeError('Screening requires eight completed binary outcomes')
    opening = next(row['value'] for row in cache.store.values() if row['kind'] == 'doctor_turn'
                   and row['identity'].get('round') == 1 and row['identity'].get('path') == '')
    successes = sum(outcomes)
    row = {'identity': identity, 'case_id': case['case_id'], 'outcomes': outcomes,
           'successes': successes, 'accepted': task['min_successes'] <= successes <= 8-task['min_successes'],
           'doctor_opening': opening}
    atomic_write_json(saved, row)
    return row


def screen(task, template):
    from src.online_rl.doctor import make_doctor
    cfg = copy.deepcopy(template); home = ROOT / task['output_dir']
    cfg['schedule']['seed'] = task['seed']
    cfg['policy']['init_adapter'] = None
    cfg['prompt_optimization'].update(output_dir=str(home/'screen_servers'), rollout_replicas=1,
                                      assistant_max_inflight=128)
    doctor = make_doctor(cfg['rollout'], (ROOT/cfg['rollout']['doctor_system_prompt_file']).read_text())
    identity = screening_identity(task, template, ROOT, doctor)
    pool = load_cases(home/'pool.jsonl')
    finished = {}; accepted = 0
    for case in pool:
        saved = home/'screen'/case['case_id']/'outcomes.json'
        if saved.exists():
            row = read_json(saved)
            if row['identity'] != identity: raise RuntimeError('Screening settings changed; choose a new output directory')
            finished[case['case_id']] = row; accepted += row['accepted']
    def prefix_ready():
        count = 0
        for case in pool:
            row = finished.get(case['case_id'])
            if row is None: return False
            count += row['accepted']
            if count >= 120: return True
        return False
    if prefix_ready() and task['selection'] == 'first120':
        return finalize_dataset(task, template, ROOT, identity)
    if len(finished) == len(pool):
        return finalize_dataset(task, template, ROOT, identity)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',')
    if len(devices) < 2: raise RuntimeError('Plan B requires two visible GPUs')
    with ExitStack() as resources:
        # Both locks are acquired before importing torch or loading either model.
        for gpu in devices[:2]:
            lock = resources.enter_context(Path(f'/tmp/sherpa_tree_physical_gpu{gpu}.lock').open('a'))
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError(f'GPU {gpu} is in use; current training is untouched. Run filtering later.') from None
        import torch
        if torch.cuda.device_count() < 2: raise RuntimeError('Need two available CUDA GPUs')
        os.environ.update(SHERPA_CALL_CACHE='1', SHERPA_RATE_LIMIT_RETRIES='0', SHERPA_AGENT_REPLICAS='1')
        from src.common.api_slots import install, create
        install(create(cfg['execution']['api_limits']))
        from src.online_rl.vllm_policy import VLLMPolicy
        from src.online_rl.local_doctor import start_server
        policy = VLLMPolicy(cfg, ROOT); resources.callback(policy.close)
        server = start_server(cfg, ROOT, home, policy.doctor, devices[1]); resources.callback(server.close)
        pending = iter(case for case in pool if case['case_id'] not in finished)
        workers = task['case_workers']
        with ThreadPoolExecutor(workers) as executor:
            futures = set()
            def fill():
                while len(futures) < workers:
                    case = next(pending, None)
                    if case is None: break
                    futures.add(executor.submit(screen_case,case,task,cfg,policy,home,identity))
            fill()
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    finished[row['case_id']] = row; accepted += row['accepted']
                    atomic_write_json(home/'screen_progress.json', {'processed':len(finished),'pool_cases':len(pool),
                        'accepted':accepted,'target':120,'min_successes':task['min_successes'],
                        'max_successes':8-task['min_successes'],'complete':prefix_ready()})
                    print(f'Screen {len(finished)}/{len(pool)} | mixed {accepted}/120 | {row["case_id"]} {row["successes"]}/8',flush=True)
                # Keep the GPU queues full without a barrier between case batches.
                # Stop dispatch only when the fixed pool prefix contains 120 mixed
                # cases, not when 120 fast cases happen to finish first.
                if not (task['selection'] == 'first120' and prefix_ready()): fill()
    return finalize_dataset(task,template,ROOT,identity)


def child(command):
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        code = process.wait()
        if code: raise subprocess.CalledProcessError(code,command)
    finally:
        if process.poll() is None: process.terminate(); process.wait()


def main(argv=None):
    signal.signal(signal.SIGTERM,lambda signum,frame: sys.exit(128+signum))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/plan_b.json')
    parser.add_argument('--phase',choices=('prepare','screen','finalize','auc32','optimize','all'),default='prepare')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--source',help='Optional local normalized MedMCQA train JSONL')
    parser.add_argument('--pool-size',type=int)
    parser.add_argument('--min-successes',type=int,choices=(1,2))
    parser.add_argument('--selection',choices=('first120','random120'))
    parser.add_argument('--case-workers',type=int)
    parser.add_argument('--output-dir')
    args=parser.parse_args(argv); os.chdir(ROOT)
    task=read_json(args.config)
    for field in ('pool_size','min_successes','selection','case_workers','output_dir'):
        value=getattr(args,field)
        if value is not None: task[field]=value
    if task['pool_size'] < 120 or task['case_workers'] < 1: parser.error('pool-size >= 120 and case-workers > 0 required')
    if task['min_successes'] not in (1,2) or task['selection'] not in ('first120','random120'):
        parser.error('min-successes must be 1 or 2; selection must be first120 or random120')
    if task['dataset'] != 'openlifescienceai/medmcqa': parser.error('Plan B currently supports MedMCQA train only')
    from src.online_rl.tree.experiment import load_config
    template=load_config(task['template_config'])
    if template['rollout']['doctor_backend']['engine'] != 'vllm': parser.error('Plan B requires the local vLLM Doctor')
    home=ROOT/task['output_dir']
    print(json.dumps({'phase':args.phase,'launch':args.launch,'pool_size':task['pool_size'],
        'screen_rollouts':8,'success_range':[task['min_successes'],8-task['min_successes']],
        'selected_train_cases':120,'prompt_cases':24,'additional_rollouts_per_prompt_case':32,
        'engine':'vllm','base_no_lora':True,'case_workers':task['case_workers'],
        'assistant_max_inflight':128,'doctor_max_inflight':128,'output_dir':str(home),
        'screening_progress':read_json(home/'screen_progress.json') if (home/'screen_progress.json').exists() else {}},indent=2),flush=True)
    if not args.launch: return
    home.mkdir(parents=True,exist_ok=True)
    with (home/'plan_b.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase in ('prepare','all'): prepare_pool(task,template,ROOT,source=args.source)
        if args.phase in ('screen','all'): screen(task,template)
        if args.phase=='finalize':
            from src.online_rl.doctor import make_doctor
            doctor=make_doctor(template['rollout'],(ROOT/template['rollout']['doctor_system_prompt_file']).read_text())
            finalize_dataset(task,template,ROOT,screening_identity(task,template,ROOT,doctor))
        if args.phase in ('auc32','all'):
            child([sys.executable,'-u','scripts/online_rl/run_sherpa.py','--config',str(home/'prompt_plan_b.json'),
                   '--phase','base-rollouts','--launch'])
        if args.phase=='optimize':
            child([sys.executable,'-u','scripts/online_rl/run_prompt_optimization.py','--config',str(home/'prompt_plan_b.json'),
                   '--phase','optimize-prompt','--launch'])
        print('Plan B phase complete:',args.phase,flush=True)


if __name__=='__main__': main()
