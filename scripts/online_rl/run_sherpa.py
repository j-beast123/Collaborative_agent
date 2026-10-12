"""Explicit offline, on-policy training and fixed-split evaluation entrypoint."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from src.online_rl.tree.experiment import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sherpa.json')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--phase', choices=('base-rollouts', 'optimize-prompt', 'train', 'evaluate', 'evaluate-base'), default='train')
    parser.add_argument('--method', choices=('sherpa', 'outcome'))
    parser.add_argument('--max-steps', '--max-batches', type=int)
    parser.add_argument('--max-cases', type=int, help='Limit base rollout collection; prompt search still needs all 120')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.method:
        cfg['method'] = args.method
        cfg['output_dir'] = 'outputs/sherpa_trajectory_v1/' + args.method
    if args.eval_only:
        args.phase = 'evaluate'
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.max_cases is not None and args.max_cases < 1:
        parser.error('--max-cases must be positive')
    if not args.launch:
        # Description only: no local verification, GPU load, credentials or API requests.
        print(json.dumps({'config': args.config, 'phase': args.phase, 'method': cfg['method'],
                          'doctor': cfg['rollout']['doctor_model'], 'assistant': cfg['policy']['base_model'],
                          'schedule': cfg['schedule'], 'rollouts_per_case': 32,
                          'checkpoint_every_steps': 10, 'launch': False}, ensure_ascii=False, indent=2))
        return
    os.chdir(ROOT)
    from src.online_rl.sherpa.config import config_problems
    problems = config_problems(cfg, ROOT)
    if problems:
        parser.error('; '.join(problems))
    os.environ['SHERPA_AGENT_REPLICAS'] = '1'
    os.environ['SHERPA_AGENT_BATCH_SIZE'] = str(cfg['five_gpu']['agent_batch_size'])
    if cfg['method'] == 'sherpa' and args.phase == 'train':
        from src.online_rl.sherpa.prompt_search import frozen_prompt
        frozen_prompt(cfg, ROOT)
    home = ROOT / cfg['output_dir']
    home.mkdir(parents=True, exist_ok=True)
    os.environ['API_USAGE_LOG'] = str(home / f'api_usage_{args.phase}.jsonl')
    os.environ['API_EVENT_LOG'] = str(home / f'api_events_{args.phase}.jsonl')
    from src.common.api_slots import create, install
    from src.common.live_gpu import configure
    from src.common.progress import configure as configure_progress
    install(create(cfg['execution']['api_limits']))
    configure(home, 0)
    configure_progress(home)
    lock = (home / 'runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        device_lock = Path(f'/tmp/sherpa_tree_physical_gpu{device_id}.lock').open('a')
        try:
            fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from scripts.online_rl.run_tree import load_key
            from src.online_rl.runtime import SharedPolicy
            from src.online_rl.cases import load_cases
            policy = SharedPolicy(cfg, load_key({'api_provider': 'openai_direct'}))
            if args.phase in ('evaluate', 'evaluate-base'):
                from src.online_rl.sherpa.evaluation import evaluate
                result = evaluate(cfg, policy, ROOT, base_only=args.phase == 'evaluate-base')
            else:
                cases = load_cases(ROOT / cfg['data']['train'])
                if args.phase == 'base-rollouts':
                    from src.online_rl.sherpa.prompt_search import collect_base
                    result = collect_base(cfg, policy, cases, ROOT, max_cases=args.max_cases)
                else:
                    from src.online_rl.sherpa.backend import ChecklistBackend
                    backend = ChecklistBackend(cfg, policy, ROOT) if cfg['method'] == 'sherpa' else None
                    if args.phase == 'optimize-prompt':
                        if backend is None:
                            parser.error('Prompt optimization belongs to SHERPA')
                        from src.online_rl.sherpa.prompt_search import optimize
                        result = optimize(cfg, policy, backend, cases, ROOT)
                    else:
                        from src.online_rl.sherpa.training import train
                        result = train(cfg, policy, backend, cases, ROOT, max_steps=args.max_steps)
            from src.online_rl.integrated.records import atomic_write_json
            atomic_write_json(home / f'{args.phase}_result.json', result)
            print(json.dumps({'phase': args.phase, 'output_dir': str(home), 'status': 'returned'}, ensure_ascii=False))
        finally:
            device_lock.close()
    finally:
        lock.close()


if __name__ == '__main__':
    main()
