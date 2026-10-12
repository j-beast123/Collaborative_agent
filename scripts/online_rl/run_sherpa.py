"""Explicit offline, on-policy training and fixed-split evaluation entrypoint."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    from src.online_rl.tree.experiment import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sherpa.json')
    parser.add_argument('--no-cache', action='store_true', help='Isolated uncached base rollout / prompt optimization run')
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
    if args.no_cache and args.phase not in ('base-rollouts', 'optimize-prompt'):
        parser.error('--no-cache is supported for base-rollouts and optimize-prompt')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.max_cases is not None and args.max_cases < 1:
        parser.error('--max-cases must be positive')
    if not args.launch:
        # Description only: no local verification, GPU load, credentials or API requests.
        print(json.dumps({'config': args.config, 'phase': args.phase, 'method': cfg['method'],
                          'doctor': cfg['rollout']['doctor_model'], 'assistant': cfg['policy']['base_model'],
                          'schedule': cfg['schedule'], 'rollouts_per_case': 32,
                          'advantage_normalization': cfg['advantage_normalization'],
                          'cache_enabled': not args.no_cache and cfg['prompt_optimization'].get('cache_enabled', True),
                          'checkpoint_every_steps': 10, 'launch': False}, ensure_ascii=False, indent=2))
        return
    os.chdir(ROOT)
    if args.no_cache:
        from src.online_rl.sherpa.cache_options import fresh_prompt_run
        cfg, config_path = fresh_prompt_run(cfg, ROOT, args.phase)
        print(f'Cache disabled; isolated run config: {config_path}', flush=True)
    os.environ['SHERPA_CALL_CACHE'] = '0' if cfg['prompt_optimization'].get('cache_enabled') is False else '1'
    from src.online_rl.sherpa.config import config_problems
    problems = config_problems(cfg, ROOT)
    if problems:
        parser.error('; '.join(problems))
    search = cfg['prompt_optimization']
    training = cfg.get('training_execution', {}) if args.phase == 'train' else {}
    if training:
        os.environ['SHERPA_RATE_LIMIT_RETRIES'] = str(training.get('rate_limit_retries', 0))
    if args.phase in ('base-rollouts', 'optimize-prompt'):
        os.environ['SHERPA_RATE_LIMIT_RETRIES'] = str(search.get('rate_limit_retries', 0))
    replicas = (search.get('rollout_replicas', 1) if args.phase == 'base-rollouts'
                else training.get('rollout_replicas', 1))
    os.environ['SHERPA_AGENT_REPLICAS'] = str(replicas)
    os.environ['SHERPA_AGENT_BATCH_SIZE'] = str(
        search.get('rollout_batch_size', 4) if args.phase == 'base-rollouts'
        else training.get('rollout_batch_size', cfg['five_gpu']['agent_batch_size']))
    if training:
        os.environ['SHERPA_AGENT_MAX_INFLIGHT'] = str(training['assistant_max_inflight'])
        cfg['execution']['agent_workers'] = training['assistant_max_inflight']
    if args.phase == 'base-rollouts':
        os.environ['SHERPA_AGENT_MAX_INFLIGHT'] = str(search.get('assistant_max_inflight', 4))
        cfg['execution']['agent_workers'] = max(cfg['execution']['agent_workers'],
                                                int(os.environ['SHERPA_AGENT_MAX_INFLIGHT']))
    if cfg['method'] == 'sherpa' and args.phase == 'train':
        from src.online_rl.sherpa.prompt_search import frozen_prompt
        frozen_prompt(cfg, ROOT)
    home = ROOT / (search['output_dir'] if args.phase in ('base-rollouts', 'optimize-prompt') else cfg['output_dir'])
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
        with ExitStack() as resources:
            from scripts.online_rl.run_tree import load_key
            from src.online_rl.cases import load_cases
            local_doctor = cfg['rollout']['doctor_backend']['engine'] == 'vllm'
            if local_doctor:
                import torch
                devices = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(torch.cuda.device_count())))).split(',')
                doctor_index = cfg['doctor_execution']['visible_gpu_index']
                if torch.cuda.device_count() <= doctor_index:
                    raise RuntimeError('Local Doctor requires two visible GPUs')
                doctor_device = devices[doctor_index]
                doctor_lock = resources.enter_context(Path(f'/tmp/sherpa_tree_physical_gpu{doctor_device}.lock').open('a'))
                fcntl.flock(doctor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.phase == 'optimize-prompt':
                # Scoring cached dialogues needs no assistant model. A local
                # Doctor still owns its reserved GPU for reflection generation.
                from src.online_rl.doctor import make_doctor
                system = (ROOT / cfg['rollout']['doctor_system_prompt_file']).read_text()
                policy = SimpleNamespace(doctor=make_doctor(cfg['rollout'], system))
            else:
                import torch
                devices = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(torch.cuda.device_count())))).split(',')
                if torch.cuda.device_count() < replicas:
                    raise RuntimeError(f'Need {replicas} visible GPUs for {args.phase}')
                for device_id in devices[:replicas]:
                    device_lock = resources.enter_context(Path(f'/tmp/sherpa_tree_physical_gpu{device_id}.lock').open('a'))
                    fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.phase == 'base-rollouts' and search.get('inference_engine') == 'vllm':
                    from src.online_rl.vllm_policy import VLLMPolicy
                    policy = VLLMPolicy(cfg, ROOT)
                    resources.callback(policy.close)
                else:
                    from src.online_rl.runtime import SharedPolicy
                    policy = SharedPolicy(cfg, load_key({'api_provider': 'openai_direct'}))
                policy.collect_training_traces = args.phase != 'base-rollouts'
                def close_batcher():
                    if hasattr(policy, '_batcher'):
                        policy._batcher.close()
                resources.callback(close_batcher)
            if local_doctor:
                from src.online_rl.local_doctor import start_server
                server = start_server(cfg, ROOT, home, policy.doctor, doctor_device)
                resources.callback(server.close)
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
                    if backend is not None:
                        resources.callback(backend.close)
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
        lock.close()


if __name__ == '__main__':
    main()
