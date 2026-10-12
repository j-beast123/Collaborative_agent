"""Preflight without paid calls; explicitly launch base collection then prompt search."""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def preflight(cfg, *, phase='all', require_keys=False):
    from src.online_rl.sherpa.config import config_problems
    from src.online_rl.sherpa.rollouts import schedule
    from src.online_rl.cases import load_cases, file_sha256
    from scripts.online_rl.run_tree import load_key
    errors = config_problems(cfg, ROOT)
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'huggingface_hub', 'tokenizers'):
        try:
            packages[name] = importlib.import_module(name).__version__
        except Exception as exc:
            errors.append(f'{name} import failed ({type(exc).__name__})')
    try:
        cases = load_cases(ROOT / cfg['data']['train'])
        schedule(cfg, cases, ROOT)
    except Exception as exc:
        errors.append(f'Dataset validation failed: {exc}')
    ro = cfg['rollout']
    if file_sha256(ROOT / ro['doctor_system_prompt_file']) != ro['doctor_system_prompt_sha256']:
        errors.append('Doctor prompt hash mismatch')
    model = (ROOT / cfg['policy']['local_model_path']).resolve()
    try:
        config = json.loads((model / 'config.json').read_text())
        if config.get('model_type') != 'qwen3' or (config.get('hidden_size'), config.get('num_hidden_layers')) != (4096, 36):
            errors.append('Local model does not match the expected Qwen3-8B architecture')
        index = json.loads((model / 'model.safetensors.index.json').read_text())
        for name in {'tokenizer.json', 'tokenizer_config.json', *index['weight_map'].values()}:
            if not (model / name).is_file() or (model / name).stat().st_size == 0:
                errors.append(f'Missing local model file: {name}')
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(model, local_files_only=True)
    except Exception as exc:
        errors.append(f'Local model validation failed ({type(exc).__name__})')
    devices = []
    if phase != 'optimize-prompt' and 'torch' in packages:
        import torch
        replicas = cfg['prompt_optimization']['rollout_replicas']
        if torch.cuda.device_count() < replicas:
            errors.append(f'Need {replicas} visible CUDA GPUs')
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    credentials = {}
    if phase != 'optimize-prompt' and cfg['prompt_optimization'].get('inference_engine') == 'vllm':
        python = ROOT / cfg['prompt_optimization']['vllm']['python']
        try:
            result = subprocess.run([str(python), '-c',
                                     'import vllm; print(vllm.__version__)'],
                                    capture_output=True, text=True, timeout=60, check=True)
            packages['vllm_server'] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            errors.append('vLLM environment unavailable; install requirements-vllm.txt in .venv-vllm')
    for provider in ('openai_direct', 'openrouter'):
        try:
            credentials[provider] = bool(load_key({'api_provider': provider}).strip())
        except Exception:
            credentials[provider] = False
        if require_keys and not credentials[provider]:
            errors.append(f'Missing {provider} API key; fill .env')
    report = {'python': sys.executable, 'packages': packages, 'model_path': str(model),
              'doctor_model': ro['doctor_model'],
              'doctor_reasoning_effort': ro['doctor_backend'].get('reasoning_effort'),
              'output_dir': cfg['prompt_optimization']['output_dir'],
              'gpus': devices, 'credentials_present': credentials,
              'scheduling': {key: cfg['prompt_optimization'][key] for key in
                             ('rollout_replicas', 'rollout_case_workers', 'rollout_batch_size', 'assistant_max_inflight', 'candidate_workers', 'case_workers')},
              'rate_limit_retries': cfg['prompt_optimization'].get('rate_limit_retries', 0),
              'inference_engine': cfg['prompt_optimization'].get('inference_engine', 'transformers'),
              'vllm': cfg['prompt_optimization'].get('vllm'),
              'api_limits': cfg['execution']['api_limits'], 'errors': errors}
    if cfg['prompt_optimization'].get('inference_engine') == 'vllm':
        report['scheduling'].pop('rollout_batch_size', None)
    print(json.dumps(report, indent=2), flush=True)
    return not errors


def main():
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='Explicitly start paid API work after preflight')
    parser.add_argument('--config', default='configs/sherpa.json')
    parser.add_argument('--phase', choices=('all', 'base-rollouts', 'optimize-prompt'), default='all')
    parser.add_argument('--max-cases', type=int, help='Smoke run limit; only with --phase base-rollouts')
    args = parser.parse_args()
    if args.max_cases is not None and (args.phase != 'base-rollouts' or not 1 <= args.max_cases <= 120):
        parser.error('--max-cases requires --phase base-rollouts and a value from 1 to 120')
    os.chdir(ROOT)
    from src.online_rl.tree.experiment import load_config
    cfg = load_config(args.config)
    if cfg.get('method') != 'sherpa':
        parser.error('Prompt optimization requires method=sherpa')
    if not preflight(cfg, phase=args.phase, require_keys=args.launch):
        raise SystemExit(1)
    if not args.launch:
        print('Preflight only. No models loaded and no API requests sent. Add --launch to run.')
        return
    home = ROOT / cfg['prompt_optimization']['output_dir']
    home.mkdir(parents=True, exist_ok=True)
    with (home / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phases = ('base-rollouts', 'optimize-prompt') if args.phase == 'all' else (args.phase,)
        for phase in phases:
            command = [sys.executable, '-u', str(ROOT / 'scripts/online_rl/run_sherpa.py'),
                       '--config', args.config, '--phase', phase, '--launch']
            if args.max_cases is not None:
                command += ['--max-cases', str(args.max_cases)]
            print(f'Starting {phase}; outputs: {home}', flush=True)
            # Separate processes release both GPU replicas before API-only search.
            child = subprocess.Popen(command)
            try:
                returncode = child.wait()
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait()


if __name__ == '__main__':
    main()
