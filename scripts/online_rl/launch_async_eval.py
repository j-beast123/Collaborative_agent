"""Start the asynchronous evaluator: one dedicated Ollama Doctor and one async_eval.py process per eval GPU.
Doctor servers use the same Ollama environment as start_five_gpu_doctors.py, on ports disjoint from training.
  launch_async_eval.py --config CFG --gpus 5,6,7 --ports 11545,11546,11547 [--eval-root DIR] [--cases-limit M]
      [--checkpoint NAME] [--once] [--stop] [--eval-override JSON --include-initial]   (DEV-only evaluation)"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.tree import experiment as X

READY_TRIES = 120


def gpu_memory_mib(gpu):
    out = subprocess.run(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def alive(url):
    try:
        with urllib.request.urlopen(url + '/api/version', timeout=2): return True
    except OSError: return False


def start_doctor(cfg, gpu, port, run_dir):
    binary = os.environ.get('SHERPA_OLLAMA_BINARY', cfg['five_gpu'].get('ollama_binary', 'ollama'))
    binary = shutil.which(binary) if '/' not in binary else binary
    models = os.environ.get('SHERPA_OLLAMA_MODELS', cfg['five_gpu'].get('ollama_models_dir', str(Path.home() / '.ollama/models')))
    url = f'http://127.0.0.1:{port}'
    if alive(url): raise SystemExit(f'port {port} already serves something')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OLLAMA_HOST': f'127.0.0.1:{port}', 'OLLAMA_MODELS': models,
           'OLLAMA_MAX_LOADED_MODELS': '1', 'OLLAMA_NUM_PARALLEL': '1', 'OLLAMA_KEEP_ALIVE': '30m'}
    with (run_dir / f'doctor_gpu{gpu}.log').open('a') as log:
        proc = subprocess.Popen([binary, 'serve'], cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
    (run_dir / f'doctor_gpu{gpu}.pid').write_text(str(proc.pid))
    b = cfg['rollout']['doctor_backend']
    for _ in range(READY_TRIES):
        if proc.poll() is not None: raise SystemExit(f'eval Doctor on GPU{gpu} exited; see its log')
        try:
            with urllib.request.urlopen(url + '/api/tags', timeout=1) as r: models_ = json.load(r)['models']
            if not any(m['name'] == b['model'] and m['digest'] == b['model_digest'] for m in models_): raise SystemExit('pinned Doctor model missing')
            return url
        except OSError: time.sleep(0.5)
    raise SystemExit(f'eval Doctor on GPU{gpu} not ready')


def stop(run_dir):
    for pid_file in sorted(run_dir.glob('*.pid')):
        try: os.killpg(int(pid_file.read_text()), signal.SIGTERM)
        except (ProcessLookupError, ValueError, PermissionError): pass


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--config', required=True); a.add_argument('--gpus', default='5,6,7'); a.add_argument('--ports', default='11545,11546,11547')
    a.add_argument('--eval-root'); a.add_argument('--cases-limit'); a.add_argument('--checkpoint'); a.add_argument('--once', action='store_true')
    a.add_argument('--stop', action='store_true'); a.add_argument('--eval-override'); a.add_argument('--include-initial', action='store_true')
    x = a.parse_args()
    cfg = X.load_config(x.config); out = ROOT / cfg['output_dir']
    run_dir = (Path(x.eval_root) if x.eval_root else out / 'evaluation').resolve() / 'async_runtime'; run_dir.mkdir(parents=True, exist_ok=True)
    if x.stop: stop(run_dir); return
    gpus, ports = x.gpus.split(','), [int(p) for p in x.ports.split(',')]
    train_ports = {int(u.rsplit(':', 1)[1]) for u in cfg['five_gpu']['doctor_urls']}
    if len(gpus) != len(ports) or len(set(ports)) != len(ports): raise SystemExit('one distinct port per eval GPU')
    if set(ports) & train_ports: raise SystemExit(f'eval ports overlap training Doctor ports {sorted(train_ports)}')
    busy = [g for g in gpus if gpu_memory_mib(g) > 1000]
    if busy: raise SystemExit(f'eval GPUs already in use: {busy}')
    urls = [start_doctor(cfg, g, p, run_dir) for g, p in zip(gpus, ports)]
    for k, (g, url) in enumerate(zip(gpus, urls)):
        cmd = [sys.executable, 'scripts/online_rl/async_eval.py', '--config', x.config, '--shard', str(k), '--shards', str(len(gpus)), '--doctor-url', url]
        for flag in ('eval_root', 'cases_limit', 'checkpoint', 'eval_override'):
            if getattr(x, flag): cmd += [f"--{flag.replace('_', '-')}", str(getattr(x, flag))]
        if x.once: cmd.append('--once')
        if x.include_initial: cmd.append('--include-initial')
        with (run_dir / f'eval_gpu{g}.log').open('a') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': g}, stdout=log, stderr=log, start_new_session=True)
        (run_dir / f'eval_gpu{g}.pid').write_text(str(proc.pid))
        print(json.dumps({'gpu': g, 'doctor': url, 'evaluator_pid': proc.pid, 'shard': k}), flush=True)
    (run_dir / 'launch.json').write_text(json.dumps({'gpus': gpus, 'ports': ports, 'config': x.config, 'time': time.time()}, indent=2))


if __name__ == '__main__': main()
