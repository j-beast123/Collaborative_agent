"""Logical five-GPU topology -> physical devices. Runtime only: never part of config or experiment identity."""
import json
import os
import socket
import time
from pathlib import Path

LOGICAL_GPUS = 5


def physical_gpus(env=None) -> list[str]:
    """Physical id of logical GPU i = i-th CUDA_VISIBLE_DEVICES entry; unset -> identity."""
    vis = (os.environ if env is None else env).get('CUDA_VISIBLE_DEVICES')
    ids = [g.strip() for g in vis.split(',')] if vis else [str(i) for i in range(LOGICAL_GPUS)]
    if len(ids) != LOGICAL_GPUS or len(set(ids)) != LOGICAL_GPUS or not all(ids):
        raise ValueError(f'CUDA_VISIBLE_DEVICES must list {LOGICAL_GPUS} distinct physical GPUs (logical 0..{LOGICAL_GPUS - 1}); got {vis!r}')
    return ids


def record_runtime(out: Path, cfg: dict) -> dict:
    """Append launch provenance (devices, host, CUDA, Ollama) to runtime_provenance.jsonl."""
    import torch
    fg = cfg['five_gpu']
    rec = {'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'host': socket.gethostname(),
           'logical_to_physical_gpu': dict(enumerate(physical_gpus())), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'ollama_binary': os.environ.get('SHERPA_OLLAMA_BINARY', fg.get('ollama_binary')), 'doctor_urls': fg['doctor_urls']}
    with (Path(out) / 'runtime_provenance.jsonl').open('a') as f: f.write(json.dumps(rec) + '\n')
    return rec
