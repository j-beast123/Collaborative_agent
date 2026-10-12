"""Logical five-GPU topology -> physical devices. Runtime only: never part of config or experiment identity."""
import json
import os
import socket
import time
from pathlib import Path

LOGICAL_GPUS = 5


def gpu_count(cfg) -> int:
    fg = cfg['five_gpu']
    return max([fg['update_gpu'], *fg['rollout_gpus']]) + 1


def physical_gpus(env=None, *, count=LOGICAL_GPUS) -> list[str]:
    """Physical id of logical GPU i = i-th CUDA_VISIBLE_DEVICES entry; unset -> identity."""
    vis = (os.environ if env is None else env).get('CUDA_VISIBLE_DEVICES')
    ids = [g.strip() for g in vis.split(',')] if vis else [str(i) for i in range(count)]
    if count < 1 or len(ids) != count or len(set(ids)) != count or not all(ids):
        raise ValueError(f'CUDA_VISIBLE_DEVICES must list {count} distinct physical GPUs (logical 0..{count - 1}); got {vis!r}')
    return ids


def record_runtime(out: Path, cfg: dict) -> dict:
    """Append launch provenance (devices, host, CUDA, Doctor API) to runtime_provenance.jsonl."""
    import torch
    fg = cfg['five_gpu']
    rec = {'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'host': socket.gethostname(),
           'logical_to_physical_gpu': dict(enumerate(physical_gpus(count=gpu_count(cfg)))), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'doctor_provider': 'openai_direct', 'doctor_model': cfg['rollout']['doctor_model']}
    with (Path(out) / 'runtime_provenance.jsonl').open('a') as f: f.write(json.dumps(rec) + '\n')
    return rec
