"""Synthetic single-GPU throughput comparison using SHERPA's actual batch sampler.

No API calls or training. Refuses to run while the selected GPU's SHERPA lock
is held. Fixed input/output lengths make the five configurations comparable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GRID = ((1, 4), (1, 8), (1, 16), (2, 4), (2, 8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0, help='Physical GPU index')
    parser.add_argument('--input-tokens', type=int, default=512)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trace-mode', choices=('compact', 'training', 'both'), default='compact')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.gpu < 0 or min(args.input_tokens, args.output_tokens, args.repeats) < 1:
        parser.error('GPU must be nonnegative; token counts and repeats must be positive')
    os.chdir(ROOT)
    with Path(f'/tmp/sherpa_tree_physical_gpu{args.gpu}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, f'GPU {args.gpu} is reserved by a SHERPA run; benchmark not started.\n')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ.setdefault('OMP_NUM_THREADS', '4')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.online_rl.agent_batch import generate_batch
        from src.online_rl.tree.experiment import load_config
        from src.online_rl.integrated.records import atomic_write_json
        cfg = load_config('configs/sherpa.json')
        model_path = (ROOT / cfg['policy']['local_model_path']).resolve()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        prefix = tokenizer.encode('Compare the evidence supporting each alternative. ', add_special_tokens=False)
        input_ids = (prefix * ((args.input_tokens + len(prefix) - 1) // len(prefix)))[:args.input_tokens]

        class FixedTokenizer:
            padding_side = 'left'
            pad_token_id = pad_id
            eos_token_id = None
            def __call__(self, prompts, **kwargs):
                ids = torch.tensor([input_ids] * len(prompts))
                return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
            def decode(self, ids, **kwargs):
                return tokenizer.decode(ids, **kwargs)

        models = []
        report = {'gpu': args.gpu, 'gpu_name': torch.cuda.get_device_name(0),
                  'torch': torch.__version__, 'model': cfg['policy']['base_model'],
                  'input_tokens': args.input_tokens, 'output_tokens': args.output_tokens,
                  'repeats': args.repeats, 'sampling': {'temperature': 1., 'top_p': 1., 'top_k': 0},
                  'note': 'Synthetic fixed-length decoding; not end-to-end rollout time. EOS disabled.',
                  'rows': []}
        path = args.output or ROOT / 'outputs/prompt_benchmark' / f'gpu{args.gpu}_{int(time.time())}.json'
        modes = ('training', 'compact') if args.trace_mode == 'both' else (args.trace_mode,)
        for replicas, requests in GRID:
            while len(models) < replicas:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                            local_files_only=True).to('cuda:0').eval()
                model.generation_config.eos_token_id = None
                model.generation_config.use_cache = True
                models.append(model)
            for mode in modes:
                def one(index, repeat):
                    return generate_batch(models[index], FixedTokenizer(), ['synthetic'] * requests,
                                          [repeat * 10000 + index * 100 + i for i in range(requests)],
                                          args.output_tokens, 1., 1., 0,
                                          collect_training_traces=mode == 'training')
                durations = []
                with ThreadPoolExecutor(replicas) as pool:
                    for repeat in range(args.repeats + 1):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        futures = [pool.submit(one, index, repeat) for index in range(replicas)]
                        results = [future.result() for future in futures]
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        count = sum(len(row['completion_ids']) for batch in results for row in batch)
                        if count != replicas * requests * args.output_tokens:
                            raise RuntimeError('Fixed-length benchmark produced an unexpected token count')
                        if repeat:
                            durations.append(elapsed)
                rates = [count / duration for duration in durations]
                row = {'models_per_gpu': replicas, 'requests_per_model': requests, 'trace_mode': mode,
                       'median_tokens_per_second': statistics.median(rates),
                       'tokens_per_second': rates, 'seconds': durations}
                report['rows'].append(row)
                atomic_write_json(path, report)
                print(f'{replicas} model(s) x {requests} requests | {mode}: '
                      f'{row["median_tokens_per_second"]:.1f} tokens/s', flush=True)
        print(f'Results: {path}', flush=True)


if __name__ == '__main__':
    main()
