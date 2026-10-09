"""Asynchronous checkpoint evaluation on dedicated GPUs: one process per GPU and Doctor, evaluating a disjoint shard of
CasiMedicos622 for every committed permanent checkpoint. Reads only immutable checkpoint files; writes only the evaluation
tree (same EvaluationQueue identity/schema as the synchronous runner). Never touches training state.
  CUDA_VISIBLE_DEVICES=<one physical GPU> async_eval.py --config CFG --shard K --shards N --doctor-url URL [--eval-root DIR]
      [--cases-limit M] [--once] [--checkpoint checkpoint_000010]
      [--eval-override configs/experiments/dev_opt/dev_eval.json --include-initial]   (DEV-only: another eval file, theta0 too)"""
import argparse
import copy
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree import experiment as X
from src.online_rl.tree.five_gpu import EvaluationQueue, cache_openings, evaluation_adapter

POLL_SECONDS = 60
EPISODES_PER_GPU = 4                      # concurrent episodes; Agent rows batch 4 (per_row_rng_batch_v1)
DATASET = 'CasiMedicos all_en_622'


def due(cursor, cfg):
    """Same checkpoints the synchronous runner evaluates."""
    fg = cfg['five_gpu']
    return cursor > 0 and (cursor % fg['checkpoint_every_cases'] == 0 or cursor == cfg['schedule']['cases'])


def settings(cfg, out):
    fg = cfg['five_gpu']
    return {'seed': fg['eval_seed'], 'data_sha256': fg['eval_sha256'],
            'protocol_sha256': json.loads((out / 'run_manifest.json').read_text())['protocol_sha256'], 'dataset': fg.get('eval_dataset', DATASET)}


OVERRIDE_KEYS = ('eval_file', 'eval_sha256', 'eval_cases', 'eval_dataset')


def apply_eval_override(cfg, path):
    """DEV evaluation: replace only the evaluation data of five_gpu (file, hash, size, label); a trained case may not be in it."""
    o = json.loads((ROOT / path).read_text())
    cfg['five_gpu'].update({k: o[k] for k in OVERRIDE_KEYS})
    from src.online_rl.cases import load_cases
    from src.online_rl.tree.five_gpu import training_order
    trained = set(training_order(cfg, load_cases(ROOT / cfg['data']['train']), ROOT)[:cfg['schedule']['cases']])
    overlap = trained & {c['case_id'] for c in load_cases(ROOT / o['eval_file'])}
    if overlap: raise SystemExit(f'{len(overlap)} evaluation cases were trained on')
    return cfg


def committed_checkpoints(out, cfg, include_initial=False):
    for ckpt in sorted((out / 'checkpoints').glob('checkpoint_*')):
        receipt = ckpt / 'COMPLETE.json'
        if ckpt.name.endswith('.pending') or not receipt.exists(): continue
        r = json.loads(receipt.read_text())
        if due(r['completed_cases'], cfg) or (include_initial and r['completed_cases'] == 0): yield ckpt, r


def log_job(path, **row):
    with open(path, 'a') as f: f.write(json.dumps({'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **row}) + '\n')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--config', required=True); a.add_argument('--shard', type=int, required=True); a.add_argument('--shards', type=int, required=True)
    a.add_argument('--doctor-url', required=True); a.add_argument('--eval-root'); a.add_argument('--cases-limit', type=int)
    a.add_argument('--once', action='store_true'); a.add_argument('--checkpoint')
    a.add_argument('--eval-override'); a.add_argument('--include-initial', action='store_true')
    x = a.parse_args()
    if x.eval_override and not x.eval_root: raise SystemExit('--eval-override needs its own --eval-root')
    if not 0 <= x.shard < x.shards: raise SystemExit('shard must be in [0, shards)')
    vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not vis or ',' in vis: raise SystemExit('CUDA_VISIBLE_DEVICES must name exactly one physical GPU')
    from scripts.online_rl.run_tree import load_key
    from src.online_rl.cases import load_cases, file_sha256
    from src.online_rl.environment import final_outcome
    from src.online_rl.integrated.continuation import adapter_sha256
    from src.online_rl.runtime import SharedPolicy
    cfg = X.load_config(x.config)
    if x.eval_override: cfg = apply_eval_override(cfg, x.eval_override)
    fg = cfg['five_gpu']; out = (ROOT / cfg['output_dir']).resolve()
    eval_root = Path(x.eval_root).resolve() if x.eval_root else out / 'evaluation'
    if file_sha256(ROOT / fg['eval_file']) != fg['eval_sha256']: raise SystemExit('Evaluation data hash differs')
    cases = load_cases(ROOT / fg['eval_file'])
    if len(cases) != fg['eval_cases']: raise SystemExit('Evaluation size differs')
    if x.cases_limit: cases = cases[:x.cases_limit]
    jobs_log = eval_root / f'async_jobs_shard{x.shard}.jsonl'; eval_root.mkdir(parents=True, exist_ok=True)
    local = copy.deepcopy(cfg); local['rollout']['doctor_backend']['endpoint'] = x.doctor_url
    policy = SharedPolicy(local, load_key(), device='cuda:0')
    problems = policy.doctor.verify()['problems']
    if problems: raise SystemExit('Eval Doctor identity: ' + '; '.join(problems))
    policy.seed_all = lambda seed: None   # Each generation seeds under the model lock (as in the synchronous pool).
    cache_openings(policy.doctor, out / 'openings' / f'eval_async_shard{x.shard}.jsonl', fg['opening_identity'])
    while not (out / 'run_manifest.json').exists(): time.sleep(POLL_SECONDS)
    queue = EvaluationQueue(eval_root, cases, settings(cfg, out))
    mine = [(i, c) for i, c in enumerate(cases) if i % x.shards == x.shard]

    def run_job(ckpt, receipt):
        job = queue.register(ckpt, receipt['completed_cases'], receipt['policy_sha256'])
        manifest = json.loads((job / 'manifest.json').read_text()); job_sha = sha256_json(manifest)
        todo = [(i, c) for i, c in mine if not (job / 'cases' / f'case_{i:06d}.json').exists()]
        if not todo: return False
        (job / 'cases').mkdir(exist_ok=True); started = time.time()
        log_job(jobs_log, checkpoint=ckpt.name, shard=x.shard, gpu=vis, status='STARTED', queued_at=receipt_time(ckpt), todo=len(todo))

        def play(item):
            i, case = item
            episode = policy.generate(case, queue.settings['seed'] + i)
            return i, case, {'dialogue': episode['dialogue'], 'final_option': episode['final_option'],
                             'num_discussion_rounds': episode['num_discussion_rounds'], 'outcome': final_outcome(case, episode['final_option'])}
        # Freeze this checkpoint's adapter for the whole job; never the live training adapter.
        with evaluation_adapter(policy, manifest['checkpoint']):
            if adapter_sha256(policy.model) != manifest['policy_sha256']: raise RuntimeError('Evaluation checkpoint adapter mismatch')
            with ThreadPoolExecutor(EPISODES_PER_GPU) as pool:
                for i, case, result in pool.map(play, todo):
                    atomic_write_json(job / 'cases' / f'case_{i:06d}.json', {'case_id': case['case_id'], 'index': i, 'job_sha256': job_sha, **result})
        queue._summarize(job, manifest)
        log_job(jobs_log, checkpoint=ckpt.name, shard=x.shard, gpu=vis, status='COMPLETED', queued_at=receipt_time(ckpt),
                started_at=started, completed_at=time.time(), duration_s=time.time() - started, cases=len(todo))
        return True

    while True:
        for ckpt, receipt in committed_checkpoints(out, cfg, x.include_initial):
            if x.checkpoint and ckpt.name != x.checkpoint: continue
            try: run_job(ckpt, receipt)
            except Exception as e:                                  # an eval failure never touches training; retried next poll
                log_job(jobs_log, checkpoint=ckpt.name, shard=x.shard, gpu=vis, status='FAILED', error=repr(e), trace=traceback.format_exc()[-2000:])
        state = json.loads((out / 'run_state.json').read_text()) if (out / 'run_state.json').exists() else {}
        final = eval_root / f"checkpoint_{cfg['schedule']['cases']:06d}" / 'summary.json'
        if state.get('case_cursor') == cfg['schedule']['cases'] and final.exists() and json.loads(final.read_text()).get('status') == 'complete':
            if not x.cases_limit and not x.checkpoint and not x.eval_override:
                atomic_write_json(out / 'EXPERIMENT_COMPLETE.json', {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                  'final_checkpoint': f"checkpoint_{cfg['schedule']['cases']:06d}", 'final_eval': json.loads(final.read_text())})
            return
        if x.once: return
        time.sleep(POLL_SECONDS)


def receipt_time(ckpt): return (Path(ckpt) / 'COMPLETE.json').stat().st_mtime


if __name__ == '__main__': main()
