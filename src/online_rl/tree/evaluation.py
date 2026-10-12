"""Milestone checkpoint evaluation: RQ1 task accuracy on a frozen held-out split. One full Doctor-Agent episode per case
with the rollout protocol (plain Agent prompt, no rubric, no training signal); correct = the Doctor FINAL option is the
gold option. Results go to eval/<checkpoint>/: cases.jsonl (written per case, reused on retry) and result.json (last)."""
from __future__ import annotations
import json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
from src.online_rl.cases import load_cases
from src.online_rl.guards import question_sha256
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.records import atomic_write_json, read_json, sha256_json

EVAL_PROTOCOL = 'tree_full_episode_task_accuracy_v1'


def eval_cases(path: str | Path, split: str, manifest: dict) -> list[dict]:
    """The complete held-out split from a local file, checked against the protected manifest (ids / question hashes)."""
    cases = load_cases(path); ref = manifest['splits'][split]
    ids, hashes = set(ref['ids']), set(ref['question_sha256'])
    outside = [c['case_id'] for c in cases if c['case_id'] not in ids and question_sha256(c['question']) not in hashes]
    if outside: raise ValueError(f'{len(outside)} cases are not in protected split {split}: {outside[:3]}')
    if hashes and {question_sha256(c['question']) for c in cases} != hashes: raise ValueError(f'{path} is not the complete {split} split')
    if len({c['case_id'] for c in cases}) != len(cases): raise ValueError(f'{path} has duplicate case ids')
    return cases


class MilestoneEvaluator:
    """play(case, seed) -> trajectory with 'final_option' (and 'termination_reason', 'num_discussion_rounds'); outcome_fn(case, option)."""

    def __init__(self, play: Callable, outcome_fn: Callable, cases: list[dict], split: str, manifest_sha256: str, protocol_sha256: str, seed: int):
        self.play, self.outcome_fn, self.cases, self.seed = play, outcome_fn, cases, seed
        self.info = {'split': split, 'n_cases': len(cases), 'manifest_sha256': manifest_sha256, 'eval_protocol': EVAL_PROTOCOL,
                     'rollout_protocol_sha256': protocol_sha256, 'eval_seed': seed, 'episodes_per_case': 1}
        self.info['eval_protocol_sha256'] = sha256_json(self.info)

    def __call__(self, *, milestone: int, checkpoint: Path, out_dir: Path, provenance: dict) -> dict:
        out_dir = Path(out_dir); done_path, part_path = out_dir / 'result.json', out_dir / 'cases.jsonl'
        if done_path.exists():
            r = read_json(done_path)
            if (r['checkpoint_sha256'], r['eval_protocol_sha256']) != (provenance['checkpoint_sha256'], self.info['eval_protocol_sha256']):
                raise RuntimeError(f'{done_path} belongs to another checkpoint or protocol')
            return r
        out_dir.mkdir(parents=True, exist_ok=True); key = (provenance['checkpoint_sha256'], self.info['eval_protocol_sha256'])
        rows = {}
        if part_path.exists():
            for line in part_path.read_text().splitlines():
                r = json.loads(line)
                if (r['checkpoint_sha256'], r['eval_protocol_sha256']) == key: rows[r['case_id']] = r
        started = time.time(); lock = threading.Lock()
        with part_path.open('a') as f:
            def one(c):
                t = self.play(c, derive_seed(self.seed, c['case_id'], 'eval', 0)); o = self.outcome_fn(c, t['final_option'])
                r = {'case_id': c['case_id'], 'correct': float(o == 1.0), 'final_option': t['final_option'], 'parsed': o is not None,
                     'termination_reason': t.get('termination_reason'), 'agent_turns': t.get('num_discussion_rounds'),
                     'checkpoint_sha256': key[0], 'eval_protocol_sha256': key[1]}
                with lock: f.write(json.dumps(r) + '\n'); f.flush(); os.fsync(f.fileno()); rows[c['case_id']] = r
            todo = [c for c in self.cases if c['case_id'] not in rows]
            workers = int(getattr(self.play, 'workers', 1))           # execution only: episodes are independent given their seeds
            if workers > 1:
                with ThreadPoolExecutor(workers) as pool: list(pool.map(one, todo))
            else:
                for c in todo: one(c)
        per_case = [{k: rows[c['case_id']][k] for k in ('case_id', 'correct', 'final_option', 'parsed', 'termination_reason', 'agent_turns')}
                    for c in self.cases]
        result = {**provenance, **self.info, 'accuracy': sum(r['correct'] for r in per_case) / len(per_case), 'per_case': per_case,
                  'started_at': started, 'completed_at': time.time(), 'status': 'complete'}
        atomic_write_json(done_path, result); return result
