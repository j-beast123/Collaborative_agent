"""Async evaluator parity with the synchronous runner (CPU only, no model)."""
import json
from pathlib import Path
from scripts.online_rl import async_eval as A
from src.online_rl.integrated.records import atomic_write_json
from src.online_rl.tree.five_gpu import EvaluationQueue

ROOT = Path(__file__).resolve().parents[1]


def test_due_checkpoints_match_the_synchronous_rule():
    cfg = {'five_gpu': {'checkpoint_every_cases': 30}, 'schedule': {'cases': 300}}
    assert [c for c in range(0, 301, 5) if A.due(c, cfg)] == list(range(30, 301, 30))
    assert A.due(10, {'five_gpu': {'checkpoint_every_cases': 30}, 'schedule': {'cases': 10}})
    src = (ROOT / 'scripts/online_rl/run_five_gpu.py').read_text()
    assert "cursor > 0 and (cursor % fg['checkpoint_every_cases'] == 0 or cursor == cfg['schedule']['cases'])" in src


def test_settings_identity_matches_the_runner(tmp_path):
    cfg = {'five_gpu': {'eval_seed': 0, 'eval_sha256': 'e4'}}
    atomic_write_json(tmp_path / 'run_manifest.json', {'protocol_sha256': 'p'})
    assert A.settings(cfg, tmp_path) == {'seed': 0, 'data_sha256': 'e4', 'protocol_sha256': 'p', 'dataset': 'CasiMedicos all_en_622'}
    src = (ROOT / 'scripts/online_rl/run_five_gpu.py').read_text()
    assert "{'seed':fg['eval_seed'],\n        'data_sha256':fg['eval_sha256'],'protocol_sha256':protocol,'dataset':'CasiMedicos all_en_622'}" in src


def test_shards_cover_every_case_exactly_once():
    cases = list(range(622))
    shards = [[i for i in cases if i % 3 == k] for k in range(3)]
    assert sorted(i for s in shards for i in s) == cases and sum(map(len, shards)) == 622


def test_only_committed_due_checkpoints_are_evaluated(tmp_path):
    cfg = {'five_gpu': {'checkpoint_every_cases': 30}, 'schedule': {'cases': 300}}
    for name, cursor, done in (('checkpoint_000000', 0, True), ('checkpoint_000030', 30, True), ('checkpoint_000060', 60, False),
                               ('checkpoint_000090.pending', 90, True), ('recovery_000007', 35, True)):
        d = tmp_path / 'checkpoints' / name; d.mkdir(parents=True)
        if done: atomic_write_json(d / 'COMPLETE.json', {'completed_cases': cursor, 'policy_sha256': 's'})
    assert [c.name for c, _ in A.committed_checkpoints(tmp_path, cfg)] == ['checkpoint_000030']


def test_shard_writes_use_the_queue_schema(tmp_path):
    cases = [{'case_id': f'c{i}'} for i in range(4)]
    q = EvaluationQueue(tmp_path, cases, {'seed': 0})
    job = q.register('/ckpt', 30, 'sha'); manifest = json.loads((job / 'manifest.json').read_text())
    from src.online_rl.integrated.records import sha256_json
    for i in range(4):
        atomic_write_json(job / 'cases' / f'case_{i:06d}.json', {'case_id': f'c{i}', 'index': i, 'job_sha256': sha256_json(manifest),
                                                                'outcome': i % 2, 'num_discussion_rounds': 2, 'dialogue': [], 'final_option': 'A'})
    q._summarize(job, manifest)
    s = json.loads((job / 'summary.json').read_text())
    assert s['status'] == 'complete' and s['evaluated'] == 4 and s['correct'] == 2
