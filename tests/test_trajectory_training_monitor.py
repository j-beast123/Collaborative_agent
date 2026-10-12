import json
import os

from src.online_rl.sherpa.monitor import TrainingMonitor, render
from src.online_rl.tree.cache import CallCache


def save(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))


def test_partial_cache_and_parallel_case_stages_are_read_only(tmp_path):
    folder = tmp_path / 'steps/step_000001'
    first, second = folder / 'one', folder / 'two'
    save(tmp_path / 'progress.json', {'step': 1, 'pid': os.getpid(), 'stage': 'rollout',
                                     'batch_started_at': 900, 'run_started_at': 850})
    cache = CallCache(first / 'calls.jsonl', enabled=True)
    for i in range(3): cache.call('final', {'path': str(i)}, lambda: 'A')
    second.mkdir(parents=True)
    save(first / 'rollouts.json', {'tree': 'x' * 100000, 'outcomes': [1] * 16 + [0] * 16})
    os.utime(first / 'rollouts.json', (950, 950))
    (first / 'reflections.json').write_text('[]'); os.utime(first / 'reflections.json', (955, 955))
    cache.call('rubric_set', {'case_id': 'one'}, lambda: {'seconds': 10, 'rubrics': []})
    for i in range(4): cache.call('trajectory_score', {'trajectory': str(i)}, lambda: {})
    reader = TrainingMonitor(); before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}
    data = reader.snapshot(tmp_path, now=1000)
    assert data['cases'][0]['rollouts_complete'] == 32
    assert data['cases'][0]['phase'] == 'Scoring'
    assert data['cases'][0]['rollout_seconds'] == 50
    assert data['cases'][0]['generation_seconds'] == 10
    assert data['cases'][0]['judged_trajectories'] == 4
    assert data['cases'][1]['phase'] == 'rollout'
    assert 'Complete 32/128' in render(data)
    assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}
    path = second / 'calls.jsonl'; encoded = json.dumps({'key': 'x', 'kind': 'final', 'value': {}})
    path.write_text(encoded[:20])
    assert reader.snapshot(tmp_path, now=1000)['cases'][1]['rollouts_complete'] == 0
    with path.open('a') as f: f.write(encoded[20:] + '\n')
    assert reader.snapshot(tmp_path, now=1000)['cases'][1]['rollouts_complete'] == 1
    assert reader.snapshot(tmp_path, now=1000)['cases'][1]['rollouts_complete'] == 1


def test_missing_run_does_not_create_files(tmp_path):
    home = tmp_path / 'missing'
    data = TrainingMonitor().snapshot(home, now=1000)
    assert not home.exists()
    assert data['cases'] == [] and not data['running']
    assert 'Stopped / waiting' in render(data)
