import hashlib
import json
import os
from pathlib import Path

from src.online_rl.sherpa.prompt_monitor import OptimizationMonitor, render


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def cfg(root):
    (root / 'initial.txt').write_text('initial')
    return {'_root': str(root), 'schedule': {'cases': 120}, 'rollout': {'branching': 32},
            'prompt_optimization': {'mixed_cases': 24, 'iterations': 3, 'initial_prompt_file': 'initial.txt'}}


def test_missing_directory_does_not_create_files(tmp_path):
    out = tmp_path / 'missing'
    state = OptimizationMonitor().snapshot(out, cfg(tmp_path))
    assert state['stage'] == '시작 대기'
    assert state['base_cases'] == 0 and not state['running']
    assert not out.exists()
    assert '0/3840' in render(state)


def test_saved_cases_and_partial_finals_are_read_only(tmp_path):
    config = cfg(tmp_path)
    save(tmp_path / 'base_rollouts/a/rollouts.json', {'trajectories': [{}] * 32, 'mixed': True})
    path = tmp_path / 'base_rollouts/b/calls.jsonl'
    path.parent.mkdir(parents=True)
    row = {'kind': 'final', 'identity': {'case': 'b', 'policy': 'p', 'protocol': 'v', 'path': '0'}}
    path.write_text(json.dumps(row) + '\n')
    save(tmp_path / 'live/gpu0.json', {'pid': os.getpid(), 'active': {'agent_batch_generate': 2}})
    before = {str(p): p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}
    state = OptimizationMonitor().snapshot(tmp_path, config)
    assert state['base_cases'] == 1 and state['base_trajectories'] == 33
    assert state['mixed_cases'] == 1 and state['running']
    assert state['active_cases'][0]['case_id'] == 'b'
    assert before == {str(p): p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}


def test_incremental_candidate_progress_partial_lines_and_confirmation(tmp_path):
    config = cfg(tmp_path)
    digest = hashlib.sha256(b'initial').hexdigest()
    folder = tmp_path / 'candidates' / digest
    folder.mkdir(parents=True)
    save(folder / 'cases/a.json', {'fitness_auc': .75})
    path = tmp_path / 'optimization_calls.jsonl'
    def row(kind, **ident):
        return json.dumps({'kind': kind, 'identity': {'scope': 'offline_prompt', 'candidate': digest, **ident},
                           'value': {'valid': True}})
    first = row('rubric_set', case_id='a')
    second = row('trajectory_score', trajectory='a/0')
    path.write_text(first + '\n' + second[:15])
    reader = OptimizationMonitor()
    state = reader.snapshot(tmp_path, config)
    candidate = state['candidates'][0]
    assert candidate['rubric_cases'] == 1 and candidate['judged_trajectories'] == 0
    with path.open('a') as f: f.write(second[15:] + '\n')
    state = reader.snapshot(tmp_path, config)
    assert state['candidates'][0]['judged_trajectories'] == 1
    assert reader.snapshot(tmp_path, config)['candidates'][0]['judged_trajectories'] == 1
    save(tmp_path / 'confirmation.json', {'repeats': [{'repeat': 1, 'initial_mean_auc': .6,
                                                      'final_mean_auc': .7, 'delta_mean_auc': .1}]})
    save(tmp_path / 'final_prompt.json', {'schema': 'sherpa_auc_prompt_v1', 'mean_auc': .8})
    state = reader.snapshot(tmp_path, config)
    assert state['complete'] and 'ΔAUC +0.1000' in render(state)
    path.write_text('')
    assert reader.snapshot(tmp_path, config)['candidates'][0]['judged_trajectories'] == 0


def test_usage_is_not_double_counted_and_corrupt_artifacts_are_tolerated(tmp_path):
    config = cfg(tmp_path)
    (tmp_path / 'history.json').write_text('{')
    path = tmp_path / 'api_usage_optimize-prompt.jsonl'
    path.write_text(json.dumps({'input_tokens': 10, 'output_tokens': 20, 'cost_usd': None}) + '\n')
    reader = OptimizationMonitor()
    for _ in range(2):
        state = reader.snapshot(tmp_path, config)
        assert state['api_usage']['calls'] == 1
        assert state['api_usage']['input_tokens'] == 10
        assert state['api_usage']['unpriced'] == 1
