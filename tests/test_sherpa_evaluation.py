"""Full held-out coverage and iteration checkpoint selection; no GPU/API work."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.online_rl.integrated.records import sha256_json
from src.online_rl.sherpa import evaluation, training
from src.online_rl.tree.experiment import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['ours', 'sherpa', 'sherpa_qwen_doctor'])
def test_full_source_is_used(name):
    cfg = load_config(f'configs/{name}.json')
    cases, manifest = evaluation.evaluation_cases(cfg, ROOT)
    source = [json.loads(line) for line in (ROOT / cfg['evaluation']['file']).read_text().splitlines()]
    assert len(cases) == len(source) == 622
    assert {row['case_id'] for row in cases} == {row['case_id'] for row in source}
    assert 'curve_case_ids' not in manifest
    assert 'curve_cases' not in cfg['evaluation']


@pytest.mark.parametrize('damage', ['omit', 'duplicate', 'foreign'])
def test_incomplete_or_invalid_test_split_rejected(tmp_path, damage):
    cfg = load_config('configs/ours.json')
    manifest = json.loads((ROOT / cfg['evaluation']['split_manifest']).read_text())
    ids = manifest['test_case_ids']
    if damage == 'omit':
        del ids[-22:]
    elif damage == 'duplicate':
        ids[-1] = ids[0]
    else:
        ids[-1] = 'not_in_source'
    path = tmp_path / 'split.json'
    path.write_text(json.dumps(manifest))
    cfg['evaluation']['split_manifest'] = str(path)
    with pytest.raises(ValueError, match='every source case exactly once'):
        evaluation.evaluation_cases(cfg, ROOT)


@pytest.mark.parametrize('completed,steps', [(29, []), (45, [30]), (90, [30, 60, 90])])
def test_only_iteration_ends_use_all_622(tmp_path, monkeypatch, completed, steps):
    cfg = load_config('configs/ours.json')
    cfg['evaluation']['file'] = str(ROOT / cfg['evaluation']['file'])
    cfg['evaluation']['split_manifest'] = str(ROOT / cfg['evaluation']['split_manifest'])
    cfg['output_dir'] = 'run'
    home = tmp_path / 'run'
    home.mkdir()
    final = home / 'checkpoints/checkpoint_000090'
    (home / 'run_state.json').write_text(json.dumps({
        'completed_steps': completed, 'complete': completed == 90, 'final_checkpoint': str(final)}))
    for step in steps:
        checkpoint = home / 'checkpoints' / f'checkpoint_{step:06d}'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'COMPLETE.json').write_text(json.dumps({'policy_sha256': str(step)}))
    policy = SimpleNamespace(doctor=SimpleNamespace(protocol='doctor'))
    monkeypatch.setattr(evaluation, 'original_base_policy', lambda *a: nullcontext())
    monkeypatch.setattr(evaluation, 'evaluation_adapter', lambda *a: nullcontext())
    jobs = []
    def job(cfg, policy, cases, folder, identity):
        jobs.append((identity.get('step'), len(cases), folder.name))
        return {'identity': identity, 'case_count': len(cases)}
    monkeypatch.setattr(evaluation, 'evaluate_job', job)
    result = evaluation.evaluate(cfg, policy, tmp_path)
    assert [(step, count) for step, count, _ in jobs] == [(None, 622)] + [(step, 622) for step in steps]
    assert [row['step'] for row in result['iteration_results']] == steps
    assert 'learning_curve' not in result
    assert (result['main_result'] is not None) == (completed == 90)
    if completed == 90:
        assert jobs[-1][2] == 'final'


def test_evaluation_only_change_preserves_training_resume(tmp_path):
    original = {'protocol': 'fixed', 'config': {'evaluation': {'test_cases': 600}, 'optimizer': {'lr': 1}},
                'frozen_prompt': 'prompt', 'case_schedule': [['case']]}
    path = tmp_path / 'run_manifest.json'
    path.write_text(json.dumps(original))
    state = {'fingerprint': sha256_json(original)}
    changed = copy.deepcopy(original)
    changed['config']['evaluation'] = {'test_cases': 622, 'checkpoint_steps': [30, 60, 90]}
    assert training.resume_fingerprint(changed, state, path) == state['fingerprint']
    changed['config']['optimizer']['lr'] = 2
    with pytest.raises(RuntimeError, match='Training protocol/config changed'):
        training.resume_fingerprint(changed, state, path)
    path.write_text(json.dumps({**original, 'frozen_prompt': 'tampered'}))
    with pytest.raises(RuntimeError, match='Training protocol/config changed'):
        training.resume_fingerprint(original | {'config': {'evaluation': {}}}, state, path)
