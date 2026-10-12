from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from src.online_rl.sherpa.backend import ChecklistBackend
from src.online_rl.sherpa import prompt_search as search


@pytest.mark.parametrize('standalone', [False, True])
def test_private_reflection_system_is_separate_only_when_enabled(monkeypatch, standalone):
    captured = []
    def chat(*args):
        captured.append(args)
        return 'The assistant checked an unsupported premise.'
    backend = object.__new__(ChecklistBackend)
    backend.policy = SimpleNamespace(doctor=SimpleNamespace(system='Discussion: state a choice.', _chat=chat))
    backend.cfg = {'rubric_pipeline': {'reflection_max_tokens': 1536}}
    if standalone:
        backend.cfg['rubric_pipeline']['reflection_system_mode'] = 'standalone'
    backend.reflection_prompt = 'Private reflection only.'
    monkeypatch.setattr('src.online_rl.sherpa.backend.usage.case', lambda *a: nullcontext())
    monkeypatch.setattr('src.online_rl.sherpa.backend.with_role', lambda role, fn: fn())
    backend.reflect({'case_id': 'case'}, {'trajectory_id': 't', 'outcome': 0, 'dialogue': []}, 7)
    assert captured[0][3] == ('Private reflection only.' if standalone else
                              'Discussion: state a choice.\n\nPrivate reflection only.')


@pytest.mark.parametrize('seed_auc,expected', [(0.51, 'initial'), (0.55, 'seed')])
def test_seed_must_clear_minimum_gain_and_confirmation_keeps_original_baseline(tmp_path, monkeypatch, seed_auc, expected):
    home = tmp_path / 'search'
    home.mkdir()
    cases = [{'case_id': str(i)} for i in range(120)]
    identity = {'base': 'fixed'}
    (home / 'base_manifest.json').write_text(json.dumps({'identity': identity, 'complete': True}))
    for case in cases:
        folder = home / 'base_rollouts' / case['case_id']
        folder.mkdir(parents=True)
        (folder / 'rollouts.json').write_text(json.dumps({'mixed': True, 'trajectories': []}))
    for text in ('initial', 'seed'):
        (tmp_path / f'{text}.txt').write_text(text)
    monkeypatch.setattr(search, 'base_identity', lambda *a: identity)
    monkeypatch.setattr(search, 'schedule', lambda *a: [cases[i:i+4] for i in range(0, 120, 4)])
    monkeypatch.setattr(search, 'reflections', lambda *a: [])
    monkeypatch.setattr(search, 'assess', lambda case, trajectories, examples, text, *a:
                        {'valid': True, 'auc': seed_auc if text == 'seed' else 0.5, 'generation': {'rubrics': []}})
    backend = SimpleNamespace(identity=lambda: {}, propose=lambda parent, *a: {'text': parent})
    cfg = {'schedule': {'seed': 0}, 'prompt_optimization': {
        'output_dir': 'search', 'final_artifact': 'search/final_prompt.json', 'initial_prompt_file': 'initial.txt',
        'additional_initial_prompt_files': ['seed.txt'], 'min_auc_improvement': 0.02,
        'iterations': 2, 'candidates_per_iteration': 4, 'mixed_cases': 20,
        'confirmation_repeats': 1, 'selection_repeats': 2}}
    search.optimize(cfg, None, backend, cases, tmp_path)
    assert (home / 'final_prompt.txt').read_text() == expected
    report = json.loads((home / 'confirmation.json').read_text())
    assert report['search_initial_mean_auc'] == pytest.approx(0.5)
    assert report['repeats'][0]['initial_mean_auc'] == pytest.approx(0.5)
