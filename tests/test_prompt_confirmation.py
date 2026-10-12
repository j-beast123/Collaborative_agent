"""Offline checks: tied AUC and independent confirmation without winner reselection."""
import json
import threading
from types import SimpleNamespace

import pytest

from src.online_rl.sherpa.rubrics import auc
from src.online_rl.sherpa import prompt_search as search


@pytest.mark.parametrize('outcomes,rewards,expected', [
    ([1, 0], [0.5, 0.5], 0.5),
    ([1, 1, 0, 0], [1, 0.5, 0.5, 0], 0.875),
    ([1, 0], [0, 1], 0.0),
    ([1, 0], [1, 0], 1.0),
    ([1, 1], [0, 1], None),
])
def test_tied_auc(outcomes, rewards, expected):
    assert auc(outcomes, rewards) == expected


@pytest.mark.parametrize('duplicate', [False, True])
@pytest.mark.parametrize('mixed_cases', [20, 24])
@pytest.mark.parametrize('selection_repeats', [0, 2])
def test_confirmation_uses_fresh_scope_and_does_not_reselect(tmp_path, monkeypatch, duplicate, mixed_cases, selection_repeats):
    home = tmp_path / 'search'
    home.mkdir()
    cases = [{'case_id': str(i)} for i in range(120)]
    identity = {'test': 'fixed-base'}
    (home / 'base_manifest.json').write_text(json.dumps({'complete': True, 'identity': identity}))
    for case in cases:
        folder = home / 'base_rollouts' / case['case_id']
        folder.mkdir(parents=True)
        (folder / 'rollouts.json').write_text(json.dumps({'mixed': True, 'trajectories': []}))
    (tmp_path / 'initial.txt').write_text('initial')
    monkeypatch.setattr(search, 'base_identity', lambda *a: identity)
    monkeypatch.setattr(search, 'schedule', lambda *a: [cases[i:i+4] for i in range(0, 120, 4)])
    monkeypatch.setattr(search, 'reflections', lambda *a: [])
    calls = []
    initial_cases = threading.Barrier(4)

    def assess(case, trajectories, examples, prompt, backend, cache, ident):
        calls.append((prompt, ident['scope'], case['case_id']))
        if prompt == 'initial' and ident['scope'] == 'offline_prompt':
            initial_cases.wait(timeout=5)  # One candidate must overlap four cases.
        confirming = ident['scope'].startswith('offline_prompt_confirmation_')
        value = (0.9 if prompt == 'initial' else 0.1) if confirming else (0.5 if prompt == 'initial' else 0.8)
        if selection_repeats and not duplicate and prompt == 'candidate-1-0' and not confirming:
            value = 0.95 if ident['scope'] == 'offline_prompt' else 0.2
        return {'valid': True, 'auc': value, 'generation': {'rubrics': []}}

    monkeypatch.setattr(search, 'assess', assess)
    backend = SimpleNamespace(identity=lambda: {'fake': True},
                              propose=lambda parent, evidence, iteration, index:
                              {'text': 'candidate-1-0' if duplicate else f'candidate-{iteration}-{index}'})
    cfg = {'schedule': {'seed': 0}, 'prompt_optimization': {
        'output_dir': 'search', 'initial_prompt_file': 'initial.txt',
        'iterations': 3, 'candidates_per_iteration': 4, 'confirmation_repeats': 1, 'mixed_cases': mixed_cases,
        'selection_repeats': selection_repeats,
        'candidate_workers': 4, 'case_workers': 4,
        'final_artifact': 'search/final_prompt.json'}}
    final = search.optimize(cfg, None, backend, cases, tmp_path)
    expected = 'candidate-1-1' if selection_repeats and not duplicate else 'candidate-1-0'
    assert (home / 'final_prompt.txt').read_text() == expected
    assert final['mean_auc'] == pytest.approx(0.8)
    assert [row['case_id'] for row in final['case_scores']] == json.loads((home / 'optimization_split.json').read_text())['case_ids']
    report = json.loads((home / 'confirmation.json').read_text())
    assert report['used_for_selection'] is False
    assert report['held_out'] is False
    assert report['repeats'][0]['delta_mean_auc'] == pytest.approx(-0.8)
    extra = (2 if duplicate else 7) * selection_repeats
    assert len(calls) == ((2 if duplicate else 13) + extra + 2) * mixed_cases
    assert len(final['selection_mean_aucs']) == selection_repeats + 1
    calls.clear()
    search.optimize(cfg, None, backend, cases, tmp_path)
    assert calls == []  # Interrupted/repeated runs reuse the saved evidence.
