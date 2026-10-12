import json
from pathlib import Path

import pytest

from src.online_rl.tree.cache import CallCache
from src.online_rl.sherpa.cache_options import fresh_prompt_run


def test_uncached_calls_ignore_disk_and_memory_but_keep_error_audit(tmp_path):
    path = tmp_path / 'calls.jsonl'
    CallCache(path, enabled=True).call('doctor', {'seed': 1}, lambda: 'old')
    original = path.read_bytes()
    cache = CallCache(path, enabled=False)
    calls = []
    def generate():
        calls.append(1)
        return len(calls)
    assert cache.call('doctor', {'seed': 1}, generate) == 1
    assert cache.call('doctor', {'seed': 1}, generate) == 2
    assert path.read_bytes() == original
    assert not cache.store and not cache.hits and cache.calls == {'doctor': 2}
    def fail():
        error = RuntimeError('length')
        error.audit = {'raw': 'partial'}
        raise error
    with pytest.raises(RuntimeError): cache.call('doctor', {'seed': 2}, fail)
    assert json.loads(path.with_suffix('.errors.jsonl').read_text())['audit']['raw'] == 'partial'


def test_uncached_search_preserves_base_input_and_isolates_old_results(tmp_path):
    source = tmp_path / 'old'
    case = source / 'base_rollouts' / 'case'
    case.mkdir(parents=True)
    (source / 'base_manifest.json').write_text('{"complete":true}')
    (case / 'rollouts.json').write_text('{"trajectories":[]}')
    (case / 'calls.jsonl').write_text('old call cache')
    (source / 'history.json').write_text('old candidates')
    cfg = {'prompt_optimization': {'output_dir': 'old', 'final_artifact': 'old/final_prompt.json'}}
    fresh, path = fresh_prompt_run(cfg, tmp_path, 'optimize-prompt')
    home = Path(fresh['prompt_optimization']['output_dir'])
    assert path.is_file() and home != source
    assert not fresh['prompt_optimization']['cache_enabled']
    assert (home / 'base_manifest.json').read_bytes() == (source / 'base_manifest.json').read_bytes()
    assert (home / 'base_rollouts/case/rollouts.json').exists()
    assert not (home / 'base_rollouts/case/calls.jsonl').exists()
    assert not (home / 'history.json').exists()
    assert cfg['prompt_optimization']['output_dir'] == 'old'
    assert (source / 'history.json').read_text() == 'old candidates'


def test_uncached_full_run_recollects_everything(tmp_path):
    source = tmp_path / 'old'
    source.mkdir()
    (source / 'base_manifest.json').write_text('{"complete":true}')
    cfg = {'prompt_optimization': {'output_dir': 'old'}}
    fresh, _ = fresh_prompt_run(cfg, tmp_path, 'all')
    assert not (Path(fresh['prompt_optimization']['output_dir']) / 'base_manifest.json').exists()


def test_environment_switch_disables_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('SHERPA_CALL_CACHE', '0')
    assert not CallCache(tmp_path / 'calls.jsonl').enabled
