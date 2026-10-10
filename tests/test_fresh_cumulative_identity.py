"""The new run may score the whole bank but cannot import semantic reuse."""
from pathlib import Path

import pytest

from src.online_rl.tree import cumulative_bank, cumulative_run
from src.online_rl.tree.experiment import load_config


CONFIG = 'configs/experiments/train120_v8_fresh_source_case_exact.json'
ROOT = Path(__file__).resolve().parents[1]


def test_new_identity_starts_empty_and_scores_the_whole_bank():
    cfg = load_config(CONFIG)
    identity = cumulative_run.config_identity(cfg, ROOT)
    assert cfg['policy']['init_adapter'] is None
    assert cfg['schedule']['cases'] == 120
    assert identity['initial_snapshot_sha256'] is None
    assert identity['scoring_scope'] == 'all'
    assert identity['semantic_matcher'] == 'disabled'
    assert identity['reuse_effect_verification'] == 'new_source_witness_only'


@pytest.mark.parametrize('change', [
    {'scoring_scope': 'top_k'},
    {'initial_snapshot_file': 'old_bank.json', 'initial_snapshot_sha256': 'old'},
])
def test_new_identity_rejects_narrow_or_imported_bank(change):
    cfg = load_config(CONFIG)
    cfg['rubric']['cumulative_bank'].update(change)
    with pytest.raises(ValueError, match='empty initial bank and full frozen-bank scoring'):
        cumulative_run.config_identity(cfg, ROOT)


def test_new_round_refuses_pre_generation_semantic_reuse(monkeypatch, tmp_path):
    cfg = load_config(CONFIG)
    before = cumulative_bank.empty_snapshot()
    monkeypatch.setattr(cumulative_run, 'committed_snapshot', lambda *_: before)
    monkeypatch.setattr(cumulative_run, 'publish_snapshot', lambda *_: None)
    state = {'policy_sha256': 'policy', 'case_cursor': 0}

    def collect(phase, prompt):
        assert phase == 'bank_prepare'
        return [{'case_id': f'case{i}', 'case_cursor': i, 'policy_sha256': 'policy',
                 'cumulative_bank_input': before,
                 'cumulative_reuse': ([{'leaf_id': 'forbidden'}] if i == 0 else [])}
                for i in range(5)]

    with pytest.raises(RuntimeError, match='cannot inherit a semantic bank match'):
        cumulative_run.round_records(cfg, ROOT, tmp_path, state, 1, tmp_path / 'batch', collect)
