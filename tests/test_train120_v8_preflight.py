"""Configuration identity and offline guardrails for the fresh source-case pilot."""
from pathlib import Path

import pytest

from src.online_rl.tree import experiment


ROOT = Path(__file__).resolve().parents[1]
PILOT = 'configs/experiments/train120_v8_fresh_source_case_exact.json'


def test_new_run_identity_does_not_change_existing_run_ids():
    old = experiment.load_config('configs/experiments/train120_v7_fresh.json')
    fresh = experiment.load_config(PILOT)
    assert old['experiment']['run_id'] == 'r05_o05__p0fixed__gemma3doctor__seed0'
    assert fresh['experiment']['run_id'] == (
        'r05_o05__p0fixed__gemma3doctor__seed0__sourcecase_exact_v8')
    assert old['output_dir'] != fresh['output_dir']
    assert experiment.fingerprint(old) != experiment.fingerprint(fresh)


@pytest.mark.parametrize('label', ['', '../old', 'two words', 'UPPER', 'a' * 49, 123])
def test_run_label_cannot_escape_or_collide_with_a_directory(label):
    cfg = experiment.load_raw(PILOT)
    cfg['experiment']['run_label'] = label
    with pytest.raises(ValueError, match='run_label'):
        experiment.resolve(cfg)


def test_five_case_stop_keeps_resume_identity_and_excludes_evaluation():
    from src.online_rl.tree.five_gpu import batch_slots

    cfg = experiment.load_config(PILOT)
    assert cfg['schedule']['cases'] == 120
    assert cfg['five_gpu']['batch_cases'] == 5
    assert cfg['five_gpu']['checkpoint_every_cases'] == 30
    assert cfg['five_gpu']['checkpoint_evaluation_enabled'] is False
    assert cfg['five_gpu']['base_evaluation']['enabled'] is False
    assert cfg['evaluation'] is None
    assert cfg['policy']['init_adapter'] is None
    assert cfg['supervision_handoff'] is None
    assert cfg['rubric']['cumulative_bank']['protocol'] == 'cumulative_keep_bank_v1'
    assert cfg['rubric']['cumulative_bank']['scoring_scope'] == 'all'
    assert cfg['rubric']['cumulative_bank'].get('initial_snapshot_file') is None
    assert cfg['rubric']['cumulative_bank'].get('initial_snapshot_sha256') is None
    assert cfg['rubric']['assignment_protocol'] == 'exact_generated_source_case_v1'
    assert cfg['judges']['compact_runtime'] == 'v7_5_v7_6'
    assert [slot['index'] for slot in batch_slots(0, cfg['schedule']['cases'],
           interval=cfg['five_gpu']['checkpoint_every_cases'], workers=5, lanes=1)] == list(range(5))
    assert [slot['index'] for slot in batch_slots(5, cfg['schedule']['cases'],
           interval=cfg['five_gpu']['checkpoint_every_cases'], workers=5, lanes=1)] == list(range(5, 10))
    assert not (ROOT / cfg['output_dir'] / 'run_state.json').exists()


def test_offline_preflight_needs_exact_code_and_real_scorer_selector():
    from scripts.online_rl.preflight_train120_v8 import check

    plan = check()
    assert plan['status'] == 'offline_contract_pass'
    assert plan['first_five_case_ids'] == [
        'medmcqa_d5efdbf8-93d6-4ce4-a89e-df816fb16944',
        'medmcqa_e7d47277-58bc-496e-b118-9c9e0f82004b',
        'medmcqa_4797bb62-d4d5-46e6-860c-aa3bd6da4edb',
        'medmcqa_dbdeda4f-ff5f-456c-86cf-980e7d41b8e0',
        'medmcqa_c2482fd9-9011-4b6b-b881-8217881c1873',
    ]
    assert '--max-batches 1' in plan['pilot_launch']
    assert '--max-batches' not in plan['continuation_launch']
