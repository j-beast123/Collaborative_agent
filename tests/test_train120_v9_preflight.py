"""Offline checks for the direct-Luna source-case experiment."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.online_rl import preflight_train120_v9 as P
from src.online_rl.tree import cumulative_run, experiment


ROOT = Path(__file__).resolve().parents[1]


def test_direct_remote_roles_and_local_agent_doctor_stay_separate():
    cfg = experiment.load_config(P.CONFIG)
    assert cfg['api_provider'] == 'openai_direct'
    assert cfg['rubric']['generation']['api_provider'] == 'openai_direct'
    assert cfg['judges']['api_provider'] == 'openai_direct'
    assert cfg['rubric']['generator_model'] == 'gpt-6-luna'
    assert {cfg['judges'][key] for key in
            ('gate_model', 'scorer_model', 'intervention_audit_model')} == {'gpt-6-luna'}
    assert {cfg['prompt_update'][key] for key in ('reflection_model', 'refiner_model')} == {'gpt-6-luna'}
    assert cfg['prompt_update']['enabled'] is False
    assert cfg['policy']['base_model'] == 'Qwen/Qwen3-8B'
    assert cfg['policy']['init_adapter'] is None
    assert cfg['rollout']['doctor_model'] == cfg['feedback']['model']
    assert cfg['rollout']['doctor_model'].startswith('ollama/')


def test_exact_only_generation_keeps_empty_initial_bank_and_full_batch_scoring():
    cfg = experiment.load_config(P.CONFIG)
    assert cfg['rubric']['assignment_protocol'] == P.ASSIGNMENT
    bank = cfg['rubric']['cumulative_bank']
    assert bank['scoring_scope'] == 'all'
    assert bank.get('initial_snapshot_file') is None
    assert bank.get('initial_snapshot_sha256') is None
    identity = cumulative_run.config_identity(cfg, ROOT)
    assert identity['semantic_matcher'] == 'disabled'
    assert identity['matcher_model'] is None
    assert cfg['five_gpu']['checkpoint_evaluation_enabled'] is False
    assert cfg['five_gpu']['base_evaluation']['enabled'] is False
    assert cfg.get('evaluation') is None


def test_fresh_id_and_120_case_schedule_survive_first_five_stop_resume():
    cfg = experiment.load_config(P.CONFIG)
    old = experiment.load_config('configs/experiments/train120_v8_fresh_source_case_exact.json')
    assert cfg['experiment']['run_id'] == P.RUN_ID
    assert cfg['experiment']['run_id'] != old['experiment']['run_id']
    assert cfg['output_dir'] != old['output_dir']
    assert cfg['schedule']['cases'] == 120
    assert cfg['five_gpu']['batch_cases'] == 5
    assert experiment.fingerprint(cfg) != experiment.fingerprint(old)


def test_offline_preflight_selects_first_five_without_api_or_gpu():
    result = P.check(require_secret=False)
    assert result['status'] == 'offline_contract_pass'
    assert result['first_five_case_ids'] == [
        'medmcqa_d5efdbf8-93d6-4ce4-a89e-df816fb16944',
        'medmcqa_e7d47277-58bc-496e-b118-9c9e0f82004b',
        'medmcqa_4797bb62-d4d5-46e6-860c-aa3bd6da4edb',
        'medmcqa_dbdeda4f-ff5f-456c-86cf-980e7d41b8e0',
        'medmcqa_c2482fd9-9011-4b6b-b881-8217881c1873',
    ]
    assert '--max-batches 1' in result['pilot_launch']
    assert '--max-batches' not in result['continuation_launch']
    assert result['evaluation_executed'] is False


def test_secret_check_only_reports_presence_and_rejects_exposed_file(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    target = tmp_path / '.config/sherpa/openai.env'
    assert P.secret_available() is False
    target.parent.mkdir(parents=True)
    target.write_text('OPENAI_API_KEY=placeholder-for-test-only\n')
    target.chmod(0o644)
    assert P.secret_available() is False
    target.chmod(0o600)
    assert P.secret_available() is True
    result = P.check(require_secret=True)
    assert result['secret_available'] is True
    assert 'placeholder-for-test-only' not in repr(result)


def test_missing_direct_provider_is_rejected_offline(monkeypatch):
    original = experiment.load_config

    def changed(path):
        cfg = copy.deepcopy(original(path))
        cfg['judges']['api_provider'] = 'openrouter'
        return cfg

    monkeypatch.setattr(experiment, 'load_config', changed)
    with pytest.raises(RuntimeError, match='judge direct provider is missing'):
        P.check(require_secret=False)
