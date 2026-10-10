"""Offline contract check for fresh source-case generation and cumulative scoring.

This program does not load a model, contact an API, or start a Doctor server.
It deliberately does not replace the runner's hardware/service preflight.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.online_rl.cases import file_sha256, load_cases
from src.online_rl.tree import experiment
from src.online_rl.tree.five_gpu import training_order


CONFIG = 'configs/experiments/train120_v8_fresh_source_case_exact.json'
ASSIGNMENT = 'exact_generated_source_case_v1'
COMPACT = 'v7_5_v7_6'


def check(config_path: str = CONFIG) -> dict:
    from scripts.online_rl.run_tree import PROMPT_HASHES, blockers, make_scorer
    from src.online_rl.tree import rubric_bank as RB
    from src.online_rl.tree import cumulative_bank as CB, cumulative_run

    if config_path != CONFIG:
        raise ValueError('This preflight is pinned to the v8 fresh-source-case config')
    cfg = experiment.load_config(config_path)
    failures = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(cfg['schedule']['cases'] == 120, 'schedule.cases must stay 120 for a five-case stop/resume')
    require(cfg['five_gpu']['batch_cases'] == 5, 'batch size must be five')
    require(cfg['policy']['base_model'] == 'Qwen/Qwen3-8B', 'base model changed')
    require(cfg['policy'].get('init_adapter') is None, 'initial adapter must be absent')
    require(cfg.get('supervision_handoff') is None, 'checkpoint handoff must be absent')
    require(cfg['rubric'].get('assignment_protocol') == ASSIGNMENT, 'exact-only assignment protocol missing')
    require(getattr(RB, 'EXACT_SOURCE_CASE_PROTOCOL', None) == ASSIGNMENT,
            'exact-only assignment code is not installed')
    cumulative = cfg['rubric'].get('cumulative_bank') or {}
    require(cumulative.get('protocol') == 'cumulative_keep_bank_v1',
            'cumulative KEEP bank protocol must be enabled')
    require(cumulative.get('scoring_scope') == 'all',
            'every case must score the full frozen cumulative bank')
    require(cumulative.get('initial_snapshot_file') is None and
            cumulative.get('initial_snapshot_sha256') is None,
            'fresh run must start with an empty cumulative bank, not an imported snapshot')
    require(CB.empty_snapshot()['entries'] == [], 'initial cumulative bank is not empty')
    try:
        cumulative_run.config_identity(cfg, ROOT)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        failures.append(f'cumulative bank config is invalid: {exc}')
    require(cfg['rubric'].get('reuse') == 'assign_or_create_v1',
            'random-5 witness requires its established assignment interface')
    require(cfg['verification']['mode'] == 'random5_witness_v1', 'witness mode changed')
    require(cfg['verification']['max_valid_supports'] == 5, 'witness support count changed')
    require(cfg['prompt_update']['enabled'] is False, 'prompt evolution must be disabled')
    require(cfg['judges'].get('compact_runtime') == COMPACT, 'v7.5/v7.6 compact selector missing')
    require(cfg['five_gpu'].get('checkpoint_evaluation_enabled') is False,
            'checkpoint evaluation must be disabled')
    require(cfg['five_gpu'].get('base_evaluation', {}).get('enabled') is False,
            'base evaluation must be disabled')
    require(cfg.get('evaluation') is None, 'synchronous evaluation must be absent')
    require(cfg['experiment']['run_id'] == 'r05_o05__p0fixed__gemma3doctor__seed0__sourcecase_exact_v8',
            'automatic experiment run ID changed')
    require(cfg['experiment']['output_root'] == 'outputs/train120_v8_fresh_source_case_exact_20261011',
            'output root changed')

    # The runner's normal prompt hashes stay pinned. The compact supplements
    # are included explicitly because they are incorporated at request time.
    for section, file_key, digest_key in PROMPT_HASHES:
        values = cfg.get(section, {})
        if file_key in values and digest_key in values:
            path = ROOT / values[file_key]
            require(path.is_file(), f'missing prompt: {path}')
            if path.is_file():
                require(hashlib.sha256(path.read_bytes()).hexdigest() == values[digest_key],
                        f'prompt hash differs: {section}.{file_key}')
    expected_supplements = {
        f'prompts/sherpa/{name}.txt' for name in (
            'compact_applicability_semantics_v3', 'compact_behavior_semantics_v3',
            'compact_applicability_relation_v7_4', 'compact_applicability_referent_v7_5',
            'compact_behavior_referent_v7_6')
    }
    pinned = cfg['judges'].get('compact_prompt_sha256') or {}
    require(set(pinned) == expected_supplements, 'compact supplement hash list is incomplete')
    for rel, digest in pinned.items():
        path = ROOT / rel
        require(path.is_file(), f'missing compact supplement: {path}')
        if path.is_file():
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                    f'compact supplement hash differs: {rel}')

    train_path = ROOT / cfg['data']['train']
    order_path = ROOT / cfg['five_gpu']['train_order_file']
    require(file_sha256(train_path) == cfg['data']['train_sha256'], 'train data hash changed')
    require(file_sha256(order_path) == cfg['five_gpu']['train_order_sha256'], 'train order hash changed')
    order = training_order(cfg, load_cases(train_path), ROOT)
    first_five = order[:5]
    require(len(first_five) == len(set(first_five)) == 5, 'first five case IDs are not unique')

    # Construct the scorer without sending a request; this catches a selector
    # silently falling back to the old scorer when the new revision is unknown.
    class OfflineClient:
        model = cfg['judges']['scorer_model']
        key = '<offline-preflight-only>'
        system = (ROOT / cfg['judges']['scorer_prompt_file']).read_text()
        max_tokens = cfg['judges']['scorer_max_tokens']
        timeout_s = 180
        post = None
        sleep = None

    try:
        scorer = make_scorer(cfg, OfflineClient())
    except Exception as exc:
        failures.append(f'offline scorer construction failed: {type(exc).__name__}: {exc}')
    else:
        require(getattr(scorer, 'compact_revision', None) == COMPACT,
                'runtime did not actually select the v7.5/v7.6 compact scorer')
        require(getattr(getattr(scorer, 'review_state', None), 'applicability_contract', None) ==
                'compact_condition_v7_5', 'applicability contract is not v7.5')

    # No GPU/service test here. Any other structural config blockers still fail.
    failures.extend(blockers(cfg))
    output = ROOT / cfg['output_dir']
    require(not (output / 'run_state.json').exists(),
            'output root has an existing run; use a new root for a fresh base/empty bank')
    require(not (output / 'checkpoints').exists(),
            'output root has checkpoints; this would not be a fresh run')
    if failures:
        raise RuntimeError('\n'.join(dict.fromkeys(failures)))
    return {
        'status': 'offline_contract_pass',
        'config': config_path,
        'output_dir': cfg['output_dir'],
        'first_five_case_ids': first_five,
        'initial_adapter': None,
        'cumulative_bank': 'empty_then_full_batch_bank',
        'assignment_protocol': ASSIGNMENT,
        'compact_runtime': COMPACT,
        'evaluation_executed': False,
        'pilot_launch': f'python scripts/online_rl/run_five_gpu.py --config {config_path} --launch --max-batches 1',
        'continuation_launch': f'python scripts/online_rl/run_five_gpu.py --config {config_path} --launch',
        'remaining_checks': 'runner GPU, Doctor server identity, API credentials and one-batch save/reload',
    }


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
