"""Offline contract check for the fresh, direct-Luna source-case experiment.

This reads local configuration and a secret file's metadata only. It does not
load a model, print a credential, contact an API, or start a Doctor server.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.online_rl.cases import file_sha256, load_cases
from src.online_rl.tree import experiment
from src.online_rl.tree.five_gpu import training_order


CONFIG = 'configs/experiments/train120_v9_fresh_source_case_direct_luna.json'
MODEL = 'gpt-6-luna'
PROVIDER = 'openai_direct'
ASSIGNMENT = 'exact_generated_source_case_v1'
COMPACT = 'v7_5_v7_6'
RUN_ID = 'r05_o05__p0fixed__gemma3doctor__seed0__sourcecase_exact_directluna_v9'
OUTPUT_ROOT = 'outputs/train120_v9_fresh_source_case_direct_luna_20261011'


def secret_available() -> bool:
    """Report only availability, never the credential or its contents."""
    path = Path.home() / '.config/sherpa/openai.env'
    try:
        info = path.stat()
    except OSError:
        return False
    return (stat.S_ISREG(info.st_mode) and info.st_size > 0 and
            stat.S_IMODE(info.st_mode) & 0o077 == 0 and os.access(path, os.R_OK))


def check(*, require_secret: bool = True) -> dict:
    from scripts.online_rl.run_tree import PROMPT_HASHES, blockers, judge_client, load_key, make_scorer
    from src.online_rl.tree import cumulative_bank as CB, cumulative_run, rubric_bank as RB

    cfg = experiment.load_config(CONFIG)
    problems: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    require(cfg.get('api_provider') == PROVIDER, 'top-level direct provider is missing')
    require(cfg['rubric']['generation'].get('api_provider') == PROVIDER,
            'rubric generator direct provider is missing')
    require(cfg['judges'].get('api_provider') == PROVIDER, 'judge direct provider is missing')
    require(cfg['rubric']['generator_model'] == MODEL, 'rubric generator model differs')
    for key in ('gate_model', 'scorer_model', 'intervention_audit_model', 'luna_canonical_slug'):
        require(cfg['judges'].get(key) == MODEL, f'judges.{key} must be direct {MODEL}')
    for key in ('reflection_model', 'refiner_model'):
        require(cfg['prompt_update'].get(key) == MODEL, f'configured prompt_update.{key} differs')
    require(cfg['prompt_update']['enabled'] is False,
            'prompt evolution and sol_backend calls must be disabled')
    require(cfg['feedback']['model'].startswith('ollama/'), 'Doctor reflection must remain local')
    require(cfg['rollout']['doctor_model'] == cfg['feedback']['model'],
            'Doctor and post-FINAL feedback must use the same local model')
    require(cfg['policy']['base_model'] == 'Qwen/Qwen3-8B' and cfg['policy'].get('init_adapter') is None,
            'run must begin with the Qwen3-8B base and no adapter')
    require(cfg.get('supervision_handoff') is None, 'old checkpoint handoff must be absent')
    require(cfg['schedule']['cases'] == 120 and cfg['five_gpu']['batch_cases'] == 5,
            'keep the 120-case schedule and five-case transactions for stop/resume')
    require(cfg['experiment']['run_id'] == RUN_ID and cfg['experiment']['output_root'] == OUTPUT_ROOT,
            'new run identity or output root differs')

    require(cfg['rubric'].get('assignment_protocol') == ASSIGNMENT and
            getattr(RB, 'EXACT_SOURCE_CASE_PROTOCOL', None) == ASSIGNMENT,
            'exact source-case generation/dedup is not selected')
    bank = cfg['rubric'].get('cumulative_bank') or {}
    require(bank.get('protocol') == 'cumulative_keep_bank_v1' and bank.get('scoring_scope') == 'all',
            'all-bank cumulative scoring must be enabled')
    require(bank.get('initial_snapshot_file') is None and bank.get('initial_snapshot_sha256') is None,
            'run must start with an empty bank, not an imported snapshot')
    require(CB.empty_snapshot()['entries'] == [], 'initial bank is not empty')
    try:
        bank_identity = cumulative_run.config_identity(cfg, ROOT)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        problems.append(f'cumulative bank configuration invalid: {type(exc).__name__}: {exc}')
    else:
        require(bank_identity.get('semantic_matcher') == 'disabled' and
                bank_identity.get('matcher_model') is None,
                'semantic reuse/consolidation must be disabled for exact-only generation')
    require(cfg['judges'].get('compact_runtime') == COMPACT,
            'compact v7.5/v7.6 applicability and behavior scoring is not selected')
    require(cfg['five_gpu'].get('checkpoint_evaluation_enabled') is False and
            cfg['five_gpu'].get('base_evaluation', {}).get('enabled') is False and
            cfg.get('evaluation') is None, '622-case evaluation must not run')

    # Validate every active prompt file and the supplementary compact semantics.
    for section, file_key, digest_key in PROMPT_HASHES:
        values = cfg.get(section, {})
        if file_key in values and digest_key in values:
            path = ROOT / values[file_key]
            require(path.is_file(), f'missing prompt {section}.{file_key}')
            if path.is_file():
                require(hashlib.sha256(path.read_bytes()).hexdigest() == values[digest_key],
                        f'prompt hash changed: {section}.{file_key}')
    supplements = cfg['judges'].get('compact_prompt_sha256') or {}
    require(len(supplements) == 5, 'five compact prompt supplements must be pinned')
    for rel, digest in supplements.items():
        path = ROOT / rel
        require(path.is_file(), f'missing compact prompt: {rel}')
        if path.is_file():
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                    f'compact prompt hash changed: {rel}')

    train_path = ROOT / cfg['data']['train']
    order_path = ROOT / cfg['five_gpu']['train_order_file']
    require(file_sha256(train_path) == cfg['data']['train_sha256'], 'training data changed')
    require(file_sha256(order_path) == cfg['five_gpu']['train_order_sha256'], 'training order changed')
    first_five = training_order(cfg, load_cases(train_path), ROOT)[:5]
    require(len(first_five) == len(set(first_five)) == 5, 'first five case IDs are not unique')

    try:
        client = judge_client(MODEL, '<offline-only>',
            (ROOT / cfg['judges']['scorer_prompt_file']).read_text(),
            max_tokens=cfg['judges']['scorer_max_tokens'], provider=PROVIDER)
        require(client.provider == PROVIDER, 'judge factory did not select direct OpenAI')
        scorer = make_scorer(cfg, client)
    except Exception as exc:
        problems.append(f'offline scorer construction failed: {type(exc).__name__}: {exc}')
    else:
        require(getattr(scorer, 'compact_revision', None) == COMPACT,
                'v7.5/v7.6 scorer selector fell back to a different scorer')
        require(getattr(getattr(scorer, 'review_state', None), 'applicability_contract', None) ==
                'compact_condition_v7_5', 'v7.5 applicability contract is not active')

    problems.extend(blockers(cfg))
    output = ROOT / cfg['output_dir']
    require(not (output / 'run_state.json').exists() and not (output / 'checkpoints').exists(),
            'output directory already contains a run or checkpoint')
    if require_secret:
        require(secret_available(), 'direct OpenAI credential is unavailable in the local secret setting')
        # Exercise the same parser as the actual runner. Never print or retain
        # the returned secret in the report, even if parsing fails.
        try:
            credential_loaded = bool(load_key(cfg).strip())
        except Exception:
            credential_loaded = False
        require(credential_loaded, 'direct OpenAI credential loader rejected the local secret setting')
    if problems:
        raise RuntimeError('\n'.join(dict.fromkeys(problems)))
    return {
        'status': 'offline_contract_pass',
        'config': CONFIG,
        'run_id': RUN_ID,
        'output_dir': cfg['output_dir'],
        'first_five_case_ids': first_five,
        'provider': PROVIDER,
        'remote_model': MODEL,
        'local_agent': cfg['policy']['base_model'],
        'local_doctor': cfg['rollout']['doctor_model'],
        'secret_available': secret_available(),
        'evaluation_executed': False,
        'pilot_launch': f'python scripts/online_rl/run_five_gpu.py --config {CONFIG} --launch --max-batches 1',
        'continuation_launch': f'python scripts/online_rl/run_five_gpu.py --config {CONFIG} --launch',
        'remaining_checks': 'GPU/Doctor identity, direct API transport and one-batch save/reload',
    }


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
