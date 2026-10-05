"""Load integrated runs for RQ1-RQ3: verified provenance, prompt lineage, on-policy rubrics and an evaluation reply function.
Provenance comes from run_manifest.json + run_state.json (never from directory names) and is checked against the files."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
from src.online_rl.integrated.conditions import unmatched_keys
from src.online_rl.integrated.records import dir_sha256, read_json, sha256_json, sha256_text
from src.online_rl.integrated.rubric import STATE_SCHEMA_SHA256


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir); m = read_json(run_dir / 'run_manifest.json'); s = read_json(run_dir / 'run_state.json'); cfg = m['config']
    problems = []
    if sha256_json(cfg) != s['config_sha256'] or m['config_sha256'] != s['config_sha256']: problems.append('config hash mismatch')
    if dir_sha256(Path(s['checkpoint'])) != s['checkpoint_sha256']: problems.append('checkpoint files changed')
    if sha256_text((run_dir / 'prompts' / f"{s['prompt_sha256']}.txt").read_text()) != s['prompt_sha256']: problems.append('prompt text mismatch')
    if m['state_schema']['sha256'] != STATE_SCHEMA_SHA256: problems.append('state schema differs from the current code')
    if problems: raise RuntimeError(f'{run_dir}: ' + '; '.join(problems))
    return {'run_dir': str(run_dir), 'condition': m['condition'], 'prompt_mode': m['prompt_mode'], 'base_model': cfg['policy']['base_model'],
            'checkpoint': s['checkpoint'], 'checkpoint_sha256': s['checkpoint_sha256'], 'adapter_sha256': s['adapter_sha256'],
            'step': s['step'], 'optimizer_steps': s['optimizer_steps'], 'finished': s['finished'], 'seed': s['run_seed'],
            'initial_prompt_sha256': m['initial_prompt']['sha256'], 'prompt_id': s['prompt_id'], 'prompt_sha256': s['prompt_sha256'],
            'accepted_prompts': s['history'], 'state_schema_version': m['state_schema']['version'], 'state_schema_sha256': m['state_schema']['sha256'],
            'reward': {'w_O': cfg['reward']['w_O'], 'w_A': cfg['reward']['w_A']}, 'advantage_mode': m['advantage_mode'],
            'config_sha256': s['config_sha256'], 'config': cfg}


def base_provenance(cfg: dict) -> dict:
    """The untrained Base condition: no adapter, no prompt."""
    return {'condition': 'base', 'prompt_mode': 'none', 'base_model': cfg['policy']['base_model'], 'adapter_sha256': None, 'checkpoint': None, 'step': 0}


def public(prov: dict) -> dict:
    return {k: v for k, v in prov.items() if k != 'config'}


def prompt_lineage(run_dir: Path) -> list[dict]:
    """One row per prompt version of every committed event (immutable lineage)."""
    run_dir = Path(run_dir); cursor = read_json(run_dir / 'run_state.json')['lineage_cursor']; rows = []
    for e in range(1, cursor + 1):
        ev = read_json(run_dir / 'prompt_events' / f'event_{e:04d}.json')
        for v in ev['versions']:
            rows.append({'event': e, 'step': ev['step'], 'parent_prompt_id': v['parent_prompt_id'], 'prompt_id': v['prompt_id'], 'sha256': v['sha256'],
                         'depth': v['depth'], 'rank': v['rank'], 'status': v['status'], 'accepted': v['status'] == 'accepted',
                         'mean_paired_Q_gain': (v['acceptance'] or {}).get('mean_paired_Q_gain'), 'rejection_reason': v['rejection_reason']})
    return rows


def onpolicy_rubrics(run_dir: Path) -> list[dict]:
    """Committed training rubrics (for theme analysis only; never an evaluation criterion)."""
    run_dir = Path(run_dir); cursor = read_json(run_dir / 'run_state.json')['evidence_cursor']; out = []
    for i in range(cursor):
        e = read_json(run_dir / 'evidence' / f'step_{i:06d}.json')
        if e['rubric']['valid']: out.append({'step': i, 'prompt_id': e['prompt_id'], 'case_id': e['case_id'], 'turn': e['turn'], 'rubric': e['rubric']['rubric']})
    return out


def rq3_pair_problems(fixed: dict, evolving: dict) -> list[str]:
    """Fixed and evolving must share P0, state schema and every setting except prompt evolution."""
    out = []
    if (fixed['prompt_mode'], evolving['prompt_mode']) != ('fixed', 'evolving'): out.append('arms are not fixed/evolving')
    for k in ('initial_prompt_sha256', 'state_schema_sha256', 'base_model', 'seed', 'advantage_mode', 'reward'):
        if fixed[k] != evolving[k]: out.append(f'{k} differs')
    diff = unmatched_keys(fixed['config'], evolving['config'])
    if diff: out.append(f'config differs at {sorted(diff)}')
    return out


def policy_reply_fn(policy, seed_fn: Callable) -> Callable:
    """Evaluation reply with the plain Agent prompt (no rubric, no structured state): (case, prefix, turn) -> text."""
    return lambda case, prefix, turn: policy.natural(case, prefix, turn, seed_fn(case['case_id'], json.dumps([d['text'] for d in prefix])))
