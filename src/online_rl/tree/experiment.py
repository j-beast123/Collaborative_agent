"""Experiment configs on the five-GPU batch protocol: base + override JSON, reward mixture, Rubric-Generator prompt mode,
run identity and resume fingerprint. A config without an "experiment" block is returned unchanged."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Optional
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.team import compose_reward

ROOT = Path(__file__).resolve().parents[3]
PROMPT_MODES = ('p0_fixed', 'pfinal_fixed', 'evolving')
MODE_TAGS = {'p0_fixed': 'p0fixed', 'pfinal_fixed': 'pfinal', 'evolving': 'evolving'}
OUTCOME_ONLY_TAG = 'outcomeonly'             # zero rubric weight: no rubric prompt in use
MODEL_TAGS = {'anthropic/claude-4.6-sonnet-20260217': 'sonnet46', 'anthropic/claude-4.5-haiku-20251001': 'haiku45',
              'gpt-6-luna': 'lunadoctor'}
EXPERIMENT_KEYS = ('family', 'reward_weight_rubric', 'reward_weight_outcome', 'prompt_mode', 'pfinal_artifact',
                   'feedback_model', 'seed', 'train_order_sha256', 'output_root', 'output_dir', 'run_label')
PFINAL_SCHEMA = 'sherpa_pfinal_v2'
PFINAL_KEYS = ('source_run_id', 'source_output_dir', 'source_checkpoint', 'source_batch', 'source_completed_cases',
               'prompt_sha256', 'prompt_text_path', 'parent_prompt_sha256', 'p0_sha256', 'adoption_history',
               'source_config_sha256', 'source_protocol_sha256', 'source_feedback_model', 'source_reward_weights',
               'source_train_sha256', 'source_train_order_sha256', 'source_theta0', 'source_theta0_policy_sha256',
               'source_prompt_mode', 'source_feedback_prompt_sha256', 'source_seed', 'source_run_config_sha256')
RUBRIC_ARTIFACTS = ('feedback', 'trajectory_inputs', 'trajectory_rubrics', 'source_path_scores', 'effectiveness',
                    'selection', 'response_scores', 'rubric_weights', 'rubric_bank')
NOT_COMPUTED, NA, SCORED = 'NOT_COMPUTED', 'NA', 'SCORED'
WEIGHT_TOLERANCE = 1e-12
REWARD_TOLERANCE = 1e-12
WEIGHT_STEPS = 10                     # run-id tags name weights in tenths
MAX_EXTENDS_DEPTH = 8


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_raw(path, depth: int = 0) -> dict:
    """Resolve the "extends" chain; paths are relative to the repository root."""
    if depth > MAX_EXTENDS_DEPTH: raise ValueError(f'extends chain deeper than {MAX_EXTENDS_DEPTH}: {path}')
    cfg = json.loads((ROOT / path).read_text())
    parent = cfg.pop('extends', None)
    return deep_merge(load_raw(parent, depth + 1), cfg) if parent else cfg


def load_config(path) -> dict:
    raw = load_raw(path)
    return resolve(raw) if 'experiment' in raw else raw


def _weight_tag(w: float) -> str:
    steps = round(w * WEIGHT_STEPS)
    if abs(w - steps / WEIGHT_STEPS) > WEIGHT_TOLERANCE: raise ValueError(f'reward weight {w} is not a multiple of 1/{WEIGHT_STEPS}')
    return str(steps // WEIGHT_STEPS) if steps in (0, WEIGHT_STEPS) else f'0{steps}'


def validate_weights(w_rubric, w_outcome) -> None:
    for w in (w_rubric, w_outcome):
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not 0.0 <= w <= 1.0:
            raise ValueError(f'reward weights must be numbers in [0, 1]; got {w!r}')
    if abs(w_rubric + w_outcome - 1.0) > WEIGHT_TOLERANCE: raise ValueError('reward weights must sum to 1')


def mixture_tag(w_rubric: float, w_outcome: float) -> str:
    return f'r{_weight_tag(w_rubric)}_o{_weight_tag(w_outcome)}'


def model_tag(model: str) -> str:
    if model not in MODEL_TAGS: raise ValueError(f'unregistered feedback model {model!r}; add it to MODEL_TAGS')
    return MODEL_TAGS[model]


def theta0_identity(cfg: dict) -> dict:
    pol = cfg['policy']
    return {'base_model': pol['base_model'], 'init_adapter': pol.get('init_adapter'), 'lora': pol['lora']}


def source_identity(cfg: dict) -> dict:
    """Condition a Pfinal source run must share with its PFINAL_FIXED target."""
    return {'source_feedback_model': cfg['feedback']['model'], 'source_feedback_prompt_sha256': cfg['feedback']['prompt_sha256'],
            'source_seed': cfg['schedule']['seed'],
            'source_reward_weights': {k: cfg['agent_reward'][k] for k in ('w_outcome', 'w_rubric')},
            'source_train_sha256': cfg['data']['train_sha256'], 'source_train_order_sha256': cfg['five_gpu']['train_order_sha256'],
            'source_theta0': theta0_identity(cfg)}


def load_pfinal(path) -> dict:
    """Pfinal must be the last prompt adopted by a completed evolving run."""
    from src.online_rl.tree.rubric_bank import MARKER
    meta = json.loads((ROOT / path).read_text())
    if meta.get('schema') != PFINAL_SCHEMA: raise ValueError(f'{path}: schema must be {PFINAL_SCHEMA}')
    missing = [k for k in PFINAL_KEYS if meta.get(k) in (None, '', [], {})]
    if missing: raise ValueError(f'{path}: missing Pfinal provenance {missing}')
    text = (ROOT / meta['prompt_text_path']).read_bytes()
    if hashlib.sha256(text).hexdigest() != meta['prompt_sha256']: raise ValueError(f'{path}: prompt text hash differs')
    if MARKER not in text.decode(): raise ValueError(f'{path}: Pfinal has no {MARKER} section')
    adopted = [h for h in meta['adoption_history'] if h['accepted']]
    if not adopted or adopted[-1]['candidate_sha256'] != meta['prompt_sha256'] or adopted[-1]['parent_sha256'] != meta['parent_prompt_sha256']:
        raise ValueError(f'{path}: Pfinal is not the last adopted prompt of its adoption history')
    if meta['adoption_history'][0]['parent_sha256'] != meta['p0_sha256']: raise ValueError(f'{path}: adoption history does not start at P0')
    if meta['source_prompt_mode'] != 'evolving': raise ValueError(f'{path}: Pfinal source must be an evolving run')
    verify_source_run(meta, path)
    return meta


def verify_source_run(meta: dict, path) -> None:
    """Cross-check Pfinal against its source run when that run is on this machine."""
    src = ROOT / meta['source_output_dir']
    if not src.exists(): return
    state = json.loads((src / 'run_state.json').read_text())
    theta0 = json.loads((src / 'checkpoints/checkpoint_000000/COMPLETE.json').read_text())['policy_sha256']
    observed = {'prompt_sha256': state['prompt_sha256'], 'source_completed_cases': state['case_cursor'],
                'source_protocol_sha256': state['protocol_sha256'], 'source_run_config_sha256': state['config_sha256'],
                'source_theta0_policy_sha256': theta0}
    differs = sorted(k for k, v in observed.items() if meta[k] != v)
    stored = src / 'prompts' / f"{meta['prompt_sha256']}.txt"
    if not stored.exists() or stored.read_bytes() != (ROOT / meta['prompt_text_path']).read_bytes(): differs.append('prompt_text')
    if differs: raise ValueError(f'{path}: Pfinal differs from its source run: {differs}')


def apply_prompt_mode(cfg: dict, mode: str, pfinal_artifact: Optional[str]) -> Optional[dict]:
    """The only place that maps a prompt mode to config; returns Pfinal provenance for pfinal_fixed."""
    if mode not in PROMPT_MODES: raise ValueError(f'prompt_mode must be one of {PROMPT_MODES}; got {mode!r}')
    if (mode == 'pfinal_fixed') != bool(pfinal_artifact):
        raise ValueError('pfinal_artifact is required for pfinal_fixed and forbidden otherwise')
    if mode != 'p0_fixed' and not cfg['rubric']['enabled']: raise ValueError(f'{mode} needs a rubric reward (reward_weight_rubric > 0)')
    cfg['prompt_update']['enabled'] = mode == 'evolving'
    if mode != 'pfinal_fixed': return None
    meta = load_pfinal(pfinal_artifact)
    expected = {**source_identity(cfg), 'p0_sha256': cfg['rubric']['prompt_sha256']}
    differs = sorted(k for k, v in expected.items() if meta[k] != v)
    if differs: raise ValueError(f'{pfinal_artifact}: Pfinal source condition differs from this run: {differs}')
    cfg['rubric'].update(generator_prompt_file=meta['prompt_text_path'], prompt_sha256=meta['prompt_sha256'],
                         prompt_id=f"PFINAL-{meta['prompt_sha256'][:12]}")
    return {k: meta[k] for k in ('source_run_id', 'source_checkpoint', 'source_batch', 'prompt_sha256', 'parent_prompt_sha256',
                                 'p0_sha256', 'source_config_sha256', 'source_protocol_sha256', 'source_theta0_policy_sha256',
                                 'source_prompt_mode')}


def run_id(exp: dict, prompt_sha256: str) -> str:
    label = exp.get('run_label')
    if label is not None and (not isinstance(label, str) or
                              re.fullmatch(r'[a-z0-9_]{1,48}', label) is None):
        raise ValueError('experiment.run_label must be 1..48 lowercase letters, digits or underscores')
    mode = (OUTCOME_ONLY_TAG if exp['reward_weight_rubric'] == 0 else
            MODE_TAGS[exp['prompt_mode']] + (f'_{prompt_sha256[:8]}' if exp['prompt_mode'] == 'pfinal_fixed' else ''))
    parts = (mixture_tag(exp['reward_weight_rubric'], exp['reward_weight_outcome']), mode,
             model_tag(exp['feedback_model']), f"seed{exp['seed']}")
    return '__'.join((*parts, label) if label else parts)


def resolve(raw: dict) -> dict:
    """Experiment block -> runnable five-GPU config. Family is manifest metadata, so one run may serve two families."""
    cfg = copy.deepcopy(raw); exp = cfg.pop('experiment')
    unknown = set(exp) - set(EXPERIMENT_KEYS)
    if unknown: raise ValueError(f'unknown experiment keys {sorted(unknown)}')
    w_r, w_o = exp['reward_weight_rubric'], exp['reward_weight_outcome']
    validate_weights(w_r, w_o)
    if exp['train_order_sha256'] != cfg['five_gpu']['train_order_sha256']: raise ValueError('experiment.train_order_sha256 differs from five_gpu')
    if isinstance(exp['seed'], bool) or not isinstance(exp['seed'], int): raise ValueError('experiment.seed must be an integer')
    cfg['agent_reward'] = {**cfg['agent_reward'], 'w_outcome': w_o, 'w_rubric': w_r}
    # Zero rubric weight = Outcome-only: no rubric pipeline.
    cfg['rubric']['enabled'] = w_r > 0.0
    cfg['feedback']['model'] = exp['feedback_model']
    cfg['schedule']['seed'] = exp['seed']
    pfinal = apply_prompt_mode(cfg, exp['prompt_mode'], exp.get('pfinal_artifact'))
    if pfinal and pfinal['source_run_id'] != run_id({**exp, 'prompt_mode': 'evolving'}, ''):
        raise ValueError(f"Pfinal source must be the matching evolving run, not {pfinal['source_run_id']}")
    rid = run_id(exp, cfg['rubric']['prompt_sha256'])
    out = f"{exp['output_root']}/{rid}"
    if exp.get('output_dir') not in (None, out): raise ValueError(f"experiment.output_dir must be {out}")
    cfg.update(name=rid, condition=rid, output_dir=out)
    cfg['experiment'] = {**{k: exp.get(k) for k in EXPERIMENT_KEYS if k not in ('family', 'output_dir')},
                         'run_id': rid, 'pfinal': pfinal, 'rubric_pipeline': cfg['rubric']['enabled'],
                         'feedback_prompt_sha256': cfg['feedback']['prompt_sha256'], 'rubric_prompt_sha256': cfg['rubric']['prompt_sha256']}
    return cfg


def fingerprint(cfg: dict) -> Optional[str]:
    """Research identity a checkpoint must share to be resumed; runtime devices/paths excluded. None without an experiment block."""
    if 'experiment' not in cfg: return None
    return sha256_json({'reward': {k: cfg['agent_reward'][k] for k in ('w_outcome', 'w_rubric')},
                        **({'run_label': cfg['experiment']['run_label']}
                           if cfg['experiment'].get('run_label') is not None else {}),
                        'rubric_pipeline': cfg['rubric']['enabled'], 'prompt_mode': cfg['experiment']['prompt_mode'],
                        'rubric_prompt_sha256': cfg['rubric']['prompt_sha256'], 'prompt_evolution': cfg['prompt_update']['enabled'],
                        'feedback': {k: cfg['feedback'][k] for k in ('model', 'prompt_sha256')},
                        'data': {'train_sha256': cfg['data']['train_sha256'], 'train_order_sha256': cfg['five_gpu']['train_order_sha256'],
                                 'cases': cfg['schedule']['cases']},
                        'seed': cfg['schedule']['seed'], 'theta0': theta0_identity(cfg),
                        **({'cumulative_bank': {k:v for k,v in cfg['rubric']['cumulative_bank'].items()
                                               if not k.endswith('_file')}}
                           if cfg['rubric'].get('cumulative_bank') is not None else {})})


def receipt_fields(cfg: dict) -> dict:
    fp = fingerprint(cfg)
    return {'experiment_fingerprint': fp} if fp else {}


def state_fields(cfg: dict, theta0_sha256: str) -> dict:
    return {**receipt_fields(cfg), 'theta0_policy_sha256': theta0_sha256} if fingerprint(cfg) else {}


def check_receipt(cfg: dict, receipt: dict, what: str) -> None:
    if receipt.get('experiment_fingerprint') != fingerprint(cfg):
        raise RuntimeError(f'{what} belongs to another experiment (fingerprint differs); use its own output directory')


def check_pfinal_source(cfg: dict, theta0_sha256: str, protocol_sha256: str) -> None:
    """Pfinal must come from a run with the same initial adapter and batch protocol."""
    pfinal = (cfg.get('experiment') or {}).get('pfinal')
    if not pfinal: return
    if pfinal['source_theta0_policy_sha256'] != theta0_sha256: raise RuntimeError('Pfinal source theta0 differs from this run')
    if pfinal['source_protocol_sha256'] != protocol_sha256: raise RuntimeError('Pfinal source protocol differs from this run')


def check_resume(cfg: dict, out: Path, state: dict, theta0_sha256: str) -> None:
    """Refuse a run state or checkpoint from another reward / prompt / feedback / data / theta0 condition."""
    check_receipt(cfg, state, 'Run state')
    if fingerprint(cfg) is None: return
    if state.get('theta0_policy_sha256') != theta0_sha256: raise RuntimeError('Initial adapter (theta0) differs from this run')
    ckpt = Path(state['checkpoint']).resolve()
    if ckpt.parent != (Path(out) / 'checkpoints').resolve(): raise RuntimeError('Checkpoint is outside this run directory')
    check_receipt(cfg, json.loads((ckpt / 'COMPLETE.json').read_text()), 'Checkpoint')


def _same(a: Optional[float], b: Optional[float]) -> bool:
    return a is b if a is None or b is None else math.isclose(a, b, rel_tol=0.0, abs_tol=REWARD_TOLERANCE)


def check_batch(cfg: dict, records: list[dict]) -> list[dict]:
    """Validate a batch before the update and return per-action reward rows; no-op without an experiment block."""
    if 'experiment' not in cfg: return []
    w = {k: cfg['agent_reward'][k] for k in ('w_outcome', 'w_rubric')}; rubric_on = cfg['rubric']['enabled']
    fixed = None if cfg['prompt_update']['enabled'] else cfg['rubric']['prompt_sha256']
    rows = []
    for r in records:
        if fixed and r['prompt_sha256'] != fixed: raise RuntimeError(f"Rubric prompt changed under a fixed prompt mode (case {r['case_id']})")
        if not rubric_on and any(r.get(k) for k in RUBRIC_ARTIFACTS): raise RuntimeError(f"Rubric pipeline ran with zero rubric weight (case {r['case_id']})")
        t = r['agent_targets']
        if t.get('reward_weights') != w: raise RuntimeError(f"Record reward weights {t.get('reward_weights')} differ from config {w}")
        for nid, x in sorted(t['nodes'].items()):
            if not rubric_on and x['R_A'] is not None: raise RuntimeError(f'R_rubric computed with zero rubric weight at {nid}')
            if not _same(x['reward'], compose_reward(x['outcome_component'], x['R_A'], **w)):
                raise RuntimeError(f'Reward of {nid} differs from the configured mixture')
            rows.append({'case_cursor': r['case_cursor'], 'case_id': r['case_id'], 'node_id': nid,
                         'R_outcome': x['outcome_component'], 'R_rubric': x['R_A'],
                         'R_rubric_status': NOT_COMPUTED if not rubric_on else NA if x['R_A'] is None else SCORED,
                         'final_reward': x['reward'], 'rubric_weight': w['w_rubric'], 'outcome_weight': w['w_outcome']})
    return rows
