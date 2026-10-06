"""Integrated experiment conditions. They share one base config and differ only in the rubric/prompt-evolution switches."""
from __future__ import annotations
import copy

MAIN = ('base', 'outcome_only', 'integrated_fixed', 'integrated_evolving')
OVERRIDES = {
    'integrated_evolving': {'prompt_update': {'enabled': True}},
    'integrated_fixed': {'prompt_update': {'enabled': False}},
    'outcome_only': {'rubric': {'enabled': False}, 'reward': {'w_A': 0.0}, 'prompt_update': {'enabled': False}},
}
MATCHED_EXCEPT = {'condition', 'output_dir', 'prompt_update.enabled', 'rubric.enabled', 'reward.w_A'}


def unresolved(cfg: dict) -> list[str]:
    """Dotted paths whose value is still 'TBD', 'CALIBRATION_REQUIRED' or 'OPEN_DECISION'."""
    out = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items(): walk(v, f'{path}.{k}' if path else k)
        elif x in ('TBD', 'CALIBRATION_REQUIRED', 'OPEN_DECISION'): out.append(f'{path} ({x})')
    walk(cfg, ''); return out


def condition_config(base: dict, name: str) -> dict:
    if name not in OVERRIDES: raise ValueError(f'trainable integrated conditions are {sorted(OVERRIDES)}; got {name!r}')
    cfg = copy.deepcopy(base)
    for sec, vals in OVERRIDES[name].items(): cfg[sec] = {**cfg[sec], **vals}
    cfg['condition'] = name; cfg['output_dir'] = f"{base['output_dir']}/{name}"
    return cfg


def prompt_mode(cfg: dict) -> str:
    if not cfg['rubric']['enabled']: return 'none'
    return 'evolving' if cfg['prompt_update']['enabled'] else 'fixed'


def unmatched_keys(a: dict, b: dict, prefix: str = '') -> set[str]:
    """Dotted keys whose values differ outside MATCHED_EXCEPT."""
    out = set()
    for k in set(a) | set(b):
        key = f'{prefix}{k}'
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict): out |= unmatched_keys(a[k], b[k], key + '.')
        elif a.get(k) != b.get(k) and key not in MATCHED_EXCEPT: out.add(key)
    return out
