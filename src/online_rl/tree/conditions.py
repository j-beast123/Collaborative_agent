"""Experimental arms of the case tree: one research config, three overrides. Everything else (cases, order, seed, initial
adapter, Doctor protocol, decoding, G / T, optimizer, checkpoint and evaluation schedule) is shared."""
from __future__ import annotations
import copy

ARMS = ('outcome_only', 'rubric_only', 'fixed', 'evolving')
OVERRIDES = {'outcome_only': {'rubric': {'enabled': False}, 'prompt_update': {'enabled': False},     # reward = outcome only, no rubric work
                             'agent_reward': {'w_outcome': 1.0, 'w_rubric': 0.0}},
             'rubric_only': {'rubric': {'enabled': True}, 'prompt_update': {'enabled': False},      # reward = R_A only (outcome logged, weight 0)
                             'agent_reward': {'w_outcome': 0.0, 'w_rubric': 1.0}},
             'fixed': {'rubric': {'enabled': True}, 'prompt_update': {'enabled': False},            # outcome + R_A, P0 throughout
                       'agent_reward': {'w_outcome': 0.5, 'w_rubric': 0.5}},
             'evolving': {'rubric': {'enabled': True}, 'prompt_update': {'enabled': True},          # outcome + R_A, J / bootstrap prompt updates
                          'agent_reward': {'w_outcome': 0.5, 'w_rubric': 0.5}}}


def arm_config(base: dict, arm: str) -> dict:
    if arm not in OVERRIDES: raise ValueError(f'tree arms are {ARMS}; got {arm!r}')
    cfg = copy.deepcopy(base)
    for sec, vals in OVERRIDES[arm].items(): cfg[sec] = {**cfg.get(sec, {}), **vals}
    cfg['condition'] = arm; cfg['output_dir'] = f"{base['output_dir']}/{arm}"
    return cfg


def arm_of(cfg: dict) -> str:
    if cfg.get('condition') in ARMS: return cfg['condition']
    if not cfg['rubric'].get('enabled', True): return 'outcome_only'
    if cfg['prompt_update']['enabled']: return 'evolving'
    return 'rubric_only' if (cfg.get('agent_reward') or {}).get('w_outcome') == 0.0 else 'fixed'


# Settings an arm does not use (team decision 2026-10-07): an unresolved value there never blocks that arm, and every
# setting the arm does use still must be resolved. Rubric off -> no feedback / rubric / verification / scoring / prompt
# evolution; prompt update off -> no prompt events.
RUBRIC_ONLY_SETTINGS = ('intervention.', 'selection.', 'prompt_update.', 'acceptance.')
PROMPT_EVENT_SETTINGS = ('prompt_update.', 'acceptance.')


def active_unresolved(cfg: dict) -> list[str]:
    """unresolved(cfg) restricted to the settings the configured condition actually uses."""
    from src.online_rl.integrated.conditions import unresolved          # TBD, CALIBRATION_REQUIRED and OPEN_DECISION
    inactive = (RUBRIC_ONLY_SETTINGS if not cfg['rubric'].get('enabled', True) else ()) + \
               (PROMPT_EVENT_SETTINGS if not cfg['prompt_update'].get('enabled', False) else ())
    return [u for u in unresolved(cfg) if not u.startswith(inactive)] if inactive else unresolved(cfg)

