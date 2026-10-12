"""Compatibility metadata for the trajectory-level SHERPA pipeline."""
from src.online_rl.sherpa import PROTOCOL
from src.online_rl.sherpa.config import config_problems


def enabled(cfg):
    return cfg.get('pipeline_protocol') == PROTOCOL


def identity(cfg):
    return {'protocol': PROTOCOL, 'schedule': cfg['schedule'],
            'rubric_pipeline': cfg['rubric_pipeline'], 'prompt_optimization': cfg['prompt_optimization'],
            'reward': 'binary_checklist_fraction', 'credit': 'all_assistant_tokens_per_trajectory'}
