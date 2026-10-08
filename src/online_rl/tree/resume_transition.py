"""Narrow, audited resume compatibility checks for safe boundary transitions."""
from __future__ import annotations
import copy


def normalized_resume_config(cfg: dict) -> dict:
    """Normalize only already-supported evaluation and matcher runtime settings."""
    value = copy.deepcopy(cfg)
    value.setdefault('five_gpu', {})['checkpoint_evaluation_enabled'] = False
    value.setdefault('rubric', {})['match_batch_size'] = 1
    return value


def normalized_candidate_seed_config(cfg: dict) -> dict:
    """Normalize only the P1 evidence-scope setting for an audited transition."""
    value = normalized_resume_config(cfg)
    value.setdefault('prompt_update', {}).pop('candidate_seed', None)
    return value


def is_matcher_1_to_8_transition(old_cfg: dict, new_cfg: dict, *, case_cursor: int,
                                 batch_cursor: int, batch_cases: int) -> bool:
    """Allow only matcher 1->8 at a fully committed batch boundary."""
    old_size = (old_cfg.get('rubric') or {}).get('match_batch_size', 1)
    new_size = (new_cfg.get('rubric') or {}).get('match_batch_size', 1)
    return (old_size == 1 and new_size == 8 and batch_cases > 0 and case_cursor > 0
            and case_cursor % batch_cases == 0 and batch_cursor == case_cursor // batch_cases
            and normalized_resume_config(old_cfg) == normalized_resume_config(new_cfg))


def is_candidate_seed_transition(old_cfg: dict, new_cfg: dict, *, case_cursor: int,
                                 batch_cursor: int, batch_cases: int) -> bool:
    """Allow first-completed-P0 -> all-P0 only at a committed batch barrier."""
    old_mode = (old_cfg.get('prompt_update') or {}).get('candidate_seed')
    new_mode = (new_cfg.get('prompt_update') or {}).get('candidate_seed')
    return (old_mode == 'first_completed_P0_v1' and new_mode == 'all_p0_before_p1_v1'
            and batch_cases > 0 and case_cursor > 0 and case_cursor % batch_cases == 0
            and batch_cursor == case_cursor // batch_cases
            and normalized_candidate_seed_config(old_cfg) == normalized_candidate_seed_config(new_cfg))
