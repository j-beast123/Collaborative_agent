"""Normalize runtime-only settings for resume identity."""
import copy



def resume_config(cfg):
    result = copy.deepcopy(cfg)
    result.pop('checkpoint_upload', None)
    result.pop('shared_rollout', None)
    result.setdefault('five_gpu', {})['checkpoint_evaluation_enabled'] = False
    return result

