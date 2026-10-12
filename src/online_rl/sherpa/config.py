"""Configuration contracts for the trajectory pipeline (checked only on launch)."""
from pathlib import Path

from src.online_rl.sherpa import PROTOCOL


def config_problems(cfg, root):
    problems = []
    if cfg.get('pipeline_protocol') != PROTOCOL:
        problems.append('Unknown trajectory pipeline protocol')
    ro = cfg.get('rollout', {})
    if (ro.get('branching'), ro.get('budget'), ro.get('temperature')) != (32, 'initial_fanout', 1.0):
        problems.append('Rollout requires 32 initial branches and temperature 1.0')
    if ro.get('doctor_model') != 'gpt-6-luna' or ro.get('doctor_backend', {}).get('engine') != 'openai_direct':
        problems.append('Doctor requires OpenAI direct gpt-6-luna')
    if cfg.get('policy', {}).get('base_model') != 'Qwen/Qwen3-8B':
        problems.append('Assistant requires Qwen/Qwen3-8B')
    spec = cfg.get('schedule', {})
    if (spec.get('cases'), spec.get('iterations'), spec.get('batch_cases'), spec.get('steps_per_iteration')) != (120, 3, 4, 30):
        problems.append('Schedule requires 120 cases, three iterations, four cases per step and 30 steps per iteration')
    if cfg.get('checkpoint_every_steps') != 10:
        problems.append('Learning-curve checkpoints must be saved every ten steps')
    if cfg.get('advantage_normalization', {}).get('mode') != 'group_zscore':
        problems.append('Trajectory groups require z-score advantages')
    if cfg.get('execution', {}).get('api_limits', {}).get('doctor') != 50:
        problems.append('Doctor API capacity must be 50')
    if cfg.get('method') == 'sherpa':
        search = cfg.get('prompt_optimization', {})
        if search.get('iterations') not in (2, 3) or search.get('candidates_per_iteration') != 4 or search.get('mixed_cases') != 24:
            problems.append('Prompt search requires 24 mixed cases, four candidates and two or three iterations')
        for section, fields in (('rubric_pipeline', ('reflection_prompt_file', 'judge_prompt_file')),
                                ('prompt_optimization', ('initial_prompt_file', 'proposal_prompt_file'))):
            for field in fields:
                path = cfg.get(section, {}).get(field)
                if not path or not (Path(root) / path).is_file():
                    problems.append(f'Missing {section}.{field}')
    return problems
