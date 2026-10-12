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
    if ro.get('doctor_model') not in ('gpt-6-luna', 'gpt-4o-mini') or ro.get('doctor_backend', {}).get('engine') != 'openai_direct':
        problems.append('Doctor requires OpenAI direct gpt-6-luna or gpt-4o-mini')
    if ro.get('doctor_model') != ro.get('doctor_backend', {}).get('model'):
        problems.append('doctor_model and doctor_backend.model must match')
    if ro.get('doctor_model') == 'gpt-4o-mini' and ro.get('doctor_backend', {}).get('reasoning_effort') not in (None, 'none'):
        problems.append('gpt-4o-mini reasoning_effort must be null')
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
        if search.get('inference_engine', 'transformers') not in ('transformers', 'vllm'):
            problems.append('Unknown prompt inference_engine')
        if search.get('inference_engine') == 'vllm':
            v = search.get('vllm', {})
            memory = v.get('gpu_memory_utilization')
            if type(memory) not in (int, float) or not 0 < memory < 1:
                problems.append('vllm.gpu_memory_utilization must be between zero and one')
            for field in ('max_model_len', 'max_num_seqs', 'startup_timeout_seconds'):
                if type(v.get(field)) is not int or v[field] < 1:
                    problems.append(f'vllm.{field} must be a positive integer')
            if not v.get('python'):
                problems.append('vllm.python is required')
        retries = search.get('rate_limit_retries', 0)
        if type(retries) is not int or retries < 0:
            problems.append('prompt_optimization.rate_limit_retries must be a nonnegative integer')
        repeats = search.get('confirmation_repeats', 0)
        if type(repeats) is not int or repeats < 0:
            problems.append('prompt_optimization.confirmation_repeats must be a nonnegative integer')
        for field in ('rollout_replicas', 'rollout_case_workers', 'candidate_workers', 'case_workers', 'assistant_max_inflight'):
            if type(search.get(field, 1)) is not int or search.get(field, 1) < 1:
                problems.append(f'prompt_optimization.{field} must be a positive integer')
        if search.get('rollout_batch_size', 4) not in (1, 2, 4, 8):
            problems.append('prompt_optimization.rollout_batch_size must be 1, 2, 4 or 8')
        if search.get('iterations') not in (2, 3) or search.get('candidates_per_iteration') != 4 or search.get('mixed_cases') != 24:
            problems.append('Prompt search requires 24 mixed cases, four candidates and two or three iterations')
        for section, fields in (('rubric_pipeline', ('reflection_prompt_file', 'judge_prompt_file')),
                                ('prompt_optimization', ('initial_prompt_file', 'proposal_prompt_file'))):
            for field in fields:
                path = cfg.get(section, {}).get(field)
                if not path or not (Path(root) / path).is_file():
                    problems.append(f'Missing {section}.{field}')
    return problems
