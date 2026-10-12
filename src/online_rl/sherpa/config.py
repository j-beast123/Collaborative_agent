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
    local_doctor = (ro.get('doctor_model') == 'Qwen/Qwen3-8B' and
                    ro.get('doctor_backend', {}).get('engine') == 'vllm')
    if not local_doctor and (ro.get('doctor_model') not in ('gpt-6-luna', 'gpt-4o-mini') or ro.get('doctor_backend', {}).get('engine') != 'openai_direct'):
        problems.append('Unsupported Doctor engine/model pair')
    if local_doctor:
        execution = cfg.get('doctor_execution', {})
        if execution.get('visible_gpu_index') != 1:
            problems.append('Local Doctor requires the second visible GPU (index 1)')
        if cfg.get('training_execution', {}).get('rollout_replicas', 1) != 1 or cfg.get('prompt_optimization', {}).get('rollout_replicas', 1) != 1:
            problems.append('Reserve GPU 0 for one assistant replica and GPU 1 for the frozen Doctor')
        if not (Path(root) / ro['doctor_backend'].get('local_model_path', '') / 'config.json').is_file():
            problems.append('Local Doctor model files are missing')
        if not execution.get('vllm', {}).get('python'):
            problems.append('Doctor vLLM Python environment is required')
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
    training = cfg.get('training_execution')
    if training:
        if training.get('inference_engine') != 'transformers':
            problems.append('Training currently requires Transformers with synchronized trainable adapters')
        for field in ('rollout_replicas', 'assistant_max_inflight', 'case_workers'):
            if type(training.get(field)) is not int or training[field] < 1:
                problems.append(f'training_execution.{field} must be a positive integer')
        if training.get('case_workers') != 4:
            problems.append('ours requires four concurrent case workers')
        if training.get('rollout_batch_size') not in (1, 2, 4, 8, 16, 32, 64):
            problems.append('Training rollout_batch_size must be a power of two from 1 to 64')
        retries = training.get('rate_limit_retries', 0)
        if type(retries) is not int or retries < 0:
            problems.append('Training rate_limit_retries must be a nonnegative integer')
    if cfg.get('advantage_normalization', {}).get('mode') != 'group_zscore':
        problems.append('Trajectory groups require z-score advantages')
    doctor_capacity = 128 if local_doctor else 50
    if cfg.get('execution', {}).get('api_limits', {}).get('doctor') != doctor_capacity:
        problems.append(f'Doctor API capacity must be {doctor_capacity}')
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
        if type(search.get('selection_repeats', 0)) is not int or search.get('selection_repeats', 0) < 0:
            problems.append('prompt_optimization.selection_repeats must be a nonnegative integer')
        for field in ('rollout_replicas', 'rollout_case_workers', 'candidate_workers', 'case_workers', 'assistant_max_inflight'):
            if type(search.get(field, 1)) is not int or search.get(field, 1) < 1:
                problems.append(f'prompt_optimization.{field} must be a positive integer')
        if search.get('rollout_batch_size', 4) not in (1, 2, 4, 8):
            problems.append('prompt_optimization.rollout_batch_size must be 1, 2, 4 or 8')
        if search.get('iterations') not in (2, 3) or search.get('candidates_per_iteration') != 4:
            problems.append('Prompt search requires four candidates and two or three iterations')
        if type(search.get('mixed_cases')) is not int or not 1 <= search['mixed_cases'] <= 120:
            problems.append('Prompt search mixed_cases must be an integer from 1 to 120')
        for section, fields in (('rubric_pipeline', ('reflection_prompt_file', 'judge_prompt_file')),
                                ('prompt_optimization', ('initial_prompt_file', 'proposal_prompt_file'))):
            for field in fields:
                path = cfg.get(section, {}).get(field)
                if not path or not (Path(root) / path).is_file():
                    problems.append(f'Missing {section}.{field}')
    return problems
