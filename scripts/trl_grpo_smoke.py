"""Minimal distributed, full-parameter TRL GRPO plumbing smoke test."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

REWARD_LOG: list[dict[str, Any]] = []

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/home/kai/models/Qwen3-8B')
    p.add_argument('--train-file', default='data/interim/medmcqa/medicine_sample_500_label_corrected.jsonl')
    p.add_argument('--output-dir', default='outputs/trl_grpo_smoke')
    p.add_argument('--max-cases', type=int, default=6)
    p.add_argument('--max-steps', type=int, default=1)
    p.add_argument('--max-completion-length', type=int, default=32)
    p.add_argument('--num-generations', type=int, default=2)
    p.add_argument('--save-final', action='store_true')
    return p.parse_args()

def versions():
    import accelerate, torch, transformers, trl
    return {'trl': trl.__version__, 'transformers': transformers.__version__, 'torch': torch.__version__, 'accelerate': accelerate.__version__}

def policy_prompt(case):
    options = '\n'.join(f'{k}. {v}' for k, v in case['options'].items())
    return [{'role': 'user', 'content': f'Answer the following clinical multiple-choice question concisely.\n\nQuestion:\n{case["question"]}\n\nAnswer choices:\n{options}'}]

def load_rows(path, limit):
    rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(rows) < limit:
        raise ValueError(f'Need {limit} cases, found {len(rows)}')
    return [{'prompt': policy_prompt(row)} for row in rows[:limit]]

def completion_text(completion: Any) -> str:
    if isinstance(completion, str): return completion
    if isinstance(completion, list):
        return ' '.join(str(x.get('content', '')) if isinstance(x, dict) else str(x) for x in completion)
    if isinstance(completion, dict): return str(completion.get('content', completion))
    return str(completion)

def synthetic_reward(completions, **kwargs):
    """Deterministic non-constant smoke reward with within-group variance."""
    rewards = []
    for completion in completions:
        text = completion_text(completion).strip()
        interaction = min(len(text) / 160.0, 1.0)
        digest = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        outcome = (digest % 1000) / 999.0
        final = 0.6 * interaction + 0.4 * outcome
        REWARD_LOG.append({'completion': text, 'interaction_reward': interaction, 'outcome_reward': outcome, 'final_reward': final})
        rewards.append(final)
    return rewards

def distributed_info():
    import torch
    return {'rank': int(os.environ.get('RANK', '0')), 'local_rank': int(os.environ.get('LOCAL_RANK', '0')), 'world_size': int(os.environ.get('WORLD_SIZE', '1')), 'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count()}

def memory_snapshot():
    import torch
    info = distributed_info()
    return {**info, 'allocated_bytes': int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0, 'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}

def parameter_snapshot(model):
    import torch
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError('No trainable parameters found; this smoke test requires full-parameter training')
    name, parameter = trainable[0]
    return {'name': name, 'trainable_parameter_count': sum(p.numel() for _, p in trainable), 'total_parameter_count': sum(p.numel() for p in model.parameters()), 'first_parameter_norm': float(torch.linalg.vector_norm(parameter.detach().float()).cpu())}

def main():
    args = parse_args()
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    import torch
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size * 1 % args.num_generations != 0:
        raise ValueError(f'global batch ({world_size}) must be divisible by num_generations ({args.num_generations}); use 6 processes with num_generations=2')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; launch after the NVIDIA driver is visible')
    print('library_versions', json.dumps(versions()), flush=True)
    print('distributed_before_model', json.dumps(distributed_info()), flush=True)
    root = Path(__file__).resolve().parents[1]
    dataset = Dataset.from_list(load_rows(root / args.train_file, max(args.max_cases, world_size)))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    print('model_loaded', args.model, flush=True)
    print('full_parameter_snapshot', json.dumps(parameter_snapshot(model)), flush=True)
    print('memory_before_trainer', json.dumps(memory_snapshot()), flush=True)
    config = GRPOConfig(output_dir=str(root / args.output_dir), max_steps=args.max_steps, per_device_train_batch_size=1, num_generations=args.num_generations, generation_batch_size=world_size, max_completion_length=args.max_completion_length, learning_rate=1e-6, temperature=1.0, top_p=1.0, bf16=True, fp16=False, disable_dropout=True, report_to='none', logging_steps=1, save_strategy='no', gradient_checkpointing=True)
    trainer = GRPOTrainer(model=model, reward_funcs=synthetic_reward, args=config, train_dataset=dataset, processing_class=tokenizer)
    before = parameter_snapshot(trainer.model)
    result = trainer.train()
    after = parameter_snapshot(trainer.model)
    changed = before['first_parameter_norm'] != after['first_parameter_norm']
    rewards = [x['final_reward'] for x in REWARD_LOG]
    print('reward_examples', json.dumps(REWARD_LOG[:4], ensure_ascii=False), flush=True)
    print('reward_mean_std', statistics.mean(rewards) if rewards else None, statistics.pstdev(rewards) if len(rewards) > 1 else 0.0, flush=True)
    print('train_loss', getattr(result, 'training_loss', None), flush=True)
    print('memory_after_train', json.dumps(memory_snapshot()), flush=True)
    print('parameter_update_check', json.dumps({'before': before, 'after': after, 'changed': changed}), flush=True)
    print('trainer_log_history', json.dumps(trainer.state.log_history, default=str), flush=True)
    if args.save_final:
        trainer.save_model(str(root / args.output_dir / 'final'))
    print('smoke_status', 'PASS' if changed and len(set(rewards)) > 1 else 'FAIL', flush=True)

if __name__ == '__main__': main()
