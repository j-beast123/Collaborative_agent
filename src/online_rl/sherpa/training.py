"""Four-case on-policy steps, three iterations and ten-step curve checkpoints."""
from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common import usage
from src.common.progress import emit
from src.online_rl.integrated.continuation import adapter_sha256, derive_seed
from src.online_rl.integrated.grpo import make_updater
from src.online_rl.integrated.records import atomic_write_json, read_json, sha256_json, to_dict
from src.online_rl.sherpa import PROTOCOL
from src.online_rl.sherpa.pipeline import assess, case_metrics, reflections, training_records
from src.online_rl.sherpa.prompt_search import frozen_prompt
from src.online_rl.sherpa.rollouts import collect, schedule
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.five_gpu import load_checkpoint, save_checkpoint


def resume_fingerprint(identity, state, manifest_path):
    """Keep cached training steps when only the evaluation configuration changes."""
    fingerprint = sha256_json(identity)
    if state['fingerprint'] == fingerprint:
        return fingerprint
    previous = read_json(manifest_path)
    def without_evaluation(value):
        return {**value, 'config': {key: setting for key, setting in value['config'].items()
                                  if key != 'evaluation'}}
    if (sha256_json(previous) != state['fingerprint']
            or without_evaluation(previous) != without_evaluation(identity)):
        raise RuntimeError('Training protocol/config changed; start a fresh run')
    # Existing receipts and cached case records retain this authenticated identity.
    return state['fingerprint']


def train(cfg, policy, backend, cases, root, *, max_steps=None):
    home = Path(root) / cfg['output_dir']
    home.mkdir(parents=True, exist_ok=True)
    method = cfg.get('method', 'sherpa')
    if method not in ('sherpa', 'outcome'):
        raise ValueError('Select sherpa or outcome; other methods need their own reward implementation')
    prompt, prompt_artifact = (frozen_prompt(cfg, root, policy=policy, backend=backend)
                              if method == 'sherpa' else (None, None))
    batches = schedule(cfg, cases, root)
    identity = {'protocol': PROTOCOL, 'config': cfg,
                'frozen_prompt': prompt_artifact['prompt_sha256'] if prompt_artifact else None,
                'case_schedule': [[case['case_id'] for case in batch] for batch in batches]}
    fingerprint = sha256_json(identity)
    state_path = home / 'run_state.json'
    updater = make_updater(policy, cfg)
    if state_path.exists():
        state = read_json(state_path)
        fingerprint = resume_fingerprint(identity, state, home / 'run_manifest.json')
        load_checkpoint(updater, state['recovery_checkpoint'])
        if adapter_sha256(policy.model) != state['policy_sha256']:
            raise RuntimeError('Recovery adapter identity differs')
    else:
        state = {'fingerprint': fingerprint, 'completed_steps': 0, 'policy_sha256': adapter_sha256(policy.model)}
        atomic_write_json(home / 'run_manifest.json', identity)
    if prompt_artifact:
        # Only after resume identity validation; never replace another run's prompt.
        (home / 'frozen_generator_prompt.txt').write_text(prompt)
        atomic_write_json(home / 'frozen_generator_prompt.json', prompt_artifact)
    if len(batches) != 90:
        raise RuntimeError('Expected exactly 90 scheduled steps')
    from src.common.checkpoint_hub import enqueue_safely
    # Catch up an enqueue interrupted after a durable ten-step commit.
    if cfg.get('checkpoint_upload', {}).get('enabled'):
        for step in range(10, state['completed_steps'] + 1, 10):
            checkpoint = home / 'checkpoints' / f'checkpoint_{step:06d}'
            receipt = read_json(checkpoint / 'COMPLETE.json')
            enqueue_safely(cfg, home, {**state, 'completed_steps': step,
                           'policy_sha256': receipt['policy_sha256'], 'last_curve_checkpoint': str(checkpoint)})

    def commit(step, receipt, recovery):
        nonlocal state
        metrics = receipt['metrics']
        step_dir = home / 'steps' / f'step_{step:06d}'
        atomic_write_json(step_dir / 'metrics.json', metrics)
        old_recovery = state.get('recovery_checkpoint')
        state.update(completed_steps=step, policy_sha256=receipt['policy_sha256'], recovery_checkpoint=recovery,
                     last_update=metrics['update'], complete=step == 90)
        if step % 10 == 0:
            checkpoint = save_checkpoint(updater, home / 'checkpoints' / f'checkpoint_{step:06d}', receipt)
            state['last_curve_checkpoint'] = checkpoint
            if step == 90:
                state['final_checkpoint'] = checkpoint
        atomic_write_json(state_path, state)
        if step % 10 == 0:
            enqueue_safely(cfg, home, state)
        if old_recovery and old_recovery != recovery:
            old_path = Path(old_recovery)
            if old_path.parent.resolve() == (home / 'recovery').resolve():
                shutil.rmtree(old_path)
        emit({'stage': 'batch_committed', 'batch': step, 'step': step,
              'iteration': (step - 1) // 30 + 1, 'reward_mean': metrics['reward_mean']})

    completed_now = 0
    for step in range(state['completed_steps'] + 1, 91):
        started = time.perf_counter()
        iteration = (step - 1) // 30 + 1
        os.environ['API_USAGE_ITERATION'] = str(iteration)
        expected = adapter_sha256(policy.model)
        recovery_path = home / 'recovery' / f'step_{step:06d}'
        if recovery_path.exists():
            # A checkpoint published immediately before a process interruption
            # can be committed without repeating that optimizer step.
            receipt = read_json(recovery_path / 'COMPLETE.json')
            if (receipt['fingerprint'] != fingerprint or receipt['step'] != step
                    or receipt['source_policy_sha256'] != expected):
                raise RuntimeError('Interrupted recovery checkpoint identity changed')
            load_checkpoint(updater, recovery_path)
            if adapter_sha256(policy.model) != receipt['policy_sha256']:
                raise RuntimeError('Interrupted recovery adapter differs')
            commit(step, receipt, str(recovery_path))
            completed_now += 1
            if max_steps is not None and completed_now >= max_steps:
                break
            continue
        policy.model.eval()
        # The batched sampler does not automatically synchronize replicas.
        # Barrier before any case threads: every rollout uses this step's adapter.
        sync = getattr(policy, 'sync_replicas', None)
        replica_state = sync(expected_sha256=expected) if sync is not None else None
        step_dir = home / 'steps' / f'step_{step:06d}'
        step_dir.mkdir(parents=True, exist_ok=True)
        if replica_state is not None:
            atomic_write_json(step_dir / 'rollout_policy.json', replica_state)
        emit({'stage': 'rollout', 'batch': step, 'step': step, 'iteration': iteration, 'cases': 4, 'trajectories': 128})

        def one(case):
            folder = step_dir / case['case_id']
            folder.mkdir(parents=True, exist_ok=True)
            saved = folder / 'record.json'
            if saved.exists():
                record = read_json(saved)
                if record['policy_sha256'] != expected or record['step'] != step or record['fingerprint'] != fingerprint:
                    raise RuntimeError('Cached step belongs to another policy or protocol')
                return record
            cache = CallCache(folder / 'calls.jsonl')
            with usage.case(case['case_id']):
                tree, trajectories = collect(case, cfg, policy, cache, step)
                atomic_write_json(folder / 'rollouts.json', {'tree': to_dict(tree), 'trajectories': trajectories,
                                  'outcomes': [row['outcome'] for row in trajectories],
                                  'success_rate': sum(row['outcome'] for row in trajectories) / 32})
                if method == 'sherpa':
                    examples = reflections(case, trajectories, backend, cache,
                                           derive_seed(cfg['schedule']['seed'], case['case_id'], 'reflection_selection', step))
                    atomic_write_json(folder / 'reflections.json', examples)
                    assessed = assess(case, trajectories, examples, prompt, backend, cache,
                                      {'step': step, 'policy': expected, 'scope': 'training'})
                    atomic_write_json(folder / 'rubric_scores.json', assessed)
                    if not assessed['valid']:
                        raise RuntimeError('Checklist generation failed after its bounded repairs; no reward is fabricated')
                    scores = assessed['scores']
                else:
                    examples = []
                    scores = [{'trajectory_id': row['trajectory_id'], 'scores': [], 'reward': row['outcome']}
                              for row in trajectories]
                    assessed = {'valid': True, 'generation': None, 'scores': scores, 'auc': None}
                records = training_records(tree, trajectories, scores)
                record = {'case_id': case['case_id'], 'step': step, 'iteration': iteration,
                          'fingerprint': fingerprint, 'policy_sha256': expected,
                          'trajectories': trajectories, 'reflections': examples,
                          'rubric_assessment': assessed, 'training_records': records,
                          'metrics': case_metrics(trajectories, scores)}
                atomic_write_json(saved, record)
                return record

        # Policy weights remain frozen until all 128 trajectories are completely scored.
        with ThreadPoolExecutor(cfg.get('training_execution', {}).get('case_workers', 4),
                                thread_name_prefix='training-case') as pool:
            records = list(pool.map(one, batches[step - 1]))
        if adapter_sha256(policy.model) != expected:
            raise RuntimeError('Policy changed during the rollout/scoring barrier')
        groups = [record['training_records'] for record in records]
        if len(groups) != 4 or any(len(group) != 32 for group in groups):
            raise RuntimeError('GRPO requires four groups of 32 trajectories')
        emit({'stage': 'agent_update', 'step': step, 'iteration': iteration})
        update = updater.update_groups(groups)
        current = adapter_sha256(policy.model)
        values = [row['reward'] for group in groups for row in group]
        traces = [trace for group in groups for row in group for trace in row['agent_traces']]
        metrics = {'step': step, 'iteration': iteration, 'step_in_iteration': (step - 1) % 30 + 1,
                   'method': method, 'trajectories': 128, 'cases': 4,
                   'reward_mean': statistics.mean(values), 'reward_std': statistics.pstdev(values),
                   'zero_advantage_group_fraction': sum(row['metrics']['zero_advantage'] for row in records) / 4,
                   'case_auc': {row['case_id']: row['metrics']['auc'] for row in records},
                   'kl': update.get('kl_loss'), 'clip_fraction': update.get('ratio', {}).get('clipped_fraction'),
                   'mean_response_tokens': statistics.mean(len(trace['completion_ids']) for trace in traces),
                   'mean_trajectory_assistant_tokens': statistics.mean(
                       sum(len(trace['completion_ids']) for trace in row['agent_traces']) for group in groups for row in group),
                   'update': update, 'source_policy_sha256': expected, 'policy_sha256': current,
                   'seconds': time.perf_counter() - started}
        atomic_write_json(step_dir / 'metrics.json', metrics)
        receipt = {'fingerprint': fingerprint, 'step': step, 'iteration': iteration,
                   'policy_sha256': current, 'source_policy_sha256': expected, 'metrics': metrics}
        # Recovery state is distinct from the checkpoints used for learning curves.
        recovery = save_checkpoint(updater, home / 'recovery' / f'step_{step:06d}', receipt)
        commit(step, receipt, recovery)
        print(f'Step {step}/90 iteration={iteration} reward={metrics["reward_mean"]:.4f}', flush=True)
        completed_now += 1
        if max_steps is not None and completed_now >= max_steps:
            break
    if state.get('complete'):
        emit({'stage': 'run_finished', 'batch': 90, 'step': 90, 'iteration': 3})
    return state
