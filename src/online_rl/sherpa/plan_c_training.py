"""Isolated Plan C trainer: four cases, turn-local groups, one update per batch.

Durable recovery/checkpoint publication follows the existing training protocol.
No shared-prefix or continuation tokens enter the loss.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil
import statistics
import time

from src.common import usage
from src.online_rl.sherpa.plan_c_events import emit
from src.online_rl.integrated.continuation import adapter_sha256, derive_seed
from src.online_rl.integrated.episode import assert_single_prompt
from src.online_rl.integrated.grpo import make_updater
from src.online_rl.integrated.records import atomic_write_json, read_json, sha256_json
from src.online_rl.sherpa.pipeline import assess, case_metrics, reflections
from src.online_rl.sherpa.plan_c_backend import generator_prompt
from src.online_rl.sherpa.plan_c_rollouts import PROTOCOL, collect_turn_groups, training_records
from src.online_rl.sherpa.rollouts import schedule
from src.online_rl.sherpa.training import resume_fingerprint
from src.online_rl.sherpa.learning_signal import signal_checks, save_step_signal
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.five_gpu import load_checkpoint, save_checkpoint


def assess_turn(case, group, cfg, backend, cache, prompt, step, expected, folder):
    started = time.perf_counter()
    candidates = group['candidates']
    examples = reflections(case, candidates, backend, cache,
                           derive_seed(cfg['schedule']['seed'], group['group_id'], 'reflection', step))
    atomic_write_json(folder / 'reflections.json', examples)
    result = assess(case, candidates, examples, prompt, backend, cache,
                    {'step': step, 'policy': expected, 'scope': PROTOCOL,
                     'group_id': group['group_id'], 'state': sha256_json(group['state'])})
    atomic_write_json(folder / 'rubric_scores.json', result)
    if not result['valid']:
        raise RuntimeError('Target-turn checklist generation failed; no reward is fabricated')
    rows = training_records(group, result['scores'])
    metrics = case_metrics(candidates, result['scores'], cfg.get('advantage_normalization', {}).get('mode', 'group_zscore'))
    emit({'stage': 'turn_reward_completed', 'step': step, 'case_id': case['case_id'],
          'target_turn': group['target_turn'], 'seconds': time.perf_counter() - started,
          'generation_seconds': result['generation'].get('seconds'), 'auc': result['auc']})
    return {'group_id': group['group_id'], 'target_turn': group['target_turn'],
            'metrics': metrics, 'training_records': rows,
            'rollout_seconds': group['seconds'], 'reward_seconds': time.perf_counter() - started}


def train(cfg, policy, backend, cases, root, *, max_steps=None):
    home = Path(root) / cfg['output_dir']
    home.mkdir(parents=True, exist_ok=True)
    method = 'plan_c'
    prompt, prompt_artifact = generator_prompt(cfg, root, policy=policy)
    batches = schedule(cfg, cases, root)
    identity = {'protocol': PROTOCOL, 'config': cfg,
                'backend': backend.identity() if backend is not None else None,
                'frozen_prompt': prompt_artifact['prompt_sha256'] if prompt_artifact else None,
                'case_schedule': [[case['case_id'] for case in batch] for batch in batches]}
    fingerprint = sha256_json(identity)
    state_path = home / 'run_state.json'
    updater = make_updater(policy, cfg)
    advantage_mode = cfg.get('advantage_normalization', {}).get('mode', 'group_zscore')
    if state_path.exists():
        state = read_json(state_path)
        fingerprint = resume_fingerprint(identity, state, home / 'run_manifest.json')
        load_checkpoint(updater, state['recovery_checkpoint'])
        if adapter_sha256(policy.model) != state['policy_sha256']:
            raise RuntimeError('Recovery adapter identity differs')
    else:
        state = {'fingerprint': fingerprint, 'completed_steps': 0, 'policy_sha256': adapter_sha256(policy.model)}
        atomic_write_json(home / 'run_manifest.json', identity)
    atomic_write_json(home / 'config.json', cfg)
    emit({'stage': 'training_configuration', 'advantage_mode': advantage_mode,
          'advantage_scale': 1.0, 'exclude_zero_advantage_trajectories': advantage_mode == 'group_mean',
          'grouping': 'same_case_same_turn_same_history', 'credit_scope': 'target_response_tokens_only'})
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
        emit({'stage': 'rollout', 'batch': step, 'step': step, 'iteration': iteration, 'cases': 4, 'target_responses_per_turn': cfg['plan_c']['responses_per_turn']})

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
                # Reward work begins as soon as a turn's siblings are complete,
                # overlapping with collection at the next spine turn.
                with ThreadPoolExecutor(1, thread_name_prefix='turn-reward') as reward_pool:
                    pending = []
                    def ready(group):
                        target_folder = folder / f'turn_{group["target_turn"]:02d}'
                        atomic_write_json(target_folder / 'rollouts.json', group)
                        pending.append(reward_pool.submit(
                            assess_turn, case, group, cfg, backend, cache, prompt,
                            step, expected, target_folder))
                    collected = collect_turn_groups(case, cfg, policy, cache, step, on_group=ready)
                    assessed_groups = [future.result() for future in pending]
                atomic_write_json(folder / 'spine.json', {
                    'tree': collected['spine'], 'main_trajectory': collected['main_trajectory']})
                record = {'case_id': case['case_id'], 'step': step, 'iteration': iteration,
                          'fingerprint': fingerprint, 'policy_sha256': expected,
                          'turn_groups': assessed_groups, 'seconds': collected['seconds']}
                atomic_write_json(saved, record)
                return record

        # Policy weights remain frozen until every reached turn is fully scored.
        with ThreadPoolExecutor(cfg.get('training_execution', {}).get('case_workers', 4),
                                thread_name_prefix='training-case') as pool:
            records = list(pool.map(one, batches[step - 1]))
        if adapter_sha256(policy.model) != expected:
            raise RuntimeError('Policy changed during the rollout/scoring barrier')
        turn_results = [group for record in records for group in record['turn_groups']]
        groups = [group['training_records'] for group in turn_results]
        if not groups or any(len(group) != cfg['plan_c']['responses_per_turn'] for group in groups):
            raise RuntimeError('Every reached spine turn needs a complete sibling group')
        for group in groups:
            assert_single_prompt(group)
        emit({'stage': 'agent_update', 'step': step, 'iteration': iteration})
        update = updater.update_groups(groups)
        signal = signal_checks(cfg, home, root, step, update,
                               prompt_artifact['prompt_sha256'] if prompt_artifact else None)
        save_step_signal(step_dir, step, signal)
        if signal['alerts']:
            emit({'stage': 'learning_signal_warning', 'step': step,
                  'advantage_mode': advantage_mode, 'alerts': signal['alerts']})
        current = adapter_sha256(policy.model)
        values = [row['reward'] for group in groups for row in group]
        traces = [trace for group in groups for row in group for trace in row['agent_traces']]
        metrics = {'step': step, 'iteration': iteration, 'step_in_iteration': (step - 1) % 30 + 1,
                   'method': method, 'trajectories': len(values), 'cases': 4,
                   'target_responses': len(values), 'unique_trajectories':
                   sum(1 + (cfg['plan_c']['responses_per_turn'] - 1) * len(row['turn_groups'])
                       for row in records),
                   'turn_groups': len(groups), 'training_scope': 'target_action_only',
                   'advantage_mode': advantage_mode, 'advantage_scale': 1.0,
                   'grad_norm': update.get('grad_norm'), 'learning_signal': signal,
                   'reward_mean': statistics.mean(values), 'reward_std': statistics.pstdev(values),
                   'zero_advantage_group_fraction': sum(row['metrics']['zero_advantage'] for row in turn_results) / len(groups),
                   'turn_auc': {row['group_id']: row['metrics']['auc'] for row in turn_results},
                   'kl': update.get('kl_loss'), 'clip_fraction': update.get('ratio', {}).get('clipped_fraction'),
                   'mean_response_tokens': statistics.mean(len(trace['completion_ids']) for trace in traces),
                   'mean_target_action_tokens': statistics.mean(
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
