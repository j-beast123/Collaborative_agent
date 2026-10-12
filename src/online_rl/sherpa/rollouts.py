"""Shared rollout and deterministic case/seed schedules for every method."""
from __future__ import annotations

from functools import partial
from pathlib import Path

from src.online_rl.integrated.continuation import adapter_sha256, derive_seed
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.backends import PolicyAgent, RoleDoctor
from src.online_rl.tree.records import to_dialogue
from src.online_rl.tree.rollout import build_tree


def collect(case, cfg, policy, cache, step, *, behavior=True, branching=32, seed_override=None):
    from src.collaboration.generator import final_answer_option, stop_reason
    from src.online_rl.environment import final_outcome
    from src.online_rl.integrated.grpo import behavior_logprobs
    seed = (derive_seed(cfg['schedule']['seed'], case['case_id'], 'rollout_step', step)
            if seed_override is None else seed_override)
    protocol = sha256_json({'rollout': cfg['rollout'], 'doctor': policy.doctor.protocol,
                           'base_model': cfg['policy']['base_model'], 'sampling_step': step})
    if getattr(policy, 'original_base_only', False):
        if behavior:
            raise ValueError('Base-only vLLM policy cannot supply GRPO behavior log probabilities')
        policy_hash = base_cache_policy_hash(cache, case['case_id'], protocol, cfg['policy']['base_model'])
    else:
        policy_hash = adapter_sha256(policy.model)
    def old(traces):
        # Raw log-probabilities are already captured by the per-row sampler.
        values = [trace.get('old_policy_logprobs') for trace in traces]
        if all(value is not None for value in values):
            return values
        with policy.gen_lock:
            return behavior_logprobs(policy.model, traces, cfg['optimizer']['logprob_token_chunk_size'])
    tree = build_tree(case, max_agent_turns=cfg['rollout']['rounds'], branching=branching,
                      run_seed=seed, policy_sha256=policy_hash, protocol_sha256=protocol,
                      agent=PolicyAgent(policy), doctor=RoleDoctor(policy.doctor),
                      stop_rule=partial(stop_reason, allow_mutual_conclusion=
                                        cfg['rollout']['halting'].get('allow_mutual_conclusion', True)),
                      final_option=final_answer_option,
                      outcome_fn=lambda c, option: float(final_outcome(c, option) == 1.0),
                      cache=cache, behavior_logprobs=old if behavior else None,
                      workers=cfg['execution']['doctor_workers'],
                      agent_workers=cfg['execution']['agent_workers'], budget='initial_fanout',
                      repeat_agent_reply_after_turns=cfg['rollout']['halting'].get('repeat_agent_reply_after_turns'))
    trajectories = [{'trajectory_id': leaf.leaf_id, 'case_id': case['case_id'],
                     'dialogue': to_dialogue(leaf.dialogue), 'outcome': int(leaf.outcome),
                     'final_option': leaf.final_option, 'final_parsed': leaf.final_option in case['options'],
                     'stop_reason': leaf.stop_reason, 'path': list(leaf.path),
                     'agent_turns': len(leaf.path)} for leaf in tree.leaves]
    if len(trajectories) != branching:
        raise RuntimeError('Every opening reply must produce exactly one trajectory')
    return tree, trajectories


def base_cache_policy_hash(cache, case_id, protocol, base_model):
    """Keep the legacy cache namespace only for validated original-base evidence.

    The disabled adapter hash was formerly included even though it did not affect
    inference. collect_base verifies the manifest before accessing these calls.
    Engine provenance lives in each new trace, not in the preserved cache key.
    """
    hashes = set()
    for record in cache.store.values():
        identity = record['identity']
        if identity.get('case') != case_id or identity.get('protocol') != protocol:
            raise RuntimeError('Base cache case/protocol mismatch; refusing engine migration')
        hashes.add(identity['policy'])
    if len(hashes) > 1:
        raise RuntimeError('Multiple policy identities in base cache; refusing engine migration')
    return next(iter(hashes)) if hashes else sha256_json({'original_base': base_model, 'adapter': 'disabled'})


def schedule(cfg, cases, root):
    import json
    from src.online_rl.cases import file_sha256
    from src.online_rl.guards import assert_not_protected
    assert_not_protected(cases, 'SHERPA training and prompt optimization')
    by_id = {case['case_id']: case for case in cases}
    if len(cases) != 120 or len(by_id) != 120:
        raise ValueError('Training requires exactly 120 unique cases')
    spec = cfg['schedule']
    if (spec['iterations'], spec['batch_cases'], spec['steps_per_iteration']) != (3, 4, 30):
        raise ValueError('Schedule must be three iterations of 30 four-case steps')
    order_path = Path(root) / spec['order_file']
    if file_sha256(order_path) != spec['order_sha256']:
        raise RuntimeError('Pinned training case order changed')
    if file_sha256(Path(root) / cfg['data']['train']) != cfg['data']['train_sha256']:
        raise RuntimeError('Training data differs from the pinned 120-case set')
    order = json.loads(order_path.read_text())
    if len(order) != 120 or set(order) != set(by_id):
        raise ValueError('Pinned case order must contain every training case once')
    return [[by_id[key] for key in order[start:start + 4]]
            for _ in range(3) for start in range(0, 120, 4)]
