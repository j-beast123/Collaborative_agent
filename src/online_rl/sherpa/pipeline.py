"""Case-level reflection/checklist reward and trajectory-level training records."""
from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor

from src.common import usage
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.metrics import advantages
from src.online_rl.integrated.records import sha256_json
from src.online_rl.sherpa.rubrics import auc, select_reflections


def reflections(case, trajectories, backend, cache, seed):
    chosen = select_reflections(trajectories, seed)
    def one(trajectory):
        call_seed = derive_seed(seed, trajectory['trajectory_id'], 'doctor_reflection', 0)
        identity = {'case_id': case['case_id'], 'trajectory': sha256_json(trajectory),
                    'seed': call_seed, 'backend': backend.identity()}
        return cache.call('doctor_reflection', identity,
                          lambda: backend.reflect(case, trajectory, call_seed))
    with ThreadPoolExecutor(min(8, backend.cfg['execution']['doctor_workers'])) as pool:
        texts = list(pool.map(one, chosen))
    return [{'trajectory': row, 'doctor_reflection': reflection}
            for row, reflection in zip(chosen, texts)]


def assess(case, trajectories, examples, prompt, backend, cache, identity):
    generation_id = {**identity, 'case_id': case['case_id'], 'prompt': sha256_json(prompt),
                     'examples': sha256_json(examples), 'backend': backend.identity(),
                     'method': backend.cfg.get('method', 'sherpa')}
    with usage.case(case['case_id']):
        generation = cache.call('rubric_set', generation_id,
                                lambda: backend.generate(case, examples, prompt))
    if not generation['valid']:
        return {'valid': False, 'generation': generation, 'scores': [], 'auc': None}
    scores = backend.score_all(case, trajectories, generation['rubrics'], cache, identity)
    return {'valid': True, 'generation': generation, 'scores': scores,
            'auc': auc([row['outcome'] for row in trajectories], [row['reward'] for row in scores])}


def training_records(tree, trajectories, scores):
    nodes = {node.node_id: node for node in tree.nodes}
    if len(scores) != 32 or {row['trajectory_id'] for row in scores} != {row['trajectory_id'] for row in trajectories}:
        raise ValueError('Every trajectory needs exactly one aligned reward')
    rewards = {row['trajectory_id']: row['reward'] for row in scores}
    return [{'trajectory_id': trajectory['trajectory_id'], 'case_id': tree.case_id,
             'reward': rewards[trajectory['trajectory_id']], 'rollout_policy_sha256': tree.policy_sha256,
             'agent_traces': [nodes[node_id].trace for node_id in trajectory['path']]}
            for trajectory in trajectories]


def case_metrics(trajectories, scores):
    values = [row['reward'] for row in scores]
    adv = advantages(values, 'group_zscore')
    return {'success_rate': statistics.mean(row['outcome'] for row in trajectories),
            'reward_mean': statistics.mean(values), 'reward_std': statistics.pstdev(values),
            'advantages': adv, 'zero_advantage': all(value == 0 for value in adv),
            'auc': auc([row['outcome'] for row in trajectories], values),
            'mean_agent_turns': statistics.mean(row['agent_turns'] for row in trajectories)}
