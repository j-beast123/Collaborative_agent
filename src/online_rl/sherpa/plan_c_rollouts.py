"""One immutable spine, sibling actions at each spine history, natural suffixes.

Only target traces enter GRPO. Suffixes are evidence, never additional targets.
The existing tree builder supplies sampling, Doctor halting and FINAL semantics.
"""
from copy import deepcopy
from functools import partial
import time

from src.online_rl.sherpa.plan_c_events import emit
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.episode import assert_single_prompt
from src.online_rl.integrated.records import sha256_json, to_dict
from src.online_rl.sherpa.rollouts import collect
from src.online_rl.tree.backends import PolicyAgent, RoleDoctor
from src.online_rl.tree.records import to_dialogue
from src.online_rl.tree.rollout import _build_initial_fanout

PROTOCOL = 'spine_turn_counterfactual_v1'


def collect_turn_groups(case, cfg, policy, cache, step, on_group=None):
    """Main action + N-1 independently sampled alternatives at every reached turn.

    A suffix may stop earlier/later than the spine, under the same absolute turn
    limit. It never supplies a prefix for another group. Main suffixes are reused.
    on_group runs immediately after that turn's continuations finish.
    """
    from src.collaboration.generator import final_answer_option, stop_reason
    from src.online_rl.environment import final_outcome
    started = time.perf_counter()
    spine, main_rows = collect(case, cfg, policy, cache, step, branching=1)
    main = main_rows[0]
    groups = []
    for node in spine.nodes:
        group_started = time.perf_counter()
        history = to_dialogue(node.prefix)
        state = {'history': history, 'doctor_status': node.prefix_doctor_status,
                 'doctor_private_state': node.prefix_partner_state, 'target_turn': node.depth}
        group_id = f'{case["case_id"]}|turn_{node.depth}|{sha256_json(state)}'
        seed = derive_seed(cfg['schedule']['seed'], group_id, 'plan_c_alternatives', step)
        protocol = sha256_json({'protocol': PROTOCOL, 'config': cfg['plan_c'],
                               'rollout': cfg['rollout'], 'doctor': policy.doctor.protocol,
                               'state': state, 'step': step})
        # Deep copy the frozen state; no alternate branch may mutate the spine.
        alternatives = _build_initial_fanout(
            case, max_agent_turns=cfg['rollout']['rounds'],
            branching=cfg['plan_c']['responses_per_turn'] - 1,
            run_seed=seed, policy_sha256=spine.policy_sha256, protocol_sha256=protocol,
            agent=PolicyAgent(policy), doctor=RoleDoctor(policy.doctor), cache=cache,
            stop_rule=partial(stop_reason, allow_mutual_conclusion=
                              cfg['rollout']['halting'].get('allow_mutual_conclusion', True)),
            final_option=final_answer_option,
            outcome_fn=lambda c, opt: float(final_outcome(c, opt) == 1),
            workers=cfg['execution']['doctor_workers'], agent_workers=cfg['execution']['agent_workers'],
            repeat_agent_reply_after_turns=cfg['rollout']['halting'].get('repeat_agent_reply_after_turns'),
            budget=PROTOCOL, frozen_state=deepcopy((history, node.prefix_doctor_status,
                                                 node.prefix_partner_state)), start_depth=node.depth)
        candidates = [{**deepcopy(main), 'trajectory_id': group_id + '/main',
                       'target_response': node.response, 'target_turn': node.depth,
                       'shared_history': history, 'target_trace': node.trace,
                       'is_main': True}]
        nodes = {n.node_id: n for n in alternatives.nodes}
        for leaf in alternatives.leaves:
            target = nodes[leaf.path[0]]
            if to_dialogue(target.prefix) != history or target.depth != node.depth:
                raise RuntimeError('Counterfactual target history/turn differs from spine')
            candidates.append({'trajectory_id': group_id + '/alternative/' + leaf.leaf_id,
                               'case_id': case['case_id'], 'dialogue': to_dialogue(leaf.dialogue),
                               'outcome': int(leaf.outcome), 'final_option': leaf.final_option,
                               'stop_reason': leaf.stop_reason, 'agent_turns':
                               node.depth - 1 + len(leaf.path), 'target_turn': node.depth,
                               'shared_history': history, 'target_response': target.response,
                               'target_trace': target.trace, 'is_main': False})
        if len(candidates) != cfg['plan_c']['responses_per_turn']:
            raise RuntimeError('Incomplete target-turn sibling group')
        group = {'group_id': group_id, 'case_id': case['case_id'], 'target_turn': node.depth,
                 'state': state, 'policy_sha256': spine.policy_sha256,
                 'candidates': candidates, 'seconds': time.perf_counter() - group_started}
        # Validate exact prompt equality before spending reflection/judge calls.
        assert_single_prompt(training_records(group, [{'trajectory_id': c['trajectory_id'], 'reward': 0}
                                                      for c in candidates]))
        emit({'stage': 'turn_rollouts_completed', 'step': step, 'case_id': case['case_id'],
              'target_turn': node.depth, 'candidates': len(candidates), 'seconds': group['seconds']})
        if on_group is not None:
            on_group(group)
        groups.append(group)
    return {'spine': to_dict(spine), 'main_trajectory': main, 'groups': groups,
            'seconds': time.perf_counter() - started}


def training_records(group, scores):
    ids = [c['trajectory_id'] for c in group['candidates']]
    if len(scores) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Every candidate requires exactly one aligned reward')
    rewards = {s['trajectory_id']: s['reward'] for s in scores}
    if set(rewards) != set(ids):
        raise ValueError('Target-turn reward IDs do not match candidates')
    records = [{'trajectory_id': c['trajectory_id'], 'case_id': group['group_id'],
                'source_case_id': group['case_id'], 'target_turn': group['target_turn'],
                'outcome': c['outcome'],
                'reward': rewards[c['trajectory_id']], 'rollout_policy_sha256': group['policy_sha256'],
                'agent_traces': [c['target_trace']]} for c in group['candidates']]
    assert_single_prompt(records)
    # The main collector records these logprobs; alternate suffixes carry raw
    # sampling logprobs instead. Both must enter the same teacher-forced alignment.
    for record in records:
        trace = record['agent_traces'][0]
        if 'behavior_logprobs' not in trace:
            raw = trace.get('old_policy_logprobs')
            if raw is None or len(raw) != len(trace['completion_ids']):
                raise ValueError('Plan C target requires aligned behavior logprobs')
            record['agent_traces'] = [{**trace, 'behavior_logprobs': list(raw)}]
    return records
