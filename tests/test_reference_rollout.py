"""Rollout topology, stopping, streaming paths and immutable resume; no model/API calls."""
import copy
from collections import defaultdict
from pathlib import Path

import pytest

from src.online_rl.integrated.continuation import derive_seed, protocol_sha256
from src.online_rl.integrated.records import from_dict, to_dict
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.experiment import load_config
from src.online_rl.tree.records import Tree, to_turns
from src.online_rl.tree.rollout import build_tree, tree_counts
from src.online_rl.tree.team import node_credit

CASE = {'case_id': 'c', 'options': {'A': 'alpha', 'B': 'beta'}}
STATUS = {'current_option': 'A', 'open_issue': True, 'collaborator_objection': False,
          'more_info_could_change_decision': False}


class Agent:
    def __init__(self, events):
        self.events = events

    def sample(self, case, prefix, turn, seed):
        self.events.append(('agent', turn, seed))
        return {'completion_text': f'reply {seed}', 'prefix': [list(turn) for turn in to_turns(prefix)]}


class Doctor:
    def __init__(self, events, v2, halt_at=None):
        self.events, self.status_protocol, self.halt_at = events, v2, halt_at
        self.min_agent_turns = 1

    def turn(self, case, dialogue, round, rounds, seed=None):
        self.events.append(('doctor', round, seed))
        visible = f'doctor {round} {seed}'
        return ({'visible': visible, 'status': STATUS, 'status_error': None, 'seed': seed}
                if self.status_protocol else visible)

    def halt(self, *, agent_turns, **kwargs):
        return 'doctor_halt' if agent_turns == self.halt_at else None

    def final(self, case, dialogue, seed=None):
        self.events.append(('final', len([d for d in dialogue if d['speaker'] == 'agent']), seed))
        return 'A'


def rollout(*, v2=True, events=None, cache=None, **kwargs):
    events = events if events is not None else []
    return build_tree(CASE, max_agent_turns=4, branching=4, run_seed=11,
                      policy_sha256='policy', protocol_sha256='protocol', agent=Agent(events),
                      doctor=kwargs.pop('doctor', Doctor(events, v2)),
                      stop_rule=kwargs.pop('stop_rule', lambda *args: None),
                      final_option=lambda text: text, outcome_fn=lambda case, opt: float(opt == 'A'),
                      cache=cache if cache is not None else CallCache(None),
                      budget=kwargs.pop('budget', 'reference_spine'), **kwargs)


@pytest.mark.parametrize('v2,workers', [(False, 1), (True, 1), (True, 3)])
def test_reference_first_thirteen_paths_and_four_sibling_groups(v2, workers):
    events, streamed, completed = [], [], []
    tree = rollout(v2=v2, events=events, workers=workers, agent_workers=workers,
                   behavior_logprobs=lambda traces: [[-0.1] for trace in traces],
                   on_leaf=completed.append, on_leaf_nodes=lambda leaf, nodes: streamed.append((leaf, nodes)))
    counts = tree_counts(tree)
    assert counts['leaves'] == counts['finals'] == 13
    assert counts['agent_nodes'] == 34
    assert counts['by_depth'] == {1: 4, 2: 7, 3: 10, 4: 13}
    first_final = next(i for i, event in enumerate(events) if event[0] == 'final')
    assert [event[1] for event in events[:first_final] if event[0] == 'agent'] == [1, 2, 3, 4]
    assert tree.leaves[0].leaf_id == 'c/0.0.0.0'
    expected_paths = {'0.0.0.0'} | {
        '.'.join(['0'] * fork + [str(child)] + ['0'] * (3 - fork))
        for fork in range(4) for child in range(1, 4)}
    assert {leaf.leaf_id.removeprefix('c/') for leaf in tree.leaves} == expected_paths
    by_id = {node.node_id: node for node in tree.nodes}
    assert len(by_id) == len(tree.nodes)
    groups = defaultdict(list)
    for node in tree.nodes:
        groups[node.parent_id].append(node)
        assert node.seed == derive_seed(11, node.node_id, 'agent', 0)
        assert node.trace['behavior_logprobs'] == [-0.1]
    siblings = [group for group in groups.values() if len(group) > 1]
    assert len(siblings) == 4 and all(len(group) == 4 for group in siblings)
    assert all(len({node.prefix for node in group}) == 1 for group in siblings)
    assert set(completed) == set(tree.leaves)
    assert len(streamed) == 13
    for leaf, nodes in streamed:
        assert tuple(node.node_id for node in nodes) == leaf.path
        assert nodes == tuple(by_id[node_id] for node_id in leaf.path)
        assert nodes[-1].terminal
    credit = node_credit(1.0, 0.0)(tree, {}, {})
    assert sum(len(group) == 4 for group in credit['groups'].values()) == 4
    assert from_dict(Tree, to_dict(tree)) == tree


@pytest.mark.parametrize('v2', [False, True])
@pytest.mark.parametrize('turns', [1, 2, 3])
def test_short_reference_only_forks_at_visited_turns(v2, turns):
    tree = rollout(v2=v2, stop_rule=lambda depth, *args: 'stop' if depth == turns else None)
    assert len(tree.leaves) == 1 + 3 * turns
    assert all(len(leaf.path) == turns and leaf.stop_reason == 'stop' for leaf in tree.leaves)
    assert max(node.depth for node in tree.nodes) == turns


def test_doctor_halt_preserves_last_visible_turn_and_streamed_ancestors():
    events, streamed = [], []
    tree = rollout(doctor=Doctor(events, True, halt_at=2),
                   on_leaf_nodes=lambda leaf, nodes: streamed.append((leaf, nodes)))
    assert len(tree.leaves) == 7
    assert all(leaf.stop_reason == 'doctor_halt' for leaf in tree.leaves)
    assert all(leaf.dialogue[-2][:3] == ('doctor', 'D3', 3) for leaf in tree.leaves)
    assert all(nodes[0].prefix_doctor_status == STATUS for leaf, nodes in streamed)


def test_alternatives_can_continue_past_early_reference_and_stop_independently():
    reference_seed = derive_seed(11, 'c/0', 'agent', 0)
    tree = rollout(stop_rule=lambda depth, doctor, reply: 'reference_stop' if reply == f'reply {reference_seed}' else None)
    assert len(tree.leaves) == 4
    assert len(tree.leaves[0].path) == 1
    assert all(len(leaf.path) == 4 for leaf in tree.leaves[1:])
    assert all(len([piece for piece in leaf.leaf_id.split('.')[1:] if piece != '0']) == 0 for leaf in tree.leaves)


@pytest.mark.parametrize('branching', [1, 4])
def test_single_branch_and_frozen_opening(branching):
    events, openings = [], []
    def opening(case, seed):
        openings.append(seed)
        return {'visible': 'frozen', 'status': STATUS}
    tree = build_tree(CASE, max_agent_turns=4, branching=branching, run_seed=11, policy_sha256='policy',
                      protocol_sha256='protocol', agent=Agent(events), doctor=Doctor(events, True),
                      stop_rule=lambda *args: None, final_option=lambda text: text, outcome_fn=lambda *args: 1.0,
                      cache=CallCache(None), opening=opening, budget='reference_spine')
    assert len(tree.leaves) == 1 + 4 * (branching - 1) and len(openings) == 1
    assert not any(event[:2] == ('doctor', 1) for event in events)


def test_cache_resume_and_legacy_budget_are_separate(tmp_path):
    path = tmp_path / 'calls.jsonl'
    events = []
    tree = rollout(events=events, cache=CallCache(path))
    restored = rollout(events=events, cache=CallCache(path), workers=3, agent_workers=3)
    assert restored == tree
    assert len([event for event in events if event[0] == 'agent']) == 34
    legacy = rollout(cache=CallCache(path), budget='full_tree')
    assert len(legacy.leaves) == 256 and len(legacy.nodes) == 340
    assert from_dict(Tree, {key: value for key, value in to_dict(legacy).items() if key != 'rollout_budget'}) == legacy


def test_config_inheritance_and_protocol_identity():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config('configs/online_parallel_stream_score.json')
    assert cfg['rollout']['budget'] == 'reference_spine'
    for path in (root / 'configs').glob('online*.json'):
        variant = load_config(path)
        assert variant['rollout']['budget'] == 'reference_spine'
        assert variant['rollout']['halting']['min_agent_turns'] == 2
        assert variant['experiment']['output_root'].endswith('_reference13')
    reference_sha = protocol_sha256(cfg, root)
    legacy = copy.deepcopy(cfg)
    legacy['rollout']['budget'] = 'full_tree'
    assert protocol_sha256(legacy, root) != reference_sha
    cfg['rollout']['branching'] = 3
    assert protocol_sha256(cfg, root) != reference_sha
    cfg['rollout_protocol_sha256_override'] = 'a' * 64
    assert protocol_sha256(cfg, root) != 'a' * 64
    cfg['rollout']['budget'] = 'unknown'
    with pytest.raises(ValueError, match='rollout.budget'):
        protocol_sha256(cfg, root)


def test_invalid_budget_rejected_before_sampling():
    events = []
    with pytest.raises(ValueError, match='rollout.budget'):
        rollout(events=events, budget='unknown')
    assert events == []


@pytest.mark.parametrize('v2', [False, True])
def test_repeat_rule_uses_shared_prefix_and_stops_reference_at_second_turn(v2):
    class RepeatingAgent(Agent):
        def sample(self, case, prefix, turn, seed):
            return {'completion_text': 'Same answer.'}
    tree = build_tree(CASE, max_agent_turns=4, branching=4, run_seed=11, policy_sha256='policy',
                      protocol_sha256='protocol', agent=RepeatingAgent([]), doctor=Doctor([], v2),
                      stop_rule=lambda *args: None, final_option=lambda text: text, outcome_fn=lambda *args: 1.0,
                      cache=CallCache(None), budget='reference_spine', repeat_agent_reply_after_turns=2)
    assert len(tree.leaves) == 7
    assert all(leaf.stop_reason == 'agent_repeated_reply' for leaf in tree.leaves)


def test_minimum_turn_rule_applies_to_alternatives():
    doctor = Doctor([], True)
    doctor.min_agent_turns = 2
    tree = rollout(doctor=doctor, stop_rule=lambda *args: 'stop')
    assert len(tree.leaves) == 7
    assert all(len(leaf.path) == 2 for leaf in tree.leaves)


def test_resume_after_partial_alternative_failure_reuses_reference(tmp_path):
    path = tmp_path / 'calls.jsonl'
    cache = CallCache(path)
    events = []
    failed_seed = derive_seed(11, 'c/2', 'agent', 0)
    class FailingAgent(Agent):
        def sample(self, case, prefix, turn, seed):
            if seed == failed_seed:
                raise RuntimeError('sampling failed')
            return super().sample(case, prefix, turn, seed)
    with pytest.raises(RuntimeError, match='sampling failed'):
        build_tree(CASE, max_agent_turns=4, branching=4, run_seed=11, policy_sha256='policy',
                   protocol_sha256='protocol', agent=FailingAgent(events), doctor=Doctor(events, True),
                   stop_rule=lambda *args: None, final_option=lambda text: text, outcome_fn=lambda *args: 1.0,
                   cache=cache, budget='reference_spine')
    tree = rollout(events=events, cache=CallCache(path))
    assert len(tree.leaves) == 13
    assert len([event for event in events if event[0] == 'agent']) == 34
    assert len([event for event in events if event[0] == 'final']) == 13
