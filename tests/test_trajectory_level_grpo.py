from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.rollout import build_tree
from src.online_rl.tree.team import trajectory_credit
from src.online_rl.tree.batch import sibling_groups
from test_tree_v1 import CASES, Log, mock_env


def independent_tree():
    env=mock_env(Log(),{'v':0},k_conds=('never',))
    return build_tree(CASES[0],max_agent_turns=3,branching=4,run_seed=9,
        policy_sha256='fixed',protocol_sha256='trajectory-test',agent=env.agent,
        doctor=env.doctor,stop_rule=lambda *a: None,final_option=env.final_option,
        outcome_fn=env.outcome_fn,cache=CallCache(None),sampling_unit='independent_trajectory')


def test_independent_rollout_has_four_paths_not_exponential_tree():
    tree=independent_tree()
    assert len(tree.nodes)==12
    assert len(tree.leaves)==4
    assert all(len(leaf.path)==3 for leaf in tree.leaves)
    assert sorted(n.depth for n in tree.nodes).count(1)==4
    assert sorted(n.depth for n in tree.nodes).count(2)==4
    assert sorted(n.depth for n in tree.nodes).count(3)==4


def test_trajectory_reward_and_group_carry_every_agent_trace():
    tree=independent_tree()
    outcomes={leaf.leaf_id:float(i%2) for i,leaf in enumerate(tree.leaves)}
    ra={nid:(1.0 if i%2 else 0.0) for i,nid in enumerate(tree.leaves[0].path)}
    targets=trajectory_credit(.5,.5)(tree,outcomes,ra)
    assert targets['mode']=='trajectory_level'
    assert len(targets['trajectories'])==4
    groups=sibling_groups(tree,targets)
    assert len(groups)==1 and len(groups[0])==4
    assert all(len(record['agent_traces'])==3 for record in groups[0])
    assert [r['reward'] for r in groups[0]][1::2]==[.5,.5]


def test_rubric_only_trajectory_reward_ignores_outcome_and_excludes_unscored_paths():
    tree=independent_tree()
    outcomes={leaf.leaf_id:float(i%2) for i,leaf in enumerate(tree.leaves)}
    first, second, third, fourth=tree.leaves
    ra={first.path[0]:0.0, second.path[0]:1.0, third.path[0]:0.5}
    targets=trajectory_credit(0.0,1.0)(tree,outcomes,ra)['trajectories']
    assert targets[first.leaf_id]['reward']==0.0
    assert targets[second.leaf_id]['reward']==1.0
    assert targets[third.leaf_id]['reward']==0.5
    assert targets[fourth.leaf_id]['reward'] is None
