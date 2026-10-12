from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.rollout import build_tree
from src.online_rl.tree.team import trajectory_credit
from src.online_rl.tree import experiment
from src.online_rl.tree.backends import RoleDoctor
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


def test_experiment_batch_validation_accepts_trajectory_targets():
    tree=independent_tree()
    outcomes={leaf.leaf_id:float(i%2) for i,leaf in enumerate(tree.leaves)}
    ra={tree.leaves[0].path[0]:0.0, tree.leaves[1].path[0]:1.0,
        tree.leaves[2].path[0]:0.5}
    targets=trajectory_credit(0.0,1.0)(tree,outcomes,ra)
    cfg={'experiment':{}, 'agent_reward':{'w_outcome':0.0,'w_rubric':1.0},
         'rubric':{'enabled':True,'prompt_sha256':'fixed'},
         'prompt_update':{'enabled':True}}
    rows=experiment.check_batch(cfg,[{'case_cursor':0,'case_id':tree.case_id,
                                      'agent_targets':targets}])
    assert len(rows)==4
    assert all(row['training_unit']=='trajectory' and 'trajectory_id' in row and 'node_id' not in row
               for row in rows)
    assert sorted(row['final_reward'] for row in rows if row['final_reward'] is not None)==[0.0,0.5,1.0]


def test_independent_trajectories_share_opening_and_receive_fixed_followup_roles():
    roles=['evidence_request','counterargument','uncertainty_check','conclusion_link']
    seen=[]
    class Doctor:
        status_protocol=True
        min_agent_turns=99
        def turn(self, case, dialogue, round_, rounds, seed, followup_role=None):
            seen.append((round_,followup_role))
            return {'visible':f'D{round_}:{followup_role}','status':None,'status_error':None,'seed':seed}
        def final(self, case, dialogue, seed): return 'option A'
        def halt(self, **kwargs): return None
    env=mock_env(Log(),{'v':0},k_conds=('never',))
    tree=build_tree(CASES[0],max_agent_turns=2,branching=4,run_seed=9,
        policy_sha256='fixed',protocol_sha256='trajectory-role-test',agent=env.agent,
        doctor=Doctor(),stop_rule=lambda *a:None,final_option=env.final_option,
        outcome_fn=env.outcome_fn,cache=CallCache(None),sampling_unit='independent_trajectory',
        doctor_followup_roles=roles)
    assert seen[0]==(1,None)
    assert [role for round_,role in seen if round_==2]==roles
    assert len(tree.leaves)==4


def test_role_doctor_forwards_followup_role():
    seen=[]
    class Inner:
        model='doctor'; protocol='protocol'
        def turn(self, *args, **kwargs):
            seen.append(kwargs.get('followup_role')); return {'visible':'ok'}
        def final(self, *args): return 'option A'
        def halt(self, **kwargs): return None
    RoleDoctor(Inner()).turn({},[],2,4,7,followup_role='counterargument')
    assert seen==['counterargument']
