from src.online_rl.tree.prompt_search import paired_acceptance

ACC={'minimum_per_source':1,'minimum_trajectories':4,'bootstrap_seed':0,'bootstrap_n':20,'delta':.01,'p_min':.9,'tie_break':'candidate'}
OUTCOMES={'s1':'success','s2':'success','f1':'failure','f2':'failure'}


def test_complete_zero_tie_prefers_candidate_without_claiming_improvement():
    values={k:0. for k in OUTCOMES}
    result=paired_acceptance(values,values,ACC,OUTCOMES)
    assert result['accepted'] and result['J_gain']==0 and result['p_gain'] is None
    assert 'tie' in result['reason']


def test_incomplete_not_a_tie():
    values={k:0. for k in OUTCOMES}
    assert not paired_acceptance(values,{'s1':0.},ACC,OUTCOMES)['accepted']


def test_worse_candidate_cannot_win_tie_rule():
    parent={k:1. for k in OUTCOMES};candidate={k:.5 for k in OUTCOMES}
    assert not paired_acceptance(parent,candidate,ACC,OUTCOMES)['accepted']


def test_minimum_count_still_required():
    values={k:0. for k in OUTCOMES}
    assert not paired_acceptance(values,values,{**ACC,'minimum_trajectories':5},OUTCOMES)['accepted']
