from types import SimpleNamespace
import pytest
from src.online_rl.tree.prompt_search import objective, paired_acceptance, stratified_split, evidence_id
from src.online_rl.tree.team import trajectory_score


def test_balanced_candidate_beats_one_sided_candidate():
    outcomes = {'s': 'success', 'f': 'failure'}
    assert objective({'s': .9, 'f': .2}, outcomes)['J'] == pytest.approx(.18)
    assert objective({'s': .6, 'f': .6}, outcomes)['J'] == pytest.approx(.36)


def test_means_include_subthreshold_and_rejected_rubrics():
    result = objective({'s1': 1/3, 's2': 0, 'f': 1}, {'s1':'success','s2':'success','f':'failure'})
    assert result['J'] == pytest.approx(1/6)
    assert result['counts'] == {'success':2, 'failure':1}


def test_missing_sources_or_incomplete_evidence_cannot_score():
    assert objective({'s':1}, {'s':'success'})['J'] is None
    assert objective({'s':1}, {'s':'success','f':'failure'})['status'] == 'incomplete'
    assert trajectory_score({'effectiveness':SimpleNamespace(status='incomplete', R_O=None)}) is None
    assert trajectory_score({'effectiveness':None}) == 0


def test_acceptance_tests_product_gain_not_mean_gain():
    outcomes = {'s':'success','f':'failure'}
    cfg = {'minimum_trajectories':2, 'delta':.01, 'p_min':.95, 'bootstrap_n':100, 'bootstrap_seed':7}
    result = paired_acceptance({'s':.9,'f':.2}, {'s':.6,'f':.6}, cfg, outcomes)
    assert result['accepted'] and result['J_gain'] == pytest.approx(.18)
    assert not paired_acceptance({'s':.6,'f':.6}, {'s':.9,'f':.2}, cfg, outcomes)['accepted']
    assert not paired_acceptance({'s':.6,'f':.6}, {'s':.6,'f':.6}, cfg, outcomes)['accepted']
    assert paired_acceptance({'s':.6,'f':.6}, {'s':.9}, cfg, outcomes)['status'] == 'deferred'


def test_stratified_split_has_both_sources_and_is_order_independent(monkeypatch):
    import src.online_rl.tree.prompt_search as module
    monkeypatch.setattr(module, 'evidence_id', lambda unit: unit.uid)
    units = [SimpleNamespace(uid=f'{outcome}{i}', case_id='c', leaf=SimpleNamespace(leaf_id=f'{outcome}{i}'),
                             tin=SimpleNamespace(outcome=outcome))
             for outcome in ('success','failure') for i in range(5)]
    split = stratified_split(units, 123, .5, 2)
    assert split == stratified_split(list(reversed(units)), 123, .5, 2)
    for side in ('search', 'acceptance'):
        for outcome in ('success', 'failure'):
            assert sum(split[u.uid] == side and u.tin.outcome == outcome for u in units) >= 2
    assert stratified_split(units[:5], 123, .5) is None


def test_bootstrap_is_reproducible_and_uses_all_planned_units():
    outcomes = {'s1':'success','s2':'success','f1':'failure','f2':'failure'}
    parent = dict(s1=1.,s2=0.,f1=0.,f2=.2)
    candidate = dict(s1=.5,s2=.8,f1=.9,f2=.5)
    cfg = dict(minimum_trajectories=4,delta=0,p_min=.5,bootstrap_n=200,bootstrap_seed=11)
    assert paired_acceptance(parent,candidate,cfg,outcomes) == paired_acceptance(parent,candidate,cfg,outcomes)
    assert not paired_acceptance(parent,candidate,{**cfg,'minimum_per_source':3},outcomes)['accepted']
