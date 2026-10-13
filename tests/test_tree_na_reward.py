"""Missing rubric assessments must not become negative evidence in rubric-only GRPO."""
from types import SimpleNamespace
import pytest
import test_tree_v1 as F
from src.online_rl.tree import reward, team
from src.online_rl.integrated import grpo


def test_rubric_only_distinguishes_na_from_true_zero():
    tree, _ = F.tree_for(F.Log(), rounds=1, branching=4)
    ids = [n.node_id for n in tree.nodes]
    targets = team.node_credit(0.0, 1.0)(tree, reward.leaf_outcomes(tree), {ids[0]: 0.0, ids[1]: 1.0, ids[2]: None})
    assert [targets['nodes'][n]['reward'] for n in ids] == [0.0, 1.0, None, None]
    assert sum(len(g) for g in targets['groups'].values()) == 4


@pytest.mark.parametrize('weights', [(1.0, 0.0), (0.5, 0.5)])
def test_active_outcome_reward_remains_available_without_rubric(weights):
    tree, _ = F.tree_for(F.Log(), rounds=1, branching=4)
    targets = team.node_credit(*weights)(tree, reward.leaf_outcomes(tree), {})
    for r in targets['nodes'].values():
        assert r['R_A'] is None
        assert r['reward'] == weights[0] * r['outcome_component']


@pytest.mark.parametrize('assessed', [{}, {'first': 1.0}])
def test_na_siblings_do_not_supply_second_rewarded_response(monkeypatch, assessed):
    tree, _ = F.tree_for(F.Log(), rounds=1, branching=4)
    ra = {tree.nodes[0].node_id: assessed['first']} if assessed else {}
    targets = team.node_credit(0.0, 1.0)(tree, reward.leaf_outcomes(tree), ra)
    updater = object.__new__(grpo.IntegratedGRPOUpdater)
    updater.policy = SimpleNamespace(torch=None, model=object())
    updater.mode = 'mean_center'
    monkeypatch.setattr(grpo, 'adapter_sha256', lambda m: tree.policy_sha256)
    result = team.sibling_grpo_update(updater)(tree, targets)
    assert result == {'optimizer_step': False, 'skip_reason': 'fewer_than_two_rewarded_responses'}


def test_old_rubric_only_run_cannot_resume_with_new_reward_policy(tmp_path):
    import json
    cfg = F.cfg(cases=1)
    cfg['agent_reward'] = {'w_outcome': 0.0, 'w_rubric': 1.0}
    tr, _, _ = F.trainer(tmp_path, cfg)
    path = tmp_path / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['rubric_only_na_policy'] == team.RUBRIC_ONLY_NA_POLICY
    manifest.pop('rubric_only_na_policy')
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='NA reward policy'):
        tr._resume()
