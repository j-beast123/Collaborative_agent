"""Synthetic CPU-only regression for cumulative retrieval -> reward -> GRPO inputs.

No real judge, model, API, or optimizer is used. Production retrieval, score
assembly, reward composition, sibling grouping and advantage math are exercised.
"""
import json
from dataclasses import replace

import pytest

from src.online_rl.integrated.metrics import advantages
from src.online_rl.integrated.records import to_dict
from src.online_rl.tree import cumulative_bank as CB
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.five_gpu import update_batch
from src.online_rl.tree.pilot_evidence_scorer import evidence_scorer
from src.online_rl.tree.posthoc import case_view
from src.online_rl.tree.records import to_dialogue
from src.online_rl.tree.reward import agent_targets
from src.online_rl.tree.scoring import score_cumulative_nodes
from src.online_rl.tree.team import effectiveness_weighted_r_a, team_strategies
from tests.test_cumulative_bank import record as synthetic_bank_record
from tests.test_tree_v1 import CASES, Log, tree_for


class SyntheticEvidenceClient:
    """Known synthetic labels; these are not claims about clinical responses."""

    model = 'synthetic-judge-no-api'
    protocol_sha256 = 'synthetic-evidence-contract'

    def __init__(self, roles, response_ids):
        self.roles = roles
        self.response_ids = response_ids
        self.calls = []

    def judge(self, stage, state, items, fields, purpose):
        assert stage == 'applicability'
        assert 'target_agent_reply' not in state
        self.calls.append((stage, tuple(items)))
        return {rid: {
            'applicable': self.roles[rid] != 'na',
            'evidence_basis': 'Synthetic present trigger' if self.roles[rid] != 'na' else '',
            'missing_requirement': 'Synthetic absent trigger' if self.roles[rid] == 'na' else '',
        } for rid in items}, {}

    def judge_batch(self, stage, state, responses, items, fields, purpose):
        assert stage == 'level'
        assert len(responses) == 4
        self.calls.append((stage, tuple(items)))
        result = {}
        for response in responses:
            index = self.response_ids.index(response['response_id'])
            reply = response['target_agent_reply']
            result[response['response_id']] = {}
            for rid in items:
                role = self.roles[rid]
                if role == 'unscorable' and index == 0:
                    level = 'UNSCORABLE'
                elif role == 'unscorable':
                    level = '2'
                elif role == 'zero':
                    level = '0'
                elif role == 'varying':
                    level = ('0', '1', '2', '2')[index]
                else:
                    raise AssertionError('NA criteria must never reach behavior scoring')
                result[response['response_id']][rid] = {
                    'level': level,
                    'anchor_basis': 'Synthetic fixture: independently specified expected boundary.',
                    'minimum_contribution_quote': reply if level in ('1', '2') else '',
                    'completion_quote': reply if level == '2' else '',
                    'missing_or_conflicting': '' if level == '2' else 'Synthetic gap in the fixture.',
                }
        return result, {}


def scored_fixture(*, all_unassessed=False):
    tree, _ = tree_for(Log(), rounds=1, branching=4)
    outcomes = (0.0, 0.0, 1.0, 1.0) if all_unassessed else (0.0, 1.0, 0.0, 1.0)
    by_leaf = dict(zip((node.node_id for node in tree.nodes), outcomes))
    tree = replace(tree, leaves=tuple(replace(leaf, outcome=by_leaf[leaf.leaf_id]) for leaf in tree.leaves))
    bank = CB.extend_snapshot(CB.empty_snapshot(), [
        synthetic_bank_record(f'synthetic-source-{i}', keyword=f'synthetic-criterion-{i}')
        for i in range(9)
    ])
    assert len(bank['entries']) == 9
    query = json.dumps({'case': case_view(CASES[0]), 'prefix': to_dialogue(tree.nodes[0].prefix)},
                       sort_keys=True, ensure_ascii=False)
    selected = [entry['bank_id'] for entry in CB.shortlist(bank, query, top_k=8)]
    roles = dict.fromkeys(selected, 'na')
    roles[selected[0]] = 'unscorable' if all_unassessed else 'zero'
    if not all_unassessed:
        roles[selected[1]] = 'varying'
        roles[selected[2]] = 'unscorable'
    client = SyntheticEvidenceClient(roles, [node.node_id for node in tree.nodes])
    scorer = evidence_scorer(client, response_batch_size=4)
    scores, retrieval = score_cumulative_nodes(
        CASES[0], list(tree.nodes), bank, scorer, CallCache(None), top_k=8,
    )
    assert len(retrieval) == 1
    assert retrieval[0]['retrieval_ids'] == selected
    assert len(scores) == 9 * 4
    assert len([call for call in client.calls if call[0] == 'applicability']) == 8
    assert len([call for call in client.calls if call[0] == 'level']) == 1
    return tree, bank, scores, roles


class CaptureTrainingInputs:
    """Capture the actual production update_batch handoff, without an optimizer."""

    def update_groups(self, groups):
        self.groups = groups
        self.advantages = [advantages([row['reward'] for row in group], 'mean_center') for group in groups]
        return {'captured_groups': len(groups)}


def reward_and_capture(tree, bank, scores, weights):
    targets = agent_targets(tree, scores,
        {rubric.rubric_id: 1.0 for rubric in CB.scoring_rubrics(bank)},
        team_strategies(None, *weights))
    capture = CaptureTrainingInputs()
    update_batch(capture, [{'tree': to_dict(tree), 'policy_sha256': tree.policy_sha256,
                            'agent_targets': targets}], tree.policy_sha256, lambda: tree.policy_sha256)
    assert len(capture.groups) == 1 and len(capture.groups[0]) == 4
    for row, node in zip(capture.groups[0], tree.nodes):
        assert row['trajectory_id'] == node.node_id
        assert row['rollout_policy_sha256'] == tree.policy_sha256
        assert row['agent_traces'] == [node.trace]
    return targets, capture


def test_nine_criteria_top_eight_preserves_zero_and_excludes_all_unassessed_from_mean():
    tree, bank, scores, roles = scored_fixture()
    nr = [score for score in scores if score.scoring_status == 'not_retrieved']
    assert len(nr) == 4
    assert all(not score.match and score.score is score.hard is score.probabilities is None for score in nr)
    zero_id = next(rid for rid, role in roles.items() if role == 'zero')
    genuine_zeros = [score for score in scores if score.rubric_id == zero_id]
    assert len(genuine_zeros) == 4
    assert all(score.match and score.scoring_status == 'scored' and score.score == 0.0 for score in genuine_zeros)
    assert sum(score.scoring_status == 'not_applicable' for score in scores) == 5 * 4
    masked = [score for score in scores if score.scoring_status == 'group_unscorable']
    assert len(masked) == 4
    assert {score.individual_assessment['scoring_status'] for score in masked} == {'scored', 'rubric_unscorable'}

    targets, capture = reward_and_capture(tree, bank, scores, (0.5, 0.5))
    ordered = [targets['nodes'][node.node_id] for node in tree.nodes]
    # Only the genuine zero and varying rubric enter the mean's denominator.
    # Dividing by 8 or 9, or dropping genuine zero, would change these values.
    assert [row['R_A'] for row in ordered] == [0.0, 0.25, 0.5, 0.5]
    assert [row['reward'] for row in ordered] == [0.0, 0.625, 0.25, 0.75]
    assert [row['reward'] for row in capture.groups[0]] == [0.0, 0.625, 0.25, 0.75]
    assert capture.advantages == [[-0.40625, 0.21875, -0.15625, 0.34375]]


@pytest.mark.parametrize('weights,expected_rewards,expected_advantages', [
    ((0.5, 0.5), [0.0, 0.0, 0.5, 0.5], [-0.25, -0.25, 0.25, 0.25]),
    ((0.0, 1.0), [None, None, None, None], [None, None, None, None]),
])
def test_all_unassessed_preserves_outcome_fallback_and_rubric_only_exclusion(
        weights, expected_rewards, expected_advantages):
    tree, bank, scores, _ = scored_fixture(all_unassessed=True)
    assert {score.scoring_status for score in scores} == {
        'not_retrieved', 'not_applicable', 'group_unscorable',
    }
    assert all(score.score is None for score in scores)
    targets, capture = reward_and_capture(tree, bank, scores, weights)
    assert all(targets['nodes'][node.node_id]['R_A'] is None for node in tree.nodes)
    assert [row['reward'] for row in capture.groups[0]] == expected_rewards
    assert capture.advantages == [expected_advantages]


@pytest.mark.parametrize('corruption', [
    {'score': 0.0}, {'score': 0.5}, {'hard': 0.0},
    {'probabilities': {'0': 1.0, '1': 0.0, '2': 0.0}}, {'match': True},
])
def test_not_retrieved_cannot_hide_numeric_or_applicable_assessments(corruption):
    _, _, scores, _ = scored_fixture(all_unassessed=True)
    original = next(score for score in scores if score.scoring_status == 'not_retrieved')
    corrupt = replace(original, **corruption)
    with pytest.raises(RuntimeError):
        effectiveness_weighted_r_a([corrupt], {corrupt.rubric_id: 1.0})
