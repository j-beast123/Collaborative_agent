"""A shared scoring rubric must not share a trajectory-specific intervention.

No model/API calls: these tests inspect the actual writer/continuation arguments,
the persisted per-member contract, and the real rerun cache.
"""
import copy
import json
from dataclasses import replace

import pytest

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import effectiveness as EF
from src.online_rl.tree import rubric_bank as RB
from src.online_rl.tree import witness as W
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies
from src.online_rl.tree.member_intervention import make_member_intervention
from src.online_rl.tree.records import (AgentNode, Leaf, RubricAssignment,
                                        TrajectoryRubric, TrajectoryRubricInput,
                                        to_dialogue)


CASE = {'case_id': 'member-case', 'scenario': '', 'question': 'Which option fits?',
        'options': {'A': 'First possibility', 'B': 'Second possibility'},
        'ground_truth': {'correct_option': 'A'}}
POLICY = 'fixed-policy'


def fixture():
    first_prefix = (('doctor', 'D1', 1, 'Please compare the stated alternatives.'),)
    second_prefix = first_prefix + (('agent', 'A1', 1, 'I prefer the first one.'),
                                    ('doctor', 'D2', 2, 'But explain their relevant difference.'))
    first = AgentNode('member-case/0', None, CASE['case_id'], 1, first_prefix,
                      'I prefer the first one.', {}, POLICY, 10, None, False,
                      'But explain their relevant difference.')
    nodes = {first.node_id: first}
    leaves = []
    for i in (0, 1):
        last = AgentNode(f'member-case/0.{i}', first.node_id, CASE['case_id'], 2,
                         second_prefix, f'Original second response {i}.', {}, POLICY,
                         20 + i, 'max_turns', True, None)
        nodes[last.node_id] = last
        dialogue = second_prefix + (('agent', 'A2', 2, last.response),
                                    ('doctor', 'FINAL', 2, 'B'))
        leaves.append(Leaf(last.node_id, CASE['case_id'], (first.node_id, last.node_id),
                           dialogue, 'B', 'B', 0.0, 'max_turns'))
    rubric = TrajectoryRubric('canonical-r', leaves[0].leaf_id, 'prompt-hash', None,
        {'rubric_id': 'canonical-r', 'name': 'Compare the two stated alternatives',
         'applicability_condition': 'The clinician explicitly requests a comparison.',
         'required_behavior': 'Explain the relevant difference between both alternatives.',
         'scores': {'0': 'No comparison.', '1': 'One relevant comparison with a remaining gap.',
                    '2': 'Both alternatives compared consistently against the requested criterion.'},
         'source_provenance': {'analysis': {'selected_agent_turn_id': 1}}},
        True, (), None, intervention_instruction='Compare both alternatives at the first request.')
    return nodes, leaves, rubric


def spec(rubric, leaf, turn, instruction, origin='matched'):
    tin = TrajectoryRubricInput(leaf.leaf_id, CASE['case_id'], '', CASE['question'],
                               tuple(CASE['options'].items()), leaf.dialogue,
                               'success' if leaf.outcome == 1 else 'failure', '',
                               sha256_json(leaf.dialogue))
    return make_member_intervention(rubric, tin, selected_agent_turn_id=turn,
                                    intervention_instruction=instruction, origin=origin)


class Execution:
    intervention_instruction_template = 'Perform this behavior: {intervention_instruction}'

    def __init__(self):
        self.agent = []
        self.doctor = []

    def write(self, case, prefix, turn, rubric, level, seed, *, intervention_instruction=None):
        self.agent.append({'prefix': copy.deepcopy(prefix), 'turn': turn,
                           'rubric': copy.deepcopy(rubric), 'level': level,
                           'seed': seed, 'instruction': intervention_instruction})
        return f'Replacement response with independent seed {seed}.'

    def final(self, case, prefix, turn, reply, seed):
        self.doctor.append({'prefix': copy.deepcopy(prefix), 'turn': turn,
                            'reply': reply, 'seed': seed})
        return {'final_option': 'A', 'outcome': 1.0}


def run(execution, cache, nodes, leaf, rubric, member_spec, *, policy=POLICY):
    node = nodes[leaf.path[member_spec['selected_agent_turn_id'] - 1]]
    return EF.run_interventions(CASE, leaf, node, rubric, reruns=3, run_seed=17,
                                policy_sha256=policy, writer=execution.write,
                                continuation=execution.final, cache=cache,
                                member_spec=member_spec)


def score_all(view, prefix, rubrics, reply):
    return {r['rubric_id']: {'applicable': True, 'hard': 2,
                              'probabilities': {'0': 0.0, '1': 0.0, '2': 1.0}}
            for r in rubrics}


def test_same_canonical_uses_member_turn_and_instruction_without_mutating_rubric():
    nodes, leaves, rubric = fixture()
    before = copy.deepcopy(rubric)
    first = spec(rubric, leaves[0], 1, rubric.intervention_instruction, 'generated')
    second = spec(rubric, leaves[1], 2, 'Address the second request with a specific comparison.')
    execution = Execution()
    for leaf, member_spec in zip(leaves, (first, second)):
        runs = run(execution, CallCache(None), nodes, leaf, rubric, member_spec)
        assert len(runs) == 3
        assert {r.turn for r in runs} == {member_spec['selected_agent_turn_id']}
        assert {r.leaf_id for r in runs} == {leaf.leaf_id}
    assert [r['turn'] for r in execution.agent] == [1, 1, 1, 2, 2, 2]
    assert [r['instruction'] for r in execution.agent] == [first['intervention_instruction']] * 3 + [second['intervention_instruction']] * 3
    assert execution.agent[-1]['prefix'] == to_dialogue(nodes[leaves[1].path[1]].prefix)
    assert rubric == before


def test_three_independent_reruns_reuse_the_exact_member_guide_only():
    nodes, leaves, rubric = fixture()
    member_spec = spec(rubric, leaves[1], 2, 'Compare the specific alternatives in the current request.')
    execution = Execution()
    run(execution, CallCache(None), nodes, leaves[1], rubric, member_spec)
    assert len(execution.agent) == len(execution.doctor) == 3
    assert len({r['seed'] for r in execution.agent}) == 3
    assert len({r['seed'] for r in execution.doctor}) == 3
    assert {r['instruction'] for r in execution.agent} == {member_spec['intervention_instruction']}
    assert all(member_spec['intervention_instruction'] not in json.dumps(r) for r in execution.doctor)


def test_reused_trajectory_without_member_guide_fails_before_any_model_call():
    nodes, leaves, rubric = fixture()
    execution = Execution()
    with pytest.raises((ValueError, RuntimeError), match='member|source|instruction'):
        EF.run_interventions(CASE, leaves[1], nodes[leaves[1].path[0]], rubric,
                             reruns=3, run_seed=17, policy_sha256=POLICY,
                             writer=execution.write, continuation=execution.final,
                             cache=CallCache(None))
    assert execution.agent == execution.doctor == []


@pytest.mark.parametrize('field,value', [('leaf_id', 'other-leaf'),
                                       ('case_id', 'other-case'),
                                       ('requested_level', '0'),
                                       ('rubric_sha256', 'old-rubric'),
                                       ('prefix_sha256', 'other-prefix')])
def test_incompatible_member_specs_are_rejected_before_execution(field, value):
    nodes, leaves, rubric = fixture()
    member_spec = spec(rubric, leaves[1], 2, 'Compare both options for this exact request.')
    member_spec[field] = value
    execution = Execution()
    with pytest.raises((ValueError, RuntimeError)):
        run(execution, CallCache(None), nodes, leaves[1], rubric, member_spec)
    assert execution.agent == execution.doctor == []


def test_member_spec_cache_survives_resume_and_new_guide_invalidates_it(tmp_path):
    nodes, leaves, rubric = fixture()
    path = tmp_path / 'calls.jsonl'
    first = spec(rubric, leaves[1], 2, 'Compare the two alternatives with one explicit criterion.')
    execution = Execution()
    saved = run(execution, CallCache(path), nodes, leaves[1], rubric, first)
    assert len(execution.agent) == 3
    resumed = Execution()
    again = run(resumed, CallCache(path), nodes, leaves[1], rubric, first)
    assert saved == again and resumed.agent == resumed.doctor == []
    changed = spec(rubric, leaves[1], 2, 'State the distinguishing relationship for each alternative.')
    run(resumed, CallCache(path), nodes, leaves[1], rubric, changed)
    assert len(resumed.agent) == len(resumed.doctor) == 3
    assert {r['instruction'] for r in resumed.agent} == {changed['intervention_instruction']}


def test_evaluation_targets_new_member_turn_and_keeps_canonical_source_identity():
    nodes, leaves, rubric = fixture()
    member_spec = spec(rubric, leaves[1], 2, 'Compare the relevant difference requested in this second turn.')
    execution = Execution()
    _, effect = EF.evaluate_rubric(CASE, nodes, leaves[1], rubric,
        scorer=score_all, writer=execution.write, continuation=execution.final,
        reruns=3, run_seed=17, policy_sha256=POLICY,
        strategies=Strategies(multi_target=lambda values: sum(values.values()) / len(values)),
        cache=CallCache(None), member_spec=member_spec)
    assert effect.leaf_id == leaves[1].leaf_id
    assert effect.target_node_ids == (leaves[1].path[1],)
    assert len(effect.runs) == 3 and {r['turn'] for r in execution.agent} == {2}
    assert rubric.leaf_id == leaves[0].leaf_id


def test_all_member_verification_passes_each_persisted_instruction():
    nodes, leaves, rubric = fixture()
    first = spec(rubric, leaves[0], 1, rubric.intervention_instruction, 'generated')
    second = spec(rubric, leaves[1], 2, 'Compare both alternatives at the second request.')
    assignments = [RubricAssignment(leaf.leaf_id, CASE['case_id'], 'failure', i,
        rubric.rubric_id, 'CREATED_NEW' if i == 0 else 'REUSED_EXISTING', None, '', 'feedback',
        intervention_spec=member_spec) for i, (leaf, member_spec) in enumerate(zip(leaves, (first, second)))]
    execution = Execution()
    _, effects = RB.verify_members(CASE, nodes, {leaf.leaf_id: leaf for leaf in leaves},
        {rubric.rubric_id: rubric}, assignments, scorer=score_all, writer=execution.write,
        continuation=execution.final, reruns=3, run_seed=17, policy_sha256=POLICY,
        strategies=Strategies(multi_target=lambda values: sum(values.values()) / len(values)),
        cache=CallCache(None))
    assert effects[leaves[0].leaf_id].target_node_ids == (leaves[0].path[0],)
    assert effects[leaves[1].leaf_id].target_node_ids == (leaves[1].path[1],)
    assert [r['instruction'] for r in execution.agent] == [first['intervention_instruction']] * 3 + [second['intervention_instruction']] * 3
    assert rubric.leaf_id == leaves[0].leaf_id


def test_bounded_witness_uses_member_spec_instead_of_canonical_or_old_feedback_turn():
    nodes, leaves, rubric = fixture()
    first = spec(rubric, leaves[0], 1, rubric.intervention_instruction, 'generated')
    second = spec(rubric, leaves[1], 2, 'Complete the comparison explicitly requested at the later turn.')
    assignments = [RubricAssignment(leaf.leaf_id, CASE['case_id'], 'failure', i,
        rubric.rubric_id, 'CREATED_NEW' if i == 0 else 'REUSED_EXISTING', None, '', 'feedback',
        intervention_spec=member_spec) for i, (leaf, member_spec) in enumerate(zip(leaves, (first, second)))]

    class ByTurn(Execution):
        def final(self, case, prefix, turn, reply, seed):
            super().final(case, prefix, turn, reply, seed)
            return {'final_option': 'A' if turn == 2 else 'B', 'outcome': float(turn == 2)}

    execution = ByTurn()
    result = W.verify_witness(CASE, nodes, {leaf.leaf_id: leaf for leaf in leaves},
        {rubric.rubric_id: rubric}, assignments,
        {leaves[0].leaf_id: 2, leaves[1].leaf_id: 1},
        applicability=lambda view, prefix, rubrics: {r['rubric_id']: {'applicable': True} for r in rubrics},
        writer=execution.write, continuation=execution.final, reruns=3,
        run_seed=17, policy_sha256=POLICY, cache=CallCache(None), max_valid_supports=5)
    witness = result[rubric.rubric_id]
    assert witness.decision == 'KEEP' and witness.validated_by_trajectory_id == leaves[1].leaf_id
    assert [row['target_reply_id'] for row in witness.tested] == [1, 2]
    assert [row['instruction'] for row in execution.agent] == [first['intervention_instruction']] * 2 + [second['intervention_instruction']] * 2


def test_old_bank_missing_later_member_spec_cannot_hide_behind_early_keep():
    nodes, leaves, rubric = fixture()
    first = spec(rubric, leaves[0], 1, rubric.intervention_instruction, 'generated')
    source = RubricAssignment(leaves[0].leaf_id, CASE['case_id'], 'failure', 0,
        rubric.rubric_id, 'CREATED_NEW', None, '', 'feedback', intervention_spec=first)
    # A legacy persisted assignment must remain distinguishable from the source's guide.
    legacy_member = replace(source, leaf_id=leaves[1].leaf_id, order=1,
                            assignment='REUSED_EXISTING', intervention_spec=None)
    execution = Execution()  # Would immediately pass if only the first member were checked.
    with pytest.raises((ValueError, RuntimeError), match='member|instruction|spec'):
        W.verify_witness(CASE, nodes, {leaf.leaf_id: leaf for leaf in leaves},
            {rubric.rubric_id: rubric}, [source, legacy_member],
            {leaf.leaf_id: 1 for leaf in leaves},
            applicability=lambda view, prefix, rubrics: {r['rubric_id']: {'applicable': True} for r in rubrics},
            writer=execution.write, continuation=execution.final, reruns=3,
            run_seed=17, policy_sha256=POLICY, cache=CallCache(None), max_valid_supports=5)
    assert execution.agent == execution.doctor == []
