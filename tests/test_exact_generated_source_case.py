"""Offline checks for source-case generation before exact-content collapse."""
import copy
import dataclasses
import json

import pytest

from src.online_rl.integrated.records import sha256_json, to_dict
from src.online_rl.tree import posthoc, rubric_bank
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.member_intervention import resolve_member_intervention
from src.online_rl.tree.records import AgentNode, DoctorFeedback, Leaf, TrajectoryRubric, Tree
from src.online_rl.tree.trainer import TreeTrainer


CASE = {'case_id': 'case', 'question': 'Which option?', 'scenario': '',
        'options': {'A': 'a', 'B': 'b'}, 'ground_truth': {'correct_option': 'A'}}
PROMPT = 'Generate one criterion.\n[Assign or create]\nThis matching section must not be called.'


def sources():
    leaves, nodes, tins = [], {}, {}
    for i, outcome in enumerate((0.0, 0.0, 1.0, 1.0)):
        lid = f'L{i}'
        first = (('doctor', 'D1', 1, 'Explain the comparison.'),)
        second = first + (('agent', 'A1', 1, f'First answer {i}'),
                          ('doctor', 'D2', 2, f'Clarify the relation {i}.'))
        path = (f'{lid}/first', f'{lid}/second')
        for turn, prefix in enumerate((first, second), 1):
            nodes[path[turn - 1]] = AgentNode(path[turn - 1], None if turn == 1 else path[0],
                'case', turn, prefix, f'answer {i}', {}, 'policy', i, None, True, None)
        option = 'A' if outcome == 1.0 else 'B'
        dialogue = second + (('agent', 'A2', 2, f'Second answer {i}'), ('doctor', 'FINAL', 0, option))
        leaf = Leaf(lid, 'case', path, dialogue, option, option, outcome, 'max_turns')
        leaves.append(leaf)
        tins[lid] = posthoc.freeze_trajectory_input(CASE, leaf,
            DoctorFeedback(lid, outcome, f'Feedback {lid}', 'feedback', target_reply_id=1), True)
    return leaves, nodes, tins


def generated_criterion(tin, *, changed=False):
    app = {'user_side_context': 'Doctor requests a comparison.',
           'agent_side_state': 'No prior Agent state is required.'}
    levels = {'0': 'Gives no concrete comparison.',
              '1': 'Compares concrete features but leaves the relation unclear.',
              '2': 'Compares concrete features and explains their relation.'}
    if changed: levels['2'] += ' A changed completion condition.'
    return {'name': f'Comparison for {tin.leaf_id}', 'applicability': app,
            'required_behavior': 'Compare concrete features and explain their relation.',
            'minimum_performance': 'Compare at least two concrete features.',
            'completion_condition': 'Explain what the comparison shows.',
            'criterion_version': 'public_behavior_mc_semantic_v3',
            'scores': {k: {**app, 'behavior': v} for k, v in levels.items()},
            'source_provenance': {'analysis': {'selected_agent_turn_id': 2 if tin.leaf_id in ('L1', 'L3') else 1},
                                  'selected_turn_policy': 'generator_v1',
                                  'source_trajectory_id': tin.leaf_id, 'source_case_id': tin.case_id,
                                  'source_outcome': tin.outcome}}


def run(*, changed=(), rejected=(), abstained=()):
    leaves, nodes, tins = sources()
    calls = {'generator': [], 'gate': []}

    def generate(prompt, tin):
        calls['generator'].append(tin.leaf_id)
        rubric = generated_criterion(tin, changed=tin.leaf_id in changed)
        if tin.leaf_id in abstained:
            expected = 0 if tin.outcome == 'failure' else 2
            rubric['source_provenance'].update(
                source_anchor_contract_mismatch=True,
                source_anchor_declaration={'expected_score_from_outcome': expected,
                                           'generator_claimed_source_score': 1,
                                           'matches_expected': False})
        return {'rubric': rubric, 'raw': json.dumps(rubric),
                'intervention_instruction': f'Guide specifically written for {tin.leaf_id}.'}

    def gate(tin, rubric):
        calls['gate'].append(tin.leaf_id)
        return {'passed': tin.leaf_id not in rejected}

    def source_for(leaf):
        def check(rubric):
            return {'applicable': True,
                    'target_reply_id': rubric['source_provenance']['analysis']['selected_agent_turn_id']}
        return check

    cache = CallCache(None)
    result = rubric_bank.build_exact_generated_banks(CASE, leaves, tins, PROMPT, 'prompt',
        generate, gate, cache, source_stage1_for=source_for)
    return leaves, nodes, tins, result, calls, cache


def test_every_trajectory_is_generated_checked_then_exact_deduplicated_with_own_guide():
    leaves, nodes, _, (canon, assignments, candidates), calls, cache = run()
    assert len(candidates) == 4 and len(canon) == 2
    assert calls == {'generator': ['L0', 'L1', 'L2', 'L3'], 'gate': ['L0', 'L1', 'L2', 'L3']}
    assert cache.calls['rubric_generation'] == cache.calls['posthoc_gate'] == 4
    assert all(row['identity']['assignment_protocol'] == rubric_bank.EXACT_SOURCE_CASE_PROTOCOL
               for row in cache.store.values() if row['kind'] in ('rubric_generation', 'posthoc_gate'))
    assert [a.assignment for a in assignments] == ['CREATED_NEW', 'GENERATED_EXACT_DUPLICATE',
                                                     'CREATED_NEW', 'GENERATED_EXACT_DUPLICATE']
    for a, leaf in zip(assignments, leaves):
        spec = a.intervention_spec
        assert spec['intervention_instruction'] == f'Guide specifically written for {leaf.leaf_id}.'
        assert spec['selected_agent_turn_id'] == (2 if leaf.leaf_id in ('L1', 'L3') else 1)
        assert resolve_member_intervention(canon[a.canonical_rubric_id], leaf, spec,
                                           nodes[leaf.path[spec['selected_agent_turn_id'] - 1]]) == spec
    assert assignments[1].intervention_spec['origin'] == 'generated_exact_duplicate'
    assert assignments[1].intervention_spec['generated_candidate_proof']['candidate_sha256'] == sha256_json(to_dict(candidates[1]))
    assert assignments[0].canonical_rubric_id != assignments[2].canonical_rubric_id  # opposite outcome strata
    t = rubric_bank.assignment_telemetry(canon, assignments, candidates=candidates)
    assert (t['generation_calls'], t['exact_generated_duplicates'], t['semantic_matcher_calls'], t['gate_pass']) == (4, 2, 0, 4)


def test_changed_anchor_is_not_merged_and_invalid_candidate_is_not_reused():
    _, _, _, (canon, assignments, candidates), calls, _ = run(changed={'L1'}, rejected={'L2'})
    assert len(candidates) == 4 and len(canon) == 4
    assert [a.assignment for a in assignments] == ['CREATED_NEW', 'CREATED_NEW', 'CREATE_REJECTED', 'CREATED_NEW']
    assert assignments[2].intervention_spec is None
    assert rubric_bank.assignment_telemetry(canon, assignments, candidates=candidates)['gate_reject'] == 1
    assert len(calls['generator']) == len(calls['gate']) == 4


def test_source_anchor_mismatch_is_normal_abstention_and_skips_gate_and_rerun():
    leaves, nodes, _, (canon, assignments, candidates), calls, _ = run(abstained={'L0', 'L2'})
    by_leaf = {a.leaf_id: a for a in assignments}
    assert by_leaf['L0'].assignment == by_leaf['L2'].assignment == rubric_bank.NO_VALID_CANDIDATE
    assert by_leaf['L0'].intervention_spec is by_leaf['L2'].intervention_spec is None
    assert calls['generator'] == ['L0', 'L1', 'L2', 'L3']
    assert calls['gate'] == ['L1', 'L3']
    abstentions = [r for r in candidates if r.leaf_id in ('L0', 'L2')]
    assert all(not r.valid and r.gate is None and r.intervention_instruction is None for r in abstentions)
    assert all(r.source_check['status'] == rubric_bank.NO_VALID_CANDIDATE for r in abstentions)
    telemetry = rubric_bank.assignment_telemetry(canon, assignments, candidates=candidates)
    assert telemetry['no_valid_candidate'] == 2
    assert telemetry['create_rejected'] == telemetry['gate_reject'] == 0
    inactive = [by_leaf['L0'], by_leaf['L2']]
    scores, effs = rubric_bank.verify_members(CASE, nodes, {l.leaf_id: l for l in leaves}, canon, inactive,
        scorer=lambda *_: (_ for _ in ()).throw(AssertionError('scorer should not run for abstentions')),
        writer=lambda *_: None, continuation=lambda *_: None, reruns=3, run_seed=0,
        policy_sha256='policy', strategies=None, cache=CallCache(None))
    assert effs['L0'] is effs['L2'] is None


def test_no_valid_candidate_round_trip_is_accepted_by_resume_validator():
    leaves, nodes, tins, (canon, assignments, candidates), _, _ = run(abstained={'L0'})
    trainer = TreeTrainer.__new__(TreeTrainer)
    trainer.member_intervention_version = 'trajectory_member_instruction_exact_source_case_v1'
    trainer.reuse = trainer.exact_source_case = True
    trainer.cumulative_bank = None
    tree = Tree('case', 'policy', 2, 2, tuple(nodes.values()), tuple(leaves))
    record = {'tree': to_dict(tree),
              'feedback': [to_dict(DoctorFeedback(l.leaf_id, l.outcome, f'Feedback {l.leaf_id}',
                                                  'feedback', target_reply_id=1)) for l in leaves],
              'trajectory_rubrics': [to_dict(r) for r in canon.values()],
              'trajectory_inputs': [to_dict(t) for t in tins.values()],
              'rubric_bank': {'assignment_protocol': rubric_bank.EXACT_SOURCE_CASE_PROTOCOL,
                              'member_intervention_version': trainer.member_intervention_version,
                              'assignments': [to_dict(a) for a in assignments],
                              'generated_candidates': [to_dict(r) for r in candidates]}}
    trainer.validate_member_interventions(copy.deepcopy(record))
    row = next(a for a in record['rubric_bank']['assignments'] if a['leaf_id'] == 'L0')
    row['assignment'] = 'CREATE_REJECTED'
    with pytest.raises(RuntimeError, match='rejected candidate'):
        trainer.validate_member_interventions(record)


def test_generated_rubric_id_collision_fails_before_canonical_overwrite(monkeypatch):
    leaves, _, tins = sources()
    def collision(_prompt, _prompt_sha, tin, _generate, _cache, _gate, source_stage1=None):
        rubric = generated_criterion(tin, changed=tin.leaf_id == 'L1')
        rubric['rubric_id'] = 'colliding-id'
        return TrajectoryRubric('colliding-id', tin.leaf_id, 'prompt', 'raw', rubric, True, (),
                                {'passed': True}, {'status': 'PASS', 'first': {'applicable': True}},
                                f'Guide for {tin.leaf_id}')
    monkeypatch.setattr(rubric_bank, 'generate_rubric', collision)
    with pytest.raises(RuntimeError, match='rubric_id collision'):
        rubric_bank.build_exact_generated_banks(CASE, leaves, tins, PROMPT, 'prompt',
            lambda *_: None, lambda *_: {'passed': True}, CallCache(None),
            source_stage1_for=lambda leaf: lambda rubric: {'applicable': True})


def test_replay_checks_duplicate_candidate_not_only_its_synthetic_proof():
    leaves, nodes, tins, (canon, assignments, candidates), _, _ = run()
    trainer = TreeTrainer.__new__(TreeTrainer)
    trainer.member_intervention_version = 'trajectory_member_instruction_exact_source_case_v1'
    trainer.reuse = trainer.exact_source_case = True
    trainer.cumulative_bank = None
    tree = Tree('case', 'policy', 2, 2, tuple(nodes.values()), tuple(leaves))
    record = {'tree': to_dict(tree),
              'feedback': [to_dict(DoctorFeedback(l.leaf_id, l.outcome, f'Feedback {l.leaf_id}',
                                                  'feedback', target_reply_id=1)) for l in leaves],
              'trajectory_rubrics': [to_dict(r) for r in canon.values()],
              'trajectory_inputs': [to_dict(t) for t in tins.values()],
              'rubric_bank': {'assignment_protocol': rubric_bank.EXACT_SOURCE_CASE_PROTOCOL,
                              'member_intervention_version': trainer.member_intervention_version,
                              'assignments': [to_dict(a) for a in assignments],
                              'generated_candidates': [to_dict(r) for r in candidates]}}
    trainer.validate_member_interventions(record)
    bad = copy.deepcopy(record)
    row = bad['rubric_bank']['generated_candidates'][1]
    row['intervention_instruction'] = 'A different generated instruction.'
    with pytest.raises(RuntimeError, match='generated source'):
        trainer.validate_member_interventions(bad)
    bad = copy.deepcopy(record)
    bad['rubric_bank']['assignments'][1]['intervention_spec']['origin'] = 'matched'
    with pytest.raises(RuntimeError, match='generated source'):
        trainer.validate_member_interventions(bad)
