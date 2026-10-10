import dataclasses
import json

import pytest

from src.online_rl.integrated.records import from_dict, sha256_json, to_dict
from src.online_rl.tree import posthoc, rubric_bank
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.member_intervention import resolve_member_intervention, source_member_intervention
from src.online_rl.tree.records import AgentNode, DoctorFeedback, Leaf, RubricAssignment


CASE = {'case_id': 'case', 'question': 'Question?', 'scenario': '',
        'options': {'A': 'a', 'B': 'b'}, 'ground_truth': {'correct_option': 'A'}}
PROMPT = 'Generate rubric\n[Assign or create]\nMatch rubric'


def sources():
    leaves, nodes, tins = [], {}, {}
    for i in range(3):
        lid = f'L{i}'
        first = (('doctor', 'D1', 1, 'Explain the criterion.'),)
        second = first + (('agent', 'A1', 1, f'First answer {i}'), ('doctor', 'D2', 2, f'Clarify the relation {i}.'))
        path = (f'{lid}/first', f'{lid}/second')
        for turn, prefix in enumerate((first, second), 1):
            nodes[path[turn - 1]] = AgentNode(path[turn - 1], None if turn == 1 else path[0], 'case', turn,
                prefix, f'answer {i}', {}, 'policy', i, None, True, None)
        dialogue = second + (('agent', 'A2', 2, f'Second answer {i}'), ('doctor', 'FINAL', 0, 'B'))
        leaf = Leaf(lid, 'case', path, dialogue, 'B', 'B', 0.0, 'max_turns')
        leaves.append(leaf)
        tins[lid] = posthoc.freeze_trajectory_input(CASE, leaf,
            DoctorFeedback(lid, 0.0, f'Feedback {lid}', 'feedback', target_reply_id=1), True)
    return leaves, nodes, tins


def functions(batched, bad=None):
    calls = []
    def generate(prompt, tin):
        calls.append(('generate', tin.leaf_id))
        rubric = {'name': 'Explain relation', 'applicability_condition': 'Doctor requests explanation.',
                  'required_behavior': 'Explain the relation.', 'scores': {'0': 'No relation', '1': 'Partial relation', '2': 'Clear relation'},
                  'source_provenance': {'analysis': {'selected_agent_turn_id': 1}, 'selected_turn_policy': 'generator_v1',
                                        'source_trajectory_id': tin.leaf_id, 'source_case_id': tin.case_id,
                                        'source_outcome': tin.outcome}}
        return {'rubric': rubric, 'intervention_instruction': 'Explain the initial criterion.', 'raw': json.dumps(rubric)}
    def decision(lid, rid):
        result = {'decision': 'REUSE_EXISTING', 'rubric_id': rid,
                  'checks': {k: True for k in rubric_bank.CHECKS}, 'rationale': 'Same criterion, later interaction.',
                  'selected_agent_turn_id': 2, 'intervention_instruction': f'Explain the relation requested in {lid}.'}
        if bad == 'missing': result.pop('intervention_instruction')
        if bad == 'turn': result['selected_agent_turn_id'] = 9
        return result
    def single(package, user):
        calls.append(('match', user))
        # The previous feedback target is reference evidence, not a forced locus.
        assert 'Behavioral locus' not in user
        lid = json.loads(user.split('Source trajectory ID (JSON value; preserve type): ', 1)[1].splitlines()[0])
        bank = json.loads(user.rsplit('canonical rubrics):\n', 1)[1])
        return decision(lid, bank[0]['rubric_id'])
    def batch(package, items, bank):
        calls.append(('batch', [t.leaf_id for t in items]))
        return {'decisions': {tin.leaf_id: decision(tin.leaf_id, bank[0].rubric_id) for tin in items}}
    single.batch = batch
    single.batch_size = 8 if batched else 1
    single.member_interventions = True
    return generate, single, calls


@pytest.mark.parametrize('batched', [False, True])
def test_reuse_freezes_each_members_own_turn_and_guide_without_changing_rubric(batched):
    leaves, nodes, tins = sources()
    generate, match, calls = functions(batched)
    cache = CallCache(None)
    canon, assignments = rubric_bank.build_banks(CASE, leaves, tins, PROMPT, 'prompt', generate, match, None, cache)
    assert len(canon) == 1 and sum(call[0] == 'generate' for call in calls) == 1
    rub = next(iter(canon.values()))
    original = sha256_json(to_dict(rub))
    assert assignments[0].intervention_spec['origin'] == 'generated'
    assert assignments[0].intervention_spec['selected_agent_turn_id'] == 1
    for assignment, leaf in zip(assignments[1:], leaves[1:]):
        spec = assignment.intervention_spec
        assert spec['origin'] == 'matched' and spec['selected_agent_turn_id'] == 2
        assert spec['intervention_instruction'] == f'Explain the relation requested in {leaf.leaf_id}.'
        assert spec['intervention_instruction'] != rub.intervention_instruction
        assert spec['requested_level'] == '2' and spec['rubric_sha256'] == sha256_json(rub.rubric)
        assert resolve_member_intervention(rub, leaf, spec, nodes[leaf.path[1]]) == spec
        assert from_dict(RubricAssignment, to_dict(assignment)).intervention_spec == spec
    assert sha256_json(to_dict(rub)) == original
    count = len(calls)
    again = rubric_bank.build_banks(CASE, leaves, tins, PROMPT, 'prompt', generate, match, None, cache)
    assert len(calls) == count and again[1] == assignments


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('bad', ['missing', 'turn'])
def test_missing_or_invalid_reused_spec_fails_before_match_cache_write(batched, bad):
    leaves, _, tins = sources()
    generate, match, _ = functions(batched, bad)
    cache = CallCache(None)
    with pytest.raises(ValueError, match='Member'):
        rubric_bank.build_banks(CASE, leaves, tins, PROMPT, 'prompt', generate, match, None, cache)
    assert not any(r['kind'] in ('rubric_match', 'rubric_match_batch') for r in cache.store.values())


def test_source_fallback_cannot_be_used_for_other_member_or_changed_prefix():
    leaves, nodes, tins = sources()
    generate, match, _ = functions(False)
    canon, assignments = rubric_bank.build_banks(CASE, leaves, tins, PROMPT, 'prompt', generate, match, None, CallCache(None))
    rub = next(iter(canon.values()))
    assert source_member_intervention(rub, leaves[0]) == assignments[0].intervention_spec
    with pytest.raises(ValueError, match='another member'):
        source_member_intervention(rub, leaves[1])
    with pytest.raises(ValueError, match='requires a complete'):
        resolve_member_intervention(rub, leaves[1])
    node = nodes[leaves[1].path[1]]
    changed = dataclasses.replace(node, prefix=(('doctor', 'D2', 2, 'Different visible context'),))
    with pytest.raises(ValueError, match='prefix'):
        resolve_member_intervention(rub, leaves[1], assignments[1].intervention_spec, changed)


def test_relabeling_canonical_leaf_cannot_bypass_source_instruction_identity():
    leaves, _, tins = sources()
    generate, match, _ = functions(False)
    canon, assignments = rubric_bank.build_banks(CASE, leaves, tins, PROMPT, 'prompt', generate, match, None, CallCache(None))
    rub = next(iter(canon.values()))
    relabeled = dataclasses.replace(rub, leaf_id=leaves[1].leaf_id)
    with pytest.raises(ValueError, match='immutable source trajectory'):
        source_member_intervention(relabeled, leaves[1])
    with pytest.raises(ValueError, match='immutable source trajectory'):
        resolve_member_intervention(relabeled, leaves[1], assignments[1].intervention_spec)
    with pytest.raises(ValueError, match='different case'):
        resolve_member_intervention(rub, dataclasses.replace(leaves[1], case_id='another-case'), assignments[1].intervention_spec)
    with pytest.raises(ValueError, match='success/failure direction'):
        resolve_member_intervention(rub, dataclasses.replace(leaves[1], outcome=1.0), assignments[1].intervention_spec)
