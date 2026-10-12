import pickle
from types import SimpleNamespace

from src.online_rl.tree.doctor_reflection import DoctorReflectionError
from src.online_rl.tree.posthoc import doctor_feedback
from src.online_rl.tree.records import AgentNode, DoctorFeedback, Leaf, Tree
from src.online_rl.tree.reward import exclude_reflection_failure_trajectories


def test_doctor_reflection_error_round_trips_through_process_pool_pickle():
    error = DoctorReflectionError(
        'reflection failed after repair',
        raw=['bad response'],
        evidence=[{'attempt': 1, 'validation_error': 'invalid JSON'}],
        settings={'format_repair_attempts': 1},
    )

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is DoctorReflectionError
    assert str(restored) == str(error)
    assert restored.raw == error.raw
    assert restored.evidence == error.evidence
    assert restored.settings == error.settings
    assert restored.audit == error.audit


def test_invalid_local_reflection_is_recorded_as_ineligible_feedback():
    leaf = Leaf('case/1', 'case', ('case/1',), (), 'FINAL', 'A', 1.0, 'final')

    def feedback_fn(*_args):
        raise DoctorReflectionError('bad JSON after repair', raw=['bad', 'still bad'],
                                    evidence=[{'validation_error': 'invalid JSON'}],
                                    settings={'format_repair_attempts': 1})

    result = doctor_feedback({'case_id': 'case', 'question': 'Q', 'options': {}}, leaf,
                             feedback_fn, 'prompt', SimpleNamespace(call=lambda _kind, _identity, fn: fn()))

    assert result.leaf_id == leaf.leaf_id
    assert result.outcome == 1.0
    assert result.valid is False
    assert result.text == ''
    assert result.locus['status'] == 'reflection_invalid'
    assert result.locus['raw'] == ['bad', 'still bad']


def test_reflection_failure_leaf_is_removed_from_outcome_credit_tree():
    node_ok = AgentNode('case/1', None, 'case', 1, (), 'ok', {}, 'policy', 0, 'final', True, None)
    node_bad = AgentNode('case/2', None, 'case', 1, (), 'bad', {}, 'policy', 1, 'final', True, None)
    leaf_ok = Leaf('case/1', 'case', ('case/1',), (), 'FINAL', 'A', 1.0, 'final')
    leaf_bad = Leaf('case/2', 'case', ('case/2',), (), 'FINAL', 'B', 0.0, 'final')
    tree = Tree('case', 'policy', 1, 2, (node_ok, node_bad), (leaf_ok, leaf_bad))
    feedback = [DoctorFeedback('case/1', 1.0, 'reflection', 'prompt'),
                DoctorFeedback('case/2', 0.0, '', 'prompt', locus={'status': 'reflection_invalid'}, valid=False)]

    training_tree = exclude_reflection_failure_trajectories(tree, feedback)

    assert [leaf.leaf_id for leaf in training_tree.leaves] == ['case/1']
    assert len(training_tree.nodes) == 2  # keep the full tree structure for stable response grouping
