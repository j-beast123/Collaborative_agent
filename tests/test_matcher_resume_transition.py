from copy import deepcopy
from src.online_rl.tree.resume_transition import is_matcher_1_to_8_transition, is_candidate_seed_transition


def base_config():
    return {
        'rubric': {'reuse': 'assign_or_create_v1'},
        'five_gpu': {'checkpoint_evaluation_enabled': True, 'batch_cases': 5},
        'schedule': {'cases': 500},
        'output_dir': 'outputs/run',
    }


def test_matcher_change_is_allowed_only_at_committed_five_case_barrier():
    old = base_config()
    new = deepcopy(old)
    new['rubric']['match_batch_size'] = 8
    assert is_matcher_1_to_8_transition(old, new, case_cursor=10, batch_cursor=2, batch_cases=5)
    assert not is_matcher_1_to_8_transition(old, new, case_cursor=9, batch_cursor=1, batch_cases=5)
    assert not is_matcher_1_to_8_transition(old, new, case_cursor=10, batch_cursor=1, batch_cases=5)


def test_matcher_transition_rejects_other_protocol_edits_or_downgrade():
    old = base_config()
    new = deepcopy(old)
    new['rubric']['match_batch_size'] = 8
    new['rollout'] = {'rounds': 4}
    assert not is_matcher_1_to_8_transition(old, new, case_cursor=10, batch_cursor=2, batch_cases=5)
    new = deepcopy(old)
    new['rubric']['match_batch_size'] = 1
    assert not is_matcher_1_to_8_transition(old, new, case_cursor=10, batch_cursor=2, batch_cases=5)


def test_candidate_seed_transition_only_changes_evidence_scope_at_batch_barrier():
    old = base_config()
    old['prompt_update'] = {'candidate_seed': 'first_completed_P0_v1'}
    new = deepcopy(old)
    new['prompt_update']['candidate_seed'] = 'all_p0_before_p1_v1'
    assert is_candidate_seed_transition(old, new, case_cursor=20, batch_cursor=4, batch_cases=5)
    assert not is_candidate_seed_transition(old, new, case_cursor=19, batch_cursor=3, batch_cases=5)
    changed = deepcopy(new)
    changed['rollout'] = {'rounds': 4}
    assert not is_candidate_seed_transition(old, changed, case_cursor=20, batch_cursor=4, batch_cases=5)
