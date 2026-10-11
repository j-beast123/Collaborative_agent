import copy

import pytest

from src.online_rl.tree import cumulative_bank as bank
from src.online_rl.tree.online_working_bank import Journal, audit_summary, ordered_keys, process_trajectories


def advanced(snapshot):
    # An empty extension still advances an immutable version and is enough to
    # test transaction ordering without manufacturing KEEP evidence.
    return bank.extend_snapshot(snapshot, [])


def test_keep_is_visible_to_immediately_following_trajectory(tmp_path):
    start = bank.empty_snapshot(); after_keep = advanced(start)
    keys = ordered_keys([
        {'case_cursor': 1, 'case_id': 'b', 'leaf_id': '2'},
        {'case_cursor': 0, 'case_id': 'a', 'leaf_id': '9'},
    ])
    journal = Journal(tmp_path/'journal.json', batch=1, source_policy_sha256='p',
                      initial_snapshot=start, trajectory_keys=keys)
    journal.append(trajectory_key=keys[0], action='GENERATED_KEEP',
                   input_snapshot=start, output_snapshot=after_keep, rubric_id='r')
    row = journal.append(trajectory_key=keys[1], action='REUSED_EXISTING',
                         input_snapshot=after_keep, output_snapshot=after_keep,
                         bank_id='bank_r')
    assert row['input_bank_sha256'] == after_keep['sha256']


def test_remove_never_enters_working_bank(tmp_path):
    start = bank.empty_snapshot(); keys = [(0, 'a', '0')]
    journal = Journal(tmp_path/'journal.json', batch=1, source_policy_sha256='p',
                      initial_snapshot=start, trajectory_keys=keys)
    journal.append(trajectory_key=keys[0], action='GENERATED_REMOVE',
                   input_snapshot=start, output_snapshot=start, rubric_id='r')
    assert journal.value['working_bank_sha256'] == start['sha256']


def test_out_of_order_or_nonkeep_advance_is_rejected(tmp_path):
    start = bank.empty_snapshot(); nxt = advanced(start); keys = [(0, 'a', '0'), (0, 'a', '1')]
    journal = Journal(tmp_path/'journal.json', batch=1, source_policy_sha256='p',
                      initial_snapshot=start, trajectory_keys=keys)
    with pytest.raises(RuntimeError, match='outside deterministic'):
        journal.append(trajectory_key=keys[1], action='GENERATED_REMOVE',
                       input_snapshot=start, output_snapshot=start)
    with pytest.raises(RuntimeError, match='Only KEEP'):
        journal.append(trajectory_key=keys[0], action='REUSED_EXISTING',
                       input_snapshot=start, output_snapshot=nxt)


def test_resume_rejects_changed_identity_and_tampering(tmp_path):
    start = bank.empty_snapshot(); keys = [(0, 'a', '0')]; path = tmp_path/'journal.json'
    journal = Journal(path, batch=1, source_policy_sha256='p', initial_snapshot=start,
                      trajectory_keys=keys)
    journal.append(trajectory_key=keys[0], action='GENERATED_REMOVE',
                   input_snapshot=start, output_snapshot=start)
    assert Journal(path, batch=1, source_policy_sha256='p', initial_snapshot=start,
                   trajectory_keys=keys).completed == 1
    with pytest.raises(RuntimeError, match='identity changed'):
        Journal(path, batch=1, source_policy_sha256='changed', initial_snapshot=start,
                trajectory_keys=keys)


def test_pipeline_matches_against_each_immediately_updated_bank(tmp_path):
    start = bank.empty_snapshot(); kept = advanced(start)
    rows = [{'case_cursor': 0, 'case_id': 'a', 'leaf_id': str(i)} for i in range(3)]
    keys = ordered_keys(rows)
    journal = Journal(tmp_path/'journal.json', batch=1, source_policy_sha256='p',
                      initial_snapshot=start, trajectory_keys=keys)
    seen = []; generated = []
    def match(row, snapshot):
        seen.append((row['leaf_id'], snapshot['sha256']))
        return {'bank_id': 'bank_new'} if row['leaf_id'] == '1' else None
    def generate(row, snapshot):
        generated.append((row['leaf_id'], snapshot['sha256']))
        return {'keep': row['leaf_id'] == '0', 'rubric_id': 'new_' + row['leaf_id'],
                'gate': 'PASS', 'reruns': [1, 1, 0]}
    def add(snapshot, row, generated):
        assert row['leaf_id'] == '0' and snapshot == start
        return kept
    results, final = process_trajectories(rows, start, journal, match=match,
        generate_verify_keep=generate, add_keep=add)
    assert [r['action'] for r in results] == ['GENERATED_KEEP', 'REUSED_EXISTING', 'GENERATED_REMOVE']
    assert seen == [('0', start['sha256']), ('1', kept['sha256']), ('2', kept['sha256'])]
    # Leaf 1 matched the rubric just kept by leaf 0, so generation was never
    # called for it. Leaf 2 did not match and therefore generated once.
    assert generated == [('0', start['sha256']), ('2', kept['sha256'])]
    assert final == kept
    assert audit_summary(journal.value) == {
        'protocol': 'trajectory_online_working_bank_v1',
        'processed_trajectories': 3, 'matcher_decisions': 3,
        'generation_required': 2, 'generation_skipped_by_reuse': 1,
        'kept_and_added': 1, 'generated_removed': 1,
        'initial_bank_sha256': start['sha256'], 'working_bank_sha256': kept['sha256']}
