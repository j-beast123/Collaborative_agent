import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.online_rl.integrated.records import atomic_write_json
from src.online_rl.tree import cumulative_bank as bank, cumulative_run as run


@pytest.fixture
def transaction(tmp_path):
    prompt = tmp_path / 'matcher.txt'
    prompt.write_text('Exact equivalence only; do not invent prior requests.')
    generator = tmp_path / 'generator.txt'
    generator.write_text('Fixed generator')
    cfg = {'rubric': {'enabled': True, 'generator_model': 'existing-generator',
                     'generator_prompt_file': str(generator), 'prompt_sha256': 'p0', 'prompt_id': 'P0',
                     'cumulative_bank': {'protocol': bank.VERSION, 'retrieval_top_k': 8,
                         'matcher_prompt_file': str(prompt),
                         'matcher_prompt_sha256': hashlib.sha256(prompt.read_bytes()).hexdigest()}},
           'prompt_update': {'enabled': False}, 'agent_reward': {'r_a_weighting': 'equal'},
           'five_gpu': {'execution_mode': 'shared_ready_tasks_v1', 'batch_cases': 5}}
    out = tmp_path / 'run'
    checkpoint = out / 'checkpoints/checkpoint_000000'
    atomic_write_json(checkpoint / 'COMPLETE.json', {'policy_sha256': 'policy'})
    state = {'checkpoint': str(checkpoint), 'case_cursor': 0, 'policy_sha256': 'policy'}
    state = run.initialize_bank_state(cfg, tmp_path, out, state)
    return cfg, tmp_path, out, state


def collector(events, *, wrong_score=False):
    def collect(phase, payload):
        events.append((phase, payload['snapshot']['sha256']))
        rows = [{'case_id': str(i), 'case_cursor': i, 'policy_sha256': 'policy',
                 'cumulative_bank_input': payload['snapshot'], 'selection': [],
                 'cumulative_reuse': []} for i in range(5)]
        if phase == 'bank_score':
            for row in rows:
                row['cumulative_bank_scoring'] = {'snapshot_sha256': payload['snapshot']['sha256']}
            if wrong_score:
                rows[-1]['cumulative_bank_scoring']['snapshot_sha256'] = 'another-bank'
        return rows
    return collect


def test_all_five_verify_before_single_frozen_bank_scoring(transaction):
    cfg, root, out, state = transaction
    events = []
    before = copy.deepcopy(state)
    records, frozen = run.round_records(cfg, root, out, state, 1, out/'batches/batch_000001', collector(events))
    assert [phase for phase, _ in events] == ['bank_prepare', 'bank_verify', 'bank_score']
    assert events[0][1] == events[1][1] == state['rubric_bank_sha256']
    assert frozen['parent_sha256'] == state['rubric_bank_sha256']
    assert frozen['version'] == state['rubric_bank_version'] + 1
    assert all(r['cumulative_bank_scoring']['snapshot_sha256'] == frozen['sha256'] for r in records)
    # Frozen scoring artifacts are tentative until optimizer/checkpoint commit.
    assert state == before
    assert json.loads((out/'run_state.json').read_text()) == before
    assert run.committed_snapshot(cfg, root, out, state)['sha256'] != frozen['sha256']


def test_exact_retry_does_not_create_another_bank_version(transaction):
    cfg, root, out, state = transaction
    args = (cfg, root, out, state, 1, out/'batches/batch_000001')
    first = run.round_records(*args, collector([]))[1]
    second = run.round_records(*args, collector([]))[1]
    assert second == first


def test_inconsistent_scoring_bank_cannot_reach_update(transaction):
    cfg, root, out, state = transaction
    with pytest.raises(RuntimeError, match='same frozen bank'):
        run.round_records(cfg, root, out, state, 1, out/'batch', collector([], wrong_score=True))
    assert json.loads((out/'run_state.json').read_text()) == state


def test_checkpoint_and_committed_bank_must_agree(transaction):
    cfg, root, out, state = transaction
    next_bank = bank.extend_snapshot(run.committed_snapshot(cfg, root, out, state), [])
    run.publish_snapshot(out, next_bank)
    checkpoint = out/'checkpoints/recovery_000001'
    receipt = {'policy_sha256': 'policy', **run.bank_fields(next_bank)}
    atomic_write_json(checkpoint/'COMPLETE.json', receipt)
    new_state = {**state, 'case_cursor': 5, 'checkpoint': str(checkpoint), **run.bank_fields(next_bank)}
    assert run.committed_snapshot(cfg, root, out, new_state) == next_bank
    atomic_write_json(checkpoint/'COMPLETE.json', {**receipt, 'rubric_bank_sha256': state['rubric_bank_sha256']})
    with pytest.raises(RuntimeError, match='Checkpoint and run state'):
        run.committed_snapshot(cfg, root, out, new_state)


def test_pending_input_cannot_change_to_another_bank_or_policy(transaction):
    cfg, root, out, state = transaction
    run.round_records(cfg, root, out, state, 1, out/'batch', collector([]))
    with pytest.raises(RuntimeError, match='Pending batch'):
        run.round_records(cfg, root, out, {**state, 'policy_sha256': 'changed'}, 1, out/'batch', collector([]))


def test_config_hash_checks_and_no_silent_bank_upgrade(transaction):
    cfg, root, out, state = transaction
    identity = run.config_identity(cfg, root)
    assert identity['reuse_effect_verification'] == 'historical_witness_no_reverification'
    Path(cfg['rubric']['cumulative_bank']['matcher_prompt_file']).write_text('changed')
    with pytest.raises(ValueError, match='prompt hash'):
        run.config_identity(cfg, root)
    advanced = {**state, 'case_cursor': 5, 'checkpoint': str(out/'checkpoints/recovery_000001')}
    advanced.pop('rubric_bank_sha256')
    advanced.pop('rubric_bank_version')
    with pytest.raises(RuntimeError, match='already advanced'):
        run.initialize_bank_state(cfg, root, out, advanced)
