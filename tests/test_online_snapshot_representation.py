"""Online working bank: the coordinator must hand the scoring phase the saved (JSON) representation of the bank.

A rubric kept earlier in the batch is built in memory with `problems` as a tuple, while a later case's record is read back
from disk with a list. The scoring guard compares the two with ==, so without normalisation it fails although the content and
every hash are identical; a restart (everything read from disk) passes. The normalised state must equal that restart state."""
import copy
import json

import pytest

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import cumulative_bank as bank, cumulative_run as run
from tests.test_cumulative_run import transaction  # noqa: F401  (fixture)
from tests.test_cumulative_trainer import source_record


def saved(value):                                                # exactly atomic_write_json followed by a reload
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def in_memory_keep(label, cursor):
    record = source_record(label); record.update(case_cursor=cursor)
    record['trajectory_rubrics'][0]['problems'] = ()             # as to_dict(TrajectoryRubric) produces it
    return record


def scoring_guard(record, snapshot):                             # the guard at the start of Trainer.score_cumulative
    entries = {entry['bank_id']: entry for entry in snapshot['entries']}
    for previous in record['cumulative_bank_input']['entries']:
        current = entries.get(previous['bank_id'])
        if (current is None or current['rubric'] != previous['rubric'] or
                any(proof not in current['sources'] for proof in previous['sources'])):
            raise RuntimeError('frozen scoring bank lost or changed prior validated criteria')


def run_batch(transaction, tmp_name):
    cfg, root, out, state = transaction
    cfg['rubric']['cumulative_bank'].update(working_bank_protocol=run.ONLINE_WORKING_BANK_PROTOCOL, scoring_scope='all')
    disk = {}; memory = {}; scored_with = {}
    def collect(phase, payload, selected=None):
        if phase == 'bank_online':
            cursor = selected[0]; before = payload['snapshot']
            keeps = [in_memory_keep('explain', cursor)] if cursor == 1 else []      # case 1 adds a KEEP; cases 2-4 consume it
            after = bank.extend_snapshot(before, keeps)
            record = {'case_id': str(cursor), 'case_cursor': cursor, 'policy_sha256': 'policy',
                      'cumulative_bank_input': before, 'cumulative_bank_output': after, 'selection': [], 'cumulative_reuse': [],
                      'online_working_bank_decisions': [{'case_cursor': cursor, 'case_id': str(cursor), 'leaf_id': 'leaf',
                          'action': 'GENERATED_KEEP', 'input_bank_sha256': before['sha256'],
                          'output_bank_sha256': after['sha256'], 'bank_id': 'b', 'rubric_id': 'r'}]}
            disk[cursor] = saved(record); memory[cursor] = record   # the worker writes the file and returns the object
            return [record]
        scored_with['snapshot'] = payload['snapshot']
        for cursor in range(5): scoring_guard(disk[cursor], payload['snapshot'])    # each worker re-reads its saved record
        return [{**disk[i], 'cumulative_bank_scoring': {'snapshot_sha256': payload['snapshot']['sha256'], 'scoring_scope': 'all'}}
                for i in range(5)]
    records, frozen = run.round_records(cfg, root, out, state, 1, out / tmp_name, collect)
    return records, frozen, scored_with['snapshot'], memory, disk, out / tmp_name


def test_in_memory_bank_differs_from_its_saved_form_only_in_representation():
    built = bank.extend_snapshot(bank.empty_snapshot(), [in_memory_keep('explain', 1)])
    reloaded = saved(built)
    assert built != reloaded                                     # tuple vs list: the F failure
    assert built['sha256'] == reloaded['sha256'] and sha256_json(built) == sha256_json(reloaded)   # identical content and bank identity
    a, b = built['entries'][0], reloaded['entries'][0]
    assert a['bank_id'] == b['bank_id'] and a['content_sha256'] == b['content_sha256']
    assert bank.content_hash(a['rubric']['rubric']) == bank.content_hash(b['rubric']['rubric'])
    assert a['rubric']['rubric'] == b['rubric']['rubric']         # rubric text, criteria, applicability, score anchors
    diff = [k for k in a['rubric'] if a['rubric'][k] != b['rubric'][k]]
    assert diff == ['problems'] and list(a['rubric']['problems']) == b['rubric']['problems'] == []
    with pytest.raises(RuntimeError, match='lost or changed'):   # the guard on the raw in-memory bank
        scoring_guard({'cumulative_bank_input': reloaded}, built)


def test_scoring_phase_receives_the_saved_representation(transaction):
    records, frozen, scoring_snapshot, memory, disk, batchdir = run_batch(transaction, 'online_batch')
    assert scoring_snapshot == saved(scoring_snapshot)            # A (normalised in-memory) == B (serialize -> reload)
    assert scoring_snapshot == disk[4]['cumulative_bank_output'] == json.loads((batchdir / 'bank_frozen.json').read_text())
    assert frozen == scoring_snapshot and len(frozen['entries']) == 1 and frozen['version'] == memory[4]['cumulative_bank_output']['version']
    assert frozen['sha256'] == memory[4]['cumulative_bank_output']['sha256']      # bank identity is unchanged by the representation
    assert frozen['entries'][0]['rubric']['rubric'] == memory[1]['cumulative_bank_output']['entries'][0]['rubric']['rubric']
    assert all(r['cumulative_bank_scoring']['snapshot_sha256'] == frozen['sha256'] for r in records)


def test_restart_from_disk_gives_the_identical_state(transaction):
    cfg, root, out, state = transaction
    first = run_batch(transaction, 'online_batch')
    disk = first[4]
    def collect(phase, payload, selected=None):                  # the resume path: every record is read back from disk
        if phase == 'bank_online': return [copy.deepcopy(disk[selected[0]])]
        for cursor in range(5): scoring_guard(disk[cursor], payload['snapshot'])
        return [{**disk[i], 'cumulative_bank_scoring': {'snapshot_sha256': payload['snapshot']['sha256'], 'scoring_scope': 'all'}}
                for i in range(5)]
    records, frozen = run.round_records(cfg, root, out, state, 1, out / 'online_batch', collect)
    assert frozen == first[1] and records == first[0]
