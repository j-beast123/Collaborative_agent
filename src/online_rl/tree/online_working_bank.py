"""Deterministic trajectory-at-a-time cumulative-bank transactions.

This module deliberately contains no model calls.  The runner supplies one
trajectory transaction at a time and this journal makes the ordering and bank
transition restart-verifiable.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.cumulative_bank import validate_snapshot

VERSION = 'trajectory_online_working_bank_v1'


def ordered_keys(rows):
    """Stable order across cases and leaves; completion timing is irrelevant."""
    keys = [(int(row['case_cursor']), str(row['case_id']), str(row['leaf_id'])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate trajectory identity in online working-bank batch')
    return sorted(keys)


class Journal:
    def __init__(self, path, *, batch, source_policy_sha256, initial_snapshot, trajectory_keys):
        validate_snapshot(initial_snapshot)
        self.path = Path(path)
        identity = {'protocol': VERSION, 'batch': int(batch),
                    'source_policy_sha256': str(source_policy_sha256),
                    'initial_bank_sha256': initial_snapshot['sha256'],
                    'trajectory_keys': [list(key) for key in trajectory_keys]}
        self.value = {'identity': identity, 'decisions': [],
                      'working_bank_sha256': initial_snapshot['sha256']}
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if saved.get('identity') != identity:
                raise RuntimeError('Online working-bank journal identity changed')
            self.value = saved
        self._validate()

    def _validate(self):
        decisions = self.value.get('decisions')
        if not isinstance(decisions, list):
            raise RuntimeError('Invalid online working-bank journal')
        expected = self.value['identity']['trajectory_keys']
        if [row.get('trajectory_key') for row in decisions] != expected[:len(decisions)]:
            raise RuntimeError('Online working-bank decisions are not a deterministic prefix')
        prior = self.value['identity']['initial_bank_sha256']
        for row in decisions:
            if row.get('input_bank_sha256') != prior:
                raise RuntimeError('Online working-bank decision input is not the prior output')
            if row.get('action') not in ('REUSED_EXISTING', 'GENERATED_KEEP', 'GENERATED_REMOVE'):
                raise RuntimeError('Unknown online working-bank action')
            if row['action'] == 'GENERATED_KEEP':
                if row.get('output_bank_sha256') == prior:
                    raise RuntimeError('KEEP did not advance the working bank')
            elif row.get('output_bank_sha256') != prior:
                raise RuntimeError('Only KEEP may advance the working bank')
            prior = row['output_bank_sha256']
        if self.value.get('working_bank_sha256') != prior:
            raise RuntimeError('Online working-bank journal tail differs')

    @property
    def completed(self):
        return len(self.value['decisions'])

    def append(self, *, trajectory_key, action, input_snapshot, output_snapshot,
               bank_id=None, rubric_id=None, evidence=None):
        validate_snapshot(input_snapshot); validate_snapshot(output_snapshot)
        expected = self.value['identity']['trajectory_keys']
        if self.completed >= len(expected) or list(trajectory_key) != expected[self.completed]:
            raise RuntimeError('Trajectory processed outside deterministic working-bank order')
        row = {'trajectory_key': list(trajectory_key), 'action': action,
               'input_bank_sha256': input_snapshot['sha256'],
               'output_bank_sha256': output_snapshot['sha256'],
               'bank_id': bank_id, 'rubric_id': rubric_id,
               'evidence_sha256': sha256_json(evidence) if evidence is not None else None}
        self.value['decisions'].append(row)
        self.value['working_bank_sha256'] = output_snapshot['sha256']
        self._validate()
        atomic_write_json(self.path, self.value)
        return row


def process_trajectories(rows, initial_snapshot, journal, *, match, generate_verify_keep, add_keep):
    """Run the required MATCH -> (GENERATE/GATE/RERUN) transaction serially.

    Callbacks are injected so this policy can be tested without GPU/API calls.
    ``generate_verify_keep`` must return durable evidence and a KEEP boolean;
    ``add_keep`` may advance the snapshot only for KEEP.  A match always sees
    the snapshot produced by the immediately preceding trajectory.
    """
    validate_snapshot(initial_snapshot)
    ordered = sorted(rows, key=lambda row: (int(row['case_cursor']), str(row['case_id']), str(row['leaf_id'])))
    if ordered_keys(ordered) != [tuple(key) for key in journal.value['identity']['trajectory_keys']]:
        raise RuntimeError('Online working-bank rows differ from journal identity')
    snapshot = initial_snapshot
    # Resume replays only the immutable bank transitions, then continues with
    # the first undecided trajectory. The caller supplies snapshots by digest.
    if journal.completed:
        raise RuntimeError('Resumed execution requires snapshot resolver; refusing ambiguous replay')
    results = []
    for row in ordered:
        key = (int(row['case_cursor']), str(row['case_id']), str(row['leaf_id']))
        reused = match(row, snapshot)
        if reused is not None:
            journal.append(trajectory_key=key, action='REUSED_EXISTING',
                           input_snapshot=snapshot, output_snapshot=snapshot,
                           bank_id=reused['bank_id'], evidence=reused)
            results.append({'trajectory_key': key, 'action': 'REUSED_EXISTING', **reused})
            continue
        generated = generate_verify_keep(row, snapshot)
        if not isinstance(generated, dict) or type(generated.get('keep')) is not bool:
            raise RuntimeError('Generation transaction lacks an explicit KEEP decision')
        before = snapshot
        if generated['keep']:
            snapshot = add_keep(snapshot, row, generated)
            validate_snapshot(snapshot)
            action = 'GENERATED_KEEP'
        else:
            action = 'GENERATED_REMOVE'
        journal.append(trajectory_key=key, action=action, input_snapshot=before,
                       output_snapshot=snapshot, rubric_id=generated.get('rubric_id'),
                       evidence=generated)
        results.append({'trajectory_key': key, 'action': action,
                        'rubric_id': generated.get('rubric_id')})
    return results, snapshot


def audit_summary(journal_value):
    """Small monitor payload whose counts must reconcile with external calls."""
    rows = journal_value['decisions']
    counts = {name: sum(row['action'] == name for row in rows) for name in
              ('REUSED_EXISTING', 'GENERATED_KEEP', 'GENERATED_REMOVE')}
    return {'protocol': journal_value['identity']['protocol'],
            'processed_trajectories': len(rows),
            'matcher_decisions': len(rows),
            'generation_required': counts['GENERATED_KEEP'] + counts['GENERATED_REMOVE'],
            'generation_skipped_by_reuse': counts['REUSED_EXISTING'],
            'kept_and_added': counts['GENERATED_KEEP'],
            'generated_removed': counts['GENERATED_REMOVE'],
            'initial_bank_sha256': journal_value['identity']['initial_bank_sha256'],
            'working_bank_sha256': journal_value['working_bank_sha256']}
