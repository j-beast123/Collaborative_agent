"""Capture an immutable evaluation adapter from a live committed training run.

The source is only read. No training lock, GPU, optimizer load or model import is
needed. Files are opened before copying so rolling checkpoint cleanup cannot
turn a successful capture into a partial one.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

from src.online_rl.integrated.records import atomic_write_json, sha256_json

CONTRACT = 'idle_evaluation_adapter_snapshot_v1'
PAYLOAD = ('adapter/adapter_model.safetensors', 'adapter/adapter_config.json')


def _read(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f'Snapshot requires a JSON object: {path}')
    return value


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _check_source(manifest, state, expected_manifest_sha256):
    if expected_manifest_sha256 is not None and sha256_json(manifest) != expected_manifest_sha256:
        raise RuntimeError('Snapshot source manifest identity differs')
    cfg = copy.deepcopy(manifest['config'])
    cfg.setdefault('five_gpu', {})['checkpoint_evaluation_enabled'] = False
    if sha256_json(cfg) != state.get('config_sha256'):
        raise RuntimeError('Snapshot source config identity differs')
    if manifest.get('protocol_sha256') != state.get('protocol_sha256'):
        raise RuntimeError('Snapshot source protocol identity differs')
    if type(state.get('case_cursor')) is not int or state['case_cursor'] < 0:
        raise RuntimeError('Snapshot source cursor is invalid')


def _check_receipt(receipt, state, cursor):
    expected = {'completed_cases': cursor,
                'policy_sha256': state.get('policy_sha256'),
                'protocol_sha256': state.get('protocol_sha256'),
                'experiment_fingerprint': state.get('experiment_fingerprint')}
    if not expected['policy_sha256'] or not expected['protocol_sha256']:
        raise RuntimeError('Snapshot source lacks policy/protocol identity')
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f'Snapshot checkpoint {key} identity differs')


def _existing(dest, source, cursor, manifest):
    proof = _read(dest / 'SNAPSHOT.json')
    expected = {'contract': CONTRACT, 'source_run': str(source),
                'completed_cases': cursor, 'source_manifest_sha256': sha256_json(manifest)}
    for key, value in expected.items():
        if proof.get(key) != value:
            raise RuntimeError(f'Existing snapshot {key} identity differs')
    state, receipt = _read(dest / 'run_state.json'), _read(dest / 'COMPLETE.json')
    if (state.get('case_cursor') != cursor or
            sha256_json(state) != proof.get('source_state_sha256') or
            sha256_json(_read(dest / 'run_manifest.json')) != proof['source_manifest_sha256']):
        raise RuntimeError('Existing snapshot state/manifest identity differs')
    _check_source(manifest, state, proof['source_manifest_sha256'])
    _check_receipt(receipt, state, cursor)
    if receipt['policy_sha256'] != proof.get('policy_sha256'):
        raise RuntimeError('Existing snapshot policy identity differs')
    hashes = proof.get('file_sha256', {})
    if set(hashes) != set(PAYLOAD) | {'COMPLETE.json'}:
        raise RuntimeError('Existing snapshot payload inventory differs')
    for name, expected_hash in hashes.items():
        if _file_hash(dest / name) != expected_hash:
            raise RuntimeError(f'Existing snapshot payload hash differs: {name}')
    return dest


def snapshot(source_run, cursor, dest, *, expected_manifest_sha256=None):
    """Return a verified adapter snapshot, or None before ``cursor`` commits.

    ``dest`` is the final checkpoint directory, separate from the training run.
    Pin ``expected_manifest_sha256`` when scheduling to reject changed run
    identity. An existing snapshot remains reusable after source cleanup.
    Missing a committed cursor fails closed rather than evaluating newer weights.
    The receipt's policy hash is checked against committed state; the evaluator
    must additionally verify that hash after loading the adapter into its model.
    """
    if type(cursor) is not int or cursor < 0:
        raise ValueError('Snapshot cursor must be a non-negative integer')
    source, dest = Path(source_run).resolve(), Path(dest).resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError('Snapshot destination must be separate from the source run')
    manifest = _read(source / 'run_manifest.json')
    if expected_manifest_sha256 is not None and sha256_json(manifest) != expected_manifest_sha256:
        raise RuntimeError('Snapshot source manifest identity differs')
    if dest.exists():
        return _existing(dest, source, cursor, manifest)
    for _ in range(3):
        state = _read(source / 'run_state.json')
        _check_source(manifest, state, expected_manifest_sha256)
        committed = state['case_cursor']
        if committed < cursor:
            return None
        if committed > cursor:
            raise RuntimeError(f'C{cursor} is unavailable: source already committed C{committed}')
        checkpoint = Path(state['checkpoint'])
        if (checkpoint.is_symlink() or checkpoint.resolve().parent != source / 'checkpoints'
                or (checkpoint / 'adapter').is_symlink()):
            raise RuntimeError('Snapshot checkpoint is outside source or contains symlinks')
        pending = None
        try:
            with ExitStack() as stack:
                handles = {}
                for name in ('COMPLETE.json', *PAYLOAD):
                    fd = os.open(checkpoint / name, os.O_RDONLY | os.O_NOFOLLOW)
                    handles[name] = stack.enter_context(os.fdopen(fd, 'rb'))
                receipt_bytes = handles['COMPLETE.json'].read()
                receipt = json.loads(receipt_bytes)
                _check_receipt(receipt, state, cursor)
                # A second manifest read rejects a source identity change before publication.
                if _read(source / 'run_manifest.json') != manifest:
                    raise RuntimeError('Snapshot source manifest changed during capture')
                dest.parent.mkdir(parents=True, exist_ok=True)
                pending = Path(tempfile.mkdtemp(prefix=dest.name + '.pending-', dir=dest.parent))
                hashes = {}
                for name, stream in handles.items():
                    stream.seek(0)
                    before = os.fstat(stream.fileno())
                    target = pending / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with target.open('wb') as output:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                            output.write(chunk)
                            digest.update(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    after = os.fstat(stream.fileno())
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError(f'Snapshot source payload changed during copy: {name}')
                    hashes[name] = digest.hexdigest()
                atomic_write_json(pending / 'run_state.json', state)
                atomic_write_json(pending / 'run_manifest.json', manifest)
                atomic_write_json(pending / 'SNAPSHOT.json', {
                    'contract': CONTRACT, 'source_run': str(source),
                    'source_checkpoint': str(checkpoint.resolve()), 'completed_cases': cursor,
                    'policy_sha256': receipt['policy_sha256'],
                    'protocol_sha256': receipt['protocol_sha256'],
                    'source_state_sha256': sha256_json(state),
                    'source_manifest_sha256': sha256_json(manifest),
                    'file_sha256': hashes,
                })
                _existing(pending, source, cursor, manifest)
                try:
                    pending.rename(dest)
                except OSError:
                    if not dest.exists():
                        raise
                    _existing(dest, source, cursor, manifest)
                return dest
        except FileNotFoundError:
            # Cleanup may win before every payload has an open descriptor.
            # Re-read committed state; never retry against a later adapter.
            continue
        finally:
            if pending is not None and pending.exists():
                shutil.rmtree(pending)
    current = _read(source / 'run_state.json')['case_cursor']
    if current > cursor:
        raise RuntimeError(f'C{cursor} is unavailable: source already committed C{current}')
    raise RuntimeError(f'C{cursor} committed checkpoint payload is unavailable or incomplete')
