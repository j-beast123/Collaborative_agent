"""Explicit batch-boundary continuation under a new supervision protocol.

This imports weights, optimizer state and RNG only. Old trees, rubrics, call
caches and scores stay in the source run. It never relabels source receipts.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree import experiment
from src.online_rl.tree.five_gpu import load_checkpoint, save_checkpoint

CONTRACT = 'explicit_supervision_checkpoint_handoff_v1'
SPEC_KEYS = {
    'contract', 'source_run', 'source_checkpoint', 'source_state_sha256',
    'source_manifest_sha256', 'source_checkpoint_sha256',
}
REQUIRED_PAYLOAD = (
    'COMPLETE.json', 'adapter/adapter_model.safetensors',
    'adapter/adapter_config.json', 'trainer_state.pt', 'cuda_rng.pt',
)


def checkpoint_sha256(path):
    """The records.dir_sha256 convention, streamed for optimizer-sized files."""
    path = Path(path)
    h = hashlib.sha256()
    for p in sorted(path.rglob('*')):
        if p.is_symlink():
            raise RuntimeError(f'Handoff checkpoint must not contain symlinks: {p}')
        if not p.is_file():
            continue
        h.update(str(p.relative_to(path)).encode())
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
    return h.hexdigest()


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f'Handoff requires a JSON object: {path}')
    return value


@contextmanager
def _source_lock(source):
    path = Path(source) / 'runner.lock'
    # A missing lock means no runner has ever owned this run; require the
    # committed source run's lock instead of creating source artifacts.
    if not path.is_file():
        raise RuntimeError('Handoff source runner.lock is missing')
    with path.open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Handoff source runner is still active') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def handoff_spec(source_run):
    """Pin a stopped source's immutable identities for an explicit new config."""
    source = Path(source_run).resolve()
    with _source_lock(source):
        state = _read(source / 'run_state.json')
        manifest = _read(source / 'run_manifest.json')
        checkpoint = Path(state['checkpoint']).resolve()
        return {
            'contract': CONTRACT, 'source_run': str(source),
            'source_checkpoint': str(checkpoint),
            'source_state_sha256': sha256_json(state),
            'source_manifest_sha256': sha256_json(manifest),
            'source_checkpoint_sha256': checkpoint_sha256(checkpoint),
        }


def _compatible_identity(cfg):
    rollout = copy.deepcopy(cfg['rollout'])
    for name in ('endpoint', 'gpu'):
        rollout.get('doctor_backend', {}).pop(name, None)
    fg = cfg['five_gpu']
    return {
        'policy': cfg['policy'], 'optimizer': cfg['optimizer'],
        'advantage_normalization': cfg.get('advantage_normalization'),
        'agent_reward': cfg['agent_reward'], 'schedule': cfg['schedule'],
        'rollout': rollout, 'feedback': cfg.get('feedback'),
        'intervention': cfg.get('intervention'),
        'verification': cfg.get('verification'),
        'data': {k: cfg['data'].get(k) for k in
                 ('train_sha256', 'excluded_calibration_case_ids')},
        'batch': {k: fg.get(k) for k in
                  ('batch_cases', 'lanes_per_gpu', 'execution_mode',
                   'train_order_sha256', 'opening_identity', 'opening_mode')},
    }


def validate_handoff(cfg, out, theta0=None):
    """Read-only validation; caller must hold the source lock while importing."""
    spec = cfg.get('supervision_handoff')
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS or spec.get('contract') != CONTRACT:
        raise RuntimeError('Invalid supervision_handoff contract or fields')
    for key in SPEC_KEYS - {'contract'}:
        if not isinstance(spec[key], str) or not spec[key]:
            raise RuntimeError(f'Invalid supervision_handoff.{key}')
    source = Path(spec['source_run']).resolve()
    target = Path(out).resolve()
    if target == source or source in target.parents or target in source.parents:
        raise RuntimeError('Handoff source and destination must be separate run directories')
    checkpoint = Path(spec['source_checkpoint']).resolve()
    if checkpoint.parent != source / 'checkpoints':
        raise RuntimeError('Handoff checkpoint is outside source run')
    state = _read(source / 'run_state.json')
    manifest = _read(source / 'run_manifest.json')
    if sha256_json(state) != spec['source_state_sha256']:
        raise RuntimeError('Handoff source run state changed')
    if sha256_json(manifest) != spec['source_manifest_sha256']:
        raise RuntimeError('Handoff source manifest changed')
    if Path(state.get('checkpoint', '')).resolve() != checkpoint:
        raise RuntimeError('Handoff checkpoint is not the committed source checkpoint')
    for name in REQUIRED_PAYLOAD:
        if not (checkpoint / name).is_file():
            raise RuntimeError(f'Handoff checkpoint payload missing: {name}')
    if checkpoint_sha256(checkpoint) != spec['source_checkpoint_sha256']:
        raise RuntimeError('Handoff checkpoint payload hash differs')
    receipt = _read(checkpoint / 'COMPLETE.json')
    oldcfg = manifest.get('config', {})
    normalized = copy.deepcopy(oldcfg)
    normalized.setdefault('five_gpu', {})['checkpoint_evaluation_enabled'] = False
    if sha256_json(normalized) != state.get('config_sha256'):
        raise RuntimeError('Handoff source configuration hash differs')
    if manifest.get('protocol_sha256') != state.get('protocol_sha256'):
        raise RuntimeError('Handoff source manifest protocol differs')
    if any(x.get('enabled') for x in (cfg['prompt_update'], oldcfg['prompt_update'])):
        raise RuntimeError('Supervision handoff currently requires fixed P0 on both sides')
    differences = sorted(k for k, value in _compatible_identity(cfg).items()
                         if value != _compatible_identity(oldcfg).get(k))
    if differences:
        raise RuntimeError(f'Handoff training conditions differ: {differences}')
    cursor, batch = state.get('case_cursor'), state.get('batch_cursor')
    batch_size = cfg['five_gpu'].get('batch_cases')
    if (type(cursor) is not int or type(batch) is not int or type(batch_size) is not int
            or batch_size < 1 or cursor <= 0 or cursor % batch_size
            or batch != cursor // batch_size or cursor >= cfg['schedule']['cases']):
        raise RuntimeError('Handoff requires a committed complete batch before final case')
    for key, expected in {'completed_cases': cursor,
                          'policy_sha256': state.get('policy_sha256'),
                          'protocol_sha256': state.get('protocol_sha256')}.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f'Handoff source checkpoint {key} differs')
    experiment.check_receipt(oldcfg, state, 'Handoff source state')
    experiment.check_receipt(oldcfg, receipt, 'Handoff source checkpoint')
    initial_path = source / 'checkpoints/checkpoint_000000/COMPLETE.json'
    if initial_path.exists():
        if state.get('theta0_policy_sha256') != _read(initial_path).get('policy_sha256'):
            raise RuntimeError('Handoff source theta0 provenance differs')
    else:
        # A run continued by an earlier explicit handoff starts at handoff_N,
        # not checkpoint_0. Preserve and verify that committed provenance chain.
        inherited = _read(source / 'supervision_handoff.json')
        digest = sha256_json(inherited)
        previous_receipt = _read(source / 'checkpoints' /
                                 f"handoff_{inherited['completed_cases']:06d}" / 'COMPLETE.json')
        reachable={inherited.get('destination_protocol_sha256')}
        for row in state.get('protocol_transitions', []):
            if (isinstance(row, dict) and row.get('from') in reachable and
                    row.get('completed_cases') == inherited.get('completed_cases')):
                reachable.add(row.get('to'))
        inherited_protocol_valid = state.get('protocol_sha256') in reachable
        if (manifest.get('supervision_handoff') != inherited or
                state.get('supervision_handoff_sha256') != digest or
                previous_receipt.get('supervision_handoff_sha256') != digest or
                previous_receipt.get('protocol_sha256') != inherited.get('destination_protocol_sha256') or
                not inherited_protocol_valid or
                inherited.get('theta0_policy_sha256') != state.get('theta0_policy_sha256')):
            raise RuntimeError('Handoff source inherited theta0 provenance differs')
    if theta0 is not None and theta0 != state.get('theta0_policy_sha256'):
        raise RuntimeError('Handoff freshly initialized theta0 differs from source')
    return {'spec': spec, 'state': state, 'receipt': receipt, 'manifest': manifest,
            'compatible_training_identity': _compatible_identity(cfg)}


def prepare_handoff(updater, cfg, out, config_sha, protocol, theta0,
                    policy_sha_fn, manifest_payload=None):
    """Import a frozen source boundary; publish new manifest before run state.

    A retry after interruption accepts only the exact same handoff checkpoint.
    The named boundary checkpoint is retained across normal rolling recovery
    cleanup, unlike a recovery_N checkpoint.
    """
    out = Path(out).resolve()
    if (out / 'run_state.json').exists():
        raise RuntimeError('Handoff destination already has run state; use normal resume')
    source = Path(cfg.get('supervision_handoff', {}).get('source_run', '')).resolve()
    with _source_lock(source):
        checked = validate_handoff(cfg, out, theta0)
        state, receipt = checked['state'], checked['receipt']
        checkpoint = Path(checked['spec']['source_checkpoint'])
        cursor, batch = state['case_cursor'], state['batch_cursor']
        provenance = {
            **checked['spec'], 'completed_cases': cursor, 'batch_cursor': batch,
            'source_policy_sha256': state['policy_sha256'],
            'source_config_sha256': state['config_sha256'],
            'source_protocol_sha256': state['protocol_sha256'],
            'theta0_policy_sha256': theta0,
            'destination_config_sha256': config_sha,
            'destination_protocol_sha256': protocol,
            'source_receipt': receipt,
            'compatible_training_identity': checked['compatible_training_identity'],
            'reused': ['adapter', 'optimizer', 'optimizer_steps', 'cpu_rng', 'cuda_rng'],
            'not_reused': ['rollouts', 'reflections', 'rubrics', 'reruns', 'scores', 'call_caches'],
        }
        seed_bank = cfg.get('rubric', {}).get('cumulative_bank') or {}
        if seed_bank.get('initial_snapshot_sha256'):
            provenance['cumulative_bank_seed_sha256'] = seed_bank['initial_snapshot_sha256']
            provenance['reused'].append('evidence_validated_KEEP_rubrics_via_pinned_bank_snapshot')
            provenance['not_reused'].remove('rubrics')
            provenance['not_reused'].append('unvalidated_or_legacy_rubrics')
        target = out / 'checkpoints' / f'handoff_{cursor:06d}'
        checkpoint_root = out / 'checkpoints'
        if checkpoint_root.exists():
            allowed = {target.name, target.name + '.pending'}
            if any(p.name not in allowed for p in checkpoint_root.iterdir()):
                raise RuntimeError('Handoff destination has unrelated checkpoint artifacts')
        load_checkpoint(updater, checkpoint)
        if policy_sha_fn() != state['policy_sha256']:
            raise RuntimeError('Handoff loaded adapter differs from source receipt')
        expected_steps = receipt.get('update', {}).get('optimizer_steps')
        if expected_steps is not None and updater.steps != expected_steps:
            raise RuntimeError('Handoff loaded optimizer step differs from source receipt')
        new_receipt = {'completed_cases': cursor, 'policy_sha256': state['policy_sha256'],
                       'protocol_sha256': protocol, **experiment.receipt_fields(cfg),
                       'supervision_handoff_sha256': sha256_json(provenance),
                       'source_policy_sha256': state['policy_sha256'],
                       'imported_optimizer_steps': updater.steps}
        save_checkpoint(updater, target, new_receipt)
        # Loading the actual destination also validates an idempotent retry's
        # existing payload and restores precisely the state future resumes use.
        load_checkpoint(updater, target)
        if policy_sha_fn() != state['policy_sha256'] or (expected_steps is not None and updater.steps != expected_steps):
            raise RuntimeError('Handoff saved checkpoint failed reload verification')
        new_state = {'case_cursor': cursor, 'batch_cursor': batch,
                     'config_sha256': config_sha, 'protocol_sha256': protocol,
                     'policy_sha256': state['policy_sha256'], 'checkpoint': str(target),
                     **experiment.state_fields(cfg, theta0),
                     'supervision_handoff_sha256': sha256_json(provenance)}
        manifest = dict(manifest_payload or {'config': cfg, 'protocol_sha256': protocol})
        if manifest.get('config') != cfg or manifest.get('protocol_sha256') != protocol:
            raise RuntimeError('Handoff destination manifest does not match new config/protocol')
        manifest['supervision_handoff'] = provenance
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out / 'supervision_handoff.json', provenance)
        atomic_write_json(out / 'run_manifest.json', manifest)
        atomic_write_json(out / 'run_state.json', new_state)
        return new_state
