import copy
import json
import os
import shutil

import pytest

from src.online_rl.integrated.records import atomic_write_json, dir_sha256, sha256_json
from src.online_rl.tree import idle_eval_snapshot as capture


@pytest.fixture
def source(tmp_path):
    run = tmp_path / 'training'
    checkpoint = run / 'checkpoints/recovery_000004'
    (checkpoint / 'adapter').mkdir(parents=True)
    cfg = {'policy': {'base_model': 'base'},
           'rollout': {'rounds': 2},
           'five_gpu': {'checkpoint_evaluation_enabled': False}}
    manifest = {'config': cfg, 'protocol_sha256': 'protocol'}
    state = {'case_cursor': 20, 'batch_cursor': 4, 'checkpoint': str(checkpoint),
             'policy_sha256': 'c20-policy', 'protocol_sha256': 'protocol',
             'config_sha256': sha256_json(cfg), 'experiment_fingerprint': 'experiment'}
    receipt = {'completed_cases': 20, 'policy_sha256': 'c20-policy',
               'protocol_sha256': 'protocol', 'experiment_fingerprint': 'experiment'}
    atomic_write_json(run / 'run_state.json', state)
    atomic_write_json(run / 'run_manifest.json', manifest)
    atomic_write_json(checkpoint / 'COMPLETE.json', receipt)
    atomic_write_json(checkpoint / 'adapter/adapter_config.json', {'r': 16})
    (checkpoint / 'adapter/adapter_model.safetensors').write_bytes(b'c20-weights')
    (checkpoint / 'trainer_state.pt').write_bytes(b'optimizer-not-needed')
    return run, checkpoint, tmp_path / 'evaluation/checkpoint_000020'


def rewrite(path, **updates):
    data = json.loads(path.read_text())
    data.update(updates)
    atomic_write_json(path, data)


def test_capture_preserves_source_and_adapter_only(source):
    run, checkpoint, dest = source
    original = dir_sha256(run)
    manifest_sha = sha256_json(json.loads((run / 'run_manifest.json').read_text()))
    assert capture.snapshot(run, 20, dest, expected_manifest_sha256=manifest_sha) == dest
    assert dir_sha256(run) == original
    assert (dest / 'adapter/adapter_model.safetensors').read_bytes() == b'c20-weights'
    assert json.loads((dest / 'run_state.json').read_text())['case_cursor'] == 20
    assert not (dest / 'trainer_state.pt').exists()
    assert not list(dest.parent.glob('*.pending-*'))


def test_capture_accepts_verified_initial_base_checkpoint(source):
    run, checkpoint, _ = source
    dest = run.parent / 'base-evaluation/checkpoint_000000'
    rewrite(run / 'run_state.json', case_cursor=0, batch_cursor=0,
            policy_sha256='c0-policy')
    rewrite(checkpoint / 'COMPLETE.json', completed_cases=0,
            policy_sha256='c0-policy')
    assert capture.snapshot(run, 0, dest) == dest
    assert json.loads((dest / 'COMPLETE.json').read_text())['completed_cases'] == 0


def test_waits_before_commit_without_creating_destination(source):
    run, _, dest = source
    rewrite(run / 'run_state.json', case_cursor=15)
    assert capture.snapshot(run, 20, dest) is None
    assert not dest.parent.exists()


def test_missing_cursor_is_not_replaced_with_new_policy(source):
    run, _, dest = source
    rewrite(run / 'run_state.json', case_cursor=25, policy_sha256='c25-policy')
    with pytest.raises(RuntimeError, match='already committed C25'):
        capture.snapshot(run, 20, dest)
    assert not dest.exists()


@pytest.mark.parametrize('key,value', [('completed_cases', 15), ('policy_sha256', 'wrong'),
                                      ('protocol_sha256', 'wrong'), ('experiment_fingerprint', 'wrong')])
def test_receipt_mismatch_rejected(source, key, value):
    run, checkpoint, dest = source
    rewrite(checkpoint / 'COMPLETE.json', **{key: value})
    with pytest.raises(RuntimeError, match='identity differs'):
        capture.snapshot(run, 20, dest)
    assert not dest.exists()


def test_reuses_snapshot_after_rolling_checkpoint_cleanup(source):
    run, checkpoint, dest = source
    capture.snapshot(run, 20, dest)
    shutil.rmtree(checkpoint)
    rewrite(run / 'run_state.json', case_cursor=25, policy_sha256='c25-policy')
    assert capture.snapshot(run, 20, dest) == dest


def test_existing_snapshot_rejects_changed_payload_and_identity(source):
    run, _, dest = source
    capture.snapshot(run, 20, dest)
    with pytest.raises(RuntimeError, match='completed_cases identity'):
        capture.snapshot(run, 25, dest)
    (dest / 'adapter/adapter_model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='payload hash'):
        capture.snapshot(run, 20, dest)


def test_pinned_source_identity_and_overlapping_output_rejected(source):
    run, _, dest = source
    with pytest.raises(RuntimeError, match='manifest identity'):
        capture.snapshot(run, 20, dest, expected_manifest_sha256='wrong')
    with pytest.raises(ValueError, match='separate'):
        capture.snapshot(run, 20, run / 'snapshots/c20')


def test_config_identity_checked(source):
    run, _, dest = source
    rewrite(run / 'run_state.json', config_sha256='wrong')
    with pytest.raises(RuntimeError, match='config identity'):
        capture.snapshot(run, 20, dest)


def test_open_descriptors_survive_cleanup_during_copy(source, monkeypatch):
    run, checkpoint, dest = source
    original = capture.tempfile.mkdtemp
    def cleanup_after_files_open(*args, **kwargs):
        shutil.rmtree(checkpoint)
        rewrite(run / 'run_state.json', case_cursor=25, policy_sha256='c25-policy')
        return original(*args, **kwargs)
    monkeypatch.setattr(capture.tempfile, 'mkdtemp', cleanup_after_files_open)
    assert capture.snapshot(run, 20, dest) == dest
    assert (dest / 'adapter/adapter_model.safetensors').read_bytes() == b'c20-weights'


def test_cleanup_before_open_fails_without_partial_snapshot(source, monkeypatch):
    run, checkpoint, dest = source
    original = capture.os.open
    def cleanup_before_open(path, flags, *args, **kwargs):
        if str(path).endswith('adapter_model.safetensors'):
            shutil.rmtree(checkpoint)
            rewrite(run / 'run_state.json', case_cursor=25, policy_sha256='c25-policy')
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(capture.os, 'open', cleanup_before_open)
    with pytest.raises(RuntimeError, match='already committed C25'):
        capture.snapshot(run, 20, dest)
    assert not dest.exists()


def test_payload_symlinks_rejected(source, tmp_path):
    run, checkpoint, dest = source
    payload = checkpoint / 'adapter/adapter_model.safetensors'
    outside = tmp_path / 'outside'
    outside.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(outside)
    with pytest.raises(OSError):
        capture.snapshot(run, 20, dest)
    assert not dest.exists()
