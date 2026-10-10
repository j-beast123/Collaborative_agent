"""CPU-only proofs of completion gating and evaluation continuation."""
import fcntl
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.online_rl import run_idle_checkpoint_eval as supervisor
from src.online_rl.integrated.records import atomic_write_json, dir_sha256, sha256_json
from src.online_rl.tree import idle_eval_priority, idle_eval_worker
from src.online_rl.tree.idle_eval_priority import EvaluationDeferred, PriorityCoordinator, PriorityTimeout


@pytest.fixture
def completed(tmp_path):
    source = tmp_path / 'training'
    checkpoint = source / 'checkpoints/checkpoint_000120'
    (checkpoint / 'adapter').mkdir(parents=True)
    (source / 'runner.lock').touch()
    cfg = {'schedule': {'cases': 120}, 'five_gpu': {'checkpoint_evaluation_enabled': False}}
    manifest = {'config': cfg, 'protocol_sha256': 'protocol'}
    state = {'case_cursor': 120, 'checkpoint': str(checkpoint), 'policy_sha256': 'final-policy',
             'protocol_sha256': 'protocol', 'config_sha256': sha256_json(cfg),
             'experiment_fingerprint': 'experiment'}
    receipt = {'completed_cases': 120, 'policy_sha256': 'final-policy',
               'protocol_sha256': 'protocol', 'experiment_fingerprint': 'experiment'}
    atomic_write_json(source / 'run_manifest.json', manifest)
    atomic_write_json(source / 'run_state.json', state)
    atomic_write_json(checkpoint / 'COMPLETE.json', receipt)
    atomic_write_json(checkpoint / 'adapter/adapter_config.json', {'r': 16})
    (checkpoint / 'adapter/adapter_model.safetensors').write_bytes(b'final-adapter')
    return source, checkpoint, sha256_json(manifest), {'version': 1, 'gpus': {}}


def rewrite(path, **updates):
    value = json.loads(path.read_text())
    atomic_write_json(path, {**value, **updates})


def test_terminal_proof_is_read_only_and_binds_final_source(completed):
    source, checkpoint, manifest_sha, priority = completed
    original = dir_sha256(source)
    proof = supervisor.terminal_training_receipt(source, manifest_sha, 120, priority)
    assert proof['completed_cases'] == 120
    assert proof['policy_sha256'] == 'final-policy'
    assert proof['checkpoint_receipt_sha256'] == sha256_json(json.loads((checkpoint / 'COMPLETE.json').read_text()))
    assert dir_sha256(source) == original


def test_missing_trainer_before_final_cursor_does_not_allow_continuation(completed):
    source, _, manifest_sha, priority = completed
    rewrite(source / 'run_state.json', case_cursor=25)
    assert supervisor.terminal_training_receipt(source, manifest_sha, 120, priority) is None
    rewrite(source / 'run_state.json', case_cursor=125)
    with pytest.raises(RuntimeError, match='passed'):
        supervisor.terminal_training_receipt(source, manifest_sha, 120, priority)


@pytest.mark.parametrize('bucket', ['registrations', 'training'])
def test_final_commit_waits_for_live_training_owners(completed, bucket):
    source, _, manifest_sha, priority = completed
    priority['gpus']['4'] = {bucket: {'owner': {'pid': 1}}}
    assert supervisor.terminal_training_receipt(source, manifest_sha, 120, priority) is None


def test_final_commit_waits_for_runner_cleanup_and_requires_existing_lock(completed):
    source, _, manifest_sha, priority = completed
    with (source / 'runner.lock').open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert supervisor.terminal_training_receipt(source, manifest_sha, 120, priority) is None
    (source / 'runner.lock').unlink()
    assert supervisor.terminal_training_receipt(source, manifest_sha, 120, priority) is None


@pytest.mark.parametrize('field,value', [('completed_cases', 115), ('policy_sha256', 'other'),
                                       ('protocol_sha256', 'other'), ('experiment_fingerprint', 'other')])
def test_mismatched_final_checkpoint_receipt_fails_closed(completed, field, value):
    source, checkpoint, manifest_sha, priority = completed
    rewrite(checkpoint / 'COMPLETE.json', **{field: value})
    with pytest.raises(RuntimeError, match='identity differs'):
        supervisor.terminal_training_receipt(source, manifest_sha, 120, priority)


def test_changed_manifest_config_or_target_cannot_enable_continuation(completed):
    source, _, manifest_sha, priority = completed
    with pytest.raises(RuntimeError, match='manifest identity'):
        supervisor.terminal_training_receipt(source, 'wrong', 120, priority)
    with pytest.raises(RuntimeError, match='target differs'):
        supervisor.terminal_training_receipt(source, manifest_sha, 115, priority)
    rewrite(source / 'run_state.json', config_sha256='wrong')
    with pytest.raises(RuntimeError, match='config identity'):
        supervisor.terminal_training_receipt(source, manifest_sha, 120, priority)


def test_missing_payload_and_wrong_checkpoint_path_cannot_enable_continuation(completed):
    source, checkpoint, manifest_sha, priority = completed
    rewrite(source / 'run_state.json', checkpoint=str(source / 'checkpoints/recovery_000024'))
    with pytest.raises(RuntimeError, match='checkpoint differs'):
        supervisor.terminal_training_receipt(source, manifest_sha, 120, priority)
    rewrite(source / 'run_state.json', checkpoint=str(checkpoint))
    (checkpoint / 'adapter/adapter_model.safetensors').unlink()
    with pytest.raises(RuntimeError, match='payload'):
        supervisor.terminal_training_receipt(source, manifest_sha, 120, priority)


def test_post_training_workers_still_yield_to_activity_and_updates(tmp_path):
    coordinator = PriorityCoordinator(tmp_path, require_training=False, idle_grace_seconds=0)
    resident = coordinator.evaluation_residency(4)
    with coordinator.evaluation_call(4, resident):
        pass
    with coordinator.training_activity(4, 'restarted-trainer'):
        with pytest.raises(EvaluationDeferred):
            coordinator.evaluation_call(4, resident)
    with pytest.raises(PriorityTimeout):
        with coordinator.training_update(4, timeout=0):
            pytest.fail('Update entered before evaluation memory released')
    assert resident.revoked()
    resident.release()
    with coordinator.training_update(4, timeout=0):
        with pytest.raises(EvaluationDeferred):
            coordinator.evaluation_residency(4)


def test_supervisor_rejects_a_full_count_of_invalid_result_files(tmp_path):
    case = {'case_id': 'one', 'ground_truth': {'correct_option': 'A'}}
    job = {'job_sha256': 'job', 'policy_sha256': 'c20', 'completed_cases': 20,
           'snapshot_sha256': 'snapshot', 'eval_seed': 0, 'rollout': {'rounds': 2},
           'dataset_cases': 1, 'case_ids': ['one'], 'original_case_sha256': [sha256_json(case)]}
    assert supervisor.evaluation_progress(tmp_path, job) == (0, False)
    path = tmp_path / 'cases/case_000000.json'
    row = {'index': 0, 'case_id': 'one', 'job_sha256': 'other-job', 'policy_sha256': 'c20',
           'original_case': case, 'seed': 0, 'snapshot_sha256': 'snapshot', 'final_option': 'A',
           'outcome': 1, 'num_discussion_rounds': 2, 'max_discussion_rounds': 2}
    atomic_write_json(path, row)
    with pytest.raises(RuntimeError, match='another job'):
        supervisor.evaluation_progress(tmp_path, job)
    atomic_write_json(path, {**row, 'job_sha256': 'job'})
    assert supervisor.evaluation_progress(tmp_path, job) == (1, True)


def test_worker_propagates_terminal_mode_without_gpu_loading(tmp_path, monkeypatch):
    job = {'job_sha256': 'job'}
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    monkeypatch.setattr(idle_eval_worker.os, 'environ', dict(idle_eval_worker.os.environ))
    monkeypatch.setattr(idle_eval_worker, 'ensure_job', lambda *args: job)
    monkeypatch.setattr(idle_eval_worker, '_inputs', lambda *args: ({}, {}, {}, [], []))
    monkeypatch.setattr(idle_eval_worker, 'summarize', lambda *args: {'status': 'pending'})
    captured = {}
    def coordinator(root, *, require_training):
        captured['require_training'] = require_training
        def defer(gpu):
            raise EvaluationDeferred('test: no residency')
        return SimpleNamespace(evaluation_residency=defer)
    monkeypatch.setattr(idle_eval_priority, 'PriorityCoordinator', coordinator)
    result = idle_eval_worker.run_worker('config', 'snapshot', tmp_path, 0, 'priority', 'ollama',
                                         require_training=False)
    assert result['status'] == 'deferred'
    assert captured == {'require_training': False}


@pytest.fixture
def continued(completed, tmp_path):
    """A stopped C35 source and a proven new supervision run ending at C120."""
    source, checkpoint, source_sha, priority = completed
    import_checkpoint = checkpoint.with_name('recovery_000007')
    checkpoint.rename(import_checkpoint)
    rewrite(source/'run_state.json', case_cursor=35, checkpoint=str(import_checkpoint))
    rewrite(import_checkpoint/'COMPLETE.json', completed_cases=35)
    old_state = json.loads((source/'run_state.json').read_text())
    old_manifest = json.loads((source/'run_manifest.json').read_text())
    old_receipt = json.loads((import_checkpoint/'COMPLETE.json').read_text())
    new = tmp_path/'continued-training'
    final = new/'checkpoints/checkpoint_000120'
    (final/'adapter').mkdir(parents=True)
    (new/'runner.lock').touch()
    spec = {'contract': 'explicit_supervision_checkpoint_handoff_v1', 'source_run': str(source),
            'source_checkpoint': str(import_checkpoint), 'source_state_sha256': sha256_json(old_state),
            'source_manifest_sha256': source_sha, 'source_checkpoint_sha256': 'imported-payload-hash'}
    cfg = {**old_manifest['config'], 'supervision_handoff': spec}
    proof = {**spec, 'completed_cases': 35, 'source_policy_sha256': old_state['policy_sha256'],
             'source_protocol_sha256': old_state['protocol_sha256'], 'destination_protocol_sha256': 'new-protocol',
             'destination_config_sha256': sha256_json(cfg), 'source_receipt': old_receipt}
    manifest = {'config': cfg, 'protocol_sha256': 'new-protocol', 'supervision_handoff': proof}
    state = {'case_cursor': 120, 'checkpoint': str(final), 'policy_sha256': 'new-final-policy',
             'protocol_sha256': 'new-protocol', 'config_sha256': sha256_json(cfg),
             'experiment_fingerprint': 'new-experiment', 'supervision_handoff_sha256': sha256_json(proof)}
    atomic_write_json(new/'run_manifest.json', manifest)
    atomic_write_json(new/'run_state.json', state)
    atomic_write_json(new/'supervision_handoff.json', proof)
    atomic_write_json(new/'checkpoints/handoff_000035/COMPLETE.json', {
        'completed_cases': 35, 'policy_sha256': old_state['policy_sha256'], 'protocol_sha256': 'new-protocol',
        'supervision_handoff_sha256': sha256_json(proof), 'source_policy_sha256': old_state['policy_sha256']})
    atomic_write_json(final/'COMPLETE.json', {'completed_cases': 120, 'policy_sha256': 'new-final-policy',
        'protocol_sha256': 'new-protocol', 'experiment_fingerprint': 'new-experiment'})
    atomic_write_json(final/'adapter/adapter_config.json', {'r': 16})
    (final/'adapter/adapter_model.safetensors').write_bytes(b'new-final-adapter')
    pointer = tmp_path/'evaluation/training_continuation.json'
    atomic_write_json(pointer, {'contract': supervisor.TRAINING_CONTINUATION, 'training_run': str(new),
                               'training_manifest_sha256': sha256_json(manifest)})
    return source, source_sha, priority, new, pointer


def test_absent_continuation_keeps_original_terminal_source(completed, tmp_path):
    source, _, source_sha, _ = completed
    assert supervisor.resolve_training_control(source, source_sha, 120, tmp_path/'missing.json') == (source, source_sha, 120)


def test_continuation_changes_only_terminal_control_after_verified_handoff(continued):
    source, source_sha, priority, new, pointer = continued
    original = dir_sha256(source)
    control = supervisor.resolve_training_control(source, source_sha, 120, pointer)
    assert control == (new, sha256_json(json.loads((new/'run_manifest.json').read_text())), 120)
    assert supervisor.terminal_training_receipt(source, source_sha, 120, priority) is None
    proof = supervisor.terminal_training_receipt(*control, priority)
    assert proof['source_run'] == str(new) and proof['policy_sha256'] == 'new-final-policy'
    assert dir_sha256(source) == original


def test_continuation_never_waives_live_training_or_early_cursor(continued):
    source, source_sha, priority, new, pointer = continued
    control = supervisor.resolve_training_control(source, source_sha, 120, pointer)
    priority['gpus']['4'] = {'registrations': {'running': {'pid': 1}}}
    assert supervisor.terminal_training_receipt(*control, priority) is None
    priority['gpus'].clear()
    rewrite(new/'run_state.json', case_cursor=115)
    control = supervisor.resolve_training_control(source, source_sha, 120, pointer)
    assert supervisor.terminal_training_receipt(*control, priority) is None


@pytest.mark.parametrize('fault', ['pointer_contract', 'pointer_hash', 'relative_path', 'same_run',
                                   'source_state', 'source_receipt', 'new_state_handoff',
                                   'handoff_file', 'import_receipt', 'manifest_handoff', 'target'])
def test_invalid_continuation_fails_closed(continued, fault):
    source, source_sha, _, new, pointer = continued
    if fault == 'pointer_contract': rewrite(pointer, contract='other')
    elif fault == 'pointer_hash': rewrite(pointer, training_manifest_sha256='wrong')
    elif fault == 'relative_path': rewrite(pointer, training_run='relative/run')
    elif fault == 'same_run': rewrite(pointer, training_run=str(source))
    elif fault == 'source_state': rewrite(source/'run_state.json', case_cursor=40)
    elif fault == 'source_receipt': rewrite(source/'checkpoints/recovery_000007/COMPLETE.json', policy_sha256='wrong')
    elif fault == 'new_state_handoff': rewrite(new/'run_state.json', supervision_handoff_sha256='wrong')
    elif fault == 'handoff_file': rewrite(new/'supervision_handoff.json', completed_cases=40)
    elif fault == 'import_receipt': rewrite(new/'checkpoints/handoff_000035/COMPLETE.json', policy_sha256='wrong')
    elif fault in ('manifest_handoff', 'target'):
        manifest = json.loads((new/'run_manifest.json').read_text())
        if fault == 'manifest_handoff': manifest['supervision_handoff']['source_manifest_sha256'] = 'wrong'
        else: manifest['config']['schedule']['cases'] = 125
        atomic_write_json(new/'run_manifest.json', manifest)
        rewrite(pointer, training_manifest_sha256=sha256_json(manifest))
        rewrite(new/'run_state.json', config_sha256=sha256_json(manifest['config']))
    with pytest.raises(RuntimeError):
        supervisor.resolve_training_control(source, source_sha, 120, pointer)


def test_symlink_continuation_pointer_is_not_followed(continued):
    source, source_sha, _, _, pointer = continued
    real = pointer.with_suffix('.real')
    pointer.rename(real)
    pointer.symlink_to(real)
    with pytest.raises(RuntimeError, match='symlink'):
        supervisor.resolve_training_control(source, source_sha, 120, pointer)
