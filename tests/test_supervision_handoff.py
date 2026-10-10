import copy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.online_rl.integrated.records import atomic_write_json, dir_sha256, sha256_json
from src.online_rl.tree import experiment, supervision_handoff as handoff


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / 'old_run'
    checkpoint = source / 'checkpoints/recovery_000003'
    checkpoint.mkdir(parents=True)
    (source / 'runner.lock').touch()
    cfg = {
        'policy': {'base_model': 'base', 'init_adapter': None, 'lora': {'r': 16}},
        'optimizer': {'learning_rate': 1e-5}, 'advantage_normalization': {'mode': 'mean_center'},
        'agent_reward': {'w_outcome': .5, 'w_rubric': .5},
        'schedule': {'seed': 0, 'cases': 120},
        'rollout': {'rounds': 2, 'temperature': 1, 'doctor_backend': {'model': 'doctor', 'endpoint': 'old'}},
        'data': {'train_sha256': 'train'},
        'five_gpu': {'batch_cases': 5, 'train_order_sha256': 'order', 'checkpoint_evaluation_enabled': False},
        'feedback': {'model': 'doctor', 'prompt_sha256': 'reflection'},
        'intervention': {'reruns_per_intervention': 3},
        'verification': {'max_valid_supports': 5},
        'prompt_update': {'enabled': False}, 'rubric': {'enabled': True, 'prompt_sha256': 'old_prompt'},
        'experiment': {'prompt_mode': 'p0_fixed'},
    }
    receipt = {'completed_cases': 15, 'policy_sha256': 'trained', 'protocol_sha256': 'old_protocol',
               'update': {'optimizer_steps': 3}, **experiment.receipt_fields(cfg)}
    state = {'case_cursor': 15, 'batch_cursor': 3, 'checkpoint': str(checkpoint),
             'config_sha256': sha256_json(cfg), 'protocol_sha256': 'old_protocol',
             'policy_sha256': 'trained', **experiment.state_fields(cfg, 'theta0')}
    manifest = {'config': cfg, 'protocol_sha256': 'old_protocol'}
    for name in handoff.REQUIRED_PAYLOAD:
        p = checkpoint / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'saved payload')
    atomic_write_json(checkpoint / 'COMPLETE.json', receipt)
    atomic_write_json(source / 'checkpoints/checkpoint_000000/COMPLETE.json', {'policy_sha256': 'theta0'})
    atomic_write_json(source / 'run_state.json', state)
    atomic_write_json(source / 'run_manifest.json', manifest)
    newcfg = copy.deepcopy(cfg)
    newcfg['rubric']['prompt_sha256'] = 'new_prompt'
    newcfg['rollout']['doctor_backend']['endpoint'] = 'new'
    newcfg['supervision_handoff'] = handoff.handoff_spec(source)
    return source, checkpoint, newcfg, tmp_path / 'new_run'


def test_identity_pins_and_valid_boundary(prepared):
    source, checkpoint, cfg, out = prepared
    checked = handoff.validate_handoff(cfg, out, 'theta0')
    assert checked['state']['case_cursor'] == 15
    assert handoff.checkpoint_sha256(checkpoint) == dir_sha256(checkpoint)
    assert checked['receipt']['update']['optimizer_steps'] == 3


@pytest.mark.parametrize('section,key,value', [
    ('optimizer', 'learning_rate', 2e-5),
    ('rollout', 'rounds', 4),
    ('rollout', 'temperature', .7),
    ('data', 'train_sha256', 'other'),
    ('five_gpu', 'train_order_sha256', 'other'),
    ('schedule', 'seed', 1),
    ('agent_reward', 'w_rubric', 1),
    ('verification', 'max_valid_supports', 10),
])
def test_training_conditions_cannot_silently_change(prepared, section, key, value):
    _, _, cfg, out = prepared
    cfg[section][key] = value
    with pytest.raises(RuntimeError, match='training conditions differ'):
        handoff.validate_handoff(cfg, out, 'theta0')


def test_source_state_or_checkpoint_changes_rejected(prepared):
    source, checkpoint, cfg, out = prepared
    (checkpoint / 'trainer_state.pt').write_bytes(b'changed optimizer')
    with pytest.raises(RuntimeError, match='payload hash'):
        handoff.validate_handoff(cfg, out)
    cfg['supervision_handoff'] = handoff.handoff_spec(source)
    state = json.loads((source / 'run_state.json').read_text())
    state['case_cursor'] = 20
    atomic_write_json(source / 'run_state.json', state)
    with pytest.raises(RuntimeError, match='run state changed'):
        handoff.validate_handoff(cfg, out)


def test_source_lock_prevents_import(prepared):
    source, _, _, _ = prepared
    with (source / 'runner.lock').open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='still active'):
            handoff.handoff_spec(source)


def test_wrong_theta0_missing_rng_and_overlapping_output_rejected(prepared):
    source, checkpoint, cfg, out = prepared
    with pytest.raises(RuntimeError, match='theta0 differs'):
        handoff.validate_handoff(cfg, out, 'different_initial_policy')
    with pytest.raises(RuntimeError, match='separate run directories'):
        handoff.validate_handoff(cfg, source)
    (checkpoint / 'cuda_rng.pt').unlink()
    with pytest.raises(RuntimeError, match='payload missing'):
        handoff.validate_handoff(cfg, out)


def _fake_io(monkeypatch, source_checkpoint):
    updater = SimpleNamespace(steps=0, policy_sha='theta0')
    def load(obj, path):
        proof = json.loads((Path(path) / 'COMPLETE.json').read_text())
        obj.policy_sha = proof['policy_sha256']
        obj.steps = proof.get('imported_optimizer_steps', proof.get('update', {}).get('optimizer_steps'))
    def save(obj, path, receipt):
        path = Path(path)
        if path.exists():
            if json.loads((path / 'COMPLETE.json').read_text()) != receipt:
                raise RuntimeError('Checkpoint receipt differs')
            return str(path)
        path.mkdir(parents=True)
        for name in handoff.REQUIRED_PAYLOAD:
            p = path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((source_checkpoint / name).read_bytes())
        atomic_write_json(path / 'COMPLETE.json', receipt)
        return str(path)
    monkeypatch.setattr(handoff, 'load_checkpoint', load)
    monkeypatch.setattr(handoff, 'save_checkpoint', save)
    return updater


def test_handoff_commits_new_identity_without_rewriting_source(prepared, monkeypatch):
    source, checkpoint, cfg, out = prepared
    old_source_hash = dir_sha256(source)
    updater = _fake_io(monkeypatch, checkpoint)
    state = handoff.prepare_handoff(updater, cfg, out, 'newconfig', 'newprotocol', 'theta0',
                                  lambda: updater.policy_sha)
    assert state['case_cursor'] == 15 and state['batch_cursor'] == 3
    assert state['checkpoint'].endswith('handoff_000015')
    assert state['experiment_fingerprint'] == experiment.fingerprint(cfg)
    assert updater.steps == 3
    assert dir_sha256(source) == old_source_hash
    assert not (out / 'cases').exists() and not (out / 'calls').exists()
    provenance = json.loads((out / 'supervision_handoff.json').read_text())
    assert provenance['source_receipt']['protocol_sha256'] == 'old_protocol'
    experiment.check_resume(cfg, out, state, 'theta0')
    with pytest.raises(RuntimeError, match='already has run state'):
        handoff.prepare_handoff(updater, cfg, out, 'newconfig', 'newprotocol', 'theta0', lambda: updater.policy_sha)


def test_crash_before_state_commit_can_retry_same_checkpoint(prepared, monkeypatch):
    _, checkpoint, cfg, out = prepared
    updater = _fake_io(monkeypatch, checkpoint)
    write = handoff.atomic_write_json
    def crash_on_state(path, value):
        if Path(path).name == 'run_state.json':
            raise OSError('interrupted before commit')
        return write(path, value)
    monkeypatch.setattr(handoff, 'atomic_write_json', crash_on_state)
    with pytest.raises(OSError, match='interrupted'):
        handoff.prepare_handoff(updater, cfg, out, 'newconfig', 'newprotocol', 'theta0', lambda: updater.policy_sha)
    assert not (out / 'run_state.json').exists()
    monkeypatch.setattr(handoff, 'atomic_write_json', write)
    state = handoff.prepare_handoff(updater, cfg, out, 'newconfig', 'newprotocol', 'theta0', lambda: updater.policy_sha)
    assert state['case_cursor'] == 15


def test_loaded_policy_and_optimizer_steps_are_verified(prepared, monkeypatch):
    _, checkpoint, cfg, out = prepared
    updater = _fake_io(monkeypatch, checkpoint)
    with pytest.raises(RuntimeError, match='loaded adapter'):
        handoff.prepare_handoff(updater, cfg, out, 'newconfig', 'newprotocol', 'theta0', lambda: 'wrong')
    assert not (out / 'run_state.json').exists()
    monkeypatch.setattr(handoff, 'load_checkpoint', lambda obj, path: setattr(obj, 'steps', 100))
    with pytest.raises(RuntimeError, match='optimizer step'):
        handoff.prepare_handoff(updater, cfg, out, 'newconfig', 'newprotocol', 'theta0', lambda: 'trained')


def test_unrelated_checkpoint_artifacts_rejected(prepared, monkeypatch):
    _, checkpoint, cfg, out = prepared
    updater = _fake_io(monkeypatch, checkpoint)
    (out / 'checkpoints/checkpoint_000000').mkdir(parents=True)
    with pytest.raises(RuntimeError, match='unrelated checkpoint'):
        handoff.prepare_handoff(updater, cfg, out, 'newconfig', 'newprotocol', 'theta0', lambda: updater.policy_sha)


def test_second_handoff_preserves_verified_theta0_chain(prepared, monkeypatch):
    _, checkpoint, cfg, first = prepared
    updater = _fake_io(monkeypatch, checkpoint)
    state = handoff.prepare_handoff(updater, cfg, first, sha256_json(cfg), 'newprotocol', 'theta0',
                                   lambda: updater.policy_sha)
    (first/'runner.lock').touch()
    secondcfg = copy.deepcopy(cfg)
    secondcfg['supervision_handoff'] = handoff.handoff_spec(first)
    checked = handoff.validate_handoff(secondcfg, first.parent/'second_run', 'theta0')
    assert checked['state']['case_cursor'] == 15
    proof = json.loads((first/'supervision_handoff.json').read_text())
    proof['theta0_policy_sha256'] = 'wrong-origin'
    atomic_write_json(first/'supervision_handoff.json', proof)
    with pytest.raises(RuntimeError, match='inherited theta0'):
        handoff.validate_handoff(secondcfg, first.parent/'second_run', 'theta0')
