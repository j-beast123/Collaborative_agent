import copy
import hashlib
import json
from pathlib import Path
import pytest
from src.online_rl.tree import b200_doctor_setup as S
from src.online_rl.tree import b200_runner as R


def files(tmp_path):
    binary = tmp_path / 'ollama'; binary.write_bytes(b'fake executable')
    models = tmp_path / 'models'
    blob = b'fake weights'; digest = hashlib.sha256(blob).hexdigest()
    p = models / 'blobs' / ('sha256-' + digest); p.parent.mkdir(parents=True); p.write_bytes(blob)
    entry = {'digest': 'sha256:' + digest, 'size': len(blob)}
    manifest = models / S.MANIFEST; manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'config': entry, 'layers': [entry]}))
    return binary, models, S.sha(manifest)


def mock_service(monkeypatch, digest):
    monkeypatch.setattr(S, 'get_json', lambda route: {'version': S.VERSION} if route == '/api/version' else
        {'models': [{'name': S.MODEL, 'digest': digest, 'details': {'quantization_level': 'Q8_0'}}]})


def test_first_pull_pins_actual_digest_and_never_pulls_again(tmp_path, monkeypatch):
    binary, models, digest = files(tmp_path); mock_service(monkeypatch, digest)
    monkeypatch.setattr(S, 'require_owned_server', lambda *a: None)
    calls = []
    monkeypatch.setattr(S.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    receipt = tmp_path / 'runtime.json'
    row = S.pull_and_pin(tmp_path, binary, models, receipt)
    assert row['digest'] == digest and row['previous_digest'] == S.OLD_DIGEST
    assert row['changed_from_original'] and len(calls) == 1
    assert S.pull_and_pin(tmp_path, binary, models, receipt) == row and len(calls) == 1
    before = receipt.read_bytes()
    mock_service(monkeypatch, 'a' * 64)
    with pytest.raises(ValueError): S.pull_and_pin(tmp_path, binary, models, receipt)
    assert receipt.read_bytes() == before and len(calls) == 1


def test_receipt_updates_every_role_and_isolates_output_and_openings(tmp_path, monkeypatch):
    binary, models, digest = files(tmp_path); mock_service(monkeypatch, digest)
    monkeypatch.setattr(S, 'require_owned_server', lambda *a: None)
    monkeypatch.setattr(S.subprocess, 'run', lambda *a, **k: None)
    receipt = tmp_path / 'runtime.json'
    S.pull_and_pin(tmp_path, binary, models, receipt)
    base = json.loads((R.ROOT / R.CONFIG).read_text())
    cfg = S.apply_receipt(copy.deepcopy(base), receipt)
    assert cfg['rollout']['doctor_backend']['model_digest'] == digest
    assert cfg['five_gpu']['opening_identity']['model_digest'] == digest
    assert cfg['rollout']['doctor_model'] == cfg['feedback']['model'] == cfg['experiment']['feedback_model']
    assert cfg['output_dir'] != base['output_dir'] and digest[:12] in cfg['output_dir']
    assert cfg['agent_reward'] == base['agent_reward'] and cfg['optimizer'] == base['optimizer']
    monkeypatch.setattr(R, 'physical_devices', lambda: ['0', '1'])
    assert all(e.startswith('Missing file:') for e in R.preflight(cfg, resources=False))
    monkeypatch.setenv('SHERPA_DOCTOR_RECEIPT', str(receipt))
    monkeypatch.delenv('SHERPA_OLLAMA_BINARY', raising=False)
    monkeypatch.delenv('SHERPA_OLLAMA_MODELS', raising=False)
    assert R.load_config()['doctor_download_receipt']['digest'] == digest
    row = json.loads(receipt.read_text()); row['digest'] = 'b' * 64
    receipt.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='integrity'): R.load_config()


def test_wrong_blob_or_service_identity_cannot_be_pinned(tmp_path, monkeypatch):
    binary, models, digest = files(tmp_path); mock_service(monkeypatch, digest)
    blob = next((models / 'blobs').iterdir()); blob.write_bytes(b'xxxx weights')
    with pytest.raises(ValueError): S.checked_model(models, verify_blobs=True)


def test_unowned_server_blocks_pull_before_any_download(tmp_path, monkeypatch):
    binary, models, _ = files(tmp_path)
    monkeypatch.setattr(S.subprocess, 'run', lambda *a, **k: pytest.fail('must not download'))
    with pytest.raises(FileNotFoundError): S.pull_and_pin(tmp_path, binary, models, tmp_path / 'receipt.json')


def test_missing_explicit_receipt_does_not_fall_back_to_old_model(tmp_path, monkeypatch):
    monkeypatch.setenv('SHERPA_DOCTOR_RECEIPT', str(tmp_path / 'missing'))
    with pytest.raises(ValueError, match='missing'): R.load_config()


def test_existing_install_is_verified_without_download(tmp_path, monkeypatch):
    target = tmp_path / ('ollama-' + S.VERSION); (target / 'bin').mkdir(parents=True)
    binary = target / 'bin/ollama'; binary.write_bytes(b'executable')
    monkeypatch.setattr(S.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(S.platform, 'machine', lambda: 'x86_64')
    (target / 'install.json').write_text(json.dumps({'archive_sha256': S.ASSETS['x86_64'][1], 'binary_sha256': S.sha(binary)}))
    monkeypatch.setattr(S.subprocess, 'run', lambda *a, **kw: pytest.fail('must not download'))
    assert S.install(tmp_path) == binary
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError, match='refusing'): S.install(tmp_path)


def test_install_download_verifies_archive_and_publishes_private_binary(tmp_path, monkeypatch):
    import io
    import tarfile
    archive = tmp_path / 'fixture.tgz'
    with tarfile.open(archive, 'w:gz') as stream:
        item = tarfile.TarInfo('bin/ollama'); item.size = 4; item.mode = 0o755
        stream.addfile(item, io.BytesIO(b'fake'))
    body = archive.read_bytes()
    monkeypatch.setattr(S.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(S.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(S, 'ASSETS', {'x86_64': ('amd64', hashlib.sha256(body).hexdigest())})
    monkeypatch.setattr(S.shutil, 'disk_usage', lambda p: type('Usage', (), {'free': 40 * 1024**3})())
    calls = []
    def download(command, **kwargs):
        calls.append(command)
        Path(command[command.index('--output') + 1]).write_bytes(body)
    monkeypatch.setattr(S.subprocess, 'run', download)
    binary = S.install(tmp_path)
    assert binary.read_bytes() == b'fake' and len(calls) == 1
    assert S.install(tmp_path) == binary and len(calls) == 1
    assert json.loads((binary.parent.parent / 'install.json').read_text())['binary_sha256'] == S.sha(binary)


def test_wrong_archive_hash_never_installs(tmp_path, monkeypatch):
    monkeypatch.setattr(S.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(S.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(S.shutil, 'disk_usage', lambda p: type('Usage', (), {'free': 40 * 1024**3})())
    monkeypatch.setattr(S.subprocess, 'run', lambda cmd, **kw: Path(cmd[cmd.index('--output') + 1]).write_bytes(b'corrupt'))
    with pytest.raises(ValueError, match='SHA256 mismatch'): S.install(tmp_path)
    assert not (tmp_path / ('ollama-' + S.VERSION)).exists()
