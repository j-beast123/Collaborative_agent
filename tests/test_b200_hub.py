"""Offline fake Hub commits/downloads, checkpoint preservation and data pins."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.online_rl.tree import b200_hub as H
from src.online_rl.integrated.records import atomic_write_json as write, sha256_json


def run_fixture(root):
    out = root/'run'
    cfg = {'output_dir': str(out), 'optimizer': {'learning_rate': 1e-5}}
    state = {'checkpoint': str(out/'checkpoints/checkpoint_000000'), 'case_cursor': 0,
             'config_sha256': sha256_json(cfg), 'source_sha256': 'fixed-source',
             'policy_sha256': 'policy', 'base_model_identity': {'model': 'hash'}}
    write(out/'run_manifest.json', {'config': cfg})
    write(out/'run_state.json', state)
    write(Path(state['checkpoint'])/'COMPLETE.json', {'policy_sha256': 'policy', 'case_cursor': 0})
    (Path(state['checkpoint'])/'optimizer.pt').write_bytes(b'synthetic optimizer')
    (Path(state['checkpoint'])/'adapter_model.safetensors').write_bytes(b'synthetic adapter')
    write(out/'batches/batch_000001/case_000000/cache.jsonl', {'completed_verification_trial': 2})
    (out/'.env').write_text('SECRET_SHOULD_NOT_UPLOAD')
    (out/'runtime.local.json').write_text('SECRET_SHOULD_NOT_UPLOAD')
    return out


class Hub:
    def __init__(self, tmp): self.tmp=tmp; self.files={}; self.commits=0; self.fail=False
    def repo_info(self, **kwargs): return SimpleNamespace(sha=f'{self.commits:040x}')
    def list_repo_files(self, **kwargs): return list(self.files)
    def download(self, *, filename, **kwargs):
        p=self.tmp/'download'/filename; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.files[filename]); return str(p)
    def create_commit(self, *, operations, parent_commit, **kwargs):
        if self.fail: raise OSError('synthetic network failure')
        assert parent_commit==f'{self.commits:040x}'
        updated=copy.deepcopy(self.files)
        for op in operations:
            if hasattr(op, 'path_or_fileobj'):
                data=op.path_or_fileobj
                updated[op.path_in_repo]=data if isinstance(data,bytes) else Path(data).read_bytes()
            else: updated.pop(op.path_in_repo)
        self.files=updated;self.commits+=1
        return SimpleNamespace(oid=f'{self.commits:040x}')


def test_snapshot_preserves_optimizer_trials_and_skips_credentials(tmp_path):
    out=run_fixture(tmp_path);hub=Hub(tmp_path)
    result=H.upload_run(out,'experiment1',api=hub,download=hub.download)
    assert hub.commits==1 and result['completed_cases']==0
    assert 'runs/experiment1/artifacts/checkpoints/checkpoint_000000/optimizer.pt' in hub.files
    assert not any(b'SECRET_SHOULD_NOT_UPLOAD' in v for v in hub.files.values())
    assert H.upload_run(out,'experiment1',api=hub,download=hub.download)['unchanged']
    restored=tmp_path/'restored'
    answer=H.download_run('experiment1',result['revision'],restored,download=hub.download)
    assert answer['same_output_path'] is False
    for name in H.inventory(out)['files']:
        assert (out/name).read_bytes()==(restored/name).read_bytes()
    with pytest.raises(ValueError,match='must not exist'):
        H.download_run('experiment1',result['revision'],restored,download=hub.download)


def test_hub_failure_leaves_local_run_unchanged_and_retry_works(tmp_path):
    out=run_fixture(tmp_path);hub=Hub(tmp_path);before=H.inventory(out);hub.fail=True
    with pytest.raises(OSError): H.upload_run(out,'experiment1',api=hub,download=hub.download)
    assert H.inventory(out)==before and not hub.files
    hub.fail=False
    assert H.upload_run(out,'experiment1',api=hub,download=hub.download)['revision']


def test_refuses_run_id_collision_and_newer_progress(tmp_path):
    out=run_fixture(tmp_path);hub=Hub(tmp_path)
    H.upload_run(out,'experiment1',api=hub,download=hub.download)
    marker='runs/experiment1/snapshot.json'
    old=json.loads(hub.files[marker]);old['snapshot']['source_sha256']='different'
    old['sha256']=sha256_json(old['snapshot'])
    hub.files[marker]=json.dumps(old).encode()
    with pytest.raises(ValueError,match='another configuration'):
        H.upload_run(out,'experiment1',api=hub,download=hub.download)
    old['snapshot']['source_sha256']='fixed-source';old['snapshot']['completed_cases']=5
    old['sha256']=sha256_json(old['snapshot'])
    hub.files[marker]=json.dumps(old).encode()
    with pytest.raises(ValueError,match='newer HF progress'):
        H.upload_run(out,'experiment1',api=hub,download=hub.download)


def test_incomplete_checkpoint_or_symlink_is_rejected(tmp_path):
    out=run_fixture(tmp_path)
    p=out/'checkpoints/recovery.pending';p.mkdir()
    with pytest.raises(ValueError,match='Incomplete'): H.inventory(out)
    p.rmdir();(out/'cases').mkdir();(out/'cases/link').symlink_to(out/'.env')
    with pytest.raises(ValueError,match='Symlink'): H.inventory(out)


def test_tampered_download_never_publishes_destination(tmp_path):
    out=run_fixture(tmp_path);hub=Hub(tmp_path)
    result=H.upload_run(out,'experiment1',api=hub,download=hub.download)
    hub.files['runs/experiment1/artifacts/checkpoints/checkpoint_000000/optimizer.pt']=b'corrupt'
    with pytest.raises(ValueError,match='hash differs'):
        H.download_run('experiment1',result['revision'],tmp_path/'restore',download=hub.download)
    assert not (tmp_path/'restore').exists()


def test_data_uses_exact_revision_and_rejects_changed_bytes(tmp_path):
    called=[];wrong=tmp_path/'wrong.jsonl';wrong.write_text('{}\n')
    def download(**kw): called.append(kw);return wrong
    with pytest.raises(ValueError,match='dataset bytes differ'): H.prepare_data(download=download,root=tmp_path)
    assert called[0]['revision']==H.DATA_REVISION and called[0]['repo_type']=='dataset'


def test_enabled_backup_only_and_offline_mode_restored(monkeypatch,tmp_path):
    import huggingface_hub.constants as constants
    monkeypatch.delenv('SHERPA_HF_RUN_ID',raising=False)
    assert H.sync_barrier(tmp_path) is None
    monkeypatch.setenv('SHERPA_HF_RUN_ID','test-run')
    monkeypatch.setattr(constants,'HF_HUB_OFFLINE',True)
    def upload(*a,**kw):
        assert constants.HF_HUB_OFFLINE is False
        return {'revision':'fake'}
    monkeypatch.setattr(H,'upload_run',upload)
    assert H.sync_barrier(tmp_path)=={'revision':'fake'}
    assert constants.HF_HUB_OFFLINE is True


@pytest.mark.parametrize('value',['../escape','/abs','bad/name','', 'a\\b'])
def test_unsafe_ids_rejected(value):
    with pytest.raises(ValueError): H.run_id(value)


def test_only_pinned_commit_can_be_loaded(tmp_path):
    with pytest.raises(ValueError,match='immutable'):
        H.download_run('experiment1','main',tmp_path/'out')


def http_error(status):
    from requests import Response
    from huggingface_hub.errors import HfHubHTTPError
    response = Response(); response.status_code = status
    return HfHubHTTPError('synthetic commit error', response=response)


def test_concurrent_other_run_commit_retries_without_overwriting(tmp_path):
    out = run_fixture(tmp_path); hub = Hub(tmp_path); calls = []; waits = []
    commit = hub.create_commit
    def concurrent_commit(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            hub.files['runs/other/artifacts/cache.json'] = b'other run must survive'
            hub.commits += 1
            raise http_error(409)
        return commit(**kwargs)
    hub.create_commit = concurrent_commit
    result = H.upload_run(out, 'experiment1', api=hub, download=hub.download, sleep=waits.append)
    assert result['commit_conflict_retries'] == 1 and waits == [1]
    assert hub.files['runs/other/artifacts/cache.json'] == b'other run must survive'
    assert calls[0]['parent_commit'] != calls[1]['parent_commit']
    assert calls[0]['operations'][0] is not calls[1]['operations'][0]
    assert H.upload_run(out, 'experiment1', api=hub, download=hub.download)['unchanged']


@pytest.mark.parametrize('status,expected', [(409, 3), (412, 3), (401, 1), (403, 1)])
def test_commit_retries_are_bounded_and_auth_errors_not_retried(tmp_path, status, expected):
    out = run_fixture(tmp_path); hub = Hub(tmp_path); calls = []; waits = []
    before = H.inventory(out)
    def fail(**kwargs): calls.append(kwargs); raise http_error(status)
    hub.create_commit = fail
    with pytest.raises(Exception, match='synthetic commit error'):
        H.upload_run(out, 'experiment1', api=hub, download=hub.download,
                     max_commit_attempts=3, sleep=waits.append)
    assert len(calls) == expected and len(waits) == expected - 1
    assert H.inventory(out) == before


def test_conflict_retry_revalidates_same_run_identity(tmp_path):
    out = run_fixture(tmp_path); hub = Hub(tmp_path)
    def conflicting_commit(**kwargs):
        snapshot = H.inventory(out); snapshot['config_sha256'] = 'someone-else'
        hub.files['runs/experiment1/snapshot.json'] = json.dumps(
            {'snapshot': snapshot, 'sha256': sha256_json(snapshot)}).encode()
        hub.commits += 1
        raise http_error(409)
    hub.create_commit = conflicting_commit
    with pytest.raises(ValueError, match='another configuration'):
        H.upload_run(out, 'experiment1', api=hub, download=hub.download, sleep=lambda _: None)
    assert hub.commits == 1


def test_run_name_controls_backup_even_with_legacy_hf_name(monkeypatch, tmp_path):
    monkeypatch.setenv('SHERPA_RUN_NAME', 'new-experiment')
    monkeypatch.setenv('SHERPA_HF_RUN_ID', 'legacy')
    calls = []
    monkeypatch.setattr(H, 'upload_run', lambda out, name, **kw: calls.append(name) or {})
    H.sync_barrier(tmp_path)
    assert calls == ['new-experiment']
    assert H.storage_settings()['run_id'] == 'new-experiment'
