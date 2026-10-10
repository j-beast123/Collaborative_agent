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
