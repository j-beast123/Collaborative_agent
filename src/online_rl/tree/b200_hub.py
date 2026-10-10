"""Explicit HF data preparation and atomic, content-verified run backups.

No I/O on import. The training hook runs synchronously only at an idle batch
barrier. HF storage is independent of the local GRPO model-loading path.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time

from src.online_rl.integrated.records import atomic_write_json, sha256_json

ROOT = Path(__file__).resolve().parents[3]
REPO = 'zzu819/SHERPA'
DATA_REVISION = '7f222446084ed59885986a2a35fb6e467110da96'
DATA = {
    'train/medmcqa_case120.jsonl': '3d34929094925c822062d03839b49de8f6fb43afba71fb573de409471bb5b8ab',
    'test/casimedicos-arg.jsonl': 'e42077d006c4e8227564f99b06050b66e7df9989f9b2b17f5fc5d9dd083feb56',
}
DIRECTORIES = {'checkpoints', 'batches', 'cases', 'diagnostics', 'evaluation', 'openings'}
TOP_FILES = {'run_state.json', 'run_manifest.json', 'final_evaluation_selection.json', 'final_accuracy.json',
             'api_usage.jsonl', 'api_usage_summary.json', 'retry_audit.jsonl', 'http_attempts.jsonl'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def safe_path(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or '..' in p.parts or '\\' in value:
        raise ValueError('Invalid artifact-relative path')
    return p


def run_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value or ''):
        raise ValueError('SHERPA_HF_RUN_ID must be 1..96 letters/digits/dots/underscores/hyphens')
    return value


@contextmanager
def online_hub():
    # Called only while all model/case tasks are idle. Keep offline local model
    # loading in the parent run; temporarily allow the explicit storage operation.
    import huggingface_hub.constants as constants
    previous = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = False
    try: yield
    finally: constants.HF_HUB_OFFLINE = previous


def prepare_data(*, download=None, root=ROOT):
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    paths = {}
    for filename, expected in DATA.items():
        p = Path(download(repo_id=REPO, repo_type='dataset', revision=DATA_REVISION, filename=filename))
        if sha(p) != expected: raise ValueError('HF dataset bytes differ: ' + filename)
        paths[filename] = p
    from scripts.online_rl.prepare_train120 import prepare
    folder = root/'data/interim/medmcqa/five_gpu_train120_seed20261008'
    result = prepare(paths['train/medmcqa_case120.jsonl'], folder)
    test = root/'data/interim/casimedicos/all_en_622.jsonl'
    if test.exists() and sha(test) != DATA['test/casimedicos-arg.jsonl']:
        raise ValueError('Refusing to replace a different test file')
    test.parent.mkdir(parents=True, exist_ok=True)
    if not test.exists(): shutil.copyfile(paths['test/casimedicos-arg.jsonl'], test)
    receipt = {'repo': REPO, 'revision': DATA_REVISION, 'source_sha256': DATA,
               'train': result, 'test_file': str(test), 'test_cases': 622}
    atomic_write_json(root/'.local/hf_data_receipt.json', receipt)
    return receipt


def inventory(out):
    """Only recognized generated artifacts; no env files, key stores or raw models."""
    out = Path(out).resolve()
    state = json.loads((out/'run_state.json').read_text())
    checkpoint = Path(state['checkpoint']).resolve()
    if checkpoint.parent != out/'checkpoints' or not (checkpoint/'COMPLETE.json').is_file():
        raise ValueError('Current checkpoint is outside run or incomplete')
    proof = json.loads((checkpoint/'COMPLETE.json').read_text())
    if proof['policy_sha256'] != state['policy_sha256']:
        raise ValueError('Checkpoint and committed policy differ')
    if proof.get('completed_cases', proof.get('case_cursor')) != state['case_cursor']:
        raise ValueError('Checkpoint and committed progress differ')
    manifest = json.loads((out/'run_manifest.json').read_text())
    if sha256_json(manifest['config']) != state['config_sha256']:
        raise ValueError('Run configuration identity differs')
    files = {}
    for p in sorted(out.rglob('*')):
        relative = p.relative_to(out)
        if relative.parts[0] not in DIRECTORIES and str(relative) not in TOP_FILES: continue
        if p.is_symlink(): raise ValueError('Symlink in run artifacts: ' + str(relative))
        if any(x.endswith(('.pending', '.tmp')) for x in relative.parts):
            raise ValueError('Incomplete publication in run; recover locally before backup')
        if not p.is_file() or p.name.endswith('.lock') or p.name == 'runner.lock': continue
        if p.name.startswith('.env') or p.suffix in {'.key', '.pem'}:
            raise ValueError('Credential-like file in run artifacts')
        files[relative.as_posix()] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    return {'schema': 'sherpa_hf_run_v1', 'original_output': str(out),
            'completed_cases': state['case_cursor'], 'config_sha256': state['config_sha256'],
            'source_sha256': state['source_sha256'], 'policy_sha256': state['policy_sha256'],
            'base_model_identity': state['base_model_identity'], 'files': files}


def upload_run(out, name, *, repo=REPO, api=None, download=None):
    """Caller owns runner.lock or is the parent at a committed batch barrier."""
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, hf_hub_download
    name = run_id(name); out = Path(out).resolve(); prefix = f'runs/{name}'
    api = api or HfApi(); download = download or hf_hub_download
    snapshot = inventory(out); digest = sha256_json(snapshot)
    head = api.repo_info(repo_id=repo, repo_type='model').sha
    remote = set(api.list_repo_files(repo_id=repo, repo_type='model', revision=head))
    marker = prefix+'/snapshot.json'
    if marker in remote:
        old = json.loads(Path(download(repo_id=repo, repo_type='model', revision=head, filename=marker)).read_text())
        if sha256_json(old['snapshot']) != old['sha256']: raise ValueError('Corrupt existing HF snapshot identity')
        for key in ('config_sha256', 'source_sha256', 'original_output'):
            if old['snapshot'][key] != snapshot[key]: raise ValueError('HF run ID belongs to another configuration/path; choose a new run ID')
        if old['snapshot']['completed_cases'] > snapshot['completed_cases']:
            raise ValueError('Refusing to overwrite newer HF progress')
        if old['sha256'] == digest:
            return {'repo': repo, 'run_id': name, 'revision': head, 'unchanged': True}
    elif any(p.startswith(prefix+'/') for p in remote):
        raise ValueError('Existing HF run lacks snapshot identity')
    started = time.monotonic()
    desired = {prefix+'/artifacts/'+f for f in snapshot['files']}
    operations = [CommitOperationAdd(path_in_repo=prefix+'/artifacts/'+f, path_or_fileobj=str(out/f))
                  for f in snapshot['files']]
    operations += [CommitOperationDelete(path_in_repo=p) for p in sorted(remote)
                   if p.startswith(prefix+'/artifacts/') and p not in desired]
    operations.append(CommitOperationAdd(path_in_repo=marker,
        path_or_fileobj=json.dumps({'sha256': digest, 'snapshot': snapshot}, sort_keys=True).encode()))
    commit = api.create_commit(repo_id=repo, repo_type='model', operations=operations,
        parent_commit=head, commit_message=f'SHERPA {name}: {snapshot["completed_cases"]}/120 cases')
    receipt = {'repo': repo, 'run_id': name, 'revision': commit.oid, 'snapshot_sha256': digest,
               'files': len(snapshot['files']), 'seconds': time.monotonic()-started,
               'completed_cases': snapshot['completed_cases']}
    # Storage receipt is excluded from the next inventory, so identical snapshots
    # are idempotent. HF history retains previous commits and deleted files.
    atomic_write_json(out/'hf_sync.json', receipt)
    return receipt


def download_run(name, revision, destination, *, repo=REPO, download=None):
    """Restore verified bytes to an empty directory; never edit resume identities."""
    from huggingface_hub import hf_hub_download
    if not re.fullmatch('[0-9a-f]{40}', revision): raise ValueError('Use an immutable HF commit SHA, not main')
    name = run_id(name); prefix = f'runs/{name}'; download = download or hf_hub_download
    def get(path): return Path(download(repo_id=repo, repo_type='model', revision=revision, filename=prefix+'/'+path))
    saved = json.loads(get('snapshot.json').read_text()); snapshot = saved['snapshot']
    if snapshot.get('schema') != 'sherpa_hf_run_v1' or sha256_json(snapshot) != saved['sha256']:
        raise ValueError('Corrupt HF snapshot manifest')
    dest = Path(destination).expanduser().resolve()
    if dest.exists(): raise ValueError('Download destination must not exist; refusing overwrite')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hf-restore-', dir=dest.parent) as temporary:
        staged = Path(temporary)/'run'; staged.mkdir()
        for name, spec in snapshot['files'].items():
            rel = safe_path(name); source = get('artifacts/'+str(rel))
            if source.stat().st_size != spec['bytes'] or sha(source) != spec['sha256']:
                raise ValueError('Artifact hash differs: ' + name)
            target = staged/str(rel); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        staged.rename(dest)
    return {'destination': str(dest), 'revision': revision, 'completed_cases': snapshot['completed_cases'],
            'original_output': snapshot['original_output'],
            'same_output_path': str(dest) == snapshot['original_output'],
            'resume_note': 'Original code/config/base model/Doctor and absolute paths still required; no identity rewriting'}


def sync_barrier(out):
    name = os.environ.get('SHERPA_HF_RUN_ID')
    if not name: return None
    out = Path(out)
    if (out/'api_usage.jsonl').exists():
        from src.online_rl.tree.b200_diagnostics import usage_report
        atomic_write_json(out/'api_usage_summary.json', usage_report(out/'api_usage.jsonl'))
    try:
        with online_hub(): result = upload_run(out, name, repo=os.environ.get('SHERPA_HF_MODEL_REPO', REPO))
    except Exception as exc:
        # The batch is already committed locally. A re-run restores it; no second update.
        raise RuntimeError('HF backup failed after local commit; local checkpoint is intact. Retry upload before resuming.') from exc
    print(json.dumps({'stage': 'hf_backup', **result}), flush=True)
    return result


def storage_settings():
    name = os.environ.get('SHERPA_HF_RUN_ID')
    return {'enabled': bool(name), 'model_repo': os.environ.get('SHERPA_HF_MODEL_REPO', REPO),
            'run_id': name, 'dataset_repo': REPO, 'dataset_revision': DATA_REVISION,
            'backup_boundary': 'synchronous after each committed five-case batch and final evaluation'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('data')
    plan = sub.add_parser('plan'); plan.add_argument('--run-dir', required=True)
    upload = sub.add_parser('upload'); upload.add_argument('--run-dir', required=True)
    upload.add_argument('--run-id', required=True); upload.add_argument('--repo', default=REPO)
    pull = sub.add_parser('download'); pull.add_argument('--run-id', required=True)
    pull.add_argument('--revision', required=True); pull.add_argument('--dest', required=True)
    pull.add_argument('--repo', default=REPO)
    args = p.parse_args()
    if args.action in ('upload', 'plan'):
        out = Path(args.run_dir).resolve()
        with (out/'runner.lock').open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('Run is active; use its automatic barrier backup or stop it first') from None
            if args.action == 'plan':
                value = inventory(out); result = {'files': len(value['files']), 'bytes': sum(v['bytes'] for v in value['files'].values()),
                    'completed_cases': value['completed_cases'], 'network_used': False}
            else:
                with online_hub(): result = upload_run(out, args.run_id, repo=args.repo)
    else:
        with online_hub():
            result = prepare_data() if args.action == 'data' else download_run(args.run_id, args.revision, args.dest, repo=args.repo)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
