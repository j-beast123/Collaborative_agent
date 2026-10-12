"""Durable detached Hub uploads of immutable, restartable checkpoint bundles."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from src.online_rl.integrated.records import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
SECRET_KEY = re.compile(r'(?:api[_-]?key|access[_-]?token|hf_token|password|client_secret|authorization|private_key)$', re.I)
SECRET_VALUE = re.compile(rb'hf_[A-Za-z0-9]{15,}|sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{16,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def credentials():
    """Read credentials without putting them in config, command arguments or logs."""
    values = dict(os.environ)
    path = Path(os.environ.get('SHERPA_ENV_FILE', str(ROOT/'runtime/credentials.env')))
    if path.is_file():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().removeprefix('export ').partition('=')
            if sep and (SECRET_KEY.search(key.strip()) or key.strip() == 'HF_REPO_ID'):
                if not values.get(key.strip()):
                    values[key.strip()] = value.strip().strip('"\'')
    return values


def hf_token():
    from huggingface_hub import get_token
    return credentials().get('HF_TOKEN') or get_token()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_bundle(folder):
    """Fail closed on unexpected paths, credential files or actual secret values."""
    folder = Path(folder)
    secrets = [value.encode() for key, value in credentials().items()
               if SECRET_KEY.search(key) and isinstance(value, str) and len(value) >= 12]
    overlap = max([256] + [len(value) for value in secrets])
    rows = []
    allowed = {'run_state.json', 'run_manifest.json', 'progress.json', 'config.json', 'bundle_manifest.json'}
    for path in sorted(folder.rglob('*')):
        relative = path.relative_to(folder)
        if path.is_symlink():
            raise ValueError(f'Upload blocked: symbolic link {relative}')
        if not path.is_file():
            continue
        parts = relative.parts
        if (any(part.lower().startswith('.env') or part.lower() in
                ('credentials', 'credentials.env', 'token', 'tokens', '.git', '.ssh', '.aws', '.cache') for part in parts)
                or path.suffix.lower() in ('.env', '.pem', '.key', '.p12', '.pfx')):
            raise ValueError(f'Upload blocked: credential filename {relative}')
        permitted = (str(relative) in allowed or
                     (parts[0] == 'checkpoints' and len(parts) >= 3) or
                     (len(parts) == 3 and parts[0] == 'batches' and parts[2] == 'timing.json') or
                     (len(parts) >= 3 and parts[:2] == ('artifacts', 'prompts') and path.suffix == '.txt'))
        if not permitted:
            raise ValueError(f'Upload blocked: unexpected path {relative}')
        tail = b''
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                digest.update(chunk)
                data = tail + chunk
                if SECRET_VALUE.search(data) or any(secret in data for secret in secrets):
                    raise ValueError(f'Upload blocked: secret content in {relative}')
                tail = data[-overlap:]
        if path.suffix == '.json':
            def inspect_keys(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if SECRET_KEY.search(key) and item:
                            raise ValueError(f'Upload blocked: credential field in {relative}')
                        inspect_keys(item)
                elif isinstance(value, list):
                    for item in value: inspect_keys(item)
            inspect_keys(json.loads(path.read_text()))
        rows.append({'path': str(relative), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    return rows


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def start_job(job_path):
    env = dict(os.environ)
    env['HF_HUB_OFFLINE'] = '0'  # Only the uploader is online; model loading stays offline.
    job_path = Path(job_path)
    with (job_path.parent.parent/'worker.log').open('a') as log:
        subprocess.Popen([sys.executable, str(ROOT/'scripts/online_rl/checkpoint_hub.py'),
                          '--worker', str(job_path)], cwd=ROOT, env=env,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         close_fds=True, start_new_session=True)


def enqueue_checkpoint(cfg, out, state):
    """Pin saved files locally, then launch a detached CPU/network uploader."""
    spec = cfg.get('checkpoint_upload') or {}
    if not spec.get('enabled'):
        return None
    out = Path(out).resolve()
    trajectory_run = 'completed_steps' in state
    checkpoint = Path(state['last_curve_checkpoint'] if trajectory_run else state['checkpoint']).resolve()
    if checkpoint.parent != out/'checkpoints':
        raise ValueError('Checkpoint is outside this run')
    for required in ('COMPLETE.json', 'trainer_state.pt', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'):
        if not (checkpoint/required).is_file():
            raise ValueError(f'Incomplete checkpoint: missing {required}')
    receipt = json.loads((checkpoint/'COMPLETE.json').read_text())
    mismatch = ((receipt['step'] != state['completed_steps'] or receipt['fingerprint'] != state['fingerprint'])
                if trajectory_run else (receipt['completed_cases'] != state['case_cursor'] or
                                        receipt['protocol_sha256'] != state['protocol_sha256']))
    if mismatch or receipt['policy_sha256'] != state['policy_sha256']:
        raise ValueError('Checkpoint and committed run state disagree')
    cursor = state['completed_steps'] if trajectory_run else state['batch_cursor']
    completed_cases = cursor * 4 if trajectory_run else state['case_cursor']
    label = f"{'step' if trajectory_run else 'batch'}_{cursor:06d}"
    repo_id = spec.get('repo_id') or credentials().get('HF_REPO_ID')
    if not repo_id:
        raise ValueError('HF_REPO_ID is required for checkpoint uploads')
    queue = out/'hf_uploads'
    job_path = queue/'jobs'/f'{label}.json'
    if job_path.exists():
        job = json.loads(job_path.read_text())
        if job.get('status') != 'uploaded': start_job(job_path)
        return job_path
    payload = queue/'payloads'/label
    pending = payload.with_name(label+'.pending')
    pending.parent.mkdir(parents=True, exist_ok=True)
    if payload.exists(): shutil.rmtree(payload)  # Unpublished staging from an interrupted enqueue.
    if pending.exists(): shutil.rmtree(pending)
    saved = pending/'checkpoints'/checkpoint.name
    shutil.copytree(checkpoint, saved, copy_function=link_or_copy, symlinks=True)
    # Mutable progress files must be copied, never hardlinked to future writes.
    atomic_write_json(pending/'run_state.json', state)
    manifest = json.loads((out/'run_manifest.json').read_text())
    atomic_write_json(pending/'run_manifest.json', manifest)
    atomic_write_json(pending/'config.json', manifest['config'])
    if (out/'progress.json').is_file(): shutil.copy2(out/'progress.json', pending/'progress.json')
    timing = out/'batches'/label/'timing.json'
    if timing.is_file():
        target = pending/'batches'/label/'timing.json'
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(timing, target)
    prompts = set()
    for section in ('group_learning', 'feedback', 'judges', 'rubric'):
        for key, value in manifest['config'].get(section, {}).items():
            if key.endswith('_prompt_file') or key == 'prompt_file':
                if isinstance(value, str) and Path(value).parts[:1] == ('prompts',): prompts.add(value)
    for relative in sorted(prompts):
        source = (ROOT/relative).resolve()
        if not source.is_relative_to(ROOT/'prompts') or source.suffix != '.txt':
            raise ValueError('Unexpected prompt artifact')
        destination = pending/'artifacts'/relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
    if trajectory_run:
        destination = pending/'artifacts/prompts/frozen_generator_prompt.txt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out/'frozen_generator_prompt.txt', destination)
    # A unique immutable snapshot path means retries never overwrite newer progress.
    prefix = spec.get('path_prefix', 'sherpa-ckpt/sherpa_main').strip('/')
    if not prefix or any(part in ('.', '..') for part in prefix.split('/')):
        raise ValueError('Invalid checkpoint upload prefix')
    job = {'status': 'queued', 'repo_id': repo_id, 'repo_type': 'model',
           'path_in_repo': f'{prefix}/{out.name}/{label}', 'payload': str(payload),
           'batch': cursor, 'completed_cases': completed_cases, 'checkpoint_name': checkpoint.name,
           'queued_at': time.time(), 'attempts': 0}
    pending.rename(payload)
    atomic_write_json(job_path, job)
    start_job(job_path)
    return job_path


def upload_job(job_path):
    job_path = Path(job_path)
    with job_path.with_suffix('.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        job = json.loads(job_path.read_text())
        if job['status'] == 'uploaded': return
        payload = Path(job['payload'])
        try:
            files = audit_bundle(payload)
            files = [row for row in files if row['path'] != 'bundle_manifest.json']
            atomic_write_json(payload/'bundle_manifest.json', {'files': files,
                'checkpoint': next(str(p.relative_to(payload)) for p in (payload/'checkpoints').iterdir()),
                'completed_cases': job['completed_cases'], 'secret_audit': 'passed',
                'note': 'Requires matching SHERPA code, base model and dataset; contains no credentials.'})
            audit_bundle(payload)  # Check the exact final upload contents, including metadata.
            job.update(secret_audit='passed', file_count=len(files)+1,
                       total_bytes=sum(row['bytes'] for row in files), started_at=time.time())
        except Exception as exc:
            job.update(status='blocked', error_type=type(exc).__name__,
                       error=str(exc) if isinstance(exc, ValueError) else 'Bundle audit failed')
            atomic_write_json(job_path, job); return
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token())
        # Serialize commits within this run, without holding any training/GPU locks.
        with (job_path.parent.parent/'commit.lock').open('a') as commit_lock:
            job['status'] = 'waiting'; atomic_write_json(job_path, job)
            fcntl.flock(commit_lock, fcntl.LOCK_EX)
            for attempt in range(5):
                job.update(status='uploading', attempts=job['attempts']+1)
                atomic_write_json(job_path, job)
                try:
                    api.repo_info(job['repo_id'], repo_type=job['repo_type'])
                    result = api.upload_folder(repo_id=job['repo_id'], repo_type=job['repo_type'],
                        folder_path=payload, path_in_repo=job['path_in_repo'],
                        commit_message=f"SHERPA {job['completed_cases']} cases: full recovery snapshot")
                    job.update(status='uploaded', completed_at=time.time(), commit_oid=result.oid,
                               commit_url=result.commit_url)
                    job.pop('error_type', None); job.pop('http_status', None)
                    atomic_write_json(job_path, job)
                    shutil.rmtree(payload)  # Release pins only after the remote commit succeeds.
                    return
                except Exception as exc:
                    response = getattr(exc, 'response', None)
                    job.update(status='retrying', error_type=type(exc).__name__,
                               http_status=getattr(response, 'status_code', None))
                    atomic_write_json(job_path, job)
                    if attempt < 4: time.sleep(min(60, 5*2**attempt))
            job['status'] = 'failed'; atomic_write_json(job_path, job)


def enqueue_safely(cfg, out, state):
    try:
        return enqueue_checkpoint(cfg, out, state)
    except Exception as exc:
        # Queue failures must not abort a committed optimizer update.
        atomic_write_json(Path(out)/'hf_uploads/enqueue_error.json',
                          {'batch': state.get('completed_steps', state.get('batch_cursor')),
                           'error_type': type(exc).__name__, 'time': time.time()})
        return None


def can_prune_checkpoint(cfg, out, checkpoint):
    if not (cfg.get('checkpoint_upload') or {}).get('enabled'): return True
    for path in (Path(out)/'hf_uploads/jobs').glob('*.json'):
        job = json.loads(path.read_text())
        if job.get('checkpoint_name') != Path(checkpoint).name: continue
        if job.get('status') == 'uploaded': return True
        pinned = Path(job['payload'])/'checkpoints'/Path(checkpoint).name
        if (pinned/'COMPLETE.json').is_file() and (pinned/'trainer_state.pt').is_file(): return True
    return False
