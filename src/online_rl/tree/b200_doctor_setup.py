"""User-invoked isolated Ollama setup. Importing this module performs no I/O."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
VERSION = '0.12.5'
MODEL = 'gemma3:12b-it-q8_0'
HOST = '127.0.0.1:11985'
MANIFEST = 'manifests/registry.ollama.ai/library/gemma3/12b-it-q8_0'
OLD_DIGEST = '997a7c2c09753e9011ec18a3fcb498e27c0ba330599dfb26a2e05b58cfbdc1bc'
ASSETS = {
    'x86_64': ('amd64', 'e0d20dac3358403ef17f880168f32a59b911f3875c509d2d256bd70acce9360e'),
    'aarch64': ('arm64', '725fde4651a4281e8cfb0384c6a2976551eb56b8720112534aa463880b1a4e70'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def receipt_path():
    return Path(os.environ.get('SHERPA_DOCTOR_RECEIPT', ROOT / '.local/b200_doctor.json')).expanduser().resolve()


def read_receipt(path):
    row = json.loads(Path(path).read_text())
    payload = {k: v for k, v in row.items() if k != 'receipt_sha256'}
    if row.get('receipt_sha256') != identity(payload): raise ValueError('Doctor receipt integrity check failed')
    if row.get('version') != VERSION or row.get('model') != MODEL or row.get('quantization') != 'Q8_0':
        raise ValueError('Unexpected Doctor identity in receipt')
    if not re.fullmatch('[0-9a-f]{64}', row.get('digest', '')): raise ValueError('Invalid Doctor digest')
    return row


def apply_receipt(cfg, path):
    row = read_receipt(path)
    label = f"ollama/{MODEL}@{row['digest'][:12]}"
    cfg['rollout']['doctor_backend']['model_digest'] = row['digest']
    cfg['rollout']['doctor_model'] = label
    cfg['feedback']['model'] = label
    cfg['experiment']['feedback_model'] = label
    cfg['five_gpu']['opening_identity']['model_digest'] = row['digest']
    cfg['five_gpu']['ollama_binary'] = row['binary']
    cfg['five_gpu']['ollama_models_dir'] = row['models']
    cfg['output_dir'] += '/doctor_' + row['digest'][:12]
    cfg['experiment']['output_root'] = cfg['output_dir']
    cfg['doctor_download_receipt'] = row
    cfg['notes']['changes'].append('User-approved fresh Doctor download; actual digest pinned in local receipt; old model-dependent caches isolated')
    return cfg


def get_json(route):
    with urllib.request.urlopen('http://' + HOST + route, timeout=15) as response:
        return json.load(response)


def local_manifest(models, *, verify_blobs=False):
    path = Path(models) / MANIFEST
    data = json.loads(path.read_text())
    for blob in [data['config'], *data['layers']]:
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', blob['digest']): raise ValueError('Malformed blob digest')
        p = Path(models) / 'blobs' / blob['digest'].replace(':', '-')
        if not p.is_file() or p.stat().st_size != blob['size']: raise ValueError('Missing/incomplete Doctor blob: ' + p.name)
        if verify_blobs and sha(p) != blob['digest'][7:]: raise ValueError('Doctor blob hash mismatch: ' + p.name)
    return sha(path)


def checked_model(models, *, verify_blobs=False):
    if get_json('/api/version').get('version') != VERSION: raise ValueError('Running Ollama must be ' + VERSION)
    rows = [r for r in get_json('/api/tags').get('models', []) if r.get('name') == MODEL or r.get('model') == MODEL]
    if len(rows) != 1: raise ValueError('Expected one downloaded ' + MODEL)
    row = rows[0]
    digest = local_manifest(models, verify_blobs=verify_blobs)
    if row.get('digest', '').removeprefix('sha256:') != digest: raise ValueError('Service and local Doctor manifest differ')
    if row.get('details', {}).get('quantization_level') != 'Q8_0': raise ValueError('Doctor must be Q8_0')
    return digest


def install(root):
    if platform.system() != 'Linux' or platform.machine() not in ASSETS: raise ValueError('Supported hosts: Linux x86_64/aarch64')
    arch, expected = ASSETS[platform.machine()]
    target = root / ('ollama-' + VERSION)
    binary = target / 'bin/ollama'
    if target.exists():
        proof = json.loads((target / 'install.json').read_text())
        if proof.get('archive_sha256') != expected or proof.get('binary_sha256') != sha(binary):
            raise ValueError('Existing private installation differs; refusing to overwrite')
        return binary
    if shutil.disk_usage(root).free < 8 * 1024**3: raise ValueError('Need at least 8 GiB free for Ollama package/extraction')
    archive = root / f'ollama-linux-{arch}-v{VERSION}.tgz'
    url = f'https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-linux-{arch}.tgz'
    if not archive.exists() or sha(archive) != expected:
        subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--continue-at', '-', '--output', str(archive), url], check=True)
    if sha(archive) != expected: raise ValueError('Ollama archive SHA256 mismatch; remove the damaged private archive before retrying')
    with tempfile.TemporaryDirectory(prefix='install-', dir=root) as tmp:
        stage = Path(tmp) / 'package'; stage.mkdir()
        with tarfile.open(archive) as stream: stream.extractall(stage, filter='data')
        executable = stage / 'bin/ollama'
        if not executable.is_file(): raise ValueError('Archive has no bin/ollama')
        (stage / 'install.json').write_text(json.dumps({'url': url, 'archive_sha256': expected, 'binary_sha256': sha(executable)}, indent=2))
        stage.rename(target)
    return binary


def server_state_dir(root):
    # Shared binaries/models/receipt; process IDs belong to a particular host.
    host = socket.gethostname()
    label = re.sub(r'[^A-Za-z0-9_.-]', '_', host)[:64]
    return Path(root) / 'servers' / (label + '-' + hashlib.sha256(host.encode()).hexdigest()[:12])


def require_owned_server(root, binary, models):
    owner = json.loads((server_state_dir(root) / 'server.json').read_text())
    if owner.get('hostname') != socket.gethostname(): raise ValueError('Doctor process belongs to another host')
    proc = Path('/proc') / str(owner['pid'])
    if Path(os.readlink(proc / 'exe')).resolve() != binary.resolve(): raise ValueError('Server PID does not refer to this private Ollama')
    env = dict(item.split(b'=', 1) for item in (proc / 'environ').read_bytes().split(b'\0') if b'=' in item)
    # Read only the two relevant values; never print or persist the environment.
    if env.get(b'OLLAMA_MODELS') != str(models).encode() or env.get(b'OLLAMA_HOST') != HOST.encode():
        raise ValueError('Doctor process uses another store/port; refusing to pull into another service')
    if get_json('/api/version').get('version') != VERSION: raise ValueError('Ollama server version differs')


def pull_and_pin(root, binary, models, path):
    if path.exists():
        row = read_receipt(path)
        if row['binary'] != str(binary) or row['models'] != str(models): raise ValueError('Pinned resource paths differ')
        if sha(binary) != row['binary_sha256'] or checked_model(models) != row['digest']:
            raise ValueError('Pinned Doctor changed; refusing to pull or repin')
        return row  # Never repull a mutable registry tag after the first successful pin.
    require_owned_server(root, binary, models)
    subprocess.run([str(binary), 'pull', MODEL], env={**os.environ, 'OLLAMA_HOST': HOST, 'OLLAMA_MODELS': str(models)}, check=True)
    digest = checked_model(models, verify_blobs=True)
    payload = {'version': VERSION, 'model': MODEL, 'digest': digest, 'quantization': 'Q8_0',
               'binary': str(binary), 'binary_sha256': sha(binary), 'models': str(models),
               'previous_digest': OLD_DIGEST, 'changed_from_original': digest != OLD_DIGEST,
               'change': 'User approved downloading the current Doctor model for the B200 experiment',
               'created_at_utc': datetime.now(timezone.utc).isoformat()}
    row = {**payload, 'receipt_sha256': identity(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(row, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    try: os.link(temp, path)  # Atomic create-only publication; never replace an existing pin.
    finally: temp.unlink()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'serve', 'pull', 'verify'])
    args = parser.parse_args()
    root = Path(os.environ.get('SHERPA_OLLAMA_HOME', Path.home() / '.local/share/sherpa-b200')).expanduser().resolve()
    binary = root / ('ollama-' + VERSION) / 'bin/ollama'
    models = root / 'models'; path = receipt_path()
    if args.action == 'verify':
        row = read_receipt(path)
        if sha(row['binary']) != row['binary_sha256'] or checked_model(row['models'], verify_blobs=True) != row['digest']:
            raise ValueError('Pinned Doctor differs')
        print(json.dumps(row, indent=2)); return
    root.mkdir(parents=True, exist_ok=True)
    if args.action == 'serve':
        if not binary.is_file(): raise ValueError('Run install first')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 11985))  # Fail on occupied port; never stop another process.
        models.mkdir(exist_ok=True)
        state_dir = server_state_dir(root); state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / 'server.json').write_text(json.dumps({'pid': os.getpid(), 'hostname': socket.gethostname(),
            'binary': str(binary), 'models': str(models)}))
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OLLAMA_HOST': HOST, 'OLLAMA_MODELS': str(models),
               'OLLAMA_NUM_PARALLEL': '1', 'OLLAMA_MAX_LOADED_MODELS': '1', 'OLLAMA_KEEP_ALIVE': '-1'}
        os.execve(binary, [str(binary), 'serve'], env)
    with (root / 'setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'install': print(install(root))
        else: print(json.dumps(pull_and_pin(root, binary, models, path), indent=2))


if __name__ == '__main__': main()
