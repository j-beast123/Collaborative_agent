"""Pin existing local Gemma weights/tokenizer on CPU; never download or load a model."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from src.online_rl.integrated.records import sha256_json

ROOT = Path(__file__).resolve().parents[3]
ENGINE = 'hf_local'
TRANSFORMERS_VERSION = '4.53.3'


def receipt_path():
    return Path(os.environ.get('SHERPA_HF_DOCTOR_RECEIPT', ROOT / '.local/b200_hf_doctor.json')).expanduser().resolve()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def inventory(model):
    model = Path(model).expanduser().resolve()
    cfg = json.loads((model / 'config.json').read_text())
    text = cfg.get('text_config', {})
    if (cfg.get('model_type') != 'gemma3' or cfg.get('architectures') != ['Gemma3ForConditionalGeneration']
            or text.get('hidden_size') != 3840 or text.get('num_hidden_layers') != 48):
        raise ValueError('Expected the local Gemma-3-12B conditional-generation architecture')
    if cfg.get('quantization_config') or text.get('quantization_config'):
        raise ValueError('This Doctor backend requires unquantized local Gemma weights')
    if not (model/'tokenizer_config.json').is_file() or not any((model/f).is_file() for f in ('tokenizer.json', 'tokenizer.model')):
        raise ValueError('Local Doctor tokenizer files are incomplete')
    index = model/'model.safetensors.index.json'
    if index.is_file():
        shards = set(json.loads(index.read_text())['weight_map'].values())
        if not shards or any(Path(s).name != s or not s.endswith('.safetensors') for s in shards):
            raise ValueError('Invalid local Doctor shard index')
    else:
        shards = {'model.safetensors'}
    if any(not (model/s).is_file() for s in shards): raise ValueError('Local Doctor weight shards are incomplete')
    files = sorted(p for p in model.iterdir() if p.is_file() and
                   (p.suffix in ('.safetensors', '.json', '.model', '.jinja') or p.name in ('merges.txt', 'vocab.txt')))
    hashes = {p.name: {'sha256': file_sha(p), 'bytes': p.stat().st_size} for p in files}
    return {'schema': 'local_gemma3_12b_files_v1', 'model_path': str(model), 'files': hashes,
            'model_digest': sha256_json(hashes), 'architecture': cfg['architectures'][0],
            'source_revision': 'not inferred from folder name; local bytes are authoritative',
            'load_dtype': 'bfloat16', 'quantization': 'none'}


def read_receipt(path=None):
    row = json.loads(Path(path or receipt_path()).read_text())
    body = {k: v for k, v in row.items() if k != 'receipt_sha256'}
    if row.get('receipt_sha256') != sha256_json(body) or row.get('schema') != 'local_gemma3_12b_files_v1':
        raise ValueError('Local HF Doctor receipt integrity failed')
    if row.get('model_digest') != sha256_json(row.get('files')):
        raise ValueError('Local HF Doctor file manifest identity failed')
    if row.get('load_dtype') != 'bfloat16' or row.get('quantization') != 'none':
        raise ValueError('Local HF Doctor must be unquantized BF16')
    return row


def verify_receipt(row):
    actual = inventory(row['model_path'])
    if actual != {k: v for k, v in row.items() if k != 'receipt_sha256'}:
        raise ValueError('Local HF Doctor weights/tokenizer/config changed; refusing resume or repin')
    return row


def pin(model, path=None):
    path = Path(path or receipt_path()); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            row = read_receipt(path)
            if row['model_path'] != str(Path(model).expanduser().resolve()): raise ValueError('Pinned HF Doctor path differs')
            return verify_receipt(row)
        body = inventory(model); row = {**body, 'receipt_sha256': sha256_json(body)}
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name); json.dump(row, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
        try: os.link(tmp, path)
        finally: tmp.unlink()
        return row


def apply_receipt(cfg):
    row = read_receipt()
    configured = os.environ.get('SHERPA_DOCTOR_MODEL')
    if configured and str(Path(configured).expanduser().resolve()) != row['model_path']:
        raise ValueError('SHERPA_DOCTOR_MODEL differs from the pinned HF Doctor')
    old = cfg['rollout']['doctor_backend']; old_identity = copy.deepcopy(old)
    backend = {k: copy.deepcopy(old[k]) for k in ('options', 'final_response', 'source_model')}
    backend.update(engine=ENGINE, model=row['model_path'], model_digest=row['model_digest'], quantization='none',
                   torch_dtype='bfloat16', transformers_version=TRANSFORMERS_VERSION, device='cuda:0',
                   attention_implementation='sdpa', sampling_protocol='hf_single_gpu_seeded_v1',
                   local_files_only=True, file_manifest=row['files'])
    if cfg.get('doctor_batching'):
        from src.online_rl.doctor_batch import VERSION
        backend['batching'] = copy.deepcopy(cfg['doctor_batching'])
        backend['sampling_protocol'] = VERSION
    label = f"{ENGINE}/{row['model_path']}@{row['model_digest'][:12]}"
    cfg['rollout']['doctor_backend'] = backend
    cfg['rollout']['doctor_model'] = cfg['feedback']['model'] = cfg['experiment']['feedback_model'] = label
    opening = cfg['five_gpu']['opening_identity']
    for key in ('engine', 'ollama_version', 'quantization', 'model', 'model_digest', 'source_model', 'options'):
        opening.pop(key, None)
    opening['hf_doctor'] = copy.deepcopy(backend)
    cfg['five_gpu']['doctor_urls'] = []
    cfg['five_gpu']['ollama_binary'] = cfg['five_gpu']['ollama_models_dir'] = None
    cfg['output_dir'] += '/hf_doctor_' + row['model_digest'][:12]
    cfg['experiment']['output_root'] = cfg['output_dir']
    cfg['hf_doctor_receipt'] = row
    cfg['doctor_backend_change'] = {'previous': old_identity, 'approved': 'Existing local unquantized Gemma instead of Ollama Q8_0',
        'decoding_note': 'Same requested sampling values; Transformers SDPA/BF16 RNG and numerics differ from Ollama. num_batch is an Ollama-only execution setting.'}
    cfg['notes']['changes'].append('User-selected local unquantized Gemma Doctor; BF16 on GPU0; same Doctor for discussion, need, reflection, verification and evaluation')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pin', 'verify'))
    parser.add_argument('--model', default=os.environ.get('SHERPA_DOCTOR_MODEL'))
    args = parser.parse_args()
    if args.action == 'pin' and not args.model: parser.error('--model or SHERPA_DOCTOR_MODEL is required')
    print(json.dumps({'stage': 'hashing_local_doctor_files', 'action': args.action, 'gpu_or_network_used': False}), flush=True)
    row = pin(args.model) if args.action == 'pin' else verify_receipt(read_receipt())
    print(json.dumps({k: row[k] for k in ('model_path', 'model_digest', 'load_dtype', 'quantization')}, indent=2))
    print(json.dumps({'receipt': str(receipt_path()), 'files': len(row['files']), 'gpu_or_network_used': False}))


if __name__ == '__main__': main()
