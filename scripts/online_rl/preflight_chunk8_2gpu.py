"""Check local assets and CUDA without loading model weights, calling APIs or starting servers."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def check(cfg):
    import torch
    import transformers
    import peft
    import jsonschema
    from huggingface_hub import snapshot_download
    from scripts.online_rl.run_tree import load_key
    from src.online_rl.tree.devices import physical_gpus, gpu_count
    count = gpu_count(cfg)
    assert count >= 1, 'At least one configured GPU is required'
    physical_gpus(count=count)
    assert torch.cuda.is_available() and torch.cuda.device_count() == count, f'Exactly {count} CUDA devices must be visible'
    providers={cfg.get('api_provider', 'openrouter'), cfg.get('judges', {}).get('api_provider', cfg.get('api_provider', 'openrouter')),
               cfg.get('rubric', {}).get('generation', {}).get('api_provider', cfg.get('api_provider', 'openrouter'))}
    for provider in providers:
        assert load_key({'api_provider':provider}).strip(), f'{provider} credential is missing'
    snapshot = Path(snapshot_download(cfg['policy']['base_model'], local_files_only=True))
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        assert (snapshot/name).is_file(), f'Model asset missing: {name}'
    index = snapshot/'model.safetensors.index.json'
    weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    assert all((snapshot/name).is_file() for name in weights), 'Model weight shards missing'
    binary = Path(os.environ['SHERPA_OLLAMA_BINARY'])
    assert binary.is_file() and os.access(binary, os.X_OK), 'Ollama executable missing'
    backend = cfg['rollout']['doctor_backend']
    model, tag = backend['model'].split(':', 1)
    models = Path(os.environ['SHERPA_OLLAMA_MODELS'])
    manifest = models/'manifests/registry.ollama.ai/library'/model/tag
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == backend['model_digest'], 'Doctor model digest differs'
    data = json.loads(manifest.read_text())
    blobs = [data['config'], *data['layers']]
    assert all((models/'blobs'/row['digest'].replace(':', '-')).is_file() for row in blobs), 'Doctor model blobs missing'
    return {'local_assets': 'ready', 'credential': 'present', 'visible_gpus': count,
            'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
            'model_weights_loaded': False, 'api_requests': 0, 'servers_started': False}

if __name__ == '__main__':
    from src.online_rl.tree.experiment import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(check(load_config(args.config)), indent=2))
