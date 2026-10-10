"""User-run preparation of a dedicated revision-pinned HF cache; no model execution."""
import argparse
import json
import re
from pathlib import Path


def check_snapshot(snapshot):
    path=Path(snapshot)
    required=['config.json','tokenizer_config.json','tokenizer.json','model.safetensors.index.json']
    absent=[name for name in required if not (path/name).is_file()]
    if absent:return ['Missing snapshot files: '+', '.join(absent)]
    index=json.loads((path/'model.safetensors.index.json').read_text())
    absent=[name for name in set(index['weight_map'].values()) if not (path/name).is_file()]
    return ['Missing weight shards: '+', '.join(sorted(absent))] if absent else []


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',default='baselines/runtime.local.json')
    parser.add_argument('--download',action='store_true',help='Fetch pinned model files; no GPU/model execution')
    a=parser.parse_args();runtime=json.loads(Path(a.runtime).read_text())
    revision=runtime.get('base_model_revision') or ''
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Set the actual SHERPA base revision first')
    root=Path(runtime['hf_hub_cache']).resolve();marker=root/'comparison_cache_identity.json'
    value={'model':'Qwen/Qwen3-8B','revision':revision,'purpose':'isolated comparison cache; main pinned for original loader'}
    if marker.exists():
        if json.loads(marker.read_text())!=value:raise RuntimeError('Cache belongs to another revision')
    elif root.exists() and any(root.iterdir()):raise RuntimeError('Choose a new dedicated cache directory; refusing to repin a shared cache')
    else:
        root.mkdir(parents=True,exist_ok=True)
        with marker.open('x') as f:json.dump(value,f,indent=2)
    if a.download:
        from huggingface_hub import snapshot_download
        snapshot_download('Qwen/Qwen3-8B',revision=revision,cache_dir=str(root))
    repo=root/'models--Qwen--Qwen3-8B';snapshot=repo/'snapshots'/revision
    errors=check_snapshot(snapshot)
    if errors:raise RuntimeError('; '.join(errors)+'; rerun with --download')
    ref=repo/'refs/main';ref.parent.mkdir(parents=True,exist_ok=True)
    if ref.exists() and ref.read_text().strip()!=revision:raise RuntimeError('Pinned main ref changed')
    ref.write_text(revision)
    print(json.dumps({'snapshot':str(snapshot),'revision':revision,'model_executed':False},indent=2))


if __name__=='__main__':main()
