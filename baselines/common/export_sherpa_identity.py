"""Run in the actual SHERPA environment; export metadata only, never run training."""
import argparse
import importlib.metadata
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--base-model-revision',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.run)
    cfg=json.loads((root/'run_manifest.json').read_text())['config']
    initial=json.loads((root/'checkpoints/checkpoint_000000/COMPLETE.json').read_text())
    if cfg['schedule']['cases']!=120 or cfg['prompt_update']['enabled']:raise ValueError('Expected actual train120 / no prompt optimization run')
    value={'config':cfg,'initial_policy_sha256':initial['policy_sha256'],'base_model_revision':a.base_model_revision,
        'package_versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate','safetensors','huggingface-hub')},
        'source_run':str(root.resolve()),'base_revision_evidence':'Operator supplies actual SHERPA snapshot commit, not model name/seed'}
    with Path(a.out).open('x') as f:json.dump(value,f,indent=2)


if __name__=='__main__':main()
