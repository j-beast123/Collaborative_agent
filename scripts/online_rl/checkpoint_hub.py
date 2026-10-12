"""Queue a saved run, retry uploads, or restore a complete Hub snapshot."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.checkpoint_hub import enqueue_checkpoint, hf_token, start_job, upload_job, file_digest
from src.online_rl.integrated.records import atomic_write_json


def restore(repo, prefix):
    if Path(prefix).is_absolute() or any(part in ('.', '..') for part in Path(prefix).parts):
        raise ValueError('Invalid remote snapshot prefix')
    os.environ['HF_HUB_OFFLINE'] = '0'
    from huggingface_hub import snapshot_download
    (ROOT/'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sherpa-restore-', dir=ROOT/'outputs') as temp:
        snapshot_download(repo_id=repo, repo_type='model', token=hf_token(),
                          allow_patterns=[prefix.rstrip('/')+'/**'], local_dir=temp)
        bundle = Path(temp)/prefix
        inventory = json.loads((bundle/'bundle_manifest.json').read_text())
        for row in inventory['files']:
            path = (bundle/row['path']).resolve()
            if not path.is_relative_to(bundle.resolve()) or file_digest(path) != row['sha256']:
                raise ValueError('Downloaded bundle file hash differs')
        config = json.loads((bundle/'config.json').read_text())
        out = (ROOT/config['output_dir']).resolve()
        if (out/'run_state.json').exists() or (out/'checkpoints').exists():
            raise ValueError('Refusing to overwrite an existing run; restore on a fresh workspace')
        for path in (bundle/'artifacts/prompts').rglob('*.txt'):
            destination = ROOT/'prompts'/path.relative_to(bundle/'artifacts/prompts')
            if destination.exists() and file_digest(destination) != file_digest(path):
                raise ValueError('Local prompt differs; use the matching SHERPA checkout')
        out.mkdir(parents=True, exist_ok=True)
        for path in bundle.iterdir():
            if path.name == 'artifacts': continue
            if path.is_dir(): shutil.copytree(path, out/path.name)
            else: shutil.copy2(path, out/path.name)
        for path in (bundle/'artifacts/prompts').rglob('*.txt'):
            destination = ROOT/'prompts'/path.relative_to(bundle/'artifacts/prompts')
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
        state = json.loads((out/'run_state.json').read_text())
        state['checkpoint'] = str(out/inventory['checkpoint'])
        atomic_write_json(out/'run_state.json', state)
        print(f"Restored {inventory['completed_cases']} cases to {out}")
        print(f"Resume: SHERPA_FIVE_GPU_CONFIG={out/'config.json'} bash run.sh --launch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/group_online.json')
    parser.add_argument('--output', type=Path, help='Queue latest committed checkpoint from this run')
    parser.add_argument('--repo', default='zzu819/SHERPA')
    parser.add_argument('--prefix', default='sherpa-ckpt/sherpa_main')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--retry', action='store_true')
    parser.add_argument('--restore', help='Remote snapshot folder including batch_XXXXXX')
    args = parser.parse_args()
    if args.worker: upload_job(args.worker); return
    if args.restore: restore(args.repo, args.restore); return
    from src.online_rl.tree.experiment import load_config
    cfg = load_config(args.config)
    out = (ROOT/(args.output or cfg['output_dir'])).resolve()
    if args.retry:
        for path in sorted((out/'hf_uploads/jobs').glob('*.json')):
            if json.loads(path.read_text()).get('status') != 'uploaded': start_job(path)
        print('Pending/failed jobs restarted in background.'); return
    state = json.loads((out/'run_state.json').read_text())
    # Use the saved training configuration, not today's changed defaults.
    saved_cfg = json.loads((out/'run_manifest.json').read_text())['config']
    saved_cfg['checkpoint_upload'] = {'enabled': True, 'repo_id': args.repo, 'path_prefix': args.prefix}
    job = enqueue_checkpoint(saved_cfg, out, state)
    print(f'Queued background upload: {job}')


if __name__ == '__main__': main()
