"""Download Hugging Face assets using hf; credentials stay in the environment."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_token():
    if os.environ.get('HF_TOKEN'):
        return
    path = Path(os.environ.get('SHERPA_ENV_FILE', ROOT / '.env'))
    if path.is_file():
        for raw in path.read_text().splitlines():
            key, sep, value = raw.strip().removeprefix('export ').partition('=')
            if sep and key.strip() == 'HF_TOKEN':
                token = value.strip().strip('"\'')
                if token:
                    os.environ['HF_TOKEN'] = token
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--revision')
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--restart', action='store_true', help='Stop an existing hf download of this model in this project')
    parser.add_argument('--http', action='store_true', help='Use standard HTTP downloads instead of Xet')
    args = parser.parse_args()
    if args.restart:
        import signal
        import time
        stopped = []
        for process in Path('/proc').iterdir():
            if not process.name.isdigit() or int(process.name) == os.getpid():
                continue
            try:
                argv = (process / 'cmdline').read_bytes().decode().split('\0')
                if (any(Path(a).name == 'hf' for a in argv if a) and 'download' in argv
                        and args.model in argv and (process / 'cwd').resolve() == ROOT):
                    os.kill(int(process.name), signal.SIGTERM)
                    stopped.append(process.name)
            except (OSError, ValueError):
                continue
        if stopped:
            print(f'Stopped {len(stopped)} previous download process(es); completed cache files are retained.', flush=True)
            time.sleep(2)
    if args.http:
        os.environ['HF_HUB_DISABLE_XET'] = '1'
    load_token()
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.setdefault('HF_HOME', str(ROOT / 'runtime/huggingface'))
    os.environ.setdefault('HF_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub'))
    from huggingface_hub import HfApi
    info = HfApi().model_info(args.model, revision=args.revision, files_metadata=True)
    print(json.dumps({'model': info.id, 'revision': info.sha}), flush=True)
    metadata_dir = ROOT / 'runtime/downloads'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    files = [{'filename': row.rfilename, 'size': row.size,
              'sha256': getattr(row.lfs, 'sha256', None)} for row in info.siblings]
    (metadata_dir / (args.model.replace('/', '--') + '.json')).write_text(json.dumps({
        'model': info.id, 'revision': info.sha, 'cache_dir': os.environ['HF_HUB_CACHE'], 'files': files}))
    if args.metadata_only:
        return
    cli = Path(sys.executable).parent / 'hf'
    subprocess.run([str(cli), 'download', args.model, '--revision', info.sha,
                    '--cache-dir', os.environ['HF_HUB_CACHE'],
                    '--include', '*.json', '--include', '*.safetensors',
                    '--include', '*.jinja', '--include', '*.model',
                    '--include', '*.txt', '--max-workers', '4'], check=True)


if __name__ == '__main__':
    main()
