"""Offline progress for Hugging Face weight downloads, including partial shards."""
import argparse
import json
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def progress(metadata):
    repo = Path(metadata['cache_dir']) / ('models--' + metadata['model'].replace('/', '--'))
    snapshot = repo / 'snapshots' / metadata['revision']
    rows = []
    for row in metadata['files']:
        name, expected = row['filename'], row['size']
        if not name.endswith('.safetensors') or not expected:
            continue
        final = snapshot / name
        complete = final.is_file() and final.stat().st_size == expected
        received = final.stat().st_size if complete else 0
        if not complete and row.get('sha256'):
            parts = list((repo / 'blobs').glob(row['sha256'] + '*.incomplete'))
            # A canceled transfer can leave another partial file; count only the latest.
            if parts:
                latest = max(parts, key=lambda p: p.stat().st_mtime_ns)
                received = min(expected, latest.stat().st_size)
        rows.append((name, received, expected, complete))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    path = ROOT / 'runtime/downloads' / (args.model.replace('/', '--') + '.json')
    if not path.is_file():
        parser.error('Download metadata missing. Run prepare_models.py --metadata-only first.')
    metadata = json.loads(path.read_text())
    history = deque()
    try:
        while True:
            rows = progress(metadata)
            done = sum(r[1] for r in rows)
            total = sum(r[2] for r in rows)
            if not total:
                parser.error('No safetensors weights in download metadata')
            now = time.monotonic()
            history.append((now, done))
            while len(history) > 2 and history[1][0] < now - 30:
                history.popleft()
            delta_t, delta_bytes = now - history[0][0], done - history[0][1]
            speed = max(0, delta_bytes / delta_t) if delta_t >= 1 else 0
            if args.watch:
                print('\033[2J\033[H', end='')
            print(f'{metadata["model"]}  {time.strftime("%H:%M:%S")}')
            print(f'{100 * done / total:.1f}%  {done / 1e9:.2f} / {total / 1e9:.2f} GB'
                  f'  | completed shards: {sum(r[3] for r in rows)}/{len(rows)}')
            if speed:
                print(f'Last 30s: {speed / 1e6:.1f} MB/s  | estimated remaining: {(total - done) / speed / 60:.1f} min')
            elif args.watch:
                print('Measuring download speed...')
            for name, received, expected, complete in rows:
                print(f'  {name}: {received / 1e9:.2f}/{expected / 1e9:.2f} GB' + ('  complete' if complete else ''))
            print('Completed files are reused; canceled partial files are excluded.', flush=True)
            if not args.watch or all(r[3] for r in rows):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
