"""Fixed held-out DEV split for the DEV-only OURS optimization: the cases at positions 400..499 of the frozen train500 order.
A 300-case run trains on positions 0..299 and a DEV arm on 0..59, so no run trains on these. CasiMedicos622 is not read.
Writes the DEV jsonl (train500 rows verbatim, in order) and a manifest with positions, ids and hashes; deterministic.

    python scripts/dev_opt/make_dev_split.py [--data-dir data/interim/medmcqa/five_gpu_train500]"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = 'train500_excluding_calibration_seed20261008.jsonl'
TRAIN_SHA256 = 'cbe8db7d03cf2d06f3e7d754deabe4fcfb55daeae2b0f64881ec976c8e7af9cd'
ORDER = 'train500_order_seed20261008.json'
ORDER_SHA256 = 'b527dc1a7bb237a9271f6db575ca155cb2d5f64b5a29b8a39bf2ce41b6e3f5f0'
FIRST, LAST = 400, 499                      # inclusive order positions
MAX_TRAINED_POSITION = 299                  # the 300-case production schedule
DEV = f'dev100_order{FIRST}_{LAST}.jsonl'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(data_dir: Path) -> dict:
    train, order = data_dir / TRAIN, data_dir / ORDER
    if sha(train) != TRAIN_SHA256 or sha(order) != ORDER_SHA256: raise SystemExit('train500 data or order hash differs')
    if FIRST <= MAX_TRAINED_POSITION: raise SystemExit('DEV positions overlap the training schedule')
    rows = {json.loads(l)['case_id']: l for l in train.read_text().splitlines() if l.strip()}
    ids = json.loads(order.read_text())[FIRST:LAST + 1]
    (data_dir / DEV).write_text(''.join(rows[i] + '\n' for i in ids))
    manifest = {'schema': 'ours_dev_split_v1', 'source_train': TRAIN, 'source_train_sha256': TRAIN_SHA256, 'source_order': ORDER,
                'source_order_sha256': ORDER_SHA256, 'order_positions': [FIRST, LAST], 'max_trained_position': MAX_TRAINED_POSITION,
                'cases': len(ids), 'case_ids': ids, 'dev_file': DEV, 'dev_sha256': sha(data_dir / DEV),
                'note': 'MedMCQA held-out DEV; never trained on; CasiMedicos622 test is not used for selection'}
    (data_dir / f'{Path(DEV).stem}.manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--data-dir', default='data/interim/medmcqa/five_gpu_train500')
    m = build(ROOT / a.parse_args().data_dir)
    print(json.dumps({k: m[k] for k in ('dev_file', 'dev_sha256', 'cases', 'order_positions')}))
