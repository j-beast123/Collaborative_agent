"""Restore the pinned 120 cases from an existing canonical train500 or train120 JSONL.

Offline, standard-library only: never sample cases, download data, or change hashes.
The tracked manifest contains IDs and hashes; question text stays in data/interim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.guards import assert_not_protected, question_sha256

MANIFEST = ROOT / 'data/splits/medmcqa_train120_seed20261008.manifest.json'
DEFAULT_OUT = ROOT / 'data/interim/medmcqa/five_gpu_train120_seed20261008'


def prepare(source: str | Path, out: str | Path = DEFAULT_OUT,
            manifest_path: str | Path = MANIFEST) -> dict:
    """Verify all inputs and both output byte hashes before creating any files."""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    order = manifest['case_ids']
    if (manifest.get('schema') != 'train_subset_manifest_v1' or
            not isinstance(order, list) or not order or
            any(not isinstance(cid, str) or not cid for cid in order) or
            len(order) != manifest['count'] or len(order) != len(set(order))):
        raise ValueError('Manifest must contain a nonempty, unique pinned case order')
    if set(manifest['question_sha256']) != set(order):
        raise ValueError('Manifest question hashes must cover exactly the pinned IDs')

    source = Path(source)
    raw = source.read_bytes()
    by_id = {}
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            cid = row['case_id']
            if not isinstance(cid, str) or not cid:
                raise ValueError('Invalid case ID')
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'Invalid source record at line {line_number}') from exc
        if cid in by_id:
            raise ValueError(f'Duplicate source case ID: {cid}')
        by_id[cid] = row
    missing = set(order) - set(by_id)
    if missing:
        raise ValueError(f'Source is missing {len(missing)} pinned case IDs')
    if hashlib.sha256(raw).hexdigest() not in {
            manifest['source_train_sha256'], manifest['train_jsonl_sha256']}:
        raise ValueError('Source hash differs from the canonical train500 and train120 JSONL')

    selected = [by_id[cid] for cid in order]
    for row in selected:
        if question_sha256(row.get('question', '')) != manifest['question_sha256'][row['case_id']]:
            raise ValueError(f'Question hash differs for {row["case_id"]}')
    assert_not_protected(selected, 'pinned train120 reconstruction')
    calibration_ids = set()
    for name in ('calibration_only_v1.json', 'calibration_only_v2_1.json'):
        reservation = json.loads((ROOT / 'data/splits' / name).read_text(encoding='utf-8'))
        calibration_ids.update(reservation['positions_1_indexed'].values())
    if set(order) & calibration_ids:
        raise ValueError('Reserved calibration IDs occur in train120')

    dataset_bytes = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                            for row in selected).encode('utf-8')
    order_bytes = (json.dumps(order, indent=2) + '\n').encode('utf-8')
    out = Path(out)
    outputs = (
        (out / Path(manifest['train_jsonl_file']).name, dataset_bytes, manifest['train_jsonl_sha256']),
        (out / Path(manifest['order_file']).name, order_bytes, manifest['order_sha256']),
    )
    for path, payload, expected in outputs:
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'Reconstructed hash differs: {path.name}')
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f'Refusing to replace a different file: {path}')
    out.mkdir(parents=True, exist_ok=True)
    for path, payload, _ in outputs:
        if not path.exists():
            path.write_bytes(payload)
    return {'count': len(order), 'train_file': str(outputs[0][0]),
            'train_sha256': manifest['train_jsonl_sha256'], 'order_file': str(outputs[1][0]),
            'order_sha256': manifest['order_sha256']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Existing canonical train500 or train120 JSONL (verified by SHA256)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        result = prepare(args.source, args.out)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
