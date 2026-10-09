"""Disagreement-first training order (label-free curriculum) from the flags of scripts' disagreement pass.

flags: one row per case of positions 0..399 of the frozen train500 order: {position, case_id, blind_option, doctor_option, disagree}
(theta0's blind private option vs the Doctor's rounds-2 opening option; no gold label, outcome or test data).
New order = [disagreement cases, in their original order] + [the other cases of 0..399, in their original order] + [positions
400..499 unchanged]. Positions 400..499 are the held-out DEV split and stay outside any run of at most 400 cases.

    python scripts/dev_opt/make_disagreement_order.py <flags.jsonl> [<flags.jsonl> ...] [--data-dir data/interim/medmcqa/five_gpu_train500]"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = 'train500_order_seed20261008.json'
ORDER_SHA256 = 'b527dc1a7bb237a9271f6db575ca155cb2d5f64b5a29b8a39bf2ce41b6e3f5f0'
POOL = 400
OUT = 'train500_order_disagreement_first_v1.json'


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


def build(flag_files: list[Path], data_dir: Path) -> dict:
    if sha(data_dir / ORDER) != ORDER_SHA256: raise SystemExit('train500 order hash differs')
    order = json.loads((data_dir / ORDER).read_text())
    flags = {}
    for f in flag_files:
        for r in map(json.loads, f.read_text().splitlines()): flags[r['case_id']] = r
    pool = order[:POOL]
    if set(flags) != set(pool) or any(flags[c]['position'] != i for i, c in enumerate(pool)): raise SystemExit('flags do not cover positions 0..399 exactly')
    dis = [c for c in pool if flags[c]['disagree']]; rest = [c for c in pool if not flags[c]['disagree']]
    new = dis + rest + order[POOL:]
    if sorted(new) != sorted(order) or new[POOL:] != order[POOL:]: raise SystemExit('order invariant broken')
    (data_dir / OUT).write_text(json.dumps(new, indent=0) + '\n')
    manifest = {'schema': 'disagreement_first_order_v1', 'source_order': ORDER, 'source_order_sha256': ORDER_SHA256, 'pool_positions': [0, POOL - 1],
                'rule': 'disagree = theta0 blind private option != Doctor rounds-2 opening option (both parsed); label-free',
                'disagreement_cases': len(dis), 'pool_cases': len(pool), 'unparsed_blind': sum(flags[c]['blind_option'] is None for c in pool),
                'unparsed_doctor_opening': sum(flags[c]['doctor_option'] is None for c in pool),
                'order_file': OUT, 'order_sha256': sha(data_dir / OUT), 'dev_positions_unchanged': [POOL, len(order) - 1],
                'flags': [flags[c] for c in pool]}
    (data_dir / f'{Path(OUT).stem}.manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('flags', nargs='+'); a.add_argument('--data-dir', default='data/interim/medmcqa/five_gpu_train500')
    x = a.parse_args(); m = build([Path(f) for f in x.flags], ROOT / x.data_dir)
    print(json.dumps({k: m[k] for k in ('order_file', 'order_sha256', 'disagreement_cases', 'pool_cases', 'unparsed_blind', 'unparsed_doctor_opening')}))
