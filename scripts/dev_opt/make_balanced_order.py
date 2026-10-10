"""Balanced-disagreement training order (sampling/curriculum diagnostic; TRAIN pool only).

Strata from the theta0 flags of the disagreement pass (positions 0..399 of the frozen train500 order) and the TRAIN gold labels:
  AR_DW  theta0 blind option == gold and Doctor rounds-2 opening option != gold (parsed)
  AW_DR  theta0 blind option != gold and Doctor rounds-2 opening option == gold
Both-wrong, both-correct and unparsed cases are excluded from the first 2*N positions. N cases per stratum are drawn with
random.Random(SEED); the 2*N positions are N (AR_DW, AW_DR) pairs in random within-pair order, so neither stratum comes first and
every batch of 5 holds 2 or 3 of each. New order = [2*N balanced cases] + [the other pool cases in their original order] +
[positions 400..499 unchanged = the held-out DEV split, never trained on]. No DEV or CasiMedicos label is read.

    python scripts/dev_opt/make_balanced_order.py [--n 30] [--seed 20261010]"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; D = ROOT / 'data/interim/medmcqa/five_gpu_train500'
ORDER, ORDER_SHA256, POOL = 'train500_order_seed20261008.json', 'b527dc1a7bb237a9271f6db575ca155cb2d5f64b5a29b8a39bf2ce41b6e3f5f0', 400
FLAGS = 'train500_order_disagreement_first_v1.manifest.json'; TRAIN = 'train500_excluding_calibration_seed20261008.jsonl'
OUT = 'train500_order_balanced_disagreement_v1.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def build(n: int, seed: int) -> dict:
    if sha(D / ORDER) != ORDER_SHA256: raise SystemExit('train500 order hash differs')
    order = json.loads((D / ORDER).read_text()); pool = order[:POOL]
    flags = {f['case_id']: f for f in json.loads((D / FLAGS).read_text())['flags']}
    gold = {d['case_id']: d['ground_truth']['correct_option'] for d in map(json.loads, open(D / TRAIN))}
    if set(flags) != set(pool): raise SystemExit('flags do not cover positions 0..399 exactly')
    q = lambda c: None if not flags[c]['blind_option'] or not flags[c]['doctor_option'] else \
        ('A' + 'RW'[flags[c]['blind_option'] != gold[c]] + '_D' + 'RW'[flags[c]['doctor_option'] != gold[c]])
    strata = {k: [c for c in pool if q(c) == k] for k in ('AR_DW', 'AW_DR', 'AR_DR', 'AW_DW')}
    rng = random.Random(seed); a = rng.sample(strata['AR_DW'], n); b = rng.sample(strata['AW_DR'], n); head = []
    for x, y in zip(a, b): head += [x, y] if rng.random() < 0.5 else [y, x]
    new = head + [c for c in pool if c not in set(head)] + order[POOL:]
    if sorted(new) != sorted(order) or new[POOL:] != order[POOL:]: raise SystemExit('order invariant broken')
    (D / OUT).write_text(json.dumps(new, indent=0) + '\n')
    m = {'schema': 'balanced_disagreement_order_v1', 'source_order': ORDER, 'source_order_sha256': ORDER_SHA256, 'flags_file': FLAGS, 'seed': seed, 'n_per_stratum': n,
         'pool_strata_sizes': {k: len(v) for k, v in strata.items()}, 'unparsed_in_pool': sum(q(c) is None for c in pool),
         'head': [{'position': i, 'case_id': c, 'stratum': q(c)} for i, c in enumerate(head)],
         'head_per_batch_of_5_AR_DW': [sum(q(c) == 'AR_DW' for c in head[i:i + 5]) for i in range(0, len(head), 5)],
         'order_file': OUT, 'order_sha256': sha(D / OUT), 'dev_positions_unchanged': [POOL, len(order) - 1], 'labels_used': 'TRAIN pool gold only (positions 0..399)'}
    (D / f'{Path(OUT).stem}.manifest.json').write_text(json.dumps(m, indent=1) + '\n'); return m


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--n', type=int, default=30); a.add_argument('--seed', type=int, default=20261010); x = a.parse_args()
    m = build(x.n, x.seed); print(json.dumps({k: m[k] for k in ('order_file', 'order_sha256', 'pool_strata_sizes', 'unparsed_in_pool', 'head_per_batch_of_5_AR_DW')}))
