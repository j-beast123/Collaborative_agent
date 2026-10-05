"""RQ1 main table: Base, Outcome-only, Integrated-Fixed, Integrated-Evolving. Accuracy per evaluation split (VAL50 for
checkpoint selection, TEST100 internal, CasiMedicos external) with case-cluster bootstrap CIs and paired differences
against Outcome-only and Base."""
from __future__ import annotations
from src.evaluation.stats import cluster_bootstrap_mean, paired_difference
from src.online_rl.integrated.conditions import MAIN


def table(results: dict[str, dict[str, list[dict]]], provenance: dict[str, dict], references=('outcome_only', 'base')) -> dict:
    """results: condition -> split -> [{'case_id','correct'}]; provenance: condition -> src.rq_eval.runs.load_run/base_provenance."""
    if set(results) - set(MAIN): raise ValueError(f'unknown conditions {sorted(set(results) - set(MAIN))}; expected {MAIN}')
    if set(results) - set(provenance): raise ValueError(f'no provenance for {sorted(set(results) - set(provenance))}')
    rows = []
    for c in (c for c in MAIN if c in results):
        for split, rs in sorted(results[c].items()):
            row = {'condition': c, 'split': split, 'accuracy': cluster_bootstrap_mean([float(r['correct']) for r in rs], [r['case_id'] for r in rs])}
            for ref in references:
                if ref != c and split in results.get(ref, {}):
                    b = {r['case_id']: float(r['correct']) for r in results[ref][split]}
                    row[f'minus_{ref}'] = paired_difference([(r['case_id'], b[r['case_id']], float(r['correct'])) for r in rs if r['case_id'] in b])
            rows.append(row)
    return {'main': rows, 'provenance': {c: {k: v for k, v in p.items() if k != 'config'} for c, p in provenance.items() if c in results}}
