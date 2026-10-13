"""RQ1 main table: Base, Outcome-only, Integrated-Fixed, Integrated-Evolving. Accuracy per evaluation split (VAL50 for
checkpoint selection, TEST100 internal, CasiMedicos external) with case-cluster bootstrap CIs and paired differences
against Outcome-only and Base. Case-tree arms: RQ1 main = Outcome-only, Fixed, Evolving (Base as reference); Rubric-only is
an auxiliary RQ1 reward ablation (rubric signal without terminal-outcome reward), never a core RQ1 or an RQ3 arm; each row
carries its rq1_role."""
from __future__ import annotations
from src.evaluation.stats import cluster_bootstrap_mean, paired_difference
from src.online_rl.integrated.conditions import MAIN

RQ1_AUXILIARY = ('rubric_only', 'tree_rubric_only')     # auxiliary reward ablation (labels as passed to rq1_milestone_inputs)


def rq1_role(condition: str) -> str:
    return 'auxiliary_reward_ablation' if condition.split('@')[0] in RQ1_AUXILIARY else 'main'


def table(results: dict[str, dict[str, list[dict]]], provenance: dict[str, dict], references=('outcome_only', 'base'), conditions=MAIN) -> dict:
    """results: condition -> split -> [{'case_id','correct'}]; provenance: condition -> src.rq_eval.runs.load_run/base_provenance
    (or runs.rq1_milestone_inputs for tree milestones, with conditions = their labels in order)."""
    if set(results) - set(conditions): raise ValueError(f'unknown conditions {sorted(set(results) - set(conditions))}; expected {conditions}')
    if set(results) - set(provenance): raise ValueError(f'no provenance for {sorted(set(results) - set(provenance))}')
    rows = []
    for c in (c for c in conditions if c in results):
        for split, rs in sorted(results[c].items()):
            row = {'condition': c, 'rq1_role': rq1_role(c), 'split': split, 'accuracy': cluster_bootstrap_mean([float(r['correct']) for r in rs], [r['case_id'] for r in rs])}
            for ref in references:
                if ref != c and split in results.get(ref, {}):
                    b = {r['case_id']: float(r['correct']) for r in results[ref][split]}
                    row[f'minus_{ref}'] = paired_difference([(r['case_id'], b[r['case_id']], float(r['correct'])) for r in rs if r['case_id'] in b])
            rows.append(row)
    return {'main': rows, 'provenance': {c: {k: v for k, v in p.items() if k != 'config'} for c, p in provenance.items() if c in results}}
