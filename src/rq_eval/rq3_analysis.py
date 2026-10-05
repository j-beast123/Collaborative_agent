"""RQ3: Integrated-Fixed vs Integrated-Evolving generator prompt (same P0, structured state and training; only prompt evolution differs). Builds the comparison table from evaluation outputs:
task success (cluster bootstrap, paired by case), RQ2 behaviour appropriateness, optional final-iteration downstream
effectiveness, plus the prompt lineage of the evolving arm. Theme evolution / fixed-probe tables come from themes.py and
fixed_probe.py and are reported separately."""
from __future__ import annotations
from src.evaluation.stats import cluster_bootstrap_mean, paired_difference

ARMS = ('fixed', 'evolving')


def task_success_table(results: dict[str, list[dict]]) -> list[dict]:
    """results: arm -> [{'case_id','correct'}] on the same evaluation cases."""
    out = []
    for arm in ARMS:
        rs = results.get(arm, [])
        out.append({'arm': arm, 'accuracy': cluster_bootstrap_mean([float(r['correct']) for r in rs], [r['case_id'] for r in rs])})
    if all(results.get(a) for a in ARMS):
        f = {r['case_id']: float(r['correct']) for r in results['fixed']}
        out.append({'arm': 'evolving_minus_fixed', 'accuracy': paired_difference([(r['case_id'], f[r['case_id']], float(r['correct'])) for r in results['evolving'] if r['case_id'] in f])})
    return out


def rq3_table(task: dict, fixed: dict, evolving: dict, lineage: list[dict], appropriateness: dict | None = None, downstream: dict | None = None) -> dict:
    """fixed / evolving: src.rq_eval.runs.load_run provenance; lineage: runs.prompt_lineage of the evolving run."""
    from src.rq_eval.runs import public, rq3_pair_problems
    problems = rq3_pair_problems(fixed, evolving)
    if problems: raise ValueError(f'RQ3 arms are not matched: {problems}')
    return {'task_success': task_success_table(task), 'behavior_appropriateness': appropriateness, 'downstream_effectiveness': downstream,
            'evolving_prompt_lineage': lineage, 'accepted_prompts': evolving['accepted_prompts'],
            'provenance': {'fixed': public(fixed), 'evolving': public(evolving)}, 'differs_only_in': 'prompt_update.enabled (same P0, state schema, training)'}
