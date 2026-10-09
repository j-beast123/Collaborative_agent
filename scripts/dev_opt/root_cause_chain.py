"""Root-cause chain of a finished SHERPA tree run, from its case records only (read-only):
sibling diversity -> outcome variation -> MC witness success -> kept rubrics -> R_A coverage -> contributing groups.

    python scripts/dev_opt/root_cause_chain.py <run_dir> <out_json>"""
from __future__ import annotations
import json, re, statistics as st, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.collaboration.generator import option   # noqa: E402  the option parser the run's own diagnostics use

EPS = 1e-9
WORD = re.compile(r"[a-z0-9']+")


def jaccard_distance(a: str, b: str) -> float:
    x, y = set(WORD.findall(a.lower())), set(WORD.findall(b.lower()))
    return 1 - len(x & y) / len(x | y) if x | y else 0.0


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]: j += 1
        for k in range(i, j + 1): r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    p = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(p) < 5: return None
    ra, rb = rank([x for x, _ in p]), rank([y for _, y in p])
    ma, mb = st.mean(ra), st.mean(rb); sa, sb = st.pstdev(ra), st.pstdev(rb)
    return None if sa == 0 or sb == 0 else round(sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / len(p) / sa / sb, 3)


def rate(pairs):
    """P(B | A) and P(B | not A) for a list of (A, B) booleans."""
    a = [b for x, b in pairs if x]; n = [b for x, b in pairs if not x]
    return {'P(B|A)': round(sum(a) / len(a), 3) if a else None, 'n_A': len(a), 'P(B|notA)': round(sum(n) / len(n), 3) if n else None, 'n_notA': len(n)}


def main(run_dir: str, out: str) -> None:
    run = Path(run_dir); cases, groups = [], []
    for f in sorted(run.glob('cases/case_*.json')):
        c = json.load(open(f)); t = c['agent_targets']; nodes = {n['node_id']: n for n in c['tree']['nodes']}
        wit = c['rubric_bank'].get('witness') or []
        reruns = sum(x.get('reruns', 0) for w in wit for x in w.get('tested') or [])
        hits = sum(x.get('positive', 0) for w in wit for x in w.get('tested') or [])
        g_rows = []
        for key, ids in t['groups'].items():
            rs = [t['nodes'][i] for i in ids]; texts = [nodes[i]['response'] for i in ids]
            opts = [option(x) for x in texts]
            o = [r['outcome_component'] for r in rs]; rew = [r['reward'] for r in rs if r['reward'] is not None]
            pair_d = [jaccard_distance(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
            row = {'case': c['case_cursor'], 'depth': nodes[ids[0]]['depth'], 'text_diversity': st.mean(pair_d) if pair_d else 0.0,
                   'option_diverse': len({x for x in opts if x}) > 1, 'outcome_varies': max(o) - min(o) > EPS,
                   'correct_and_incorrect': any(x == 1 for x in o) and any(x == 0 for x in o),
                   'R_A_available': sum(r['R_A'] is not None for r in rs) / len(rs),
                   'R_A_varies': len({r['R_A'] for r in rs if r['R_A'] is not None}) > 1,
                   'nonzero_advantage': len(rew) >= 2 and max(rew) - min(rew) > EPS}
            g_rows.append(row)
        groups += g_rows
        leaf = [l['outcome'] for l in c['tree']['leaves'] if l['outcome'] is not None]
        p = sum(leaf) / len(leaf) if leaf else None
        cases.append({'case': c['case_cursor'], 'text_diversity': st.mean(g['text_diversity'] for g in g_rows),
                      'option_diverse_groups': st.mean(g['option_diverse'] for g in g_rows),
                      'leaf_accuracy': p, 'leaf_outcome_variance': p * (1 - p) if p is not None else None,
                      'outcome_varying_groups': st.mean(g['outcome_varies'] for g in g_rows),
                      'mc_target_hit_rate': hits / reruns if reruns else None, 'mc_reruns': reruns,
                      'candidates_exhausted': sum(w['stop_reason'] == 'candidates_exhausted' for w in wit),
                      'rubrics_tested': len(wit), 'kept_rubrics': len(c.get('training_rubrics') or []),
                      'R_A_available': st.mean(g['R_A_available'] for g in g_rows),
                      'contributing_groups': st.mean(g['nonzero_advantage'] for g in g_rows)})
    col = lambda k: [x[k] for x in cases]
    chain = {
        'diversity -> outcome variation (case, spearman)': {
            'text_diversity vs outcome_varying_groups': spearman(col('text_diversity'), col('outcome_varying_groups')),
            'option_diverse_groups vs outcome_varying_groups': spearman(col('option_diverse_groups'), col('outcome_varying_groups')),
            'text_diversity vs leaf_outcome_variance': spearman(col('text_diversity'), col('leaf_outcome_variance'))},
        'diversity -> outcome variation (group)': {
            'option_diverse -> outcome_varies': rate([(g['option_diverse'], g['outcome_varies']) for g in groups]),
            'text_diversity above median -> outcome_varies': rate([(g['text_diversity'] > st.median(x['text_diversity'] for x in groups), g['outcome_varies']) for g in groups])},
        'outcome variation -> MC success (case, spearman)': {
            'leaf_outcome_variance vs mc_target_hit_rate': spearman(col('leaf_outcome_variance'), col('mc_target_hit_rate')),
            'outcome_varying_groups vs mc_target_hit_rate': spearman(col('outcome_varying_groups'), col('mc_target_hit_rate'))},
        'MC success -> kept rubric (case, spearman)': {
            'mc_target_hit_rate vs kept_rubrics': spearman(col('mc_target_hit_rate'), col('kept_rubrics')),
            'any MC hit -> kept>0': rate([((x['mc_target_hit_rate'] or 0) > 0, x['kept_rubrics'] > 0) for x in cases if x['rubrics_tested']])},
        'kept rubric -> R_A coverage (case)': {
            'kept_rubrics vs R_A_available (spearman)': spearman(col('kept_rubrics'), col('R_A_available')),
            'R_A_available when kept=0': st.mean(x['R_A_available'] for x in cases if x['kept_rubrics'] == 0) if any(x['kept_rubrics'] == 0 for x in cases) else None,
            'R_A_available when kept>0': st.mean(x['R_A_available'] for x in cases if x['kept_rubrics'] > 0)},
        'R_A coverage -> contributing groups': {
            'case spearman R_A_available vs contributing_groups': spearman(col('R_A_available'), col('contributing_groups')),
            'group: R_A_varies -> nonzero_advantage': rate([(g['R_A_varies'], g['nonzero_advantage']) for g in groups]),
            'group: outcome_varies -> nonzero_advantage': rate([(g['outcome_varies'], g['nonzero_advantage']) for g in groups]),
            'contributing_groups when kept=0 vs kept>0': [round(st.mean(x['contributing_groups'] for x in cases if (x['kept_rubrics'] > 0) == k), 3) for k in (False, True)]},
        'baseline rates': {'groups': len(groups), 'option_diverse': round(st.mean(g['option_diverse'] for g in groups), 3),
                           'outcome_varies': round(st.mean(g['outcome_varies'] for g in groups), 3),
                           'correct_and_incorrect_siblings': round(st.mean(g['correct_and_incorrect'] for g in groups), 3),
                           'mean_text_diversity': round(st.mean(g['text_diversity'] for g in groups), 3),
                           'cases_kept0': sum(x['kept_rubrics'] == 0 for x in cases), 'cases': len(cases)},
    }
    Path(out).write_text(json.dumps({'chain': chain, 'cases': cases}, indent=1))
    print(json.dumps(chain, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
