"""Behavioural-theme evolution for contextual rubrics. Each generated rubric is normalised to ONE semantic theme from the
frozen taxonomy (configs/online_rl/behavior_theme_taxonomy_v1.json) by a frozen judge reading its required behaviour and
level-2 description (never keyword counts). Output: tidy rows and theme x iteration bubble data (size = proportion).
On-policy rubric distributions (what the generator produced for the states the policy visited) and fixed-probe analyses
are kept in separate tables (source='on_policy' vs source='fixed_probe')."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

TAXONOMY = json.loads((Path(__file__).resolve().parents[2] / 'configs/online_rl/behavior_theme_taxonomy_v1.json').read_text())
THEMES = tuple(TAXONOMY['themes'])


def theme_question() -> dict:
    return {'theme': {'type': 'choice', 'instructions': 'Classify the main behaviour that this rubric asks of the Agent (its required behaviour and level-2 description) into exactly one theme. Themes: ' +
                      ' | '.join(f'{k}: {v}' for k, v in TAXONOMY['themes'].items()), 'criteria': {k: v for k, v in TAXONOMY['themes'].items()}}}


def normalize_theme(decide: Callable, rubric: dict) -> str:
    vis = {k: rubric.get(k) for k in ('name', 'applicability_condition', 'required_behavior', 'scores')}
    t = decide(json.dumps({'candidate_rubric': vis}, ensure_ascii=False), theme_question())['theme']['value']
    if t not in THEMES: raise ValueError(f'judge returned unknown theme {t}')
    return t


def tidy_rows(items: list[dict], decide: Callable, source: str) -> list[dict]:
    """items: [{'rubric','prompt_id','prompt_sha256','policy_iteration','case_id','state', optional 'C','A','E'}]."""
    if source not in ('on_policy', 'fixed_probe'): raise ValueError('source must be on_policy or fixed_probe')
    rows = []
    for it in items:
        rows.append({'source': source, 'policy_iteration': it['policy_iteration'], 'prompt_id': it.get('prompt_id'), 'prompt_sha256': it.get('prompt_sha256'),
                     'case_id': it['case_id'], 'state': it.get('state'), 'raw_rubric': it['rubric'], 'theme': normalize_theme(decide, it['rubric']),
                     'taxonomy_version': TAXONOMY['version'], **{k: it.get(k) for k in ('C', 'A', 'E')}})
    return rows


def bubble_data(rows: list[dict]) -> list[dict]:
    """theme x policy_iteration with count, proportion within the iteration (bubble size) and mean C/A/E when present."""
    by_it = Counter(r['policy_iteration'] for r in rows); cells = defaultdict(list)
    for r in rows: cells[(r['source'], r['theme'], r['policy_iteration'])].append(r)
    out = []
    for (src, theme, it), rs in sorted(cells.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        def mean(k):
            v = [r[k] for r in rs if r.get(k) is not None]; return sum(v) / len(v) if v else None
        out.append({'source': src, 'theme': theme, 'policy_iteration': it, 'count': len(rs), 'proportion': len(rs) / by_it[it], 'C_mean': mean('C'), 'A_mean': mean('A'), 'E_mean': mean('E')})
    return out
