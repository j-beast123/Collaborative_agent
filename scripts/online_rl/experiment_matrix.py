"""Generate experiment configs and the run manifest from a matrix spec; --check only verifies that they are current.
A run already defined by an earlier family is referenced, not duplicated. Reward ratios are fixed in the spec; nothing
here reads evaluation results."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import experiment as X

MANIFEST_SCHEMA = 'sherpa_experiment_manifest_v2'
BLOCKED_PFINAL = 'blocked_pfinal_missing'
OUTCOME_ONLY = 'N/A_OUTCOME_ONLY'
PROMPT_LABELS = {'p0_fixed': 'fixed P0 rubric prompt', 'pfinal_fixed': 'fixed Pfinal rubric prompt', 'evolving': 'evolving rubric prompt'}


def dump(obj, sort_keys=False) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=sort_keys) + '\n'


def arm_label(w_r: float, w_o: float, mode: str) -> str:
    """Report name; rubric-reward-only arms still use outcomes inside prompt evolution and witness reruns."""
    if w_r == 0: return 'Outcome-only'
    reward = 'Rubric-reward-only' if w_o == 0 else f'Rubric {w_r} + Outcome {w_o} reward'
    return f'{reward} + {PROMPT_LABELS[mode]}'


def run_entry(cfg: dict) -> dict:
    e = cfg['experiment']; rubric = e['rubric_pipeline']
    return {'run_id': e['run_id'], 'output_dir': cfg['output_dir'],
            'arm_label': arm_label(e['reward_weight_rubric'], e['reward_weight_outcome'], e['prompt_mode']),
            'reward_weight_rubric': e['reward_weight_rubric'], 'reward_weight_outcome': e['reward_weight_outcome'],
            'prompt_mode': e['prompt_mode'], 'effective_prompt_mode': e['prompt_mode'] if rubric else OUTCOME_ONLY,
            'prompt_evolution_applicable': rubric, 'prompt_evolution': cfg['prompt_update']['enabled'], 'rubric_pipeline': rubric,
            'rubric_prompt_sha256': e['rubric_prompt_sha256'], 'pfinal': e['pfinal'],
            'feedback_model': e['feedback_model'], 'feedback_prompt_sha256': e['feedback_prompt_sha256'], 'seed': e['seed'],
            'train_sha256': cfg['data']['train_sha256'], 'train_order_sha256': cfg['five_gpu']['train_order_sha256'],
            'schedule_cases': cfg['schedule']['cases'], 'batch_cases': cfg['five_gpu']['batch_cases'],
            'experiment_fingerprint': X.fingerprint(cfg), 'resolved_config_sha256': sha256_json(cfg), 'families': []}


def build(matrix: str) -> dict[Path, str]:
    """Return {path: text} for every leaf config and the manifest."""
    mdir = (ROOT / matrix).parent; spec = json.loads((ROOT / matrix).read_text())
    base = X.load_raw(spec['base']); anchor = spec['anchor_reward_rubric_outcome']
    out, families, runs, by_file = {}, {}, {}, {}
    for family, fam in spec['families'].items():
        families[family] = []
        for r in fam['runs']:
            w_r, w_o = r.get('reward_rubric_outcome', anchor)
            # Anchor families never take a per-run reward.
            if fam.get('reward') == 'anchor' and [w_r, w_o] != anchor: raise ValueError(f"{family}/{r['file']}: reward must be the anchor {anchor}")
            mode = r['prompt_mode']; priority = r.get('priority', fam['priority'])
            for seed in spec['seeds']:
                exp = {**base['experiment'], 'family': family, 'reward_weight_rubric': w_r, 'reward_weight_outcome': w_o,
                       'prompt_mode': mode, 'seed': seed, 'pfinal_artifact': spec['pfinal_artifact'] if mode == 'pfinal_fixed' else None}
                name = r['file'] if len(spec['seeds']) == 1 else r['file'].replace('.json', f'_seed{seed}.json')
                rel = str((mdir / family / name).relative_to(ROOT))
                blocked = mode == 'pfinal_fixed' and not exp['pfinal_artifact']
                cfg = None if blocked else X.resolve(X.deep_merge(base, {'experiment': exp}))
                rid = cfg['experiment']['run_id'] if cfg else None
                if rid in runs:
                    # Reuse the earlier config of an identical run.
                    if runs[rid]['resolved_config_sha256'] != sha256_json(cfg): raise ValueError(f'{rid}: two configs share a run id but differ')
                    rel = runs[rid]['config']
                else:
                    exp['output_dir'] = cfg['output_dir'] if cfg else None
                    out[ROOT / rel] = dump({'extends': spec['base'],
                                            'experiment': {k: exp[k] for k in X.EXPERIMENT_KEYS if k != 'output_root' and (k != 'variant' or k in exp)}})
                    if cfg: runs[rid] = {**run_entry(cfg), 'config': rel, 'priority': priority}
                if cfg:
                    runs[rid]['families'] = sorted({*runs[rid]['families'], family})
                    runs[rid]['priority'] = min(runs[rid]['priority'], priority)
                by_file[f'{family}/{name}'] = rid
                families[family].append({'config': rel, 'run_id': rid, 'priority': priority, 'prompt_mode': mode,
                                         'status': BLOCKED_PFINAL if blocked else 'ready'})
    source = by_file[spec['pfinal_source']]
    for rid, run in runs.items():
        if run['pfinal'] and run['pfinal']['source_run_id'] != source:
            raise ValueError(f"{rid}: Pfinal must come from {source}, not {run['pfinal']['source_run_id']}")
    base_file = json.loads((ROOT / spec['base']).read_text())
    out[mdir / 'manifest.json'] = dump({'schema': MANIFEST_SCHEMA, 'matrix': matrix, 'base': spec['base'],
                                        'protocol_reference': base_file['extends'], 'anchor_reward_rubric_outcome': anchor,
                                        'anchor_policy': spec['anchor_policy'], 'pfinal_artifact': spec['pfinal_artifact'],
                                        'pfinal_source_run_id': source,
                                        'run_count_by_priority': {str(p): sum(r['priority'] == p for r in runs.values())
                                                                  for p in sorted({r['priority'] for r in runs.values()})},
                                        'families': families, 'runs': runs}, sort_keys=True)
    return out


def stale(matrix: str, files: dict[Path, str]) -> list[Path]:
    mdir = (ROOT / matrix).parent; spec = json.loads((ROOT / matrix).read_text())
    return sorted(p for f in spec['families'] for p in (mdir / f).glob('*.json') if p not in files)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('--matrix', default='configs/experiments/reward_prompt_matrix/matrix.json')
    a.add_argument('--check', action='store_true', help='fail if generated files are missing, stale or edited')
    x = a.parse_args(argv)
    files = build(x.matrix); old = stale(x.matrix, files)
    if x.check:
        bad = [p for p, t in files.items() if not p.exists() or p.read_text() != t] + old
        if bad: raise SystemExit('out of date: ' + ', '.join(str(p.relative_to(ROOT)) for p in bad))
    else:
        if old: raise SystemExit('stale configs not in the matrix (remove by hand): ' + ', '.join(map(str, old)))
        for p, t in files.items():
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text(t)
    m = json.loads(files[(ROOT / x.matrix).parent / 'manifest.json'])
    print(dump({'families': {f: [(r['config'].split('/', 3)[-1], r['run_id'], r['priority'], r['status']) for r in rows]
                             for f, rows in m['families'].items()},
                'run_count_by_priority': m['run_count_by_priority'], 'unique_runs': len(m['runs'])}))


if __name__ == '__main__': main()
