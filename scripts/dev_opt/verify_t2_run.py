"""Invariants and learning health of a FIXED_T2 run directory (smoke, optimizer DEV arm or 120-case arm); read-only.

    python scripts/dev_opt/verify_t2_run.py <run_dir> [--log <runner log>] [--json out.json]

ENGINEERING: identical scorer requests answered identically (siblings share their Stage-1 request), per-response audit
records (only that response's own decisions), exact reward recomputation, one policy per batch, at most one optimizer step
per batch, tracebacks / OOM in the log. MAXTURN2: Agent depth, leaves, stop reasons. P0_FIXED: one prompt hash, no prompt
events. LEARNING HEALTH: R_O / R_A, kept rubrics, zero-advantage / contributing groups, |advantage|, grad norm, KL."""
from __future__ import annotations
import argparse, json, math, statistics as st, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.tree import experiment as X        # noqa: E402
from src.online_rl.tree.team import compose_reward     # noqa: E402

EPS = 1e-9
P0 = 'f8d6e39470d7f8a93ce8224cea9e5aa2ed2fa07ced9d982cc5e16d0e495d9bad'


def q(xs):
    xs = [x for x in xs if x is not None]
    return {'n': len(xs), 'mean': round(st.mean(xs), 4), 'std': round(st.pstdev(xs), 4)} if xs else {'n': 0}


def calls_of(run: Path, cursor: int):
    path = run / f'batches/batch_{cursor // 5 + 1:06d}/case_{cursor:06d}/calls.jsonl'
    return [json.loads(l) for l in open(path)] if path.exists() else None


def main(run_dir: str, log: str | None) -> dict:
    run = Path(run_dir); manifest = json.loads((run / 'run_manifest.json').read_text()); cfg = manifest['config']
    w = {k: cfg['agent_reward'][k] for k in ('w_outcome', 'w_rubric')}
    eng, health, turns, prompt = Counter(), defaultdict(list), Counter(), Counter()
    batches = {}
    for f in sorted(run.glob('cases/case_*.json')):
        c = json.load(open(f)); t = c['agent_targets']; nodes = {n['node_id']: n for n in c['tree']['nodes']}
        prompt[c.get('prompt_sha256')] += 1
        # rewards and policy
        for nid, x in t['nodes'].items():
            eng['rewards'] += 1
            r = compose_reward(x['outcome_component'], x['R_A'], **w)
            eng['reward_mismatch'] += not (r is x['reward'] if r is None or x['reward'] is None else math.isclose(r, x['reward'], abs_tol=1e-12))
        eng['nodes_policy_differs_from_case'] += sum(n['policy_sha256'] != c['policy_sha256'] for n in nodes.values())
        b = c['case_cursor'] // 5; batches.setdefault(b, {'policies': set(), 'update': c.get('batch_update') or {}})['policies'].add(c['policy_sha256'])
        # tree shape
        turns['max_depth'] = max(turns['max_depth'], max(n['depth'] for n in nodes.values()))
        for l in c['tree']['leaves']: turns[f"leaf_path_{len(l['path'])}"] += 1; turns[f"stop_{l['stop_reason']}"] += 1
        turns['cases'] += 1; turns['leaves'] += len(c['tree']['leaves']); turns['agent_nodes'] += len(nodes)
        # scorer calls: identical Stage-1 requests (same rubric set, same prefix = siblings) must agree; audit records own only
        calls = calls_of(run, c['case_cursor'])
        if calls is not None:
            eng['cases_with_call_log'] += 1
            by_set = defaultdict(dict)
            for d in calls:
                if d['kind'] != 'rubric_score': continue
                by_set[tuple(sorted(d['identity']['rubrics']))][d['identity']['node']] = d['value']
            for rset, vals in by_set.items():
                for nid, v in vals.items():
                    dec = v.get('_decisions') or []
                    stages = [x.get('stage') for x in dec]
                    eng['score_records'] += 1
                    applicable = any(v[r]['applicable'] for r in rset if r in v)
                    if stages != (['applicability', 'level'] if applicable else ['applicability']): eng['audit_record_contaminated'] += 1
                    for x in dec:
                        for rq in x.get('requests') or []:
                            eng['scorer_requests'] += 1; eng['scorer_request_cache_hits'] += bool(rq.get('request_cache_hit'))
                for key, ids in t['groups'].items():
                    present = [i for i in ids if i in vals]
                    if len(present) < 2: continue
                    eng['scored_sibling_groups'] += 1
                    apps = {json.dumps({r: vals[i][r]['applicable'] for r in rset}, sort_keys=True) for i in present}
                    eng['identical_request_inconsistency'] += len(apps) > 1
        # learning health
        health['kept_rubrics'].append(len(c.get('rubric_weights') or {}))   # kept set; training_rubrics is filled only by PromptOpt
        health['candidate_rubrics'].append((c['rubric_bank'].get('telemetry') or {}).get('rubrics', 0))
        for key, ids in t['groups'].items():
            rs = [t['nodes'][i] for i in ids]; rew = [r['reward'] for r in rs if r['reward'] is not None]
            contributing = len(rew) >= 2 and max(rew) - min(rew) > EPS
            health['contributing'].append(contributing); health['outcome_varies'].append(max(r['outcome_component'] for r in rs) - min(r['outcome_component'] for r in rs) > EPS)
            if contributing: m = st.mean(rew); health['abs_adv'] += [abs(x - m) for x in rew]
            for r in rs: health['R_O'].append(r['outcome_component']); health['R_A'].append(r['R_A'])
    steps = [b['update'] for _, b in sorted(batches.items())]
    eng['batches'] = len(batches); eng['batches_with_two_policies'] = sum(len(b['policies']) > 1 for b in batches.values())
    eng['optimizer_steps'] = sum(bool(u.get('optimizer_step')) for u in steps)
    eng['batches_more_than_one_step'] = sum((u.get('optimizer_steps', 0) or 0) > b * 1 + 1 for b, u in enumerate(steps))
    if log:
        text = Path(log).read_text(errors='replace')
        eng['traceback'] = text.count('Traceback'); eng['oom'] = text.count('out of memory') + text.count('OutOfMemory')
    events = list(run.glob('prompt_events/*.json'))
    ra = health['R_A']
    out = {'run': str(run), 'config_variant': (cfg.get('experiment') or {}).get('variant'), 'reward_weights': w,
           'engineering': dict(eng),
           'maxturn2': {**dict(turns), 'rounds_config': cfg['rollout']['rounds']},
           'p0_fixed': {'prompt_hashes': {k[:12] if k else k: v for k, v in prompt.items()}, 'prompt_events': len(events),
                        'prompt_update_enabled': cfg['prompt_update']['enabled'], 'rubric_prompt_sha256_is_P0': cfg['rubric']['prompt_sha256'] == P0},
           'health': {'R_O': q(health['R_O']), 'R_A': q(ra), 'R_A_available': round(sum(x is not None for x in ra) / len(ra), 4) if ra else None,
                      'kept_rubrics_per_case': q(health['kept_rubrics']), 'candidate_rubrics_per_case': q(health['candidate_rubrics']),
                      'zero_kept_cases': sum(k == 0 for k in health['kept_rubrics']),
                      'contributing_groups': round(st.mean(health['contributing']), 4) if health['contributing'] else None,
                      'zero_advantage_groups': round(1 - st.mean(health['contributing']), 4) if health['contributing'] else None,
                      'outcome_varying_groups': round(st.mean(health['outcome_varies']), 4) if health['outcome_varies'] else None,
                      'mean_abs_advantage': q(health['abs_adv']).get('mean'),
                      'grad_norm': [round(u.get('grad_norm'), 4) for u in steps if u.get('grad_norm') is not None],
                      'kl_to_theta0_by_update': [u.get('kl_loss') for u in steps if u.get('kl_loss') is not None]}}
    e = out['engineering']
    out['PASS'] = {'reward_recomputation': e.get('reward_mismatch', 0) == 0,
                   'same_policy_batches': e.get('batches_with_two_policies', 0) == 0 and e.get('nodes_policy_differs_from_case', 0) == 0,
                   'optimizer_at_most_one_per_batch': e['optimizer_steps'] <= e['batches'],
                   'identical_request_inconsistency_0': e.get('identical_request_inconsistency', 0) == 0,
                   'audit_contamination_0': e.get('audit_record_contaminated', 0) == 0,
                   'scorer_cache_active': e.get('scorer_request_cache_hits', 0) > 0 or e.get('scorer_requests', 0) == 0,
                   'maxturn2': out['maxturn2']['max_depth'] <= 2 and cfg['rollout']['rounds'] == 2,
                   'p0_fixed': set(prompt) <= {P0, None} and not events and not cfg['prompt_update']['enabled'],
                   'traceback_oom_0': (e.get('traceback', 0) + e.get('oom', 0)) == 0 if log else None}
    return out


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('run_dir'); a.add_argument('--log'); a.add_argument('--json')
    x = a.parse_args(); r = main(x.run_dir, x.log)
    if x.json: Path(x.json).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
