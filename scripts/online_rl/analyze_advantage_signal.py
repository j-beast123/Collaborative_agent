"""Rebuild per-step/type signal tables from saved case or target-turn training."""
import argparse
import csv
import io
from pathlib import Path
import statistics
import sys
import math

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.online_rl.integrated.grpo import prepare_advantage_groups, advantage_summary, signal_by_group_type
from src.online_rl.integrated.records import read_json, atomic_write_json
from src.online_rl.sherpa import PROTOCOL
from src.online_rl.sherpa.plan_c_rollouts import PROTOCOL as PLAN_C_PROTOCOL


def step_groups(home, metrics, cfg):
    update = metrics['update']
    diagnostics = update.get('advantage_diagnostics', {})
    mode = diagnostics.get('mode', cfg['advantage_normalization']['mode'])
    details = diagnostics.get('groups', [])
    if details and all('advantage' in g and g.get('group_type') in
                       ('mixed','all_success','all_failure') for g in details):
        return details, mode
    # Old logs contain exact rewards/advantages but no outcome-type annotation.
    # Read original case evidence; never infer outcome from a rubric reward.
    rebuilt = []
    folder = Path(home) / 'steps' / f'step_{metrics["step"]:06d}'
    for old in details:
        record = read_json(folder / old['group'] / 'record.json')
        outcomes = {t['trajectory_id']: t['outcome'] for t in record['trajectories']}
        rows = [{**r, 'outcome': outcomes[r['trajectory_id']]} for r in record['training_records']]
        if [r['trajectory_id'] for r in rows] != old['trajectory_ids']:
            raise ValueError('Historical trajectory ordering differs from the optimizer log')
        if [r['reward'] for r in rows] != old['rewards']:
            raise ValueError('Historical rewards differ from the optimizer log')
        _, _, _, group = prepare_advantage_groups([rows], mode)
        computed = group[0]['advantages']
        recorded = old['advantages']
        if len(computed) != len(recorded) or any(
                (a is None) != (b is None) or (a is not None and not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10))
                for a,b in zip(computed,recorded)):
            raise ValueError('Historical advantage formula differs; refusing to label a recomputation as observed')
        if group[0]['used'] != old['used']:
            raise ValueError('Historical group inclusion differs')
        rebuilt.extend(group)
    if not rebuilt:
        raise ValueError('No saved trajectory advantage diagnostics found')
    return rebuilt, mode


def analyze_run(home, max_step=30):
    home = Path(home)
    manifest = read_json(home / 'run_manifest.json')
    protocol = manifest.get('protocol')
    if protocol not in (PROTOCOL, PLAN_C_PROTOCOL):
        raise ValueError('Unknown trajectory/turn training protocol')
    cfg = manifest['config']
    state = read_json(home / 'run_state.json')
    group_rows, type_rows, curve_rows = [], [], []
    for path in sorted((home/'steps').glob('step_*/metrics.json')):
        metrics = read_json(path)
        step = metrics['step']
        if step > max_step or step > state['completed_steps']:
            continue
        details, mode = step_groups(home, metrics, cfg)
        if protocol == PROTOCOL:
            if len(details) != 4 or any(len(g['trajectory_ids']) != 32 for g in details):
                raise ValueError('Expected exactly four groups of 32 trajectories per step')
        elif (not 4 <= len(details) <= 4 * cfg['rollout']['rounds']
              or any(len(g['trajectory_ids']) != cfg['plan_c']['responses_per_turn'] for g in details)):
            raise ValueError('Incomplete Plan C turn groups')
        summaries = signal_by_group_type(details)
        for kind, values in summaries.items():
            type_rows.append({'run':home.name,'step':step,'mode':mode,'group_type':kind,**values})
        for g in details:
            rewards = [r for r in g['rewards'] if r is not None]
            group_rows.append({'run':home.name,'step':step,'mode':mode,'case_id':g['group'],
                'group_type':g['group_type'], 'outcome_successes':sum(g['outcomes']),
                'group_used':g['used'], 'reward_std':statistics.pstdev(rewards),
                'auc':metrics.get('case_auc',metrics.get('turn_auc',{})).get(g['group']),
                'sum_abs_advantage':g['advantage']['sum_abs'],
                'mean_abs_advantage':g['advantage']['mean_abs'],
                'max_abs_advantage':g['advantage']['max_abs'],
                'trajectories':g['advantage']['count'], 'used_trajectories':len(g['used_trajectory_ids']),
                'used_assistant_tokens':g['used_assistant_tokens']})
        distribution = advantage_summary([a for g in details for a in g['advantages']])
        skipped = sum(not g['used'] for g in details)
        curve_rows.append({'run':home.name,'step':step,'mode':mode,
            'mean_abs_advantage':distribution['mean_abs'],'max_advantage':distribution['max'],
            'max_abs_advantage':distribution['max_abs'], 'grad_norm':metrics['update'].get('grad_norm'),
            'kl':metrics.get('kl'), 'groups_skipped':skipped,'skipped_group_fraction':skipped/len(details)})
    summaries = []
    total_abs = sum(g['sum_abs_advantage'] for g in group_rows)
    total_tokens = sum(g['used_assistant_tokens'] for g in group_rows)
    for kind in ('mixed','all_success','all_failure'):
        groups = [g for g in group_rows if g['group_type'] == kind]
        abs_sum = sum(g['sum_abs_advantage'] for g in groups)
        tokens = sum(g['used_assistant_tokens'] for g in groups)
        count = sum(g['trajectories'] for g in groups)
        aucs = [g['auc'] for g in groups if g['auc'] is not None]
        summaries.append({'run':home.name, 'mode':cfg['advantage_normalization']['mode'], 'group_type':kind,
            'groups_total':len(groups), 'groups_used':sum(g['group_used'] for g in groups),
            'groups_skipped':sum(not g['group_used'] for g in groups),
            'mean_auc':statistics.mean(aucs) if aucs else None,
            'mean_reward_std':statistics.mean(g['reward_std'] for g in groups) if groups else None,
            'sum_abs_advantage':abs_sum,'share_abs_advantage':abs_sum/total_abs if total_abs else None,
            'mean_abs_advantage':abs_sum/count if count else None,
            'max_abs_advantage':max((g['max_abs_advantage'] for g in groups),default=None),
            'used_assistant_tokens':tokens, 'share_used_tokens':tokens/total_tokens if total_tokens else None})
    return {'source':str(home.resolve()),'completed_steps':state['completed_steps'],
            'group_scope':'case_turn_shared_history' if protocol == PLAN_C_PROTOCOL else 'case',
            'analyzed_steps':len(curve_rows),'max_requested_step':max_step,
            'advantage_mode':cfg['advantage_normalization']['mode'],
            'signal_definition':'sum of absolute trajectory advantages, not measured gradient attribution',
            'group_rows':group_rows, 'type_rows':type_rows, 'curve_rows':curve_rows,'summary':summaries}


def write_csv(path, rows):
    if not rows:
        return
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    path.write_text(stream.getvalue())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--reference-dir')
    parser.add_argument('--max-step', type=int, default=30)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    if args.max_step < 1:
        parser.error('--max-step must be positive')
    sources = [Path(args.run_dir).resolve()]
    if args.reference_dir:
        sources.append(Path(args.reference_dir).resolve())
    output = Path(args.output_dir).resolve()
    if any(output == p or p in output.parents or output in p.parents for p in sources):
        parser.error('Report output must be separate from source training directories')
    reports = [analyze_run(path,args.max_step) for path in sources]
    output.mkdir(parents=True,exist_ok=True)
    atomic_write_json(output/'report.json', {'runs':reports,'held_out_evaluation':False})
    for name in ('group_rows','type_rows','curve_rows','summary'):
        write_csv(output/f'{name}.csv',[row for report in reports for row in report[name]])
    print(f'Wrote signal comparison to {output}; analyzed steps: '+', '.join(str(r['analyzed_steps']) for r in reports))


if __name__ == '__main__':
    main()
