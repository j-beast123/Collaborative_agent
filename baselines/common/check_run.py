"""Read-only check after a user-run one-batch smoke, or later committed batch."""
import argparse,json
from pathlib import Path
from src.online_rl.tree.team import node_credit
from src.online_rl.tree.records import Tree
from src.online_rl.integrated.records import from_dict


def check(root,expected_cases=None):
    out=Path(root);cfg=json.loads((out/'run_manifest.json').read_text())['config']
    state=json.loads((out/'run_state.json').read_text());cursor=state['case_cursor']
    if expected_cases is not None and cursor!=expected_cases:raise RuntimeError('Unexpected committed case count')
    if cursor<5 or cursor%5:raise RuntimeError('Expected complete five-case batches')
    order=json.loads(Path(cfg['five_gpu']['train_order_file']).read_text())
    rows=[json.loads((out/'cases'/f'case_{i:06d}.json').read_text()) for i in range(cursor)]
    if [r['case_id'] for r in rows]!=order[:cursor]:raise RuntimeError('Case order differs')
    steps=0
    for start in range(0,cursor,5):
        batch=rows[start:start+5]
        if len({r['policy_sha256'] for r in batch})!=1:raise RuntimeError('Mixed policy batch')
        updates=[r['batch_update'] for r in batch]
        update=updates[0]
        if any(u!=update for u in updates) or not isinstance(update.get('optimizer_step'),bool):raise RuntimeError('Expected one shared batch update record')
        steps+=int(update['optimizer_step'])
        if update['optimizer_step'] and update.get('optimizer_steps')!=steps:raise RuntimeError('Cumulative optimizer step count differs from committed batches')
    weights=cfg['comparison_method']['actual_reward_weights']
    for row in rows:
        targets=row['agent_targets'];tree=from_dict(Tree,row['tree'])
        direct={k:v['R_A'] for k,v in targets['nodes'].items()}
        if targets!=node_credit(**weights)(tree,{},direct):raise RuntimeError('Stored reward mixture differs')
    if (out/'evaluation').exists() and list((out/'evaluation').glob('checkpoint_*')):raise RuntimeError('Smoke/training should not have test evaluation')
    return {'committed_cases':cursor,'five_case_batches':cursor//5,'optimizer_steps':steps,'reward_weights':weights,
        'role_placement':cfg.get('four_gpu'),'test_evaluated':False,'check':'passed',
        'note':'Zero-advantage/invalid groups can produce a legitimate skipped update; inspect per-case group diagnostics.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--expected-cases',type=int)
    a=p.parse_args();print(json.dumps(check(a.run,a.expected_cases),indent=2))
