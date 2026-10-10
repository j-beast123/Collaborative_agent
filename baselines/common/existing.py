"""Read-only eligibility audit for an existing outcome-only result."""
import json
from pathlib import Path
from baselines.common.protocol import identity
from baselines.common.data import _cases
from src.online_rl.integrated.records import sha256_json


def audit(root,cfg):
    root=Path(root)
    manifest=json.loads((root/'run_manifest.json').read_text())
    remote=manifest['config'];state=json.loads((root/'run_state.json').read_text())
    initial=json.loads((root/'checkpoints/checkpoint_000000/COMPLETE.json').read_text())
    evidence=remote.get('comparison_method',manifest)
    problems=[]
    if identity(remote)!=identity(cfg):problems.append('Common conditions differ')
    if any(remote['agent_reward'][k]!=v for k,v in {'w_outcome':1.,'w_rubric':0.}.items()):problems.append('Not outcome-only')
    if remote['prompt_update']['enabled']:problems.append('Prompt optimization enabled')
    if state['case_cursor']!=120:problems.append('Training incomplete')
    from baselines.common.initialization import expected_initial
    expected=expected_initial(cfg)
    if expected is None:problems.append('Initial adapter comparison pending: no first-run local hash recorded yet')
    elif initial['policy_sha256']!=expected:problems.append('Initial adapter differs')
    if evidence.get('base_model_revision')!=cfg['comparison_method']['base_model_revision']:problems.append('Base revision missing/different')
    if evidence.get('package_versions')!=cfg['comparison_method']['package_versions']:problems.append('Package evidence missing/different')
    folder=root/'evaluation/checkpoint_000120'
    em=json.loads((folder/'manifest.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    if em['policy_sha256']!=state['policy_sha256']:problems.append('Evaluated adapter differs')
    settings=em.get('settings',{})
    if settings.get('data_sha256')!=cfg['five_gpu']['eval_sha256'] or settings.get('seed')!=cfg['five_gpu']['eval_seed']:problems.append('Test identity differs')
    if summary.get('status')!='complete' or summary.get('evaluated')!=622:problems.append('Test622 incomplete')
    if any(p.name!='checkpoint_000120' for p in (root/'evaluation').glob('checkpoint_*')):problems.append('Intermediate test evaluation exists')
    test_ids=[c['case_id'] for c in _cases(cfg['five_gpu']['eval_file'])]
    if em.get('case_ids')!=test_ids:problems.append('Test row order differs')
    final=json.loads((root/'checkpoints/checkpoint_000120/COMPLETE.json').read_text())
    if final.get('policy_sha256')!=state['policy_sha256'] or final.get('completed_cases')!=120:problems.append('Final checkpoint receipt differs')
    rows=[json.loads(p.read_text()) for p in sorted((folder/'cases').glob('case_*.json'))]
    if len(rows)!=622 or [r.get('index') for r in rows]!=list(range(622)):problems.append('Evaluation case rows missing/duplicated')
    elif any(r.get('job_sha256')!=sha256_json(em) or r.get('case_id')!=test_ids[i] for i,r in enumerate(rows)):problems.append('Evaluation row identity differs')
    if len(rows)==622:
        correct=sum(r['outcome']==1 for r in rows);unparsed=sum(r['outcome'] is None for r in rows)
        if (summary['correct'],summary['unparsed'],summary['accuracy'])!=(correct,unparsed,correct/622):problems.append('Summary/denominator differs')
    result={'eligible_for_reuse':not problems,'issues':problems,'source':str(root),'summary':summary}
    if problems:raise RuntimeError(json.dumps(result))
    return result
