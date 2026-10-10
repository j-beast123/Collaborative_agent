"""Separate reward provider attempts from original rollout/training logs."""
import argparse
import json
from pathlib import Path
from src.online_rl.integrated.records import sha256_json


def summarize(roots):
    paths = sorted({p.resolve() for root in roots for p in Path(root).rglob('reward_api_attempts.jsonl')})
    attempts = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
    costs = [r.get('cost_usd') for r in attempts]
    by_kind = {}
    for row in attempts:
        by_kind[row['kind']] = by_kind.get(row['kind'],0)+1
    unique_trees, private, visible = set(), 0, 0
    for root in roots:
        for path in (Path(root)/'cases').glob('case_*.json'):
            record=json.loads(path.read_text());tree=record['tree'];key=sha256_json(tree)
            if key in unique_trees:continue
            unique_trees.add(key)
            for node in tree['nodes']:
                for segment in node['trace'].get('segments',[]):
                    n=len(segment.get('completion_ids',[]))
                    if segment.get('role')=='private_assessment':private+=n
                    elif segment.get('role')=='visible_reply':visible+=n
    reference_paths=sorted({p.resolve() for root in roots for p in Path(root).rglob('reference_generation.jsonl')})
    references=[json.loads(line) for path in reference_paths for line in path.read_text().splitlines() if line.strip()]
    initial_imports=sum(len(p.read_text().splitlines()) for root in roots for p in Path(root).rglob('initial_rubric_imports.jsonl'))
    return {'reference_generation_attempts':len(references),
        'reference_generation_successes':sum(r['status']=='ok' for r in references),
        'reference_generation_failures':sum(r['status']!='ok' for r in references),
        'reference_private_tokens':sum(r.get('private_tokens',0) for r in references),
        'reference_visible_tokens':sum(r.get('visible_tokens',0) for r in references),
        'reference_generation_seconds':sum(r['seconds'] for r in references),
        'reference_doctor_continuations':sum(r['doctor_calls'] for r in references),
        'initial_rubrics_imported_without_API':initial_imports,
        'roots':[str(Path(p).resolve()) for p in roots],'reward_api_attempts':len(attempts),
        'reward_successful_attempts':sum(r['status']=='ok' for r in attempts),
        'reward_failed_attempts':sum(r['status']!='ok' for r in attempts),
        'unique_successful_request_hashes':len({r['request_sha256'] for r in attempts if r['status']=='ok'}),
        'attempts_by_operation':by_kind,'reward_api_latency_s':sum(r['latency_s'] for r in attempts),
        'known_reported_cost_usd':sum(c for c in costs if isinstance(c,(int,float))),
        'total_reward_api_cost_usd':sum(costs) if costs and all(isinstance(c,(int,float)) for c in costs) else None,
        'unreported_attempt_cost_count':sum(c is None for c in costs),
        'unique_generated_case_trees':len(unique_trees),'private_action_tokens':private,'visible_action_tokens':visible,
        'gpu_hours':None,'doctor_tokens':None,
        'note':'Provider attempts include retries. No price means unknown, not zero. GPU-hours/Doctor tokens require original runtime logs; API latency is not GPU time. Reference generation attempts/visible and private tokens are separate from current-policy rollouts. Reference seconds are measured generation wall time, not total GPU allocation hours. Failed requests may have unreported token costs. Imported initial rubrics incur no local API cost but retain source provenance.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();Path(a.out).write_text(json.dumps(summarize(a.runs),indent=2)+'\n')
