"""Read-only provider-separated API audit summary; missing dollar cost stays unknown."""
import argparse
import json
from pathlib import Path


def summarize(root):
    path=Path(root)/'reward_api_attempts.jsonl'; groups={}
    for line in path.read_text().splitlines() if path.exists() else []:
        row=json.loads(line)
        provider=row.get('provider') or ('openrouter' if str(row.get('requested_model','')).startswith(('openai/','google/')) else 'unknown')
        if 'openrouter.ai' in row.get('base_url',''):provider='openrouter'
        groups.setdefault(provider,[]).append(row)
    result={}
    for provider,rows in groups.items():
        known=[r['cost_usd'] for r in rows if isinstance(r.get('cost_usd'),(int,float))]
        result[provider]={'attempts':len(rows),'successful_attempts':sum(r['status']=='ok' for r in rows),
            'failed_attempts':sum(r['status']!='ok' for r in rows),
            'reported_cost_usd':sum(known) if len(known)==len(rows) else None,
            'known_reported_cost_subtotal_usd':sum(known),'unreported_cost_attempts':len(rows)-len(known),
            'prompt_tokens_reported':sum((r.get('usage') or {}).get('prompt_tokens',0) for r in rows),
            'completion_tokens_reported':sum((r.get('usage') or {}).get('completion_tokens',0) for r in rows),
            'usage_missing_attempts':sum(not r.get('usage') for r in rows)}
    return {'providers':result,'note':'Token totals cover only reported usage. Unknown charges are not zero; reconcile with provider billing. Concurrent request latencies cannot be summed as elapsed time.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args()
    print(json.dumps(summarize(a.run),indent=2))
