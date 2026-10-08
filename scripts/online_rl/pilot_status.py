"""Read pilot progress without GPU or API calls."""
import argparse,json
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('--run-dir',default='outputs/latest_pilot');args=p.parse_args()
root=Path(args.run_dir).resolve()
state=json.loads((root/'run_state.json').read_text()) if (root/'run_state.json').exists() else {}
counts=Counter()
if (root/'calls.jsonl').exists():
    for line in (root/'calls.jsonl').read_text().splitlines():
        if line.strip():counts[json.loads(line)['kind']]+=1
print(json.dumps({'optimization_cases_completed':state.get('case_cursor',0),'optimization_cases_total':4,
 'heldout_completed':(root/'heldout/result.json').exists(),'prompt_events':state.get('lineage_cursor',0),
 'current_prompt':state.get('prompt_id'),'completed_calls':dict(counts),
 'log':str(root/'run.log'),'report':str(root/'review/report.md')},indent=2))
