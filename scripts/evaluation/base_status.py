"""Aggregate baseline results without loading a model or calling an API."""
import argparse
from collections import Counter
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--run-dir',default='outputs/test/base/casimedicos_exp_5turn_seed20261006')
a=p.parse_args();root=Path(a.run_dir)
rows=[json.loads(p.read_text()) for p in sorted(root.glob('gpu*/cases/case_*.json'))]
summary={'completed_cases':len(rows),'total_cases':622,'shards':{},'splits':{}}
for gpu in range(1,5):
    folder=root/f'gpu{gpu}'
    summary['shards'][f'gpu{gpu}']={'saved_cases':len(list(folder.glob('cases/case_*.json'))),
       'technical_errors':len(list(folder.glob('errors/case_*.json')))}
for split in ['train','dev','test','all']:
    selected=[r for r in rows if split=='all' or r['source_split']==split]
    valid=[r for r in selected if r['status']=='complete']
    correct=sum(r['outcome']==1.0 for r in valid)
    summary['splits'][split]={'saved':len(selected),'scorable':len(valid),'correct':correct,
        'accuracy':correct/len(valid) if valid else None,'unparseable_final':len(selected)-len(valid)}
if root.exists(): (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
