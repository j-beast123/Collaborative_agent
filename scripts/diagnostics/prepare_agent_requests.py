"""Read-only extraction of actual precommit Agent actions for parallel timing."""
import hashlib
import json
from pathlib import Path
from scripts.diagnostics.doctor_parallel_bench import ROOT,rows


def prepare(out,count=64):
    target=Path(out)/'requests.json';data=json.loads(target.read_text());run=Path(data['source_run']);actions=[]
    for manifest_path in sorted(run.glob('batches/batch_*/manifest.json')):
        manifest=json.loads(manifest_path.read_text())
        for a,cid in zip(manifest['assignments'],manifest['case_ids']):
            opening=next(r for r in rows(run/'openings'/f'train_gpu{a["worker"]}.jsonl') if r['identity']['case']['case_id']==cid)
            cache=manifest_path.parent/f'case_{a["index"]:06d}'/'calls.jsonl';records=list(rows(cache))
            agents={r['identity']['path']:r for r in records if r['kind']=='agent'}
            doctors={r['identity']['path']:r['value'] for r in records if r['kind']=='doctor_turn'}
            for path,row in agents.items():
                prefix=[{'speaker':'doctor','stage':'D1','round':1,'text':opening['value']['visible']}]
                parts=path.split('.');valid=True
                for depth in range(1,len(parts)):
                    ancestor='.'.join(parts[:depth])
                    if ancestor not in agents or ancestor not in doctors:valid=False;break
                    prefix.extend([{'speaker':'agent','stage':f'A{depth}','round':depth,'text':agents[ancestor]['value']['completion_text']},
                                   {'speaker':'doctor','stage':f'D{depth+1}','round':depth+1,'text':doctors[ancestor]['visible']}])
                if valid:actions.append({'case':opening['identity']['case'],'prefix':prefix,'turn':len(parts),'seed':row['identity']['seed'],'path':path})
    actions.sort(key=lambda r:hashlib.sha256((r['case']['case_id']+'/'+r['path']).encode()).hexdigest())
    if len(actions)<count:raise RuntimeError('Not enough completed Agent inputs')
    data['agent_requests']=actions[:count];target.write_text(json.dumps(data,ensure_ascii=False));print(json.dumps({'agent_actions':count,'segments_per_action':2}))

if __name__=='__main__':
    import sys
    prepare(ROOT/sys.argv[1])
