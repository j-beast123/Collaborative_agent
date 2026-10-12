"""Review sparse candidate tags; keep the outcome-blind initial audit intact."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from scripts.online_rl.analyze_base_behaviors import TrainingGate, write_report
from src.online_rl.integrated.records import atomic_write_json,read_json,sha256_json,sha256_text
from src.online_rl.sherpa.behavior_analysis import (
    decode_annotations,dialogue_view,evidence_passages,numbered_dialogue,tag_schema)

SPARSE={'contradiction_flag','unverified_fact_assertion'}


def apply_review(row,behaviors,rejected):
    candidates=set(row['tags'])&SPARSE
    accepted=[b['tag'] for b in behaviors]
    if (len(set(accepted))!=len(accepted) or len(set(rejected))!=len(rejected)
            or set(accepted)&set(rejected) or set(accepted)|set(rejected)!=candidates):
        raise ValueError('Every original sparse candidate must be accepted or rejected exactly once')
    result=copy.deepcopy(row)
    result['evidence']=[e for e in row['evidence'] if e['tag'] not in SPARSE]+behaviors
    result['tags']=[e['tag'] for e in result['evidence']]
    result['sparse_review']={'accepted':accepted,'rejected':rejected}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/ours.json')
    p.add_argument('--analysis-dir',default='outputs/base_behavior_luna_v2')
    p.add_argument('--concurrency',type=int,default=64)
    p.add_argument('--launch',action='store_true')
    a=p.parse_args()
    if not 1<=a.concurrency<=100:p.error('Invalid concurrency')
    os.chdir(ROOT)
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.cases import load_cases
    cfg=load_config(a.config);home=ROOT/a.analysis_dir;out=home/'verified'
    source=read_json(home/'annotations.json')
    if not read_json(home/'report.json')['complete'] or len(source)!=3840:
        raise ValueError('Finish all 3840 first-pass annotations before verification')
    cases={c['case_id']:c for c in load_cases(ROOT/cfg['data']['train'])}
    rows={r['trajectory_id']:r for r in source};jobs=[]
    for cid in cases:
        candidates=[r for r in source if r['case_id']==cid and set(r['tags'])&SPARSE]
        if not candidates:continue
        trajectories={t['trajectory_id']:t for t in read_json(ROOT/cfg['prompt_optimization']['output_dir']/'base_rollouts'/cid/'rollouts.json')['trajectories']}
        for start in range(0,len(candidates),8):
            group=candidates[start:start+8];mapping={f'dialogue_{i}':r for i,r in enumerate(group)}
            views={key:dialogue_view(trajectories[r['trajectory_id']]) for key,r in mapping.items()}
            request={'case':{k:cases[cid].get(k) for k in ('scenario','question','options')},'dialogues':[
                {'dialogue_id':key,'candidate_tags':sorted(set(r['tags'])&SPARSE),'dialogue':numbered_dialogue(views[key])}
                for key,r in mapping.items()]}
            jobs.append((mapping,views,request))
    prompt=(ROOT/'prompts/sherpa/behavior_sparse_review_v1.txt').read_text()
    identity={'source':sha256_json(source),'prompt':sha256_text(prompt),'model':'gpt-6-luna'}
    print(json.dumps({'candidate_dialogues':sum(bool(set(r['tags'])&SPARSE) for r in source),'batches':len(jobs),'launch':a.launch}),flush=True)
    if not a.launch:return
    out.mkdir(exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=out/'manifest.json'
        if path.exists() and read_json(path)!=identity:raise RuntimeError('Sparse review identity changed')
        atomic_write_json(path,identity)
        os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl');os.environ['API_USAGE_STAGE']='behavior_sparse_review'
        os.environ['SHERPA_RATE_LIMIT_RETRIES']='0'
        from scripts.online_rl.run_tree import load_key
        from src.online_rl.tree.openai_direct import responses_json
        key=load_key({'api_provider':'openai_direct'});gate=TrainingGate(ROOT/cfg['output_dir'])
        settings={'max_tokens':8192,'reasoning_effort':'low','response_format':'json_schema_strict'}
        def review(job):
            mapping,views,request=job
            cache_id={'protocol':identity,'request':sha256_json(request)}
            path=out/'batches'/f'{sha256_json(cache_id)}.json'
            if path.exists():record=read_json(path)
            else:
                schema=tag_schema(views,sorted({key for view in views.values() for key in evidence_passages(view)}))
                properties=schema['properties']['annotations']['items']['properties']
                properties['rejected_tags']={'type':'array','items':{'type':'string','enum':sorted(SPARSE)},'maxItems':2}
                schema['properties']['annotations']['items']['required'].append('rejected_tags')
                for attempt in range(2):
                    gate.wait()
                    raw,meta=responses_json(key,'gpt-6-luna',prompt,json.dumps(request,ensure_ascii=False),settings,schema,'sparse_tag_review','behavior_sparse_review')
                    try:
                        value=json.loads(raw)
                        decoded=decode_annotations({'annotations':[{'dialogue_id':r['dialogue_id'],'behaviors':r['behaviors']} for r in value['annotations']]},views)
                        by={r['dialogue_id']:r for r in decoded['annotations']}
                        reviewed=[apply_review(mapping[r['dialogue_id']],by[r['dialogue_id']]['behaviors'],r['rejected_tags']) for r in value['annotations']]
                    except (ValueError,TypeError,KeyError) as exc:
                        atomic_write_json(out/'validation_errors'/f'{path.stem}_{attempt}.json',{'error':str(exc),'raw':raw})
                        if attempt:raise
                        request['repair']={'error':str(exc)}
                        continue
                    record={'identity':cache_id,'rows':reviewed,'meta':{k:meta.get(k) for k in ('response_id','usage')}}
                    atomic_write_json(path,record);break
            if record['identity']!=cache_id:raise RuntimeError('Sparse review cache identity changed')
            return record['rows']
        with ThreadPoolExecutor(a.concurrency) as pool:
            for future in as_completed([pool.submit(review,j) for j in jobs]):
                for row in future.result():rows[row['trajectory_id']]=row
        result=write_report(out,list(rows.values()),True)
        result['sparse_review']={'candidates':sum(bool(set(r['tags'])&SPARSE) for r in source),
                                'original_counts':{tag:sum(tag in r['tags'] for r in source) for tag in SPARSE},
                                'verified_counts':{tag:sum(tag in r['tags'] for r in rows.values()) for tag in SPARSE},
                                'note':'Only original positive candidates for these two tags were reviewed; false negatives were not exhaustively audited.'}
        atomic_write_json(out/'report.json',result)
        print(json.dumps(result['sparse_review'],indent=2),flush=True)


if __name__=='__main__':main()
