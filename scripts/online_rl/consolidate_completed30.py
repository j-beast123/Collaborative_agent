"""Consolidate saved P0/P1 candidates without new rollouts or interventions."""
import json,sys,os,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import from_dict,to_dict,atomic_write_json,sha256_json
from src.online_rl.tree.records import TrajectoryRubric
from src.online_rl.tree.consolidation import consolidate,PROMPT,serializable_consolidation
from src.online_rl.tree.backends import api_rubric_consolidator
from src.online_rl.tree.cache import CallCache
from scripts.online_rl.run_tree import load_key


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/online_rl/tree_batch12_100cases_haiku_gemini_luna.json').read_text())
    model=cfg['pilot']['prompt_model'];os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl')
    generate=api_rubric_consolidator(load_key(),model,8192);cache=CallCache(out/'calls.jsonl')
    atomic_write_json(out/'manifest.json',{'model':model,'prompt':PROMPT.read_text(),'prompt_sha256':sha256_json(PROMPT.read_text()),'source':'completed30 P0 and P1','reruns':False})
    allresults={};excluded=[];summary={}
    for version in ['P0','P1']:
        results=[];total=0;eligible=0
        folder=ROOT/'outputs/pilots/completed30_seed20261007/prompt_comparison'/version
        for f in sorted(folder.glob('*.json')):
            candidates=[]
            for uid,row in json.loads(f.read_text())['results'].items():
                total+=1;r=row['rubric']
                if r['valid']:
                    candidates.append(from_dict(TrajectoryRubric,r))
                else:
                    excluded.append({'version':version,'case_id':row['case_id'],'leaf_id':r['leaf_id'],'rubric_id':r['rubric_id'],'problems':r['problems'],'gate':r['gate']})
            eligible+=len(candidates)
            if not candidates:continue
            cid=f.stem
            result=consolidate(cid,candidates,generate,cache,model_identity={'model':model,'version':version})
            atomic_write_json(out/version/(cid+'.json'),serializable_consolidation(result));results.append(result)
            print(json.dumps({'version':version,'case_id':cid,'candidates':len(candidates),'representatives':len(result['groups'])}),flush=True)
        groups=[g for r in results for g in r['groups']]
        summary[version]={'trajectories':total,'eligible_candidates':eligible,'excluded':total-eligible,'representatives':len(groups),'merged_groups':sum(len(g['member_ids'])>1 for g in groups)}
        allresults[version]=results
        atomic_write_json(out/'progress.json',summary)
    atomic_write_json(out/'excluded.json',excluded);atomic_write_json(out/'summary.json',summary)
    text=['P0/P1 case-local consolidation. No reruns or keep decisions performed.',json.dumps(summary,ensure_ascii=False,indent=2)]
    for version,results in allresults.items():
        for result in results:
            byid={r.rubric_id:r for r in result['originals']}
            for g in result['groups']:
                text += [f'\n{version} | {result["case_id"]}', 'Representative: '+g['representative_id'],'Members: '+', '.join(g['member_ids']), 'Reason: '+g['reason'],'Original trajectory/turn mapping: '+json.dumps(g['source_targets'],ensure_ascii=False,indent=2),'Representative original rubric: '+json.dumps(byid[g['representative_id']].rubric['candidate_rubric'],ensure_ascii=False,indent=2)]
    (out/'consolidation_review.txt').write_text('\n\n'.join(text))
    import zipfile
    with zipfile.ZipFile(out/'consolidation_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*.json'):z.write(p,p.relative_to(out))
        z.write(out/'consolidation_review.txt','consolidation_review.txt')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        arg=sys.argv[sys.argv.index('--out')+1]
        atomic_write_json(ROOT/arg/'error.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
