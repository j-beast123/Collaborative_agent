"""Verify consolidated representatives at every distinct linked source target."""
import json,sys,os,copy,traceback
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import atomic_write_json,from_dict,sha256_json
from scripts.online_rl.run_completed30 import jsonable
from src.online_rl.tree.records import AgentNode,Leaf,TrajectoryRubric


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source');p.add_argument('--doctor-port',type=int,default=11555);a=p.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    base=ROOT/'outputs/pilots/completed30_seed20261007';src=ROOT/a.source if a.source else base/'consolidation/P0_P1_v1'
    selection=json.loads((base/'selection.json').read_text());sources={s['leaf']['leaf_id']:s for s in selection['selected']}
    plan=[]
    for version in ['P0','P1']:
        for f in sorted((src/version).glob('*.json')):
            bundle=json.loads(f.read_text());originals={r['rubric_id']:r for r in bundle['originals']}
            for group in bundle['groups']:
                seen={};targets=[]
                for member in group['member_ids']:
                    original=originals[member];source=sources[original['leaf_id']];node=next(n for n in source['nodes'] if n['node_id']==original['selected_agent_turn_id'])
                    key=sha256_json([node['node_id'],node['prefix'],node['policy_sha256'],source['leaf']['outcome']])
                    if key in seen:
                        targets[seen[key]]['linked_members'].append(member);targets[seen[key]]['linked_leaf_ids'].append(original['leaf_id']);continue
                    seen[key]=len(targets);targets.append({'key':key,'member':member,'linked_members':[member],'linked_leaf_ids':[original['leaf_id']],'source':source,'node':node})
                plan.append({'version':version,'case_id':bundle['case_id'],'group':group,'representative':originals[group['representative_id']],'targets':targets})
    cfg=copy.deepcopy(selection['config']);cfg['output_dir']=str(out.relative_to(ROOT));cfg['gpu']['rollout_devices']=['cuda:0']
    cfg['batch']['doctor_urls']=[f'http://127.0.0.1:{a.doctor_port}'];cfg['batch']['doctor_gpu_ids']=[0];cfg['batch']['doctor_parallelism']=1
    # Fresh guidance and current scorer/reviewer, fixed original Agent adapter.
    current=json.loads((ROOT/'configs/online_rl/tree_batch12_100cases_haiku_gemini_luna.json').read_text());cfg['intervention']=current['intervention'];cfg['judge']=current['judge']
    os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl')
    total=sum(len(g['targets']) for g in plan)
    atomic_write_json(out/'manifest.json',{'config':cfg,'source':str(src),'policy_sha256':selection['policy_sha256'],'checkpoint':selection['checkpoint'],'groups':len(plan),'distinct_targets':total,'maximum_reruns':3*total,'retention':'any applicable target with >=2/3 desired outcomes; incomplete unresolved groups deferred','agent_update':False})
    from scripts.online_rl.run_tree import load_key
    from scripts.online_rl.start_batch_doctors import start
    from src.online_rl.tree import batch_worker as W
    from src.online_rl.tree.effectiveness import evaluate_rubric
    start(cfg);W.initialize(cfg,0,load_key());W.load_checkpoint(selection['checkpoint'],selection['policy_sha256'])
    completed=0;reports=[]
    try:
        for item in plan:
            folder=out/item['version']/item['representative']['rubric_id'];folder.mkdir(parents=True,exist_ok=True)
            results=[]
            for target in item['targets']:
                source=target['source'];case=source['case'];leaf=from_dict(Leaf,source['leaf']);node=from_dict(AgentNode,target['node'])
                rep=from_dict(TrajectoryRubric,item['representative']);payload=copy.deepcopy(rep.rubric)
                payload['source_trajectory_id']=leaf.leaf_id
                if isinstance(payload.get('analysis'),dict):payload['analysis']['selected_agent_turn_id']=node.node_id
                projected=replace(rep,leaf_id=leaf.leaf_id,selected_agent_turn_id=node.node_id,rubric=payload)
                cache,doctor,iv,generate,scorer,gate,backend,strategies=W.components(case,folder)
                scores,effect=evaluate_rubric(case,{node.node_id:node},leaf,projected,scorer=scorer,writer=iv.write,continuation=iv.continue_,reruns=3,run_seed=cfg['schedule']['seed'],policy_sha256=selection['policy_sha256'],strategies=strategies,cache=cache)
                result={'target_key':target['key'],'node_id':node.node_id,'linked_member_ids':target['linked_members'],'linked_trajectory_ids':target['linked_leaf_ids'],'source_outcome':leaf.outcome,'instruction':iv.instruction(payload,effect.level) if effect else None,'source_scores':jsonable(scores),'effectiveness':jsonable(effect)}
                results.append(result);atomic_write_json(folder/'targets'/(target['key']+'.json'),result);completed+=1
                atomic_write_json(out/'progress.json',{'stage':'reruns','completed_targets':completed,'total_targets':total,'completed_groups':len(reports),'total_groups':len(plan)})
                print(json.dumps({'version':item['version'],'rubric_id':rep.rubric_id,'completed_targets':completed,'total_targets':total,'status':effect.status if effect else None,'R_O':effect.R_O if effect else None}),flush=True)
            keep=any(r['effectiveness'] and r['effectiveness']['R_O'] is not None and r['effectiveness']['R_O']>=2/3 for r in results)
            incomplete=any(r['effectiveness'] is None or r['effectiveness']['status']=='incomplete' for r in results)
            report={'version':item['version'],'case_id':item['case_id'],'representative':item['representative'],'group':item['group'],'targets':results,'decision':'keep' if keep else 'deferred' if incomplete else 'discard'}
            reports.append(report);atomic_write_json(folder/'result.json',report)
        summary={'stage':'complete','groups':len(reports),'targets':total,'decisions':{v:{k:sum(r['version']==v and r['decision']==k for r in reports) for k in ['keep','discard','deferred']} for v in ['P0','P1']},'reruns':sum(len(t['effectiveness']['runs']) if t['effectiveness'] else 0 for r in reports for t in r['targets'])}
        atomic_write_json(out/'results.json',reports);atomic_write_json(out/'summary.json',summary);atomic_write_json(out/'progress.json',summary)
        text=[json.dumps(summary,ensure_ascii=False,indent=2)]
        for r in reports:
            text.append(f'\n{r["version"]} | {r["group"]["representative_id"]} | {r["decision"]}')
            for t in r['targets']:
                text.extend(['Target: '+t['node_id'],'Instruction: '+str(t['instruction']),json.dumps(t['effectiveness'],ensure_ascii=False,indent=2)])
        (out/'rerun_review.txt').write_text('\n\n'.join(text))
    finally:W.CTX['policy'].batcher.close()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        dest=ROOT/sys.argv[sys.argv.index('--out')+1];atomic_write_json(dest/'error.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
