"""After the 30-source P0 pilot, generate one case-local P1 and compare both."""
import argparse,json,os,sys,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.online_rl.run_completed30 import jsonable
from src.online_rl.integrated.records import from_dict,atomic_write_json,to_dict
from src.online_rl.tree.records import Leaf,AgentNode,TrajectoryRubricInput


def main(out):
    rec=json.loads((out/'selection.json').read_text())
    while not (out/'summary.json').exists():
        if (out/'error.json').exists():raise RuntimeError('P0 pilot failed; inspect error.json')
        pid=int((out/'runner.pid').read_text())
        status=Path('/proc')/str(pid)/'status'
        if not status.exists() or 'Z (zombie)' in status.read_text():raise RuntimeError('P0 process stopped before completion')
        time.sleep(5)
    pid=int((out/'runner.pid').read_text())
    while (Path('/proc')/str(pid)/'status').exists():
        if 'Z (zombie)' in (Path('/proc')/str(pid)/'status').read_text():break
        time.sleep(2)
    from scripts.online_rl.run_tree import load_key
    from src.online_rl.tree import batch_worker as W
    from scripts.online_rl.start_batch_doctors import start
    from src.online_rl.tree.prompt_search import EvidenceUnit,ScoringDeps,run_event,score_prompt,objective,unit_values
    from src.online_rl.integrated.prompt_search import PromptStore
    from src.online_rl.tree.effectiveness import select
    from src.online_rl.tree.scoring import score_nodes
    cfg=copy.deepcopy(rec['config']);cfg['prompt_update']['candidate_count']=1;cfg['prompt_update']['refinement_depth']=1
    key=load_key();os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl')
    start(cfg);W.initialize(cfg,0,key);W.load_checkpoint(rec['checkpoint'],rec['policy_sha256'])
    cases={};bycase={};units=[]
    for index,item in enumerate(rec['selected']):
        frozen=json.loads((out/'trajectories'/f'{index:03d}.json').read_text())
        case=item['case'];cid=case['case_id'];cases[cid]=case
        leaf=from_dict(Leaf,item['leaf']);nodes=tuple(from_dict(AgentNode,n) for n in item['nodes'])
        tin=from_dict(TrajectoryRubricInput,frozen['input']);unit=EvidenceUnit(cid,leaf,nodes,tin,rec['policy_sha256'])
        units.append(unit);bycase.setdefault(cid,[]).append(unit)
    eligible=[cid for cid,us in bycase.items() if all(sum(u.tin.outcome==label for u in us)>=2 for label in ['success','failure'])]
    if not eligible:raise RuntimeError('No case has both sources for case-local prompt optimization')
    cid=max(eligible,key=lambda cid:len(bycase[cid]));folder=out/'cases'/cid
    cache,doctor,iv,generate,scorer,gate,backend,strategies=W.components(cases[cid],folder)
    deps=ScoringDeps(3,cfg['schedule']['seed'],generate,gate,scorer,iv.write,iv.continue_,rec['policy_sha256'],strategies,cache)
    store=PromptStore(out/'prompt_comparison'/'prompts');text=(ROOT/cfg['rubric']['generator_prompt_file']).read_text();p0=store.put(text)
    atomic_write_json(out/'p1_progress.json',{'stage':'prompt_optimization','optimization_case':cid})
    event=run_event(1,'P0',p0,bycase[cid],{cid:cases[cid]},store,backend,deps,cfg,cfg['schedule']['seed'],0,[])
    atomic_write_json(out/'prompt_comparison'/'prompt_event.json',jsonable(event))
    candidates=[v for v in event['versions'] if v.sha256 and v.sha256!=p0 and (store.dir/(v.sha256+'.txt')).exists()]
    if not candidates:
        atomic_write_json(out/'p1_progress.json',{'stage':'no_valid_P1','decision':event['decision']});return
    p1=candidates[0].sha256
    atomic_write_json(out/'prompt_comparison'/'P0.txt.json',{'sha256':p0,'text':text})
    atomic_write_json(out/'prompt_comparison'/'P1.txt.json',{'sha256':p1,'text':store.get(p1)})
    comparison=[]
    for label,sha in [('P0',p0),('P1',p1)]:
        values={};outcomes={};retained=0;pairs=0
        for cid,us in bycase.items():
            cache,doctor,iv,generate,scorer,gate,backend,strategies=W.components(cases[cid],out/'cases'/cid)
            deps=ScoringDeps(3,cfg['schedule']['seed'],generate,gate,scorer,iv.write,iv.continue_,rec['policy_sha256'],strategies,cache)
            res=score_prompt(store.get(sha),sha,us,cases,deps)
            values.update(unit_values(res,strategies));outcomes.update({k:r['source_outcome'] for k,r in res.items()})
            effects=[r['effectiveness'] for r in res.values() if r['effectiveness'] is not None]
            decisions=select(effects,cfg['selection']);ids={d.rubric_id for d in decisions if d.retained};retained+=len(ids)
            nodes={n.node_id:n for u in us for n in u.path_nodes}
            scores=score_nodes(cases[cid],list(nodes.values()),[r['rubric'] for r in res.values() if r['rubric'].rubric_id in ids],scorer,cache);pairs+=len(scores)
            atomic_write_json(out/'prompt_comparison'/label/f'{cid}.json',{'results':jsonable(res),'selection':jsonable(decisions),'response_scores':jsonable(scores)})
            atomic_write_json(out/'p1_progress.json',{'stage':'comparing','prompt':label,'case_id':cid,'evaluated_trajectories':len(outcomes),'total':30})
            print(json.dumps({'stage':'comparison','prompt':label,'case':cid,'evaluated':len(outcomes)}),flush=True)
        comparison.append({'prompt_id':label,'prompt_sha256':sha,'retained':retained,'response_rubric_pairs':pairs,**objective(values,outcomes)})
    summary={'stage':'complete','prompt_scores':comparison,'acceptance':event['decision'],'optimization_case':eligible,'note':'Case-local search/acceptance determines adoption. Full 30-trajectory comparison is diagnostic, not untouched holdout. No Agent update.'}
    atomic_write_json(out/'prompt_comparison'/'summary.json',summary);atomic_write_json(out/'p1_progress.json',summary)
    W.CTX['policy'].batcher.close();print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/pilots/completed30_seed20261007');a=p.parse_args();out=(ROOT/a.out).resolve()
    try:main(out)
    except Exception as e:atomic_write_json(out/'p1_error.json',{'type':type(e).__name__,'error':str(e)[:1000]});raise
