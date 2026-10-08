"""Evaluate 30 completed cached branches without resuming rollout or training."""
import argparse,json,os,random,sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import atomic_write_json,to_dict,from_dict
from src.online_rl.tree.records import AgentNode,Leaf,to_turns

def jsonable(value):
    import dataclasses
    if dataclasses.is_dataclass(value):return to_dict(value)
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value



def prepare(source,out,count):
    from src.online_rl.cases import load_cases
    from src.online_rl.environment import final_outcome
    from src.online_rl.tree.backends import parse_final_option
    cfg=json.loads((source/'run_manifest.json').read_text())['config']
    cases={c['case_id']:c for c in load_cases(ROOT/cfg['data']['train'])}
    manifest=json.loads((source/'batches/batch_000001/manifest.json').read_text())
    groups={0:[],1:[]}
    for f in sorted((source/'batches/batch_000001/cases').glob('case_*/calls.jsonl')):
        records={}
        for line in f.open():
            try:r=json.loads(line)
            except ValueError:continue
            kind=r.get('kind');ident=r.get('identity',{})
            if kind in ('agent','doctor_turn','final'):records[(kind,ident['path'])]=r
        for (kind,path),r in records.items():
            if kind!='final':continue
            cid=r['identity']['case'];case=cases[cid]
            opt=parse_final_option(r['value'],dict(case['options'])) or ''
            outcome=final_outcome(case,opt)
            if outcome is None:continue
            parts=path.split('.');dialogue=[];nodes=[]
            for depth in range(1,len(parts)+1):
                parent='.'.join(parts[:depth-1]);p='.'.join(parts[:depth]);nid=f'{cid}/{p}'
                d=records[('doctor_turn',parent)]['value'];a=records[('agent',p)]
                dialogue.append({'speaker':'doctor','stage':f'D{depth}','round':depth,'text':d})
                trace=a['value'];nodes.append(AgentNode(nid,f'{cid}/{parent}' if parent else None,cid,depth,to_turns(dialogue),trace['completion_text'],trace,r['identity']['policy'],a['identity']['seed'],None,depth==len(parts),None))
                dialogue.append({'speaker':'agent','stage':f'A{depth}','round':depth,'text':trace['completion_text']})
            dialogue.append({'speaker':'doctor','stage':'FINAL','round':0,'text':r['value']})
            leaf=Leaf(nodes[-1].node_id,cid,tuple(n.node_id for n in nodes),to_turns(dialogue),r['value'],opt,outcome,'cached_terminal')
            groups[int(outcome)].append({'case':case,'leaf':to_dict(leaf),'nodes':[to_dict(n) for n in nodes],'source_cache':str(f),'source_protocol':r['identity']['protocol']})
    rng=random.Random(20261007)
    for g in groups.values():rng.shuffle(g)
    # Equal source quotas where available; fill the remainder from either group.
    picked=groups[0][:count//2]+groups[1][:count//2]
    chosen={r['leaf']['leaf_id'] for r in picked}
    extra=[r for g in groups.values() for r in g if r['leaf']['leaf_id'] not in chosen]
    picked+=extra[:count-len(picked)]
    if len(picked)!=count:raise ValueError('Not enough completed parseable branches')
    out.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(cfg);cfg['output_dir']=str(out.relative_to(ROOT));cfg['batch']['doctor_urls']=['http://127.0.0.1:11545'];cfg['batch']['doctor_parallelism']=1
    record={'source_run':str(source),'checkpoint':manifest['checkpoint'],'policy_sha256':manifest['policy_sha256'],'available':{str(k):len(v) for k,v in groups.items()},'selected':picked,'config':cfg,'agent_updates':False,'prompt_optimization':False}
    atomic_write_json(out/'selection.json',record)
    return record


def run(out):
    from scripts.online_rl.run_tree import load_key
    from scripts.online_rl.start_batch_doctors import start
    from src.online_rl.tree import batch_worker as W
    from src.online_rl.tree.posthoc import doctor_feedback,freeze_trajectory_input
    from src.online_rl.tree.backends import api_feedback
    from src.online_rl.tree.prompt_search import EvidenceUnit,ScoringDeps,score_prompt,evidence_id,unit_values,objective
    from src.online_rl.tree.effectiveness import select
    from src.online_rl.tree.scoring import score_nodes
    rec=json.loads((out/'selection.json').read_text());cfg=rec['config'];key=load_key()
    os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl')
    start(cfg);W.initialize(cfg,0,key);W.load_checkpoint(rec['checkpoint'],rec['policy_sha256'])
    ptext=(ROOT/cfg['rubric']['generator_prompt_file']).read_text();psha=cfg['rubric']['prompt_sha256']
    import hashlib
    if hashlib.sha256(ptext.encode()).hexdigest()!=psha:raise ValueError('Pinned generator prompt changed')
    atomic_write_json(out/'prompt.json',{'prompt_sha256':psha,'text':ptext})
    bycase={};results=[];values={};outcomes={}
    for index,item in enumerate(rec['selected']):
        case=item['case'];leaf=from_dict(Leaf,item['leaf']);nodes=[from_dict(AgentNode,n) for n in item['nodes']]
        folder=out/'cases'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
        cache,doctor,iv,generate,scorer,gate,backend,strategies=W.components(case,folder)
        fbfn=api_feedback(key,cfg['rollout']['doctor_model'],(ROOT/cfg['feedback']['prompt_file']).read_text(),cfg['feedback']['max_tokens'],call=doctor.call)
        fb=doctor_feedback(case,leaf,fbfn,cfg['feedback']['prompt_sha256'],cache)
        tin=freeze_trajectory_input(case,leaf,fb,{n.node_id:n for n in nodes})
        unit=EvidenceUnit(case['case_id'],leaf,tuple(nodes),tin,rec['policy_sha256'])
        deps=ScoringDeps(3,cfg['schedule']['seed'],generate,gate,scorer,iv.write,iv.continue_,rec['policy_sha256'],strategies,cache)
        scored=score_prompt(ptext,psha,[unit],{case['case_id']:case},deps)
        value=scored[evidence_id(unit)];results.append(value)
        values.update(unit_values(scored,strategies));outcomes[evidence_id(unit)]=tin.outcome
        bundle=bycase.setdefault(case['case_id'],{'case':case,'nodes':{},'rubrics':[],'effects':[]})
        bundle['nodes'].update({n.node_id:n for n in nodes});bundle['rubrics'].append(value['rubric'])
        if value['effectiveness'] is not None:bundle['effects'].append(value['effectiveness'])
        atomic_write_json(out/'trajectories'/f'{index:03d}.json',{'source':item,'doctor_feedback':to_dict(fb),'input':to_dict(tin),'evaluation':jsonable(value)})
        atomic_write_json(out/'progress.json',{'stage':'rubric_reruns','completed':index+1,'total':len(rec['selected'])})
        print(json.dumps({'stage':'rubric_reruns','completed':index+1,'total':len(rec['selected'])}),flush=True)
    scores=[];selections=[]
    for cid,b in bycase.items():
        decisions=select(b['effects'],cfg['selection']);selections+=decisions
        ids={d.rubric_id for d in decisions if d.retained}
        cache,doctor,iv,generate,scorer,*_=W.components(b['case'],out/'cases'/cid)
        scores+=score_nodes(b['case'],list(b['nodes'].values()),[r for r in b['rubrics'] if r.rubric_id in ids],scorer,cache)
        atomic_write_json(out/'scores'/f'{cid}.json',{'selection':jsonable(decisions),'response_scores':jsonable(scores_by_case(scores,cid,b))})
    atomic_write_json(out/'rubrics.json',[to_dict(r['rubric']) for r in results])
    atomic_write_json(out/'response_scores.json',[to_dict(s) for s in scores])
    summary={'stage':'complete','trajectories':len(results),'valid_rubrics':sum(r['rubric'].valid for r in results),'retained_rubrics':sum(s.retained for s in selections),'response_rubric_pairs':len(scores),'objective':objective(values,outcomes),'agent_updates':False,'prompt_optimization':False}
    atomic_write_json(out/'summary.json',summary);atomic_write_json(out/'progress.json',summary);print(json.dumps(summary),flush=True)
    W.CTX['policy'].batcher.close()


def scores_by_case(scores,cid,b):
    return [s for s in scores if s.response_node_id in b['nodes']]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='outputs/latest_training');p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=30);p.add_argument('--run',action='store_true');a=p.parse_args();out=(ROOT/a.out).resolve()
    if not (out/'selection.json').exists():
        r=prepare((ROOT/a.source).resolve(),out,a.count)
        print(json.dumps({'selected':len(r['selected']),'available':r['available'],'selected_success':sum(x['leaf']['outcome']==1 for x in r['selected'])}),flush=True)
    if a.run:
        try:run(out)
        except Exception as e:
            atomic_write_json(out/'error.json',{'type':type(e).__name__,'error':str(e)[:1000]});raise
