"""Independent P0/P1 success/failure reuse banks on seven saved trajectories."""
import json,sys,os,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import from_dict,to_dict,atomic_write_json,sha256_json
from src.online_rl.tree.records import TrajectoryRubricInput
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.reuse_bank import packet,validate_reuse,PROMPT,PROTOCOL
from src.online_rl.tree.posthoc import generate_rubric
from src.online_rl.tree.backends import api_rubric_generator
from scripts.online_rl.run_tree import load_key


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    base=ROOT/'outputs/pilots/completed30_seed20261007';cid='medmcqa_e70be528-ae55-4317-ba07-e54dbdbdd6fe'
    inputs=[from_dict(TrajectoryRubricInput,json.loads(f.read_text())['input']) for f in sorted((base/'trajectories').glob('*.json')) if json.loads(f.read_text())['input']['case_id']==cid]
    cfg=json.loads((ROOT/'configs/online_rl/tree_batch12_100cases_haiku_gemini_luna.json').read_text());model=cfg['pilot']['prompt_model'];key=load_key();system=PROMPT.read_text();cache=CallCache(out/'calls.jsonl')
    os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl')
    from src.prompt_opt import harness as H
    generate=api_rubric_generator(key,model,8192);summaries={};allrows=[]
    atomic_write_json(out/'manifest.json',{'case_id':cid,'model':model,'protocol':PROTOCOL,'prompt':system,'trajectory_order':[t.leaf_id for t in inputs],'separate_banks':['prompt_version','case_id','source_outcome'],'reruns':False,'gate_checked':False,'candidate_limit':None})
    for version in ['P0','P1']:
        banks={'success':[],'failure':[]};links={'success':{},'failure':{}};counts={'generated':0,'reused':0,'ineligible':0,'invalid_generated':0}
        prompt=json.loads((base/'prompt_comparison'/f'{version}.txt.json').read_text())
        for index,tin in enumerate(inputs):
            bank=banks[tin.outcome];state=packet(tin,bank);inspection=None
            if bank:
                def inspect():
                    msg=json.dumps(state,ensure_ascii=False)
                    for attempt in range(2):
                        raw=H.doctor_call(key,model,system,msg,4096,0)
                        try:
                            text=raw.strip()
                            if text.startswith('```'):
                                lines=text.splitlines()
                                if lines[-1].strip()=='```':text='\n'.join(lines[1:-1])
                            return {'raw':raw,'result':validate_reuse(json.loads(text),state)}
                        except (ValueError,KeyError,TypeError):
                            if attempt:raise
                            msg=json.dumps(state,ensure_ascii=False)+'\nReturn the exact schema and exact quotations. Previous output:\n'+raw
                inspected=cache.call('rubric_reuse_inspection',{'protocol':PROTOCOL,'version':version,'prompt_sha256':prompt['sha256'],'inspection_prompt':sha256_json(system),'state':sha256_json(state),'model':model},inspect)
                inspection=inspected['result']
            if inspection and not inspection['eligible']:
                counts['ineligible']+=1;row={'trajectory_id':tin.leaf_id,'source_outcome':tin.outcome,'action':'ineligible','inspection':inspection}
            elif inspection and inspection['reuse']:
                counts['reused']+=1;rid=inspection['rubric_id'];link={'trajectory_id':tin.leaf_id,'evidence_agent_turn_id':inspection['agent_turn_id'],'doctor_quote':inspection['doctor_quote'],'agent_quote':inspection['agent_quote'],'reason':inspection['reason'],'effectiveness_verified':False}
                links[tin.outcome][rid].append(link);row={'trajectory_id':tin.leaf_id,'source_outcome':tin.outcome,'action':'reuse','rubric_id':rid,'inspection':inspection}
            else:
                r=generate_rubric(prompt['text'],prompt['sha256'],tin,generate,cache);counts['generated']+=1
                row={'trajectory_id':tin.leaf_id,'source_outcome':tin.outcome,'action':'generate','inspection':inspection,'rubric':to_dict(r),'gate_checked':False}
                if r.gate and r.gate.get('eligible') is False:
                    counts['ineligible']+=1
                    row['action']='ineligible'
                    row['eligibility_reason']=r.gate['reason']
                elif r.valid:
                    bank.append(r);nodes=state['full_dialogue'];i=next(i for i,d in enumerate(nodes) if d.get('node_id')==r.selected_agent_turn_id)
                    preceding=next(d for d in reversed(nodes[:i]) if d['speaker']=='doctor')
                    links[tin.outcome][r.rubric_id]=[{'trajectory_id':tin.leaf_id,'selected_agent_turn_id':r.selected_agent_turn_id,'evidence_agent_turn_id':r.selected_agent_turn_id,'doctor_quote':preceding['text'],'agent_quote':nodes[i]['text'],'reason':'Generator-selected source turn; full source text retained','effectiveness_verified':False}]
                else:counts['invalid_generated']+=1
            row['version']=version;allrows.append(row);atomic_write_json(out/version/'steps'/f'{index:02d}.json',row)
            atomic_write_json(out/'progress.json',{'stage':'reuse_and_generation','version':version,'completed':index+1,'total':len(inputs),'counts':counts});print(json.dumps({'version':version,'index':index,'action':row['action']}),flush=True)
        atomic_write_json(out/version/'banks.json',{'banks':{k:[to_dict(r) for r in v] for k,v in banks.items()},'member_evidence':links})
        summaries[version]={**counts,'success_bank_size':len(banks['success']),'failure_bank_size':len(banks['failure']),'generation_reduction':1-counts['generated']/len(inputs),'baseline_generations':len(inputs)}
    summary={'stage':'complete','case_id':cid,'trajectories':len(inputs),'prompts':summaries,'note':'No gate, reruns or retention decisions. Structural validity and reuse claims are not efficacy verification; independent source behavior review remains necessary.'}
    atomic_write_json(out/'summary.json',summary);atomic_write_json(out/'progress.json',summary);atomic_write_json(out/'results.json',allrows)
    (out/'reuse_review.txt').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'+'\n\n'.join(json.dumps(r,ensure_ascii=False,indent=2) for r in allrows))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        out=ROOT/sys.argv[sys.argv.index('--out')+1];atomic_write_json(out/'error.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
