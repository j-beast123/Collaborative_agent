"""Score unique original responses with rerun-retained sequential reuse rubrics."""
import sys,json,os,csv,collections,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import atomic_write_json,sha256_json

def main(out):
 from scripts.online_rl.run_tree import load_key
 from src.judges.openrouter_choice import OpenRouterChoiceJudge
 from src.judges.applicability_review import ApplicabilityReviewer
 from src.judges.rubric_scoring import behavior_questions
 from src.online_rl.tree.backends import visible_behavior_rubric
 from src.online_rl.tree.cache import CallCache
 base=ROOT/'outputs/pilots/completed30_seed20261007';out.mkdir(parents=True,exist_ok=True)
 bank=json.loads((base/'reuse_bank/single_case_P0_P1_v1/P0/banks.json').read_text());rubrics=[]
 for rs in bank['banks'].values():
  for r in rs:
   folder=base/'reuse_bank/reruns_v1/P0'/(r['rubric_id']+'-member0')
   effects=[json.loads(f.read_text()).get('effectiveness') for f in (folder/'targets').glob('*.json')]
   if any(e and e.get('R_O') is not None and e['R_O']>=2/3 for e in effects):rubrics.append(r['rubric'])
 if len(rubrics)!=2:raise RuntimeError('Expected two completed retained P0 rubrics')
 cid=rubrics[0]['source_case_id'];nodes={};case=None
 selection=json.loads((base/'selection.json').read_text())
 for s in selection['selected']:
  if s['case']['case_id']==cid:case=s['case'];nodes.update({n['node_id']:n for n in s['nodes']})
 client=OpenRouterChoiceJudge(load_key(),model='openai/gpt-6-luna',max_tokens=4096,batch_size=8,audit_path=out/'scorer_calls_raw.jsonl')
 reviewer=ApplicabilityReviewer(client,out/'reviewer_calls_raw.jsonl');cache=CallCache(out/'calls.jsonl');rows=[];matches={}
 os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl')
 atomic_write_json(out/'rubrics.json',rubrics);atomic_write_json(out/'manifest.json',{'case_id':cid,'version':'P0','rubrics':len(rubrics),'unique_responses':len(nodes),'model':client.model,'agent_update':False,'scope':'All unique Agent nodes in seven saved source trajectories; independent prefix applicability review before scoring'})
 atomic_write_json(out/'prompt_snapshot.json',{'reviewer_system':reviewer.system,'scorer_identity':client.identity(),'behavior_questions':[behavior_questions([visible_behavior_rubric(r)]) for r in rubrics]})
 for n in sorted(nodes.values(),key=lambda n:n['node_id']):
  prefix=[{'speaker':d[0],'stage':d[1],'round':d[2],'text':d[3]} for d in n['prefix']];packet=n.get('trace',{}).get('pre_response_belief');recorded=packet.get('state') if packet else None
  for r in rubrics:
   candidate=r['candidate_rubric'];visible=visible_behavior_rubric(r);identity={'prefix_sha256':sha256_json(prefix),'conditions':candidate['applicability'],'recorded_state':recorded,'review_prompt_sha256':sha256_json(reviewer.system),'model':client.model}
   review=cache.call('independent_match',identity,lambda:reviewer.review(prefix,candidate['applicability'],recorded));label=None
   if review['applicable']:
    state={'dialogue_before_target_turn':prefix,'target_turn':{'speaker':'agent','text':n['response']},'case_context':{'question':case['question'],'options':case['options']}}
    if recorded is not None:state['agent_pre_response_state']=recorded
    q=behavior_questions([visible]);answer=cache.call('reviewed_behavior_score',{**identity,'reply_sha256':sha256_json(n['response']),'questions':q},lambda:client.decide(json.dumps(state,ensure_ascii=False),q,purpose='reviewed_response_behavior')[0]);label=int(answer[visible['rubric_id']]['value'])
   row={'node_id':n['node_id'],'agent_turn':n['depth'],'rubric_id':visible['rubric_id'],'rubric_name':candidate['name'],'applicable':review['applicable'],'score':label,'normalized_reward':label/2 if label is not None else None,'review':review,'prefix':prefix,'target_response':n['response']};rows.append(row);atomic_write_json(out/'rows'/f'{len(rows):03d}.json',row);atomic_write_json(out/'progress.json',{'completed_pairs':len(rows),'total_pairs':len(nodes)*len(rubrics)});print(json.dumps({k:row[k] for k in ['node_id','rubric_id','applicable','score']}),flush=True)
 atomic_write_json(out/'results.json',rows);summary={'stage':'complete','unique_responses':len(nodes),'rubrics':len(rubrics),'evaluated_pairs':len(rows),'score_counts':dict(collections.Counter(str(r['score']) if r['applicable'] else 'NA' for r in rows))};atomic_write_json(out/'summary.json',summary)
 fields=['node_id','agent_turn','rubric_id','rubric_name','applicable','score','normalized_reward','target_response']
 with (out/'scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 (out/'scoring_review.txt').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'+'\n\n'.join(json.dumps(r,ensure_ascii=False,indent=2) for r in rows))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=ROOT/a.out
 try:main(out)
 except Exception as e:atomic_write_json(out/'error.json',{'error':str(e),'traceback':traceback.format_exc()});raise
