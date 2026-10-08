"""Independent applicability review then Luna hard scoring of 22 saved replies."""
import sys,json,os,collections,csv,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import atomic_write_json


def main(output_dir=None):
    from scripts.online_rl.run_tree import load_key
    from src.judges.openrouter_choice import OpenRouterChoiceJudge
    from src.judges.applicability_review import ApplicabilityReviewer
    from src.judges.rubric_scoring import behavior_questions
    from src.online_rl.tree.backends import visible_behavior_rubric
    from src.online_rl.tree.cache import CallCache
    from src.online_rl.integrated.records import sha256_json
    base=ROOT/'outputs/pilots/completed30_seed20261007';out=(ROOT/output_dir).resolve() if output_dir else base/'rescoring/independent_applicability_v1';out.mkdir(parents=True,exist_ok=True)
    cid='medmcqa_e70be528-ae55-4317-ba07-e54dbdbdd6fe';src=json.loads((base/'prompt_comparison/P1'/f'{cid}.json').read_text())
    kept={d['rubric_id'] for d in src['selection'] if d['retained']};envelope=next(v['rubric']['rubric'] for v in src['results'].values() if v['rubric']['rubric_id'] in kept)
    candidate=envelope['candidate_rubric'];visible=visible_behavior_rubric(envelope)
    nodes={};case=None
    for f in (base/'trajectories').glob('*.json'):
        r=json.loads(f.read_text())
        if r['source']['case']['case_id']==cid:
            case=r['source']['case'];nodes.update({n['node_id']:n for n in r['source']['nodes']})
    client=OpenRouterChoiceJudge(load_key(),model='openai/gpt-6-luna',max_tokens=4096,batch_size=1,audit_path=out/'scorer_calls_raw.jsonl')
    reviewer=ApplicabilityReviewer(client,out/'reviewer_calls_raw.jsonl');cache=CallCache(out/'calls.jsonl');rows=[]
    atomic_write_json(out/'prompt_snapshot.json',{'reviewer_system':reviewer.system,'scorer_identity':client.identity(),'behavior_questions':behavior_questions([visible]),'source':str(base/'prompt_comparison/P1'/f'{cid}.json')})
    os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl')
    atomic_write_json(out/'rubric.json',envelope)
    for old in src['response_scores']:
        n=nodes[old['response_node_id']];prefix=[{'speaker':d[0],'stage':d[1],'round':d[2],'text':d[3]} for d in n['prefix']];packet=n['trace'].get('pre_response_belief');recorded=packet.get('state') if packet else None
        identity={'node':n['node_id'],'prefix_sha256':sha256_json(prefix),'conditions':candidate['applicability'],'recorded_state':recorded,'review_prompt_sha256':sha256_json(reviewer.system),'model':client.model}
        review=cache.call('independent_match',identity,lambda:reviewer.review(prefix,candidate['applicability'],recorded))
        label=None
        if review['applicable']:
            state={'dialogue_before_target_turn':prefix,'target_turn':{'speaker':'agent','text':n['response']},'case_context':{'question':case['question'],'options':case['options']}}
            if recorded is not None:state['agent_pre_response_state']=recorded
            q=behavior_questions([visible]);judgeid={**identity,'reply_sha256':sha256_json(n['response']),'questions':q}
            answer=cache.call('reviewed_behavior_score',judgeid,lambda:client.decide(json.dumps(state,ensure_ascii=False),q,purpose='reviewed_response_behavior')[0])
            label=int(answer[visible['rubric_id']]['value'])
        row={'node_id':n['node_id'],'rubric_id':visible['rubric_id'],'prefix':prefix,'target_response':n['response'],'review':review,
             'old_applicable':old['match'],'old_score_0_to_2':old['hard']*2 if old['hard'] is not None else None,
             'new_applicable':review['applicable'],'new_score_0_to_2':label,'new_normalized_reward':label/2 if label is not None else None}
        rows.append(row);atomic_write_json(out/'rows'/f'{len(rows):02d}.json',row)
        atomic_write_json(out/'progress.json',{'completed':len(rows),'total':len(src['response_scores'])});print(json.dumps({'completed':len(rows),'applicable':review['applicable'],'score':label}),flush=True)
    atomic_write_json(out/'results.json',rows)
    summary={'completed':len(rows),'total':len(src['response_scores']),'old_matches':sum(r['old_applicable'] for r in rows),'new_matches':sum(r['new_applicable'] for r in rows),
             'removed_matches':sum(r['old_applicable'] and not r['new_applicable'] for r in rows),'score_counts':dict(collections.Counter(str(r['new_score_0_to_2']) if r['new_applicable'] else 'NA' for r in rows)),
             'note':'Independent review experiment only; original rewards/selection/training unchanged. Human verification is needed to establish whether changed matching is more accurate.'}
    atomic_write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='outputs/pilots/completed30_seed20261007/rescoring/independent_applicability_v1');args=parser.parse_args()
    try:main(args.out)
    except Exception as e:atomic_write_json((ROOT/args.out).resolve()/'error.json',{'error':str(e)});raise
