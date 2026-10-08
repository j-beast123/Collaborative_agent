"""Blind pre-generation grouping of the 30 saved trajectories, then P0/P1 reps."""
import sys,json,os,traceback
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.online_rl.integrated.records import atomic_write_json,sha256_json,from_dict,to_dict
from src.online_rl.tree.records import TrajectoryRubricInput
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.posthoc import generate_rubric
from src.online_rl.tree.backends import api_rubric_generator
from scripts.online_rl.run_tree import load_key

SYSTEM='''You are the Rubric Generator performing PRE-GENERATION grouping within one case.
You have completed trajectories, success/failure labels, and Doctor feedback, but no pre-existing rubrics or gold answer. Feedback is fallible: prioritize actual dialogue evidence.
For EACH trajectory extract ONE supported collaborative success/failure reason and its evidence Agent node ID. Quote the Doctor request/context BEFORE that Agent response and the exact Agent response fragment. The node ID must lie on that trajectory's path. Do not use final correctness alone as behavioral evidence. If no supported collaborative reason exists, mark eligible=false, node/quotes=null, and explain why; do not invent a reason.
Group eligible trajectories only when their Doctor-side need, pre-response Agent situation and observable behavior are sufficiently equivalent to share the same rubric applicability and 0/1/2 behavior distinctions. Same topic or final answer is insufficient. Keep success and failure sources separate. If uncertain, use singleton groups. Choose an existing representative trajectory that best illustrates the group's criterion, not merely the first. Do not write rubrics yet. Retain each member's own evidence location.
Return ONLY JSON: {"members":[{"trajectory_id":"existing","eligible":true,"evidence_agent_turn_id":"existing-node","doctor_quote":"exact earlier quote","agent_quote":"exact target quote","reason":"supported behavior reason"}],"groups":[{"group_id":"g1","representative_trajectory_id":"existing","member_trajectory_ids":["existing"],"reason":"equivalence and representative rationale"}]}.
Every trajectory appears once in members. Every eligible trajectory appears once in groups, and no ineligible trajectory is grouped.'''


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    base=ROOT/'outputs/pilots/completed30_seed20261007';cfg=json.loads((ROOT/'configs/online_rl/tree_batch12_100cases_haiku_gemini_luna.json').read_text());model=cfg['pilot']['prompt_model'];key=load_key()
    os.environ['API_USAGE_LOG']=str(out/'api_usage.jsonl');cache=CallCache(out/'calls.jsonl')
    from src.prompt_opt import harness as H
    bycase=defaultdict(list);originals={}
    for f in sorted((base/'trajectories').glob('*.json')):
        tin=json.loads(f.read_text())['input'];originals[tin['leaf_id']]=tin
        nodes=iter(tin['agent_node_ids']);dialogue=[]
        for speaker,stage,turn,text in tin['dialogue']:
            row={'speaker':speaker,'stage':stage,'turn':turn,'text':text}
            if speaker=='agent':row['node_id']=next(nodes)
            dialogue.append(row)
        bycase[tin['case_id']].append({'trajectory_id':tin['leaf_id'],'source_outcome':tin['outcome'],'doctor_feedback':tin['feedback'],'dialogue':dialogue,'agent_pre_response_states':tin['agent_pre_response_states']})
    atomic_write_json(out/'manifest.json',{'model':model,'prompt':SYSTEM,'source_trajectories':len(originals),'existing_rubrics_shown':False,'reruns':False,'grouping_separates_source_outcomes':True})
    allgroups=[];members={}
    for cid,rows in sorted(bycase.items()):
        tin=originals[rows[0]['trajectory_id']];packet={'case_id':cid,'question':tin['question'],'options':tin['options'],'trajectories':rows};lookup={r['trajectory_id']:r for r in rows}
        def call():
            raw=H.doctor_call(key,model,SYSTEM,json.dumps(packet,ensure_ascii=False),16384,0)
            value=raw.strip()
            if value.startswith('```'):
                parts=value.splitlines()
                if parts[-1].strip()=='```':value='\n'.join(parts[1:-1])
            return {'raw':raw,'parsed':json.loads(value)}
        raw=cache.call('pre_generation_grouping',{'model':model,'system':sha256_json(SYSTEM),'input':sha256_json(packet)},call)
        value=raw['parsed'];m=value['members'];groups=value['groups']
        if len(m)!=len(rows) or {r['trajectory_id'] for r in m}!=set(lookup):raise ValueError('Incomplete/duplicate member extraction')
        eligible=set()
        for r in m:
            tid=r['trajectory_id'];d=lookup[tid]['dialogue']
            if type(r.get('eligible')) is not bool or not r.get('reason'):raise ValueError('Invalid evidence member')
            if r['eligible']:
                positions=[i for i,x in enumerate(d) if x.get('node_id')==r['evidence_agent_turn_id']]
                if len(positions)!=1:raise ValueError('Evidence node outside trajectory')
                i=positions[0]
                if not r.get('agent_quote') or r['agent_quote'] not in d[i]['text']:raise ValueError('Agent quote not exact')
                if not r.get('doctor_quote') or not any(x['speaker']=='doctor' and r['doctor_quote'] in x['text'] for x in d[:i]):raise ValueError('Doctor quote not exact before target')
                eligible.add(tid)
            members[tid]=r
        seen=set()
        for g in groups:
            ids=g['member_trajectory_ids'];rep=g['representative_trajectory_id']
            if not ids or rep not in ids or len(ids)!=len(set(ids)) or seen.intersection(ids) or not set(ids)<=eligible:raise ValueError('Invalid grouping membership')
            if len({lookup[t]['source_outcome'] for t in ids})!=1:raise ValueError('Mixed success/failure group')
            seen.update(ids);g['case_id']=cid;g['group_id']=cid+':'+g['group_id'];g['member_evidence']=[members[t] for t in ids];allgroups.append(g)
        if seen!=eligible:raise ValueError('Grouping omitted eligible trajectories')
        atomic_write_json(out/'groups'/(cid+'.json'),value)
        print(json.dumps({'case_id':cid,'trajectories':len(rows),'eligible':len(eligible),'groups':len(groups)}),flush=True)
    atomic_write_json(out/'grouping.json',{'groups':allgroups,'members':list(members.values())})
    # Compare old partitions only on shared source trajectory IDs, not denominators.
    comparisons={}
    for version in ['P0','P1']:
        oldmap={};oldcount=0
        for f in (base/'consolidation/P0_P1_v1'/version).glob('*.json'):
            x=json.loads(f.read_text());byid={r['rubric_id']:r for r in x['originals']}
            for g in x['groups']:
                oldcount+=1
                for rid in g['member_ids']:oldmap[byid[rid]['leaf_id']]=g['representative_id']
        newmap={tid:g['group_id'] for g in allgroups for tid in g['member_trajectory_ids']};common=sorted(set(oldmap)&set(newmap));pairs=[(a,b) for i,a in enumerate(common) for b in common[i+1:] if originals[a]['case_id']==originals[b]['case_id']]
        agreements=sum((oldmap[a]==oldmap[b])==(newmap[a]==newmap[b]) for a,b in pairs)
        comparisons[version]={'old_candidates':len(oldmap),'old_groups':oldcount,'shared_trajectories':len(common),'same_case_pairs':len(pairs),'pair_agreements':agreements,'pair_agreement_rate':agreements/len(pairs) if pairs else None,'note':'Old pipeline excludes parse/gate failures; new grouping uses all 30. Source-outcome separation is new.'}
        prompt=json.loads((base/'prompt_comparison'/f'{version}.txt.json').read_text());generate=api_rubric_generator(key,model,8192);valid=0
        for i,g in enumerate(allgroups):
            tid=g['representative_trajectory_id'];tin=from_dict(TrajectoryRubricInput,originals[tid])
            rubric=generate_rubric(prompt['text'],prompt['sha256'],tin,generate,cache)
            atomic_write_json(out/'rubrics'/version/(sha256_json(g['group_id'])+'.json'),{'group':g,'rubric':to_dict(rubric),'gate_checked':False})
            valid+=int(rubric.valid)
            atomic_write_json(out/'progress.json',{'stage':'representative_generation','version':version,'completed':i+1,'total':len(allgroups)})
        comparisons[version]['new_representative_generations']=len(allgroups);comparisons[version]['structurally_valid']=valid
    summary={'stage':'complete','trajectories':len(originals),'eligible':sum(r['eligible'] for r in members.values()),'groups':len(allgroups),'comparisons':comparisons,'reruns':False,'gate_checked':False}
    atomic_write_json(out/'summary.json',summary);atomic_write_json(out/'progress.json',summary)
    text=[json.dumps(summary,ensure_ascii=False,indent=2)]
    for g in allgroups:text.extend([g['group_id'],'Representative: '+g['representative_trajectory_id'],'Grouping reason: '+g['reason'],json.dumps(g['member_evidence'],ensure_ascii=False,indent=2)])
    (out/'grouping_review.txt').write_text('\n\n'.join(text))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        out=ROOT/sys.argv[sys.argv.index('--out')+1];atomic_write_json(out/'error.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
