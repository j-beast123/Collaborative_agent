"""Export committed scoring rows next to rubric anchors and the exact target context."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import zipfile
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def display_score(record):
    if not record['match']:return 'No Match'
    probabilities=record['probabilities']
    return int(max(probabilities,key=probabilities.get))


def history_key(dialogue):
    return tuple((d.get('speaker'),d.get('text')) for d in dialogue)


def export(run,out):
    out.mkdir(parents=True,exist_ok=True)
    audit=[]
    for line in (run/'judge_calls_raw.jsonl').read_text().splitlines():
        try:audit.append(json.loads(line))
        except ValueError:pass
    index={}
    for i,a in enumerate(audit):
        if a.get('error') or 'rubric_score_' not in a.get('purpose',''):continue
        state=a['state'];kind='behavior' if 'target_turn' in state else 'applicability'
        target=state.get('target_turn',{}).get('text','')
        for rid in a['questions']:
            index[(kind,rid,history_key(state.get('dialogue_before_target_turn',[])),target)]=i
    reviews=[];rubricsets=[];issues=[]
    for path in sorted((run/'cases').glob('case_*.json')):
        case=json.loads(path.read_text());nodes={n['node_id']:n for n in case['tree']['nodes']}
        rubrics={r['rubric_id']:r for r in case['trajectory_rubrics']}
        tins={r['leaf_id']:r for r in case['trajectory_inputs']}
        effects={r['rubric_id']:r for r in case['effectiveness']}
        kept={r['rubric_id'] for r in case['selection'] if r['retained']}
        for rid,r in rubrics.items():
            rubricsets.append({'case_id':case['case_id'],'rubric_id':rid,'retained':rid in kept,
                'rubric':r['rubric'],'selected_agent_turn_id':r['selected_agent_turn_id'],
                'source_trajectory':tins[r['leaf_id']], 'effectiveness':effects.get(rid)})
        for stage,key in [('retained_rubric_response_scoring','response_scores'),('source_turn_validation','source_path_scores')]:
            for score in case[key]:
                rid=score['rubric_id'];rub=rubrics[rid];node=nodes[score['response_node_id']]
                candidate=rub['rubric'].get('candidate_rubric',rub['rubric'])
                prefix=[{'speaker':d[0],'stage':d[1],'round':d[2],'text':d[3]} for d in node['prefix']]
                tin=tins[rub['leaf_id']];packet=node['trace'].get('pre_response_belief') or {}
                ids={kind:index.get((kind,rid,history_key(prefix),node['response'] if kind=='behavior' else '')) for kind in ['applicability','behavior']}
                outputs={}
                for kind,i in ids.items():
                    if i is None:outputs[kind]=None;continue
                    msg=audit[i]['raw_response']['choices'][0]['message']
                    outputs[kind]={'content':msg.get('content'),'reasoning':msg.get('reasoning'),'audit_line':i+1}
                level=display_score(score)
                if not score['match'] and any(score.get(k) is not None for k in ['score','hard','probabilities']):issues.append(f'{rid}/{node["node_id"]}: No Match has a score')
                if score['match'] and abs(score['score']-level/2)>1e-8:issues.append(f'{rid}/{node["node_id"]}: normalized score disagrees with categorical level')
                row={'stage':stage,'case_id':case['case_id'],'node_id':node['node_id'],'agent_turn':node['depth'],
                     'rubric_id':rid,'rubric_name':candidate['name'],'retained':rid in kept,
                     'applicable':score['match'],'score_0_2':level,'score_normalized_0_1':score['score'],
                     'user_side_context':candidate['applicability']['user_side_context'],
                     'agent_side_state_condition':candidate['applicability']['agent_side_state'],
                     **{f'behavior_{n}':candidate['scores'][n]['behavior'] for n in ['0','1','2']},
                     'question':tin['question'],'options':dict(tin['options']),
                     'doctor_latest':next((d['text'] for d in reversed(prefix) if d['speaker']=='doctor'),''),
                     'agent_pre_response_state':packet.get('state'), 'history_before_response':prefix,
                     'agent_response':node['response'],'source_outcome':tin['outcome'],
                     'selected_agent_turn_id':rub['selected_agent_turn_id'],
                     'source_doctor_feedback':tin['feedback'],'judge_model':'google/gemini-3.8-flash',
                     'judge_outputs':outputs,'human_match':'','human_score':'','human_note':''}
                reviews.append(row)
        shutil.copy2(path,out/path.name)
    (out/'scoring_review.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2)+'\n')
    (out/'rubrics_with_sources_and_reruns.json').write_text(json.dumps(rubricsets,ensure_ascii=False,indent=2)+'\n')
    with (out/'scoring_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(reviews[0]) if reviews else []
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in reviews:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    summary={'snapshot_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'source_run':str(run),'committed_cases':len(list((run/'cases').glob('case_*.json'))),
             'judge':'google/gemini-3.8-flash','scoring_unit':'one response × one rubric',
             'distribution':{stage:dict(Counter(str(r['score_0_2']) for r in reviews if r['stage']==stage)) for stage in ['retained_rubric_response_scoring','source_turn_validation']},
             'structural_issues':issues,'note':'No Match is not score zero. display score is 0/1/2; internal score is normalized 0..1. Source outcome/feedback is reviewer context, excluded from scoring inputs. Judge reasons are shown only when returned; missing reasoning is not invented.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(run/'judge_calls_raw.jsonl',out/'judge_calls_raw.jsonl')
    html=HTML.replace('__DATA__',json.dumps(reviews,ensure_ascii=False).replace('<','\\u003c')).replace('__SUMMARY__',json.dumps(summary,ensure_ascii=False).replace('<','\\u003c'))
    (out/'review.html').write_text(html)
    bundle=out.with_suffix('.zip')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.is_file():z.write(f,f.name)
    print(json.dumps({'directory':str(out),'zip':str(bundle),'summary':summary},ensure_ascii=False,indent=2))


HTML='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>루브릭 채점 검토</title>
<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:24px auto;padding:0 20px;background:#f4f6f8;color:#16202b}header{position:sticky;top:0;background:#f4f6f8;padding:12px 0}select,input{padding:8px;margin:4px;border:1px solid #bbb;border-radius:6px}article{background:white;padding:20px;margin:16px 0;border:1px solid #ccd3db;border-radius:10px}pre{white-space:pre-wrap;word-break:break-word;font:14px/1.6 system-ui}h3{margin-top:8px}.score{font-weight:bold;color:#075a9c}details{margin:12px 0}label{display:block;font-weight:600}.anchors{background:#edf3f9;padding:12px}button{padding:8px}</style>
<header><h2>루브릭과 Agent 응답 채점 검토</h2><div id="summary"></div><select id="stage"><option value="retained_rubric_response_scoring">유지 루브릭 → 응답 채점</option><option value="source_turn_validation">원본 선택 턴 검증</option></select><select id="score"><option value="all">모든 점수</option><option>No Match</option><option>0</option><option>1</option><option>2</option></select><select id="rubric"><option value="all">모든 루브릭</option></select><input id="query" placeholder="응답·조건·루브릭 검색"><span id="count"></span></header><main id="rows"></main>
<script>const data=__DATA__,summary=__SUMMARY__;const $=x=>document.getElementById(x);const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('summary').textContent=`${summary.snapshot_kst} | ${summary.committed_cases}문제 | ${summary.judge} | No Match는 점수 없음. 내부 정규화 값과 0/1/2점을 구분하세요.`;
[...new Map(data.map(r=>[r.rubric_id,r.rubric_name])).entries()].forEach(([id,name])=>{const o=document.createElement('option');o.value=id;o.textContent=name+' · '+id;$('rubric').append(o)});
const section=(title,text)=>`<label>${esc(title)}</label><pre>${esc(typeof text==='string'?text:JSON.stringify(text,null,2))}</pre>`;
function render(){const q=$('query').value.toLowerCase();const selected=data.filter(r=>r.stage===$('stage').value&&($('score').value==='all'||String(r.score_0_2)===$('score').value)&&($('rubric').value==='all'||r.rubric_id===$('rubric').value)&&(!q||JSON.stringify(r).toLowerCase().includes(q)));$('count').textContent=selected.length+'건';$('rows').innerHTML=selected.map(r=>`<article><h3>${esc(r.rubric_name)} <span class="score">${esc(r.score_0_2)}${r.score_0_2==='No Match'?'':'점'}</span></h3><small>${esc(r.node_id)} · Agent ${r.agent_turn}턴 · ${esc(r.rubric_id)}</small><details><summary>문제와 보기</summary>${section('문제',r.question)}${section('보기',r.options)}</details><div class="anchors">${section('적용 조건 — User side',r.user_side_context)}${section('적용 조건 — Agent side',r.agent_side_state_condition)}${section('0점 행동',r.behavior_0)}${section('1점 행동',r.behavior_1)}${section('2점 행동',r.behavior_2)}</div>${section('직전 Doctor 발화',r.doctor_latest)}${section('채점 대상 Agent 응답',r.agent_response)}<details><summary>응답 직전 전체 대화·Agent 상태</summary>${section('대화',r.history_before_response)}${section('Agent 상태',r.agent_pre_response_state)}</details><details><summary>Gemini 실제 출력·원본 로그 줄 번호</summary>${section('출력',r.judge_outputs)}<p>별도 판단 이유가 반환되지 않았다면 원본 JSON 판정만 표시됩니다.</p></details><details><summary>루브릭 출처 — 채점 모델에 제공하지 않은 검토 정보</summary>${section('출처 결과',r.source_outcome)}${section('선택 턴',r.selected_agent_turn_id)}${section('Doctor 피드백',r.source_doctor_feedback)}</details></article>`).join('')}
['stage','score','rubric','query'].forEach(id=>$(id).addEventListener('input',render));render();</script></html>'''

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',default='outputs/latest_pilot');parser.add_argument('--out-dir')
    args=parser.parse_args();run=(ROOT/args.run_dir).resolve()
    out=Path(args.out_dir).resolve() if args.out_dir else ROOT/'outputs/exports'/('scoring_review_'+datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_KST'))
    export(run,out)
