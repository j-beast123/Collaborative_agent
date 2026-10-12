#!/usr/bin/env python3
"""Read-only terminal monitor. Works with existing live files and new stage timestamps."""
import argparse
from datetime import datetime
import json
import math
import os
import shutil
import socket
from pathlib import Path
import sys
import time
import unicodedata

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.common.rollout_monitor import RolloutCounters
from src.common.bank_monitor import BankCounters, activity as bank_activity
_BANK=BankCounters()
_COUNTERS=RolloutCounters()
STAGES={'initializing':'모델 초기화', 'rollout':'Rollout 준비', 'bank_rollout':'Rollout + Doctor reflection',
        'bank_pipeline':'Rollout + rubric 처리', 'bank_stream':'Rollout + trajectory bank 처리', 'bank_online':'Rubric 생성·재사용·검증', 'bank_score':'전체 bank 채점',
        'agent_update':'GRPO 업데이트', 'batch_committed':'배치 저장 완료', 'run_finished':'실행 완료',
        'ready':'대기', 'fixed':'Rollout'}


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def alive(pid):
    return isinstance(pid,int) and (Path('/proc')/str(pid)).exists()


def duration(value):
    if value is None:return '알 수 없음'
    seconds=max(0,int(value));hours,seconds=divmod(seconds,3600);minutes,seconds=divmod(seconds,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def snapshot(out, total=120, batch_size=5, now=None, discovery_cap=48, group_mode=False):
    out=Path(out);now=time.time() if now is None else now
    state=read(out/'run_state.json');completed=state.get('case_cursor',0)
    progress=read(out/'progress.json')
    remote_host=bool(progress.get('host') and progress['host']!=socket.gethostname())
    rows=[]
    for p in sorted((out/'live').glob('gpu*.json')):
        row=read(p)
        if row:
            row['alive']=(now-row.get('updated_at',0)<120 and progress.get('stage')!='run_finished') if remote_host else alive(row.get('pid'))
            row['server']='서버1'
            rows.append(row)
    remote_worker=read(out/'distributed/worker/heartbeat.json')
    worker_alive=bool(remote_worker and now-remote_worker.get('timestamp',0)<15
                      and remote_worker.get('status') not in ('error','stopped'))
    for path in sorted((out/'distributed/worker/live').glob('gpu*.json')):
        row=read(path)
        if row:
            row.update(alive=worker_alive,server='서버2')
            rows.append(row)
    exact=bool(progress and (any(r['alive'] for r in rows) or now-progress.get('timestamp',0)<120
                           if remote_host else alive(progress.get('pid'))))
    batch=progress.get('batch') if exact else None
    batch=batch or (state.get('batch_cursor', 1) if progress.get('stage')=='run_finished'
                   else state.get('batch_cursor',0)+1)
    batch=max(1,batch)
    batchdir=out/'batches'/f'batch_{batch:06d}'
    live=[r for r in rows if r['alive']]
    if exact:
        stage=progress.get('stage','initializing');since=progress.get('stage_started_at')
        batch_since=progress.get('batch_started_at');timing='기록값'
    else:
        active=[r for r in live if r.get('active')]
        phase_order={'agent_update':4,'bank_score':3,'bank_online':2,'bank_stream':2,'bank_rollout':1}
        current=max(active,key=lambda r:phase_order.get(r.get('phase'),0),default={})
        stage=current.get('phase','ready' if live else 'stopped')
        starts=[v for r in active if r.get('phase')==stage for k,v in r.get('active_since',{}).items()
                if k.startswith(stage+'.') or k==stage]
        since=min(starts) if starts else None
        try:batch_since=(batchdir/'manifest.json').stat().st_mtime
        except OSError:batch_since=None
        if stage=='bank_rollout' and batch_since is not None:
            since=batch_since  # Existing run: no global stage timestamp was recorded.
        timing='추정값'
    if not live and progress.get('stage')=='run_finished':
        stage='run_finished'
        since=progress.get('stage_started_at');batch_since=progress.get('batch_started_at')
    if completed>=total and not live:
        stage='run_finished';batch=math.ceil(total/batch_size)
    elif exact and stage=='run_finished':
        batch=state.get('batch_cursor',batch)
    manifest=read(batchdir/'manifest.json')
    indices=[assignment['index'] for assignment in manifest.get('assignments',[]) if 'index' in assignment]
    if not indices:
        indices=list(range((batch-1)*batch_size,min(batch*batch_size,total)))
    cases=[]
    for index in indices:
        directory=batchdir/f'case_{index:06d}'
        counts=_COUNTERS.read(directory/'calls.jsonl')
        finished=(directory/'bank_rollout.json').is_file() or (directory/'group_rollout.json').is_file()
        current=next((r for r in live if r.get('case_index')==index),{})
        bank=_BANK.read(directory,discovery_cap=discovery_cap)
        finished=finished or bank.get('stream',{}).get('rollout_done',False)
        operation,operation_seconds=bank_activity(current,now)
        if bank.get('stream',{}).get('protocol')=='parallel_chunk_delta_commit_v1' and not bank['done']:
            operation=None
        if not operation and bank.get('stream') and not bank['done']:
            operation=bank['stream'].get('stage')
            started=bank['stream'].get('stage_started')
            operation_seconds=now-started if isinstance(started,(int,float)) else None
        bank.update(operation=operation,operation_seconds=operation_seconds)
        group_progress=read(directory/'group_progress.json')
        group_metrics=read(directory/'group_metrics.json')
        cases.append({'index':index,**counts,'bank':bank,'rollout_done':finished,
                      'group_progress':group_progress, 'group_metrics':group_metrics,
                      'server':next((assignment.get('server','local') for assignment in manifest.get('assignments',[]) if assignment.get('index')==index),'local'),
                      'phase':current.get('phase'), 'started':(directory/'calls.jsonl').exists()})
    totals={key:sum(case[key] for case in cases) for key in ('agent_nodes','doctor_turns','trajectories','reflections')}
    files={phase:len(list(batchdir.glob(f'case_*/{phase}.json')))
           for phase in ('bank_rollout','bank_online','bank_score')}
    files['bank_rollout']=sum(case['rollout_done'] for case in cases)
    if not live and stage!='run_finished':stage='stopped'
    timing_record=read(batchdir/'timing.json')
    batch_seconds=now-batch_since if batch_since is not None else None
    if stage in ('run_finished','batch_committed'):
        batch_seconds=timing_record.get('batch_seconds',progress.get('batch_seconds',batch_seconds))
    latest_timing=timing_record
    if not latest_timing:
        paths=sorted((out/'batches').glob('batch_*/timing.json'))
        if paths:latest_timing=read(paths[-1])
    latest_update={}
    if group_mode:
        records=sorted((out/'cases').glob('case_*.json'))
        if records:latest_update=read(records[-1]).get('batch_update') or {}
    upload_counts={}
    for path in (out/'hf_uploads/jobs').glob('*.json'):
        status=read(path).get('status','queued')
        upload_counts[status]=upload_counts.get(status,0)+1
    enqueue_error=read(out/'hf_uploads/enqueue_error.json')
    return {'out':str(out),'now':now,'total':total,'completed':completed,'batch':batch,
            'batches':math.ceil(total/batch_size),'batch_size':batch_size,'stage':stage,
            'stage_started_at':since,'batch_started_at':batch_since,'timing':timing,
            'stage_seconds':now-since if since is not None else None,
            'batch_seconds':batch_seconds,'latest_timing':latest_timing,'latest_update':latest_update,
            'upload_counts':upload_counts,'upload_enqueue_error':enqueue_error,
            'remote_worker':remote_worker,
            'files':files,'rollout_cases':cases,'rollout_totals':totals,'gpus':rows,'progress':progress,'exact':exact,
            'group_mode':group_mode or any(case['group_progress'] or case['group_metrics'] for case in cases)}


def render(s, *, details=False, width=110):
    if s.get('group_mode'):return render_groups(s,details=details,width=width)
    now=s['now'];stage=s['stage'];label=STAGES.get(stage,'프로세스 종료' if stage=='stopped' else stage)
    lines=[f"SHERPA ONLINE | {datetime.fromtimestamp(now).astimezone():%Y-%m-%d %H:%M:%S %Z}",
           f"배치 {s['batch']}/{s['batches']} | 저장 완료 {s['completed']}/{s['total']} 케이스",
           f"현재 단계: {label} | 경과 {duration(s['stage_seconds'])} ({s['timing']})",
           f"현재 배치 경과: {duration(s['batch_seconds'])}",
           '케이스 완료 파일: '+ ' | '.join(f"{phase.replace('bank_','')} {count}/{s['batch_size']}" for phase,count in s['files'].items()),'']
    totals=s['rollout_totals']
    lines.append(f"이번 배치 rollout: trajectory 완료 {totals['trajectories']} | Agent 응답 {totals['agent_nodes']} | Doctor 논의 {totals['doctor_turns']} | Reflection 완료 {totals['reflections']}")
    lines.append('케이스   상태             Trajectory   Agent 응답   Reflection   종료 깊이별 개수')
    for case in s['rollout_cases']:
        status='rollout 완료' if case['rollout_done'] else ('rollout 진행' if case.get('phase') in ('bank_rollout','bank_stream') else ('처리 중' if case['started'] else '대기'))
        depths=', '.join(f"{depth}턴:{count}" for depth,count in sorted(case['leaf_depths'].items(),key=lambda item:int(item[0]))) or '-'
        lines.append(f"{case['index']+1:>5}   {status:<12} {case['trajectories']:>9} {case['agent_nodes']:>12} {case['reflections']:>12}   {depths}")
    lines.append('Trajectory는 FINAL까지 끝난 대화 경로입니다. 조기 종료에 따라 총량이 달라져 고정 퍼센트로 표시하지 않습니다.\n')
    flows=[case for case in s['rollout_cases'] if case['bank'].get('flow')]
    if flows:
        lines.append('Trajectory 흐름 | reflection 준비 → 8개 묶음 대기 → bank 처리 → 반영 완료')
        lines.append('케이스   준비 누적   묶음 대기   Lock 대기   bank 처리 중   bank 완료 trajectory / chunk')
        for case in flows:
            flow=case['bank']['flow']
            lines.append(f"{case['index']+1:>5} {flow['ready']:>11} {flow['waiting_trajectories']:>11} {flow['lock_wait_trajectories']:>11} {flow['processing_trajectories']:>14} {flow['committed_trajectories']:>16} / {flow['committed_chunks']}")
        lines.append('준비 누적은 유효 reflection 수, bank 완료는 생성·검증까지 처리한 trajectory 수입니다. KEEP rubric 수와 다릅니다.\n')
    lines.append('케이스별 bank 진행 (완료 캐시 기준)')
    for case in s['rollout_cases']:
        bank=case['bank']
        if bank['score_done']:status='채점 완료'
        elif bank['done']:status='Bank 반영 완료'
        elif bank['operation']:status=bank['operation']
        elif bank['chunks_generated']:status='Bank 처리 대기 / 완료 캐시 있음'
        elif case['rollout_done']:status='Bank 대기'
        else:status='Rollout 대기 / 진행 중'
        elapsed=f" | 현재 작업 {duration(bank['operation_seconds'])}" if bank['operation'] else ''
        count=lambda value:'?' if value is None else str(value)
        lines.append(f"  케이스 {case['index']+1}: {status}{elapsed}")
        if bank.get('stream'):
            stream=bank['stream']
            lines.append(f"    Reflection 준비 {stream.get('ready_trajectories',0)} | bank chunk 반영 {stream.get('chunks_committed',0)} | rollout {'완료' if stream.get('rollout_done') else '진행 중'}")
            flow=bank.get('flow')
            if flow:
                marks=[]
                start=max(0,len(flow['chunk_sizes'])-6)
                for index in range(start,len(flow['chunk_sizes'])):
                    state='완료' if index in flow['completed_indices'] else ('Lock 대기' if flow['lock_wait_trajectories'] else '처리 중') if index in flow.get('active_indices',[])  else '대기'
                    marks.append(f"#{index+1} [{state} {flow['chunk_sizes'][index]}개]")
                if start:marks.insert(0,f'앞 {start}개 chunk 생략')
                if marks:lines.append('    Chunk: '+' → '.join(marks))
                waiting=flow['waiting_trajectories']
                if waiting:
                    detail=(f'남은 {waiting}개 처리 대기 (마지막 1~7개도 처리)' if flow['rollout_done']
                            else f'다음 묶음 준비 {min(waiting,8)}/8 | 대기 {waiting}개')
                    lines.append('    '+detail)
        if case['rollout_done'] or bank['chunks_generated'] or bank['done']:
            proposals=bank['proposals']
            lines.append(f"    대상 trajectory {count(bank['selected'])} | chunk 생성 응답 {bank['chunks_generated']}/{count(bank['chunks_total'])} | 제안 CREATE {proposals['CREATE']} · REUSE {proposals['REUSE_BANK']} · PEER {proposals['PEER']}")
            lines.append(f"    Gate 완료 {bank['gate_done']} (통과 {bank['gate_passed']} / 거절 {bank['gate_rejected']} / 미판정 {bank['gate_unresolved']}) | Applicability {bank['source_checks']} | PEER 검사 {bank['peer_checks']} | Delta 중복 비교 {bank.get('delta_checks',0)}")
            lines.append(f"    Intervention 검사 {bank['instruction_audits']} | Verification rerun 완료 {bank['reruns']} | 응답 채점 호출 완료 {bank['score_calls']}")
            if bank['done']:
                lines.append(f"    최종 KEEP {count(bank['kept'])} / 제외 {count(bank['removed'])} | Bank rubric 증가 {count(bank['bank_added'])}")
    lines.append('생성 응답·Gate·rerun 수는 각각 별도 단계입니다. 최종 KEEP과 bank 반영 수는 케이스 처리 완료 후 확정됩니다.\n')
    if s['exact'] and s['stage_started_at']:
        lines.append(f"단계 시작: {datetime.fromtimestamp(s['stage_started_at']).astimezone():%H:%M:%S}")
    for row in s['gpus']:
        phase=STAGES.get(row.get('phase'),row.get('phase','-'))
        case=row.get('case_index');case=f'케이스 {case+1}' if isinstance(case,int) else '-'
        age=duration(now-row['updated_at']) if row.get('updated_at') else '-'
        lines.append(f"GPU {row.get('gpu')} | {phase} | {case} | PID {row.get('pid')} {'실행 중' if row['alive'] else '종료'} | 마지막 기록 {age} 전")
        activities=row.get('active',{})
        for operation in sorted(activities,key=lambda name:row.get('active_since',{}).get(name,now))[:5]:
            since=row.get('active_since',{}).get(operation)
            lines.append(f"  {operation} ×{activities[operation]} | {duration(now-since if since else None)}")
        if row.get('last_error'):lines.append(f"  마지막 오류: {row['last_error']}")
    if not s['exact']:
        lines.append('\n기존 실행: 단계 시간은 live/배치 파일 기준 추정값입니다. 다음 실행부터 정확한 단계 시작 시각이 기록됩니다.')
    lines.extend(['',f"출력: {s['out']}",'Ctrl+C: 모니터만 종료 (학습은 계속 진행)'])
    return '\n'.join(lines)


GROUP_LABELS={'reflection':'회고 중','generation':'생성 중','verification':'검증 중',
              'reward':'판정 중','no_candidate_rubric':'후보 없음',
              'no_verified_rubric':'검증 탈락','reflection_invalid':'회고 오류',
              'generation_invalid':'생성 오류','reward_unresolved':'판정 오류'}
GROUP_DONE={'scored','no_candidate_rubric','no_verified_rubric','reflection_invalid',
            'generation_invalid','reward_unresolved'}


def cell(value, width):
    """Pad by terminal columns so Korean and Latin columns line up."""
    value=str(value)
    columns=sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in value)
    return value+' '*max(0,width-columns)


def fraction(numerator, denominator):
    return f'{numerator}/{denominator} ({100*numerator/denominator:.0f}%)' if denominator else '대기'


def group_label(group):
    if group.get('status')=='scored':
        return '동점' if all(value==0 for value in group.get('advantages',[])) else '학습 신호'
    return GROUP_LABELS.get(group.get('status'),group.get('status','대기'))


def short_group_label(group):
    labels={'reflection':'회고','generation':'생성','verification':'검증','reward':'판정',
            'no_candidate_rubric':'후보없음','no_verified_rubric':'탈락',
            'reflection_invalid':'회고오류','generation_invalid':'생성오류','reward_unresolved':'판정오류'}
    if group.get('status')=='scored':
        return '동점' if all(v==0 for v in group.get('advantages',[])) else '학습'
    return labels.get(group.get('status'),'대기')


def case_phase(case):
    """Report simultaneous active stages, rather than implying one sequential stage."""
    groups=case['group_progress'].get('groups',[])
    stages=[]
    if not case['rollout_done']:
        stages.append('대화·분기' if case.get('started') else '대기')
    for key,label in (('reflection','회고'),('generation','생성'),('verification','검증'),('reward','판정')):
        count=sum(group.get('status')==key for group in groups)
        if count:stages.append(f'{label} {count}')
    if stages:return ' + '.join(stages)
    if groups and all(group.get('status') in GROUP_DONE for group in groups):
        if any(group.get('status') in ('reflection_invalid','generation_invalid','reward_unresolved') for group in groups):
            return '처리 완료 · 오류 있음'
        if any(group.get('status')=='scored' and any(a!=0 for a in group.get('advantages',[])) for group in groups):
            return '처리 완료 · 학습 준비'
        return '처리 완료 · 학습 신호 없음'
    return '분기 완료 · 그룹 준비'


def terminal_frame(screen, width, height):
    """Fit Unicode text to one screen, preventing wrap/scroll and stale line tails."""
    lines=screen.splitlines()
    budget=max(1,height-1)
    if len(lines)>budget:lines=[line for line in lines if line.strip()]
    if len(lines)>budget:lines=[line for line in lines if not set(line)<=set('=-')]
    if len(lines)>budget:lines=[line for line in lines if not line.startswith(('출력:','[케이스별','[이번 배치'))]
    if len(lines)>budget:
        lines=lines[:max(0,budget-1)]+['상세 전체 보기: --once --details (화면 높이 제한)']
    clipped=[]
    for line in lines:
        result='';columns=0
        for char in line:
            size=0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in 'WF' else 1
            if columns+size>max(1,width-2):
                result+='…';break
            result+=char;columns+=size
        clipped.append(result)
    return '\r\n'.join(clipped)


def operation_label(operation):
    if 'verification_partner_state' in operation:return '즉시 선택지 검증'
    if 'group_verification' in operation:return '검증 이어가기'
    if 'group_contrast' in operation:return '대비 응답 생성'
    if 'group_reflection' in operation:return 'partner 회고'
    if 'group_generation' in operation:return 'rubric 생성'
    if 'group_reward' in operation:return 'judge 판정'
    if 'agent_update' in operation or 'grpo' in operation:return 'GRPO 업데이트'
    if 'doctor' in operation:return 'partner 대화'
    if 'final' in operation:return '대화 결과 확인'
    if 'agent' in operation:return 'assistant 응답'
    if 'gemini' in operation or 'luna' in operation or 'api.' in operation:return '외부 모델 호출'
    return '기타 처리'


def render_groups(s, *, details=False, width=110):
    cases=s['rollout_cases']
    groups=[group for case in cases for group in case['group_progress'].get('groups',[])]
    done=[g for g in groups if g.get('status') in GROUP_DONE]
    scored=[g for g in groups if g.get('status')=='scored']
    usable=[g for g in scored if any(value!=0 for value in g.get('advantages',[]))]
    zero=len(scored)-len(usable)
    candidates=[c for g in groups for c in g.get('candidates',[])]
    resolved=[c for c in candidates if c.get('s') is not None]
    passed=sum(bool(c.get('retained')) for c in resolved)
    rubric_groups=sum(bool(g.get('kept_rubric_ids')) for g in done)
    stage=s['stage'];label=STAGES.get(stage,'프로세스 종료' if stage=='stopped' else stage)
    active_labels=sorted({group_label(g) for g in groups if g.get('status') not in GROUP_DONE})
    if active_labels and stage not in ('agent_update','run_finished','stopped','batch_committed'):
        label=' / '.join((['대화 진행'] if any(not c['rollout_done'] for c in cases) else [])+active_labels)
    bar_size=20;filled=min(bar_size,int(bar_size*s['completed']/max(1,s['total'])))
    lines=[f"SHERPA 학습 모니터  |  {datetime.fromtimestamp(s['now']).astimezone():%m-%d %H:%M:%S}",
           '='*min(width,108),
           f"전체 [{'#'*filled}{'.'*(bar_size-filled)}]  {s['completed']}/{s['total']} 케이스 저장  |  배치 {s['batch']}/{s['batches']}",
           f"현재: {label}  |  이번 배치 {duration(s['batch_seconds'])}"]
    if s.get('verification_mode') == 'next_partner_state_v1':
        lines.append('검증 기준: 대비 직후 partner의 비밀 선택지 | 3회 중 2회 이상')
    timing=s.get('latest_timing') or {}
    if timing.get('batch_seconds') is not None:
        projected=timing.get('projected_full_run_seconds',timing.get('projected_24_batches_seconds',24*timing['batch_seconds']))
        planned=timing.get('planned_batches',24)
        lines.append(f"최근 완료 배치: {duration(timing['batch_seconds'])}  |  ×{planned} 전체 예상: {duration(projected)}")
    update=s.get('latest_update') or {}
    if update:
        label='업데이트 완료' if update.get('optimizer_step') else '업데이트 없음'
        lines.append(f"최근 GRPO: {label}  |  실제 사용 {update.get('groups_used','?')}그룹")
        health=[]
        for key,title in (('loss','loss'),('kl_loss','KL'),('grad_norm','gradient')):
            if isinstance(update.get(key),(int,float)):health.append(f'{title} {update[key]:.4g}')
        ratio=update.get('ratio') or {}
        if isinstance(ratio.get('clipped_fraction'),(int,float)):
            health.append(f"clipping {100*ratio['clipped_fraction']:.1f}%")
        if health:lines.append('  '+'  |  '.join(health))
    lines.extend(['','[이번 배치 학습 신호]',
                  f"학습 가능 {len(usable)}그룹  |  rubric 보유 {fraction(rubric_groups,len(done))}  |  동점 {fraction(zero,len(scored))}",
                  f"검증 통과 {fraction(passed,len(resolved))}  |  후보 없음 {sum(g.get('status')=='no_candidate_rubric' for g in done)}그룹  |  처리 오류 {sum(g.get('status') in ('reflection_invalid','generation_invalid','reward_unresolved') for g in done)}그룹",
                  '', '[케이스별 진행]'])
    headers=('케이스','대화','그룹 완료','rubric','학습','현재 진행')
    widths=(8,6,11,8,6,0)
    lines.append(''.join(cell(h,w) for h,w in zip(headers,widths)))
    lines.append('-'*min(width,108))
    for case in cases:
        case_groups=sorted(case['group_progress'].get('groups',[]),key=lambda g:g['anchor'])
        finished=sum(g.get('status') in GROUP_DONE for g in case_groups)
        count=sum(bool(g.get('kept_rubric_ids')) for g in case_groups)
        train=sum(g.get('status')=='scored' and any(a!=0 for a in g.get('advantages',[])) for g in case_groups)
        server='2' if case.get('server')=='remote' else '1'
        values=(f"#{case['index']+1:02d}/S{server}",case['trajectories'],f'{finished}/{len(case_groups)}',count,train,case_phase(case))
        turns='  '.join(f"{g['anchor']}:{short_group_label(g)}" for g in case_groups)
        if not turns:turns='대화·분기 준비 중' if case.get('started') else '대기'
        row=''.join(cell(v,w) for v,w in zip(values,widths))
        lines.append(row)
        if details:
            lines.append('  턴별: '+turns)
            for group in case_groups:
                if group.get('ranks'):
                    lines.append(f"  T{group['anchor']} 형제 원래 순서의 순위: {group['ranks']} | reward: {[round(v,3) for v in group['rewards']]}")
                for candidate in group.get('candidates',[]):
                    lines.append(f"  T{group['anchor']} 검증 s={candidate.get('s')} | {candidate.get('status','진행 중')}")
    if details:
        lines.extend(['완료/준비: 분기 준비가 끝난 그룹 중 처리가 끝난 그룹. 준비 중인 턴은 분모에 포함하지 않습니다.',
                      '학습: judge 판정 완료 + nonzero advantage. 실제 사용 수는 최근 GRPO에 표시됩니다.',
                      '비율 기준: rubric은 처리 완료 그룹, 동점은 judge 판정 완료 그룹, 검증은 s 확정 후보.'])
    lines.append('')
    gpu_labels=[]
    for row in s['gpus']:
        labels=sorted({operation_label(op) for op in row.get('active',{})}) if row['alive'] else []
        text=' / '.join(labels) or ('대기' if row['alive'] else '종료')
        gpu_labels.append(f"{row.get('server','서버1')} GPU{row.get('gpu')}: {text}")
        if row.get('last_error'):lines.append(f"  오류: {row['last_error']}")
        if details:
            for operation,count in sorted(row.get('active',{}).items()):
                since=row.get('active_since',{}).get(operation)
                lines.append(f"  {operation} ×{count} | {duration(s['now']-since if since else None)}")
    if gpu_labels:
        for start in range(0,len(gpu_labels),2):lines.append('  |  '.join(gpu_labels[start:start+2]))
    remote=s.get('remote_worker') or {}
    if remote:
        labels={'initializing':'모델 준비 중','ready':'준비 완료','running':'처리 중','error':'오류','stopped':'종료'}
        lines.append(f"서버2 worker: {labels.get(remote.get('status'),remote.get('status'))} | heartbeat {duration(s['now']-remote.get('timestamp',s['now']))} 전")
    uploads=s.get('upload_counts') or {}
    if uploads:
        waiting=sum(uploads.get(key,0) for key in ('queued','waiting','retrying'))
        failed=sum(uploads.get(key,0) for key in ('failed','blocked'))
        lines.append(f"HF 업로드: 완료 {uploads.get('uploaded',0)} | 진행 {uploads.get('uploading',0)} | 대기 {waiting} | 실패 {failed}")
    if s.get('upload_enqueue_error'):
        lines.append(f"HF 큐 오류: {s['upload_enqueue_error'].get('error_type')} (학습은 진행)")
    lines.extend(['',f"출력: {Path(s['out']).name}",
                  'Ctrl+C: 모니터 종료  |  --details: 순위·검증 결과와 내부 작업 상세'])
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/sherpa.json')
    parser.add_argument('--output',type=Path,help='실행 output 디렉터리 직접 지정')
    parser.add_argument('--interval',type=float,default=2)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--json',action='store_true',help='한 번 JSON 상태 출력')
    parser.add_argument('--details',action='store_true',help='순위·검증 결과와 내부 작업명도 표시')
    args=parser.parse_args()
    if args.interval<=0:parser.error('--interval must be positive')
    from src.online_rl.tree.experiment import load_config
    cfg=load_config(args.config)
    out=args.output or ROOT/cfg['output_dir']
    if cfg.get('pipeline_protocol') == 'sherpa_trajectory_checklist_v1':
        from src.online_rl.sherpa.monitor import snapshot as trajectory_snapshot, render as trajectory_render
        try:
            while True:
                data = trajectory_snapshot(out)
                screen = json.dumps(data, ensure_ascii=False, indent=2) if args.json else trajectory_render(data)
                if sys.stdout.isatty() and not args.once and not args.json:
                    sys.stdout.write('\x1b[2J\x1b[H')
                print(screen, flush=True)
                if args.once or args.json:
                    return
                time.sleep(args.interval)
        except KeyboardInterrupt:
            return
    saved_cfg=read(out/'run_manifest.json').get('config') or cfg
    fullscreen=sys.stdout.isatty() and not args.once and not args.json and os.environ.get('TERM')!='dumb'
    if fullscreen:
        sys.stdout.write('\033[?1049h\033[?25l\033[2J\033[H');sys.stdout.flush()
    try:
        while True:
            data=snapshot(out,saved_cfg['schedule']['cases'],saved_cfg['five_gpu']['batch_cases'],
                          discovery_cap=(cfg.get('rubric',{}).get('cumulative_bank') or {}).get('discovery_trajectory_cap',48),
                          group_mode=bool(cfg.get('group_learning')))
            data['verification_mode'] = saved_cfg.get('group_learning', {}).get('verification_mode', 'final_outcome_v1')
            if args.json:print(json.dumps(data,ensure_ascii=False,indent=2));return
            size=shutil.get_terminal_size((110,40))
            screen=render(data,details=args.details,width=size.columns)
            if fullscreen:
                sys.stdout.write('\033[2J\033[H'+terminal_frame(screen,size.columns,size.lines)+'\033[J')
                sys.stdout.flush()
            else:print(screen,flush=True)
            if args.once:return
            time.sleep(args.interval)
    except KeyboardInterrupt:pass
    finally:
        if fullscreen:
            sys.stdout.write('\033[?25h\033[?1049l');sys.stdout.flush()

if __name__=='__main__':main()
