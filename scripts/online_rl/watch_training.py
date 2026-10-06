"""Incremental live terminal monitor for the 12-case trainer and evaluation queue."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


class Journal:
    def __init__(self):self.offset=0;self.counts=Counter();self.depths=Counter();self.last_stage={};self.batches=Counter();self.final_paths=set()
    def update(self,path):
        if not path.exists():return
        if path.stat().st_size<self.offset:self.__init__()
        with path.open('rb') as f:
            f.seek(self.offset)
            while True:
                begin=f.tell();line=f.readline()
                if not line:break
                if not line.endswith(b'\n'):f.seek(begin);break
                self.offset=f.tell()
                try:r=json.loads(line)
                except (ValueError,UnicodeDecodeError):continue
                if not isinstance(r,dict):continue
                kind=r.get('kind')
                if kind:
                    self.counts[kind]+=1
                    if kind=='final':self.final_paths.add(r.get('identity',{}).get('path',''))
                    if kind=='agent':
                        depth=len(r.get('identity',{}).get('path','').split('.'))
                        self.depths[depth]+=1
                if r.get('stage')=='agent_generation_batch':self.batches[r['batch_size']]+=1
                elif r.get('stage'):self.last_stage=r


def duration(seconds):
    if seconds is None:return '측정 전'
    return f'{seconds/60:.0f}분' if seconds<3600 else f'{seconds/3600:.1f}시간'


def rollout_work(journal, turns):
    # Terminal prefixes prune their entire subtree; deeper-than-assumption work
    # is excluded from this scenario rather than inflating its progress.
    actual=sum(n for d,n in journal.depths.items() if d<=turns)
    pruned=sum(sum(4**k for k in range(1,turns-len(p.split('.'))+1))
               for p in journal.final_paths if p)
    budget=sum(4**k for k in range(1,turns+1))
    return min(budget,actual+pruned),budget


def estimates(out, active, cases, now=None):
    now=time.time() if now is None else now
    path=out/'monitor_progress.json'
    data=read(path)
    samples=data.get('samples',[]) if data.get('batch')==active.name else []
    current={str(index):{'agent':j.counts['agent'],
                        'rubric_generation':j.counts['rubric_generation'],
                        'rerun':j.counts['rerun'],'rubric_score':j.counts['rubric_score'],
                        'work4':rollout_work(j,4)[0],
                        'work5':rollout_work(j,5)[0]}
             for index,j,gpu,complete in cases}
    # Persist observations so restarting the monitor preserves the window.
    if not samples or now-samples[-1]['time']>=15:
        samples.append({'time':now,'cases':current})
    samples=[r for r in samples if r['time']>=now-660]
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps({'batch':active.name,'samples':samples}))
    temp.replace(path)
    baseline=samples[0];elapsed=now-baseline['time']
    print('\n롤아웃 시간 추정 (최근 최대 10분 관측; 루브릭·리런·채점 제외)')
    if elapsed<60:
        print('  처리 속도: 측정 중 — 1분 이후 표시, 10분 누적 시 안정화')
    else:
        generated=sum(v['agent']-baseline['cases'].get(k,v)['agent'] for k,v in current.items())
        print(f'  최근 {elapsed/60:.1f}분 Agent 생성 속도: {generated*60/elapsed:.1f}응답/분')
    for turns in (4,5):
        finished=sum(rollout_work(j,turns)[0] for _,j,_,_ in cases)
        budget=sum(rollout_work(j,turns)[1] for _,j,_,_ in cases)
        gpu_work={};blocked=[]
        for index,j,gpu,complete in cases:
            work,limit=rollout_work(j,turns)
            if complete:work=limit
            if work>=limit:continue
            old=baseline['cases'].get(str(index),current[str(index)])
            rate=(current[str(index)][f'work{turns}']-old[f'work{turns}'])/elapsed if elapsed>=60 else 0
            if rate<=0:blocked.append(index)
            remaining,velocity=gpu_work.get(gpu,(0,0))
            gpu_work[gpu]=(remaining+limit-work,velocity+max(0,rate))
        eta=max((remaining/rate for remaining,rate in gpu_work.values() if rate>0),default=None)
        label=duration(eta) if not blocked and elapsed>=60 else '산출 보류 (관측 부족 또는 진행 없는 case)'
        print(f'  {turns}턴 가정: {finished}/{budget} 상당 ({finished/budget:.1%}) · 남은 시간 {label}')
    stale=[]
    for index,j,gpu,complete in cases:
        f=active/'cases'/f'case_{index:06d}'/'calls.jsonl'
        if not complete and f.exists() and now-f.stat().st_mtime>=600:
            stale.append(f'{index:03d} ({(now-f.stat().st_mtime)/60:.0f}분 기록 없음)')
    print('  정체 의심 (10분 이상 호출 기록 없음): '+(', '.join(stale) or '없음'))
    for kind,label in [('rubric_generation','루브릭 생성'),('rerun','리런'),('rubric_score','채점')]:
        count=sum(v[kind] for v in current.values())
        delta=sum(v[kind]-baseline['cases'].get(k,v).get(kind,v[kind]) for k,v in current.items())
        speed=f'{delta*60/elapsed:.1f}회/분' if elapsed>=60 and delta>0 else '측정 전 또는 최근 완료 없음'
        print(f'  {label}: 완료 호출 {count} · 속도 {speed} · ETA는 예정량 확정 전')
    print('  LoRA / 첫 업데이트 ETA: 측정 전 — 전체 완료 시간으로 환산하지 않음')


def process(path):
    try:
        pid=int(path.read_text());s=(Path('/proc')/str(pid)/'status').read_text()
        state=next((l for l in s.splitlines() if l.startswith('State:')),'')
        return f'실행 중 PID {pid}' if 'Z (zombie)' not in state else '중지'
    except (OSError,ValueError):return '중지 또는 확인 불가'


def render(out,journals,gpu=False):
    state=read(out/'run_state.json');manifest=read(out/'run_manifest.json');cfg=manifest.get('config',{})
    total=cfg.get('schedule',{}).get('cases',500);cursor=state.get('case_cursor',0);done=state.get('batch_cursor',0)
    log=journals.setdefault('runner',Journal());log.update(out/'runner.log')
    print(datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST'))
    print('학습 실행기: '+process(out/'runner.pid'))
    print(f'학습 완료 {cursor}/{total}문제 ({cursor/total:.1%}) | 업데이트 완료 묶음 {done} | 프롬프트 {state.get("prompt_id","P0")}')
    print('최근 단계: '+str(log.last_stage.get('stage','모델 로딩/준비'))+' | 정책 '+state.get('policy_sha256','미발행')[:12])
    folders=sorted((out/'batches').glob('batch_*'));active=next((p for p in folders if int(p.name.split('_')[-1])>done),None)
    if active:
        m=read(active/'manifest.json');assignments={r['index']:r for r in m.get('assignments',[])}
        print(f'\n현재 {active.name}: {len(m.get("case_indices",[]))}문제 | 완료는 롤아웃·검증·채점 뒤 업데이트까지 끝난 기준')
        case_stats=[]
        for index in m.get('case_indices',[]):
            case=active/'cases'/f'case_{index:06d}';journal=journals.setdefault(str(case),Journal());journal.update(case/'calls.jsonl');c=journal.counts
            status='롤아웃 중'
            if (case/'source_rollout.json').exists():status='원본 생성 완료 / 최적화 대기'
            if c['rubric_generation']:status='루브릭 생성·리런·채점 중'
            if (case/'case.json').exists():status='case 처리 완료 / 묶음 업데이트 대기'
            assignment=assignments.get(index,{})
            case_stats.append((index,journal,assignment.get('gpu','?'),(case/'source_rollout.json').exists()))
            depth=max(journal.depths,default=0)
            print(f'  case {index:03d} GPU{assignment.get("gpu","?")} 대화{assignment.get("lane",0)+1} | {status}')
            print(f'    Agent {c["agent"]}응답 · 최대 {depth}/5턴 · 원본 결정 {c["final"]} · Doctor 피드백 {c["feedback"]} · belief {c["agent_belief"]}')
            print(f'    루브릭 {c["rubric_generation"]} · 리런 {c["rerun"]} · 채점 호출 {c["rubric_score"]}')
        if case_stats:estimates(out,active,case_stats)
    print('\nAgent 생성 배치 크기별 누적 호출: '+str(dict(sorted(log.batches.items()))))
    evaluation=read(out/'evaluation/worker_status.json');queued=list((out/'evaluation/queue').glob('case_*.json'))
    results=list((out/'evaluation/results').glob('case_*/summary.json'));finished={r.parent.name for r in results};pending=[p for p in queued if p.stem not in finished]
    eval_stage={'waiting':'체크포인트 대기','evaluating':'평가 중'}.get(evaluation.get('stage'),'미시작')
    print(f'\nGPU4 평가: {eval_stage} | 대기 {len(pending)}개 · 완료 {len(results)}개 체크포인트')
    if evaluation.get('checkpoint_cases'):
        count=evaluation['checkpoint_cases'];progress=read(out/'evaluation/results'/f'case_{count:06d}'/'progress.json')
        print(f'  {count}문제 체크포인트: {progress.get("completed",0)}/{progress.get("total",622)}문제 평가')
    if results:
        summary=read(sorted(results)[-1]);s=summary.get('splits',{}).get('all',{});acc=s.get('accuracy')
        print(f'  최근 평가 {summary.get("checkpoint_cases")}문제 checkpoint | 정답률 {acc:.2%}' if acc is not None else '  최근 평가 정답률 없음')
    error=read(out/'evaluation/last_error.json')
    if error:print('  최근 평가 오류 기록: '+str(error.get('error',''))[:160])
    if gpu:
        try:
            r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=3)
            if r.returncode==0:print('\nGPU 번호 / 메모리 / 사용률\n'+r.stdout.strip())
        except (OSError,subprocess.TimeoutExpired):pass
    print('\nCtrl+C는 모니터만 종료합니다. 학습·평가는 계속 실행됩니다.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='outputs/latest_training');p.add_argument('--interval',type=float,default=3);p.add_argument('--once',action='store_true');p.add_argument('--gpu',action='store_true');a=p.parse_args()
    if a.interval<=0:p.error('interval must be positive')
    out=(ROOT/a.run_dir).resolve();journals={}
    try:
        while True:
            if not a.once and sys.stdout.isatty():print('\033[2J\033[H',end='')
            render(out,journals,a.gpu)
            if a.once:break
            time.sleep(a.interval)
    except KeyboardInterrupt:pass

if __name__=='__main__':main()
