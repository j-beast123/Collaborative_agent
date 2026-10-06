"""Local terminal monitor; reads saved results only, no GPU or API calls."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]


def rows(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass  # append-in-progress line
    return result


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def process(folder,name):
    try:
        pid=(folder/name).read_text().strip()
        status=Path('/proc')/pid/'status'
        if not status.exists():return '중지'
        state=next((line for line in status.read_text().splitlines() if line.startswith('State:')),'')
        return '중지' if 'Z (zombie)' in state else f'실행 중 (PID {pid})'
    except OSError:return '확인 불가'


def snapshot(pilot_only=False):
    p=(ROOT/'outputs/latest_pilot').resolve()
    state=read(p/'run_state.json');calls=Counter(r['kind'] for r in rows(p/'calls.jsonl'))
    print(datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST'))
    print('\nGPU 0 파일럿 — '+process(p,'pilot.pid'))
    print(f"완료 문제 {state.get('case_cursor',0)}/4 | heldout {'완료' if (p/'heldout/result.json').exists() else '대기'} | 프롬프트 {state.get('prompt_id','P0')} | 최적화 이벤트 {state.get('lineage_cursor',0)}")
    for kind,label in [('doctor_turn','Doctor 대화'),('agent','Agent 응답'),('agent_belief','belief'),('final','최종 결정'),('feedback','닥터 피드백'),('rubric_generation','루브릭 생성'),('posthoc_gate','루브릭 검증'),('rubric_score','응답 채점 캐시'),('rerun','리런')]:
        print(f'  {label}: {calls[kind]}')
    usage=rows(p/'api_usage.jsonl')
    if usage:print(f"최근 API 기록: {usage[-1].get('timestamp')} | {usage[-1].get('model')} | {usage[-1].get('purpose')}")
    log=p/'run.log'
    if log.exists():
        lines=[x for x in log.read_text().splitlines() if x.strip()]
        if lines:print('로그 마지막 줄: '+lines[-1][:160])
    print('출력: '+str(p))
    if pilot_only:
        print('\nCtrl+C로 모니터만 종료합니다. 실험은 계속 실행됩니다.')
        return
    b=ROOT/'outputs/test/base/casimedicos_exp_5turn_seed20261006'
    results=[]
    print('\nGPU 1~4 CasiMedicos baseline')
    for gpu in range(1,5):
        f=b/f'gpu{gpu}'
        saved=[read(x) for x in f.glob('cases/case_*.json')]
        valid=[x for x in saved if x.get('status')=='complete']
        correct=sum(x.get('outcome')==1 for x in valid)
        accuracy=f'{correct/len(valid):.1%}' if valid else '—'
        errors=len(list(f.glob('errors/case_*.json')))
        print(f"  GPU {gpu}: {len(saved)}/{156 if gpu<3 else 155} | 정답률 {accuracy} | 기술 오류 {errors} | {process(f,'worker.pid')}")
        results+=saved
    valid=[x for x in results if x.get('status')=='complete']
    correct=sum(x.get('outcome')==1 for x in valid)
    accuracy=f'{correct/len(valid):.1%}' if valid else '—'
    print(f'전체: {len(results)}/622 ({len(results)/622:.1%}) | 정답률 {accuracy} ({correct}/{len(valid)})')
    print('\n카운터는 저장된 누적 호출 수이며 문제 완료 수와 다릅니다. Ctrl+C로 모니터만 종료합니다.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--interval',type=float,default=3);parser.add_argument('--once',action='store_true');parser.add_argument('--pilot-only',action='store_true')
    args=parser.parse_args()
    if args.interval<=0:parser.error('--interval must be positive')
    try:
        while True:
            if not args.once and sys.stdout.isatty():print('\033[2J\033[H',end='')
            snapshot(args.pilot_only)
            if args.once:break
            time.sleep(args.interval)
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
