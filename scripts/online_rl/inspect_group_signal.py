"""Read-only inspection of verification and training signal, including partial batches."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def inspect(out, batch=None):
    folders = sorted((out / 'batches').glob('batch_*'))
    if batch is None:
        if not folders:
            return '아직 시작된 배치가 없습니다. 실행 로그의 준비 검사·모델 로딩 상태를 확인하세요.'
        directory = folders[-1]
    else:
        directory = out / 'batches' / f'batch_{batch:06d}'
    manifest = read(directory / 'manifest.json')
    if not manifest:
        return f'배치 manifest가 아직 없습니다: {directory}'
    assignments = manifest['assignments']
    reports, committed = [], []
    finished = total_groups = 0
    reference_turns = []
    for assignment in assignments:
        index = assignment['index']
        case_dir = directory / f'case_{index:06d}'
        rollout = read(case_dir / 'group_rollout.json')
        leaves = rollout.get('tree', {}).get('leaves', [])
        if leaves:
            total_groups += len(leaves[0]['path'])
            reference_turns.append(len(leaves[0]['path']))
        reports.extend(read(case_dir / 'group_progress.json').get('groups', []))
        finished += (case_dir / 'group_metrics.json').is_file()
        record = read(out / 'cases' / f'case_{index:06d}.json')
        if record.get('batch_update') is not None:
            committed.append(record)
    candidates = [item for report in reports for item in report.get('candidates', [])]
    resolved = [item for item in candidates if item.get('s') is not None]
    kept = [item for item in candidates if item.get('retained')]
    scored = [report for report in reports if report.get('status') == 'scored']
    zero = sum(all(value == 0 for value in report['advantages']) for report in scored)
    usable = len(scored) - zero
    groups_with_rubric = sum(bool(report.get('kept_rubric_ids')) for report in reports)
    histogram = Counter(f'{round(item["s"] * 3)}/3' for item in resolved)
    lines = [f'배치: {directory.name} | case 완료 {finished}/{len(assignments)}',
             f'기준 대화에서 확인된 그룹 {total_groups} | 현재 기록된 그룹 {len(reports)}',
             f'후보 {len(candidates)} | verification 결과 확정 {len(resolved)} | KEEP(s >= 2/3) {len(kept)}',
             's 분포: ' + (' | '.join(f'{key}: {histogram[key]}' for key in ('0/3', '1/3', '2/3', '3/3')) if resolved else '아직 확정 결과 없음'),
             f'rubric 있는 그룹 {groups_with_rubric} | 채점 완료 그룹 {len(scored)}',
             f'zero advantage 그룹 {zero}/{len(scored)} | 업데이트에 기여할 그룹 {usable}',
             '그룹 상태: ' + str(dict(Counter(report.get('status', 'unknown') for report in reports))),
             '후보 상태: ' + str(dict(Counter(item.get('status', 'pending') for item in candidates)))]
    if reference_turns:
        lines.append(f'기준 대화 평균 턴: {sum(reference_turns) / len(reference_turns):.2f} | 분포: {dict(Counter(reference_turns))}')
    if reports:
        invalid = sum(report.get('status') == 'reflection_invalid' for report in reports)
        lines.append(f'회고 오류로 제외된 그룹: {invalid}/{len(reports)} ({invalid / len(reports):.1%})')
    if total_groups and finished == len(assignments):
        lines.append(f'rubric 있는 그룹 비율: {groups_with_rubric / total_groups:.1%}')
    if scored:
        lines.append(f'채점된 그룹 중 zero advantage 비율: {zero / len(scored):.1%}')
    timing = read(directory / 'timing.json')
    if timing:
        lines.append(f'배치 실제 소요 시간: {timing["batch_seconds"]:.1f}초')
        lines.append(f'배치 시간 × 24: {timing["projected_24_batches_seconds"] / 3600:.2f}시간 (한 배치 기준 추정)')
    if committed:
        update = committed[0]['batch_update']
        lines.append('실제 GRPO: ' + json.dumps({key: update.get(key) for key in (
            'optimizer_step', 'optimizer_steps', 'groups_total', 'groups_used', 'groups_skipped',
            'used_agent_completions', 'skip_reason', 'skip_counts')}, ensure_ascii=False))
    else:
        lines.append('실제 GRPO 결과: 아직 배치가 저장되지 않았습니다.')
    if finished == len(assignments):
        if not kept:
            lines.append('확인 필요: 완료된 배치에서 KEEP rubric이 0개입니다. 후보 상태와 s 분포를 먼저 보세요.')
        elif not scored:
            lines.append('확인 필요: KEEP rubric은 있지만 정상 채점된 그룹이 없습니다. reward_unresolved를 확인하세요.')
        elif not usable:
            lines.append('확인 필요: 채점된 그룹의 advantage가 모두 0이라 optimizer 업데이트 신호가 없습니다.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/group_online.json')
    parser.add_argument('--batch', type=int)
    args = parser.parse_args()
    if args.batch is not None and args.batch < 1:
        parser.error('--batch must be positive')
    from src.online_rl.tree.experiment import load_config
    cfg = load_config(args.config)
    print(inspect(ROOT / cfg['output_dir'], args.batch))


if __name__ == '__main__':
    main()
