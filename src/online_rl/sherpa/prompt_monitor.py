"""Read-only, incremental monitoring of existing offline prompt-search artifacts."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class Tail:
    def __init__(self):
        self.files = {}

    def scan(self, path, consume, reset):
        try:
            stat = path.stat()
            previous = self.files.get(str(path))
            if previous is None or previous[0] != stat.st_ino or stat.st_size < previous[1]:
                reset()
                previous = (stat.st_ino, 0)
            with path.open('rb') as stream:
                stream.seek(previous[1])
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b'\n'):
                        self.files[str(path)] = (stat.st_ino, offset)
                        break
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict): consume(row)
                    except (ValueError, TypeError, KeyError):
                        continue
        except OSError:
            return


class OptimizationMonitor:
    def __init__(self):
        from src.common.rollout_monitor import RolloutCounters
        self.rollouts = RolloutCounters()
        self.tail = Tail()
        self.json_cache = {}
        self.groups = defaultdict(lambda: {'rubric_cases': set(), 'judged': set()})
        self.reflections = set()
        self.usage = {}

    def summary(self, path, project):
        try:
            signature = (path.stat().st_mtime_ns, path.stat().st_size)
        except OSError:
            return {}
        cached = self.json_cache.get(str(path))
        if cached is None or cached[0] != signature:
            cached = (signature, project(read(path)))
            self.json_cache[str(path)] = cached
        return cached[1]

    def cache_row(self, row):
        ident = row.get('identity') or {}
        value = row.get('value') or {}
        kind = row.get('kind')
        if kind == 'doctor_reflection':
            self.reflections.add(row.get('key'))
        if kind not in ('rubric_set', 'trajectory_score'):
            return
        group = self.groups[(ident.get('scope'), ident.get('candidate'))]
        if kind == 'rubric_set':
            group['rubric_cases'].add(ident.get('case_id'))
        else:
            group['judged'].add(ident.get('trajectory') or value.get('trajectory_id'))

    def snapshot(self, out, cfg, *, now=None, gpu_stats=False):
        out = Path(out)
        now = time.time() if now is None else now
        spec = cfg['prompt_optimization']
        total = cfg['schedule']['cases']
        branches = cfg['rollout']['branching']
        rows = []
        for folder in sorted((out / 'base_rollouts').glob('*')):
            saved = self.summary(folder / 'rollouts.json', lambda r: {
                'done': len(r.get('trajectories', [])) == branches,
                'mixed': r.get('mixed', False), 'success_rate': r.get('success_rate')})
            # Finished cases may have very large token-trace caches; their
            # persisted rollout summary already supplies the FINAL count.
            counts = ({'agent_nodes': 0, 'doctor_turns': 0, 'trajectories': branches}
                      if saved.get('done') else self.rollouts.read(folder / 'calls.jsonl'))
            rows.append({'case_id': folder.name, **counts, **saved})
        completed = sum(bool(row.get('done')) for row in rows)
        trajectories = sum(branches if row.get('done') else row['trajectories'] for row in rows)
        manifest = read(out / 'base_manifest.json')
        split = read(out / 'optimization_split.json')
        def reset():
            self.groups.clear()
            self.reflections.clear()
        self.tail.scan(out / 'optimization_calls.jsonl', self.cache_row, reset)

        labels = defaultdict(list)
        initial = Path(cfg.get('_root', '.')) / spec['initial_prompt_file']
        try: labels[hashlib.sha256(initial.read_bytes()).hexdigest()].append('initial')
        except OSError: pass
        for path in sorted((out / 'proposals').glob('*.json')):
            text = read(path).get('text')
            if isinstance(text, str): labels[hashlib.sha256(text.encode()).hexdigest()].append(path.stem)
        candidates = []
        for folder in sorted((out / 'candidates').glob('*')):
            if not folder.is_dir(): continue
            score = self.summary(folder / 'score.json', lambda r: {
                'mean_auc': r.get('mean_auc'), 'candidate_id': r.get('candidate_id'),
                'invalid_cases': sum(not case.get('valid', False) for case in r.get('case_scores', []))})
            finished = sum(bool(self.summary(path, lambda r: {'done': 'fitness_auc' in r}).get('done'))
                           for path in (folder / 'cases').glob('*.json'))
            counts = self.groups[('offline_prompt', folder.name)]
            candidates.append({'prompt_sha256': folder.name,
                               'labels': labels.get(folder.name) or [score.get('candidate_id') or folder.name[:10]],
                               'completed_cases': finished, 'rubric_cases': len(counts['rubric_cases']),
                               'judged_trajectories': len(counts['judged']), **score})
        candidates.sort(key=lambda row: (row['labels'][0] != 'initial', row['labels'][0]))
        history = self.summary(out / 'history.json', lambda r: {
            'completed_iterations': r.get('completed_iterations', 0), 'winner': r.get('winner')})
        final = self.summary(out / 'final_prompt.json', lambda r: {
            'complete': r.get('schema') == 'sherpa_auc_prompt_v1', 'mean_auc': r.get('mean_auc')})
        confirmation = read(out / 'confirmation.json')
        confirming = [{'scope': scope, 'prompt_sha256': digest, 'labels': labels.get(digest, [str(digest)[:10]]),
                       'rubric_cases': len(counts['rubric_cases']), 'judged_trajectories': len(counts['judged'])}
                      for (scope, digest), counts in self.groups.items()
                      if scope and scope.startswith('offline_prompt_confirmation_')]
        live = [read(path) for path in (out / 'live').glob('gpu*.json')]
        running = any(type(row.get('pid')) is int and Path(f'/proc/{row["pid"]}').exists() for row in live)
        active = Counter()
        for row in live:
            if type(row.get('pid')) is int and Path(f'/proc/{row["pid"]}').exists():
                active.update(row.get('active') or {})
        if final.get('complete'): stage = '최적화 완료'
        elif confirming: stage = '초기·최종 프롬프트 재평가'
        elif candidates: stage = '프롬프트 후보 평가'
        elif split.get('complete'): stage = 'Doctor reflection 준비'
        elif completed == total or manifest.get('complete'): stage = 'mixed case 선택 / 최적화 준비'
        elif rows or live: stage = 'Base rollout 수집 / 모델 준비'
        else: stage = '시작 대기'
        usage = Counter()
        for path in out.glob('api_usage_*.jsonl'):
            key = str(path)
            def reset_usage(key=key): self.usage[key] = Counter()
            def add_usage(row, key=key):
                counts = self.usage[key]
                counts['calls'] += 1
                for name in ('input_tokens', 'output_tokens', 'retry_count'):
                    counts[name] += row.get(name) or 0
                counts['cost_usd'] += row.get('cost_usd') or 0
                counts['unpriced'] += row.get('cost_usd') is None
            self.tail.scan(path, add_usage, reset_usage)
            usage.update(self.usage.get(key, {}))
        gpu_rows = []
        if gpu_stats:
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    gpu_rows = [line.strip() for line in result.stdout.splitlines()]
            except (OSError, subprocess.TimeoutExpired): pass
        return {'out': str(out), 'now': now, 'stage': stage, 'running': running, 'complete': final.get('complete', False),
                'base_cases': completed, 'total_cases': total, 'base_trajectories': trajectories,
                'total_trajectories': total * branches, 'mixed_cases': sum(bool(row.get('mixed')) for row in rows),
                'selected_cases': len(split.get('case_ids', [])), 'required_mixed_cases': spec['mixed_cases'],
                'active_cases': [row for row in rows if not row.get('done')], 'reflections': len(self.reflections),
                'candidates': candidates, 'history': history, 'iterations': spec['iterations'],
                'confirmation': confirmation, 'confirmation_progress': confirming, 'final': final,
                'activity': dict(active), 'api_usage': dict(usage), 'gpu_rows': gpu_rows,
                'last_errors': [row['last_error'] for row in live if row.get('last_error')]}


def render(state, *, details=False):
    status = '완료' if state['complete'] else ('실행 중' if state['running'] else '실행 프로세스 미확인')
    lines = [f'SHERPA PROMPT OPTIMIZATION | {datetime.fromtimestamp(state["now"]):%Y-%m-%d %H:%M:%S}',
             f'현재 단계: {state["stage"]} | {status}',
             f'Base case {state["base_cases"]}/{state["total_cases"]} | FINAL trajectory {state["base_trajectories"]}/{state["total_trajectories"]}',
             f'mixed case {state["mixed_cases"]} (필요 {state["required_mixed_cases"]}) | 선택 {state["selected_cases"]} | reflection 완료 {state["reflections"]}',
             f'탐색 iteration {state["history"].get("completed_iterations", 0)}/{state["iterations"]}', '']
    for case in state['active_cases'][:(120 if details else 8)]:
        lines.append(f'  {case["case_id"]}: FINAL {case["trajectories"]} | assistant {case["agent_nodes"]} | doctor {case["doctor_turns"]}')
    if state['candidates']: lines.append('후보 | case 완료 | rubric 생성 case | trajectory 채점 | 평균 AUC')
    for row in state['candidates']:
        auc = '-' if row.get('mean_auc') is None else f'{row["mean_auc"]:.4f}'
        winner = ' [현재 winner]' if row['prompt_sha256'] == state['history'].get('winner') else ''
        lines.append(f'  {", ".join(row["labels"])}{winner} | {row["completed_cases"]}/{state["required_mixed_cases"]} | {row["rubric_cases"]} | {row["judged_trajectories"]} | {auc}')
    for row in state['confirmation_progress']:
        lines.append(f'재평가 {row["scope"].rsplit("_", 1)[-1]} {row["prompt_sha256"][:10]}: rubric {row["rubric_cases"]} | 채점 {row["judged_trajectories"]}')
    for row in state['confirmation'].get('repeats', []):
        lines.append(f'재평가 #{row["repeat"]}: initial {row["initial_mean_auc"]:.4f} → final {row["final_mean_auc"]:.4f} | ΔAUC {row["delta_mean_auc"]:+.4f}')
    usage = state['api_usage']
    lines.append(f'API 완료 {usage.get("calls", 0)} | 입력/출력 토큰 {usage.get("input_tokens", 0)}/{usage.get("output_tokens", 0)} | 재시도 {usage.get("retry_count", 0)}')
    if usage: lines.append(f'기록된 비용 ${usage.get("cost_usd", 0):.4f} (가격 미기록 {usage.get("unpriced", 0)}건 제외)')
    if state['activity']: lines.append('현재 작업: ' + ' | '.join(f'{key} ×{value}' for key, value in sorted(state['activity'].items()) if value))
    if state['gpu_rows']:
        lines.append('서버 GPU 전체 사용량: index, GPU %, 사용 MiB, 전체 MiB')
        lines.extend('  ' + row for row in state['gpu_rows'])
    if state['last_errors']: lines.append('마지막 기록 오류: ' + ', '.join(state['last_errors']))
    lines.extend(['', '재평가는 동일 최적화 case를 재사용하며 독립 테스트 성능이 아닙니다.',
                  f'출력: {state["out"]}', 'Ctrl+C: 모니터만 종료 | --details 상세 | --once 한 번 | --json JSON'])
    return '\n'.join(lines)
