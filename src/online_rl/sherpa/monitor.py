"""Read-only training dashboard; bounded incremental reads never acquire training locks."""
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def tail_field(path, field):
    """Read trailing summary fields without loading token traces in large records."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 65536))
            text = stream.read().decode('utf-8', errors='replace')
        marker = json.dumps(field) + ': '
        index = text.rfind(marker)
        return json.JSONDecoder().raw_decode(text[index + len(marker):])[0] if index >= 0 else None
    except (OSError, ValueError):
        return None


def mtime(path):
    try: return Path(path).stat().st_mtime
    except OSError: return None


def duration(seconds):
    if seconds is None: return '-'
    seconds = max(0, int(seconds))
    return f'{seconds // 3600}h {seconds % 3600 // 60:02d}m' if seconds >= 3600 else f'{seconds // 60}m {seconds % 60:02d}s'


def bar(value, total, width=22):
    filled = min(width, int(width * value / max(1, total)))
    return '[' + '#' * filled + '.' * (width - filled) + ']'


def spark(values):
    values = [v for v in values if isinstance(v, (float, int))][-30:]
    if not values: return '-'
    low, high = min(values), max(values)
    return ''.join(' .:-=+*#'[int(7 * (v - low) / (high - low)) if high > low else 3] for v in values)


class TrainingMonitor:
    def __init__(self):
        self.files = {}
        self.gpu_checked = 0
        self.gpus = []
        self.summaries = {}

    def calls(self, path):
        try: stat = path.stat()
        except OSError: return {'counts': Counter(), 'generation_seconds': None, 'remaining_bytes': 0}
        state = self.files.get(str(path))
        if state is None or stat.st_ino != state['inode'] or stat.st_size < state['offset']:
            state = {'inode': stat.st_ino, 'offset': 0, 'seen': set(), 'counts': Counter(), 'generation_seconds': None}
            self.files[str(path)] = state
        # Four current cases read at most ~4 MiB each per refresh. Token-array
        # values are skipped entirely except for the single rubric-set response.
        with path.open('rb') as stream:
            stream.seek(state['offset']); budget = stream.tell() + 4 * 1024 * 1024
            while stream.tell() < budget:
                start = stream.tell(); line = stream.readline()
                if not line or not line.endswith(b'\n'): state['offset'] = start; break
                state['offset'] = stream.tell()
                head = line[:16384].decode('utf-8', errors='replace')
                key = re.search(r'"key":\s*"([^"]+)"', head)
                kind = re.search(r'"kind":\s*"([^"]+)"', head)
                if not key or not kind or key[1] in state['seen']: continue
                state['seen'].add(key[1]); state['counts'][kind[1]] += 1
                if kind[1] == 'rubric_set':
                    try:
                        generation = json.loads(line)['value']
                        state['generation_seconds'] = generation.get('seconds')
                    except (ValueError, KeyError): pass
        return {**state, 'remaining_bytes': max(0, stat.st_size - state['offset'])}

    def summary(self, path):
        try: signature = (path.stat().st_mtime_ns, path.stat().st_size)
        except OSError: return {}
        old = self.summaries.get(str(path))
        if old is None or old[0] != signature:
            old = (signature, read(path)); self.summaries[str(path)] = old
        return old[1]

    def snapshot(self, home, selected_step=None, *, now=None, gpu_stats=False):
        home = Path(home); now = time.time() if now is None else now
        state = read(home / 'run_state.json'); progress = read(home / 'progress.json')
        step = selected_step if selected_step is not None else progress.get('step') or state.get('completed_steps', 0)
        folder = home / 'steps' / f'step_{step:06d}'
        started = progress.get('batch_started_at') if step == progress.get('step') else None
        cases = []
        for case in sorted(p for p in folder.glob('*') if p.is_dir()):
            calls = self.calls(case / 'calls.jsonl'); counts = calls['counts']
            rolled = mtime(case / 'rollouts.json'); reflected = mtime(case / 'reflections.json')
            scored = mtime(case / 'rubric_scores.json'); recorded = mtime(case / 'record.json')
            outcomes = tail_field(case / 'rollouts.json', 'outcomes') or []
            positive = sum(outcomes); negative = len(outcomes) - positive
            reflection_target = min(4, positive) + min(4, negative) if positive and negative else min(4, len(outcomes))
            complete = 32 if rolled or recorded else min(32, counts['final'])
            phase = ('Complete' if recorded else 'Scoring' if counts['rubric_set'] else 'Rubric generation' if reflected
                     else 'Doctor reflection' if rolled else 'rollout')
            case_metrics = tail_field(case / 'record.json', 'metrics') or {}
            generation_seconds = calls['generation_seconds']
            judge_start = reflected + generation_seconds if reflected is not None and generation_seconds is not None else None
            cases.append({'case_id': case.name, **case_metrics, 'phase': phase,
                          'rollouts_complete': complete, 'rollouts_total': 32,
                          'assistant_turns': counts['agent'], 'doctor_turns': counts['doctor_turn'],
                          'rollout_seconds': max(0, (rolled or now) - started) if started else None,
                          'rollout_done': bool(rolled or recorded),
                          'reflections': reflection_target if reflected else counts['doctor_reflection'],
                          'reflection_target': reflection_target or None,
                          'reflection_seconds_estimate': max(0, (reflected or now) - rolled) if rolled else None,
                          'generation_seconds': generation_seconds,
                          'generation_elapsed_estimate': max(0, now - reflected) if reflected and generation_seconds is None else None,
                          'judged_trajectories': 32 if scored or recorded else min(32, counts['trajectory_score']),
                          'judge_seconds_estimate': max(0, (scored or recorded or now) - judge_start) if judge_start else None,
                          'cache_bytes_pending': calls['remaining_bytes']})
        history = [self.summary(p) for p in sorted((home / 'steps').glob('step_*/metrics.json'))]
        history = [r for r in history if r]
        metrics = read(folder / 'metrics.json')
        latest = metrics or (history[-1] if history else {})
        completed = state.get('completed_steps', 0)
        times = [r['seconds'] for r in history if isinstance(r.get('seconds'), (int, float))]
        average = sum(times[-10:]) / len(times[-10:]) if times else None
        live = [read(p) for p in (home / 'live').glob('gpu*.json')]
        active = Counter()
        errors = []
        for row in live:
            if isinstance(row.get('pid'), int) and Path(f'/proc/{row["pid"]}').exists(): active.update(row.get('active') or {})
            if row.get('last_error'): errors.append(str(row['last_error']))
        running = isinstance(progress.get('pid'), int) and Path(f'/proc/{progress["pid"]}').exists()
        uploads = Counter(read(p).get('status', 'pending') for p in (home / 'hf_uploads/jobs').glob('*.json'))
        if gpu_stats and now - self.gpu_checked >= 5:
            self.gpu_checked = now
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                self.gpus = result.stdout.splitlines() if result.returncode == 0 else []
            except (OSError, subprocess.TimeoutExpired): self.gpus = []
        return {'completed_steps': completed, 'total_steps': 90, 'step': step,
                'iteration': (step - 1) // 30 + 1 if step else 0, 'stage': progress.get('stage', 'not_started'),
                'running': running, 'complete': state.get('complete', False), 'cases': cases,
                'metrics': latest, 'metrics_step': latest.get('step'), 'history': history,
                'step_seconds': now - started if started else None,
                'run_seconds': now - progress['run_started_at'] if progress.get('run_started_at') else None,
                'average_step_seconds': average, 'eta_seconds': average * max(0, 90 - completed) if average else None,
                'activity': dict(active), 'gpu_rows': self.gpus, 'uploads': dict(uploads), 'errors': errors, 'output': str(home)}


def snapshot(home, selected_step=None):
    return TrainingMonitor().snapshot(home, selected_step)


def render(state):
    done = state['completed_steps']; cases = state['cases']; total = sum(c['rollouts_complete'] for c in cases)
    status = 'Complete' if state['complete'] else 'Running' if state.get('running') else 'Stopped / waiting'
    lines = [f'OURS | {status} | iteration {state["iteration"]}/3 | STEP {done}/90',
             f'{bar(done, 90)} {done / 90:.1%}  | Runtime {duration(state.get("run_seconds"))} | ETA ~ {duration(state.get("eta_seconds"))}',
             f'Current step {state["step"]} | {state["stage"]} | Elapsed {duration(state.get("step_seconds"))}',
             f'ROLLOUT {bar(total, 128)} Complete {total}/128 | Pending {128 - total} | Ready cases {sum(c["phase"] == "Complete" for c in cases)}/4', '']
    for index, case in enumerate(cases, 1):
        label = case['case_id'].replace('medmcqa_', '')
        lines.append(f'{index:02d} {label[:8]}...{label[-8:]}  [{case["phase"]}]  rollout {bar(case["rollouts_complete"], 32, 12)} {case["rollouts_complete"]}/32')
        lines.append(f'   rollout ~ {duration(case["rollout_seconds"])} | Reflection {case["reflections"]}/{case["reflection_target"] or "?"} ~ {duration(case["reflection_seconds_estimate"])} | rubric {duration(case["generation_seconds"])}'
                     + (f' (Elapsed ~ {duration(case["generation_elapsed_estimate"])})' if case['generation_elapsed_estimate'] is not None else ''))
        lines.append(f'   Scoring {case["judged_trajectories"]}/32 ~ {duration(case["judge_seconds_estimate"])} | Assistant turns {case["assistant_turns"]} | Doctor turns {case["doctor_turns"]}')
        if case.get('reward_mean') is not None:
            auc = case.get('auc'); lines.append(f'   success {case.get("success_rate", 0):.1%} | reward {case["reward_mean"]:.3f} | AUC {auc if auc is not None else "-"}')
        if case['cache_bytes_pending']: lines.append('   Reading saved records (counts continue next refresh)')
    metrics = state['metrics']
    if metrics:
        lines += ['', f'Latest training metrics | step {state.get("metrics_step")}',
                  f'reward {metrics.get("reward_mean", 0):.4f} +/- {metrics.get("reward_std", 0):.4f} | KL {metrics.get("kl")} | clip {metrics.get("clip_fraction")} | advantage 0 {metrics.get("zero_advantage_group_fraction", 0):.1%}',
                  f'Reward trend {spark([r.get("reward_mean") for r in state.get("history", [])])} | Average step {duration(state.get("average_step_seconds"))}']
    activity = state.get('activity', {})
    updating = state.get('running') and state['stage'] == 'agent_update'
    lines += ['', f'Assistant requests (including queued) {activity.get("cache.agent", 0)} | Doctor requests {activity.get("local_doctor_generate", 0)} | GRPO {"Computing" if updating else "Waiting"}']
    if activity: lines.append('Activity: ' + ' | '.join(f'{k}={v}' for k, v in sorted(activity.items()) if v))
    if state.get('gpu_rows'):
        lines.append('GPU | index / utilization % / used MiB / total MiB'); lines.extend('  ' + r for r in state['gpu_rows'])
    lines.append('HF uploads: ' + (', '.join(f'{k} {v}' for k, v in state.get('uploads', {}).items()) or 'Waiting for first 10-step checkpoint'))
    if state.get('errors'): lines.append('Latest error: ' + ' | '.join(state['errors'])[:250])
    lines += ['', 'Per case: 32 rollouts -> reflection -> rubrics -> 32 scores (cases overlap)',
              'Pending includes running and queued work. ~ times use file timestamps.',
              'Counts use saved records. GPU numbers include all server processes.',
              'Ctrl+C stops only the monitor.', f'Output: {state.get("output", "")}']
    return '\n'.join(lines)
