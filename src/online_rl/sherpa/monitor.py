"""Read-only progress view for the trajectory pipeline."""
import json
from pathlib import Path


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def snapshot(home, selected_step=None):
    home = Path(home)
    state = read(home / 'run_state.json')
    progress = read(home / 'progress.json')
    step = selected_step if selected_step is not None else progress.get('step') or state.get('completed_steps', 0)
    folder = home / 'steps' / f'step_{step:06d}'
    metrics = read(folder / 'metrics.json')
    cases = []
    for path in sorted(folder.glob('*/record.json')):
        row = read(path)
        generation = row.get('rubric_assessment', {}).get('generation') or {}
        cases.append({'case_id': row.get('case_id'), **row.get('metrics', {}),
                      'rubrics': len(generation.get('rubrics', [])),
                      'leakage_removed_fraction': generation.get('removed_fraction', 0),
                      'reflections': len(row.get('reflections', []))})
    return {'completed_steps': state.get('completed_steps', 0), 'total_steps': 90,
            'step': step, 'iteration': (step - 1) // 30 + 1 if step else 0,
            'stage': progress.get('stage', 'not_started'), 'metrics': metrics,
            'cases': cases, 'complete': state.get('complete', False)}


def render(snapshot):
    metrics = snapshot['metrics']
    lines = [f'SHERPA step {snapshot["completed_steps"]}/90 | iteration {snapshot["iteration"]}/3 | {snapshot["stage"]}',
             f'현재 step {snapshot["step"]}: case 완료 {len(snapshot["cases"])}/4 | 32 trajectory/case']
    if metrics:
        lines.append(f'reward {metrics["reward_mean"]:.4f} ± {metrics["reward_std"]:.4f} | '
                     f'advantage 0 그룹 {metrics["zero_advantage_group_fraction"]:.1%} | '
                     f'KL {metrics.get("kl")} | clip {metrics.get("clip_fraction")}')
    for case in snapshot['cases']:
        value = case.get('auc')
        lines.append(f'{case["case_id"]}: success={case["success_rate"]:.3f} '
                     f'reward={case["reward_mean"]:.3f} AUC={value if value is not None else "undefined"} '
                     f'rubric={case["rubrics"]} reflection={case["reflections"]} '
                     f'누출 제거={case["leakage_removed_fraction"]:.1%}')
    return '\n'.join(lines)
