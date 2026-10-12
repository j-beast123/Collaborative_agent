"""Read-only reference comparison and persisted learning-signal diagnostics.

    Advantage mass is descriptive, not a decomposition of parameter gradients.
    Alerts never change the learning rate, advantage scale or training decisions.
"""
import csv
import io
from pathlib import Path

from src.online_rl.integrated.records import atomic_write_json, read_json

COMPARABLE_FIELDS = ('policy', 'data', 'rollout', 'optimizer', 'schedule', 'rubric',
                     'judges', 'rubric_pipeline', 'prompt_optimization', 'training_execution', 'method')


def signal_checks(cfg, home, root, step, update, prompt_hash=None):
    settings = cfg.get('learning_signal_monitor', {})
    diagnostics = update.get('advantage_diagnostics', {})
    current = diagnostics.get('all', diagnostics.get('used', {}))
    result = {'mode': cfg.get('advantage_normalization', {}).get('mode', 'group_zscore'),
              'advantage': current, 'same_rewards_group_zscore': diagnostics.get('same_rewards_group_zscore'),
              'by_group_type': diagnostics.get('by_group_type', {}),
              'groups_total': update.get('groups_total'), 'groups_skipped': update.get('groups_skipped'),
              'grad_norm': update.get('grad_norm'), 'kl': update.get('kl_loss'),
              'kl_measured_at': 'before_optimizer_step_against_frozen_base',
              'alerts': [], 'reference': {'status': 'not_configured'}}
    total, skipped = update.get('groups_total', 0), update.get('groups_skipped', 0)
    result['skipped_group_fraction'] = skipped / total if total else None
    reference_path = settings.get('reference_output_dir')
    if reference_path:
        reference = Path(root) / reference_path
        manifest_path = reference / 'run_manifest.json'
        metrics_path = reference / 'steps' / f'step_{step:06d}' / 'metrics.json'
        if manifest_path.exists() and metrics_path.exists():
            manifest = read_json(manifest_path)
            compatible = all(manifest['config'].get(k) == cfg.get(k) for k in COMPARABLE_FIELDS)
            compatible &= prompt_hash is None or manifest.get('frozen_prompt') == prompt_hash
            prompt_difference = False
            if not compatible and settings.get('allow_prompt_difference') is True:
                old_pipeline = manifest['config'].get('rubric_pipeline', {})
                new_pipeline = cfg.get('rubric_pipeline', {})
                omit = lambda p: {k:v for k,v in p.items() if k not in ('generator_prompt_file','judge_prompt_file')}
                prompt_difference = (all(manifest['config'].get(k) == cfg.get(k)
                                         for k in COMPARABLE_FIELDS if k != 'rubric_pipeline')
                                     and omit(old_pipeline) == omit(new_pipeline))
            if compatible or prompt_difference:
                old = read_json(metrics_path)
                prior = old.get('update', {})
                old_adv = prior.get('advantage_diagnostics', {})
                old_grad, grad = prior.get('grad_norm'), update.get('grad_norm')
                ratio = grad / old_grad if grad is not None and old_grad is not None and old_grad > 0 else None
                result['reference'] = {'status': ('different_prompts_descriptive_comparison'
                                                  if prompt_difference else 'comparable'),
                    'prompt_difference': prompt_difference, 'output_dir': str(reference),
                    'mode': manifest['config']['advantage_normalization']['mode'],
                    'step': step, 'grad_norm': old_grad, 'gradient_norm_ratio': ratio,
                    'kl': old.get('kl'), 'advantage': old_adv.get('all', old_adv.get('used', {}))}
                if ratio is not None and ratio < settings.get('gradient_norm_ratio_threshold', .1):
                    result['alerts'].append({'kind': 'gradient_norm_more_than_10x_smaller',
                        'gradient_norm_ratio': ratio, 'threshold': settings.get('gradient_norm_ratio_threshold', .1)})
            else:
                result['reference'] = {'status': 'incompatible_settings_or_prompt', 'output_dir': str(reference)}
        else:
            result['reference'] = {'status': 'matching_reference_step_unavailable', 'output_dir': str(reference)}
    patience = settings.get('kl_patience_steps', 3)
    threshold = settings.get('kl_near_zero_threshold', 1e-8)
    window = []
    for previous_step in range(max(1, step - patience + 1), step + 1):
        if previous_step == step:
            value = update.get('kl_loss') if update.get('optimizer_step') else None
        else:
            path = Path(home) / 'steps' / f'step_{previous_step:06d}' / 'metrics.json'
            prior = read_json(path).get('update', {}) if path.exists() else {}
            value = prior.get('kl_loss') if prior.get('optimizer_step') else None
        window.append(value)
    if (len(window) == patience and all(v is not None and abs(v) <= threshold for v in window)):
        result['alerts'].append({'kind': 'kl_near_zero_for_consecutive_steps',
                                'steps': patience, 'threshold': threshold, 'values': window})
    return result


def save_step_signal(folder, step, signal):
    folder = Path(folder)
    atomic_write_json(folder / 'learning_signal.json', {'step': step, **signal})
    rows = [{'step': step, 'mode': signal['mode'], 'group_type': kind, **values}
            for kind, values in signal.get('by_group_type', {}).items()]
    if rows:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (folder / 'signal_by_group_type.csv').write_text(stream.getvalue())
