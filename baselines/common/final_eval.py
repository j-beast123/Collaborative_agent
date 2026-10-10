"""Use the original evaluator, registering only the completed final checkpoint."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from src.online_rl.integrated.records import atomic_write_json
from src.online_rl.tree.five_gpu import EvaluationQueue


def install_final_only_queue(runner, cfg):
    final_cursor = cfg['schedule']['cases']
    class FinalOnlyQueue(EvaluationQueue):
        def __init__(self, root, cases, settings):
            state = json.loads((Path(cfg['output_dir']) / 'run_state.json').read_text())
            if state['case_cursor'] != final_cursor:
                raise RuntimeError('Test evaluation is forbidden until all training cases are committed')
            super().__init__(root, cases, settings)
            if any(p.name != f'checkpoint_{final_cursor:06d}' for p in self.root.glob('checkpoint_*')):
                raise RuntimeError('Non-final test jobs exist; cannot claim final-checkpoint-only evaluation')
        def register(self, checkpoint, cursor, policy_sha):
            if cursor != final_cursor:
                return None
            if Path(checkpoint).name != f'checkpoint_{final_cursor:06d}':
                raise RuntimeError('Evaluation requires the final checkpoint')
            return super().register(checkpoint, cursor, policy_sha)
    runner.EvaluationQueue = FinalOnlyQueue


def evaluate_completed_run(runner, cfg, *, method_identity):
    out = Path(cfg['output_dir'])
    state = json.loads((out / 'run_state.json').read_text())
    if state['case_cursor'] != cfg['schedule']['cases']:
        return None
    selection = {'method_identity': method_identity, 'checkpoint': state['checkpoint'],
                 'policy_sha256': state['policy_sha256'], 'completed_cases': state['case_cursor'],
                 'test_sha256': cfg['five_gpu']['eval_sha256'], 'test_seed': cfg['five_gpu']['eval_seed'],
                 'selection': 'final_training_checkpoint_no_test_selection'}
    path = out / 'final_evaluation_selection.json'
    if path.exists():
        stored = json.loads(path.read_text())
        if any(stored.get(k) != v for k, v in selection.items()):
            raise RuntimeError('Final evaluation identity changed')
    else:
        selection['selected_at_utc'] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(selection, stream, indent=2)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg['five_gpu']['checkpoint_evaluation_enabled'] = True
    install_final_only_queue(runner, eval_cfg)
    runner.run(eval_cfg, eval_only=True)
    summary_path = out / 'evaluation' / f"checkpoint_{state['case_cursor']:06d}" / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['status'] != 'complete' or summary['evaluated'] != cfg['five_gpu']['eval_cases']:
        raise RuntimeError('Final evaluation is incomplete')
    atomic_write_json(out / 'final_accuracy.json', {**summary, 'selection': json.loads(path.read_text()),
        'metric': 'Doctor FINAL accuracy; original denominator includes unparsed responses'})
    return summary
