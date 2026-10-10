"""Read-only progress summary; no imports of model or API clients."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.online_rl.tree.b200_runner import load_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default=load_config()['output_dir'])
args = parser.parse_args()
root = Path(args.output)
def read(path):
    return json.loads(path.read_text()) if path.is_file() else None
state = read(root / 'run_state.json') or {}
diags = sorted((root / 'diagnostics').glob('batch_*.json'))
last = read(diags[-1]) if diags else {}
print(json.dumps({'output': str(root), 'completed_cases': state.get('case_cursor', 0),
                  'completed_batches': state.get('batch_cursor', 0), 'total_cases': 120, 'total_batches': 24,
                  'checkpoint': state.get('checkpoint'),
                  'last_batch_optimizer': last.get('optimizer_update'),
                  'last_batch_rates': last.get('rates'), 'last_batch_needs': last.get('needs'),
                  'last_batch_accuracy': last.get('accuracy_score_distribution'),
                  'last_batch_partner': last.get('partner_score_distribution'),
                  'last_batch_verification': last.get('verification_decisions'),
                  'final_accuracy': read(root / 'final_accuracy.json')}, indent=2))
