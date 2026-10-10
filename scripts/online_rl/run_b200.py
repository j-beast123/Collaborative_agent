"""Explicit CPU preflight, train/resume or final-only evaluation. No service management."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from src.online_rl.tree.b200_runner import CONFIG, load_config, preflight, physical_devices, run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--action', choices=('preflight', 'train', 'evaluate'), default='preflight')
    parser.add_argument('--cpu-only', action='store_true', help='Validate scientific config/data/prompts; omit server resources and credentials')
    parser.add_argument('--max-batches', type=int)
    args = parser.parse_args()
    if args.cpu_only and args.action != 'preflight': parser.error('--cpu-only is for preflight only')
    if args.max_batches is not None and (args.action != 'train' or args.max_batches < 1): parser.error('--max-batches is positive and train-only')
    cfg = load_config(args.config)
    errors = preflight(cfg, resources=not args.cpu_only)
    print(json.dumps({'action': args.action, 'blockers': errors, 'output': cfg['output_dir'],
                      'topology': cfg['two_gpu'], 'gpu_mapping': physical_devices() if not errors else None,
                      'train_cases': cfg['schedule']['cases'], 'reward': cfg['agent_reward'],
                      'bank_mode': cfg['rubric'].get('bank_mode'),
                      'candidate_batch_size': cfg['rubric'].get('candidate_batch_size'),
                      'global_api_max_in_flight': cfg['execution'].get('global_api_max_in_flight'),
                      'remote_model_support': 'not live-checked; strict provider parameters, no fallback; fail on rejection',
                      'resource_checks': not args.cpu_only, 'live_work_started': False}, indent=2))
    if errors: raise SystemExit(1)
    if args.action != 'preflight': run(cfg, max_batches=args.max_batches, evaluate=args.action == 'evaluate')


if __name__ == '__main__':
    main()
