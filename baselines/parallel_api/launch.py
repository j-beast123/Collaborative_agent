"""Opt-in API concurrency adapter; retains original scientific/resume/cache identities.

Run this entrypoint only after the previous launcher has exited. Original GPU and
runner locks remain enforced. Nothing is installed into running Python processes.
"""
import argparse
import functools
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from baselines.common.runner import source_hash
from baselines.parallel_api.io import append_json

BASE_SOURCE = '4787fc56ebb99afeffcf62cb6b890c5857af3cefcc67db3948d93e126643a745'


def execution_identity(workers, grader_workers=None):
    root = Path(__file__).parent
    files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py')) if not p.name.startswith('test_')}
    return {'execution':'phased_reward_api_threads_v1', 'generator_workers_per_method':workers, 'grader_workers_per_method':grader_workers or workers,
        'base_scientific_source_sha256':BASE_SOURCE, 'execution_source_sha256':files,
        'rollout_barrier':'original five frozen-policy cases, all completed before rewards',
        'reference_execution':'serial original group order, frozen initial adapter on original Assistant GPU',
        'api_limit_scope':'separate generator/grader phases; per method process',
        'cache_identity':'unchanged; existing successful results reused',
        'provider_reproducibility':'Request completion order changes; unseedable remote sampling can differ on cache misses'}


def install(workers, grader_workers=None):
    if source_hash() != BASE_SOURCE:
        raise RuntimeError('Base source changed; review this execution adapter before running')
    from baselines.common import rubric_launch
    from baselines.parallel_api.engine import ParallelRubricEngine
    from baselines.parallel_api.backend import ParallelAuditedBackend
    original_hook = rubric_launch.RubricHook
    class ExecutionHook(original_hook):
        def check_pfinal_source(self, cfg, theta0, protocol):
            super().check_pfinal_source(cfg, theta0, protocol)
            if not getattr(self, '_parallel_recorded', False):
                # Called inside the existing GPU ownership + runner locks, after
                # initial weight verification. Never writes run state/checkpoints.
                state = Path(cfg['output_dir'])/'run_state.json'
                cursor = json.loads(state.read_text())['case_cursor'] if state.exists() else 0
                append_json(Path(cfg['output_dir'])/'api_execution_history.jsonl', {
                    **execution_identity(workers, grader_workers), 'utc':datetime.now(timezone.utc).isoformat(),
                    'method':cfg['comparison_method']['method'], 'experiment_fingerprint':self.fingerprint,
                    'committed_cases_at_start':cursor})
                self._parallel_recorded = True
    rubric_launch.RubricEngine = functools.partial(ParallelRubricEngine, api_workers=workers, grader_workers=grader_workers)
    rubric_launch.AuditedBackend = ParallelAuditedBackend
    rubric_launch.RubricHook = ExecutionHook
    return rubric_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--method', required=True, choices=('rar','online_rubrics'))
    parser.add_argument('--api-workers', type=int, default=20, choices=range(1,21), metavar='1..20')
    parser.add_argument('--grader-workers', type=int, default=None, choices=range(1,101), metavar='1..100')
    parser.add_argument('--openai-grader', action='store_true')
    args, rest = parser.parse_known_args()
    if args.grader_workers and args.grader_workers > 20 and not args.openai_grader:
        parser.error('OpenRouter mode is limited to the approved 20 requests per method')
    grader_workers = args.grader_workers or (100 if args.openai_grader else args.api_workers)
    launcher = install(args.api_workers, grader_workers)
    if args.openai_grader:
        from baselines.parallel_api.openai_profile import install_profile
        install_profile(launcher)
        if '--config' not in rest:
            rest = ['--config', f'baselines/parallel_api/{args.method}_openai.json', *rest]
    print(json.dumps({'api_execution':execution_identity(args.api_workers, grader_workers), 'live_work_started':False}, indent=2), flush=True)
    sys.argv = [sys.argv[0], *rest]
    launcher.main(args.method)


if __name__ == '__main__':
    main()
