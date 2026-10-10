"""Opt-in fresh Luna-generator runs. Does not alter an existing run or launcher."""
import argparse
import json
import sys
from pathlib import Path
from baselines.luna_generator.generator_profile import MODEL, PROFILE, install_profile, source_identity


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('--method', required=True, choices=('rar', 'online_rubrics'))
    p.add_argument('--api-workers', type=int, default=20, choices=range(1,21), metavar='1..20')
    p.add_argument('--grader-workers', type=int, default=100, choices=range(1,101), metavar='1..100')
    args, rest = p.parse_known_args()
    from baselines.parallel_api.launch import install
    from baselines.parallel_api.openai_profile import install_profile as grader_profile
    launcher = install(args.api_workers, args.grader_workers)
    grader_profile(launcher)
    install_profile(launcher)
    if not any(x == '--config' or x.startswith('--config=') for x in rest):
        rest = ['--config', str(Path(__file__).with_name(args.method + '.json')), *rest]
    print(json.dumps({'profile': PROFILE, 'generator_model': MODEL, 'generator_reasoning': 'none',
        'grader': 'gpt-5.6-luna via direct OpenAI (unchanged)',
        'api_workers': args.api_workers, 'grader_workers': args.grader_workers,
        'profile_source': source_identity(), 'live_work_started': False}, indent=2), flush=True)
    sys.argv = [sys.argv[0], *rest]
    launcher.main(args.method)


if __name__ == '__main__':
    main()
