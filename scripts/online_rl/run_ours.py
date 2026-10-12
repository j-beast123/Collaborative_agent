"""Prepare ours independently of the running prompt search; launch only explicitly."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ours.json')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.sherpa.config import config_problems
    from src.online_rl.sherpa.rollouts import schedule
    from src.online_rl.cases import load_cases
    from src.online_rl.sherpa.prompt_search import frozen_prompt
    cfg = load_config(args.config)
    problems = config_problems(cfg, ROOT)
    if cfg.get('method') != 'sherpa':
        problems.append('ours requires the rubric reward (method=sherpa)')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    try:
        batches = schedule(cfg, load_cases(ROOT / cfg['data']['train']), ROOT)
    except Exception as exc:
        problems.append(str(exc))
        batches = []
    artifact_path = ROOT / cfg['prompt_optimization']['final_artifact']
    from src.common.checkpoint_hub import credentials, hf_token
    upload = cfg.get('checkpoint_upload', {})
    repo = upload.get('repo_id') or credentials().get('HF_REPO_ID')
    upload_ready = bool(repo and hf_token()) if upload.get('enabled') else True
    if args.launch and not upload_ready:
        problems.append('Fill HF_REPO_ID (existing model repository) and HF_TOKEN in .env for ten-step uploads')
    prompt_status = 'pending optimization'
    prompt_file = str(artifact_path)
    prompt_source = None
    if artifact_path.exists():
        try:
            _, artifact = frozen_prompt(cfg, ROOT)
            prompt_source = artifact.get('prompt_source', 'auc_optimized')
            prompt_status = 'ready: ' + prompt_source + ': ' + artifact['prompt_sha256']
            prompt_file = artifact['prompt_file']
        except Exception as exc:
            problems.append(str(exc))
            prompt_status = 'invalid'
    elif args.launch:
        problems.append('Prompt optimization is not complete: final_prompt.json is required. No initial/intermediate prompt fallback.')
    print(json.dumps({'method': 'ours', 'output_dir': cfg['output_dir'],
        'steps': len(batches), 'iterations': 3, 'steps_per_iteration': 30,
        'cases_per_step': 4, 'rollouts_per_case': 32, 'rollouts_per_step': 128,
        'temperature': cfg['rollout']['temperature'],
        'doctor': cfg['rollout']['doctor_model'], 'reward': 'mean binary rubric score',
        'advantage_normalization': cfg['advantage_normalization'],
        'advantage_grouping': 'within each case, 32 trajectories',
        'exclude_zero_advantage_trajectories': cfg['advantage_normalization']['mode'] == 'group_mean',
        'frozen_prompt': prompt_file, 'prompt_status': prompt_status,
        'prompt_source': prompt_source, 'source_optimized_artifact': str(artifact_path),
        'judge_prompt_file': cfg['rubric_pipeline']['judge_prompt_file'],
        'execution': cfg['training_execution'], 'checkpoint_every_steps': 10,
        'checkpoint_upload': {'enabled': upload.get('enabled', False), 'ready': upload_ready, 'repo_id': repo},
        'problems': problems, 'launch': args.launch}, indent=2), flush=True)
    if problems:
        raise SystemExit(1)
    if not args.launch:
        return
    from scripts.online_rl.run_sherpa import main as run
    command = ['--config', args.config, '--phase', 'train', '--launch']
    if args.max_steps is not None:
        command += ['--max-steps', str(args.max_steps)]
    run(command)


if __name__ == '__main__':
    main()
