"""Dry-run by default; explicitly launch isolated target-turn GRPO later."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/plan_c.json')
    parser.add_argument('--responses-per-turn', type=int,
                        help='Total siblings INCLUDING the main response (default 32)')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.sherpa.plan_c_config import config_problems
    from src.online_rl.sherpa.plan_c_backend import generator_prompt
    cfg = load_config(args.config)
    if args.responses_per_turn is not None:
        cfg['plan_c']['responses_per_turn'] = args.responses_per_turn
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    problems = config_problems(cfg, ROOT)
    if problems:
        parser.error('; '.join(problems))
    _, prompt_artifact = generator_prompt(cfg, ROOT)
    n, rounds = cfg['plan_c']['responses_per_turn'], cfg['rollout']['rounds']
    print(json.dumps({'method': 'plan_c', 'launch': args.launch,
                      'output_dir': cfg['output_dir'], 'main_dialogues_per_case': 1,
                      'responses_per_turn': n, 'alternatives_per_turn': n - 1,
                      'groups_per_case': f'1..{rounds}, only turns reached by the main dialogue',
                      'max_groups_per_step': rounds * 4, 'max_target_responses_per_step': n * rounds * 4,
                      'training_scope': 'target response only; advantages within identical history',
                      'advantage_normalization': cfg['advantage_normalization'],
                      'exclude_zero_advantage_responses': cfg['advantage_normalization']['mode'] == 'group_mean',
                      'judge_prompt_file': cfg['rubric_pipeline']['judge_prompt_file'],
                      'steps': 90, 'iterations': 3, 'cases_per_step': 4,
                      'assistant_max_inflight': cfg['training_execution']['assistant_max_inflight'],
                      'doctor_max_inflight': cfg['execution']['api_limits']['doctor'],
                      'generator_prompt': prompt_artifact,
                      'checkpoint_every_steps': 10, 'checkpoint_upload': cfg['checkpoint_upload']}, indent=2))
    if not args.launch:
        return
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    # Verify the pinned train-only schedule before reserving GPUs or loading weights.
    from src.online_rl.cases import load_cases
    from src.online_rl.sherpa.rollouts import schedule
    cases = load_cases(ROOT / cfg['data']['train'])
    schedule(cfg, cases, ROOT)
    home = ROOT / cfg['output_dir']
    home.mkdir(parents=True, exist_ok=True)
    training = cfg['training_execution']
    os.environ.update(SHERPA_CALL_CACHE='1', SHERPA_AGENT_REPLICAS='1',
                      SHERPA_AGENT_MAX_INFLIGHT=str(training['assistant_max_inflight']),
                      SHERPA_AGENT_BATCH_SIZE=str(training['rollout_batch_size']),
                      SHERPA_RATE_LIMIT_RETRIES=str(training.get('rate_limit_retries', 0)),
                      API_USAGE_LOG=str(home / 'api_usage_train.jsonl'),
                      API_EVENT_LOG=str(home / 'api_events_train.jsonl'))
    cfg['execution']['agent_workers'] = training['assistant_max_inflight']
    devices = [d.strip() for d in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')]
    if len(devices) != 2 or len(set(devices)) != 2 or any(not d.isdigit() for d in devices):
        parser.error('Expose exactly two distinct GPU indices for Plan C')
    devices = [str(int(d)) for d in devices]
    if len(set(devices)) != 2:
        parser.error('Expose two distinct physical GPUs')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
    with ExitStack() as resources:
        runner_lock = resources.enter_context((home / 'runner.lock').open('a'))
        try:
            fcntl.flock(runner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for device in devices:
                lock = resources.enter_context(Path(f'/tmp/sherpa_tree_physical_gpu{device}.lock').open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('A runner or GPU is occupied. Current training is untouched; launch Plan C later.')
        # Physical locks precede all CUDA initialization and server startup.
        import torch
        if torch.cuda.device_count() != 2:
            parser.error('Plan C requires two visible CUDA GPUs')
        from src.common.api_slots import create, install
        from src.common.live_gpu import configure
        from src.common.progress import configure as configure_progress
        install(create(cfg['execution']['api_limits']))
        configure(home, 0)
        configure_progress(home)
        from scripts.online_rl.run_tree import load_key
        from src.online_rl.runtime import SharedPolicy
        policy = SharedPolicy(cfg, load_key({'api_provider': 'openai_direct'}))
        policy.collect_training_traces = True
        def close_batcher():
            if hasattr(policy, '_batcher'):
                policy._batcher.close()
        resources.callback(close_batcher)
        from src.online_rl.local_doctor import start_server
        server = start_server(cfg, ROOT, home, policy.doctor, devices[1])
        resources.callback(server.close)
        from src.online_rl.sherpa.plan_c_backend import TurnChecklistBackend
        backend = TurnChecklistBackend(cfg, policy, ROOT)
        resources.callback(backend.close)
        from src.online_rl.sherpa.plan_c_training import train
        from src.online_rl.integrated.records import atomic_write_json
        result = train(cfg, policy, backend, cases, ROOT, max_steps=args.max_steps)
        atomic_write_json(home / 'train_result.json', result)


if __name__ == '__main__':
    main()
