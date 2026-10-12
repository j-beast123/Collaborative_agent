"""Plan C validation without loading models, credentials or requesting inference."""
from copy import deepcopy
from pathlib import Path

from src.online_rl.sherpa.config import config_problems as base_problems
from src.online_rl.sherpa.plan_c_rollouts import PROTOCOL


def config_problems(cfg, root):
    # Reuse environment/model/schedule checks; Plan C has its own rollout contract.
    baseline = deepcopy(cfg)
    baseline['rollout'].update(branching=32, budget='initial_fanout')
    problems = base_problems(baseline, root)
    spec = cfg.get('plan_c', {})
    if spec.get('protocol') != PROTOCOL or cfg['rollout'].get('budget') != PROTOCOL:
        problems.append('Plan C requires its isolated counterfactual protocol')
    if cfg['rollout'].get('branching') != 1:
        problems.append('Plan C requires exactly one main dialogue')
    if type(cfg['rollout'].get('rounds')) is not int or cfg['rollout']['rounds'] < 1:
        problems.append('Plan C rounds must be a positive integer')
    if type(spec.get('responses_per_turn')) is not int or spec['responses_per_turn'] < 2:
        problems.append('responses_per_turn must be an integer >= 2 including the main response')
    if spec.get('include_main_response') is not True or spec.get('turn_workers') != 1:
        problems.append('Plan C includes the main action and collects turns sequentially')
    if cfg.get('method') != 'sherpa' or cfg['policy'].get('init_adapter') is not None:
        problems.append('Plan C starts a fresh LoRA experiment with rubric rewards')
    if cfg['rollout'].get('agent_protocol') is not None:
        problems.append('Plan C currently supports visible single-response actions only')
    if not cfg['optimizer'].get('require_raw_on_policy_sampling'):
        problems.append('Plan C requires recorded raw on-policy sampling and teacher-forced alignment')
    if cfg['rollout']['doctor_backend']['engine'] != 'vllm':
        problems.append('Plan C currently requires the frozen local vLLM Doctor')
    if cfg['rubric_pipeline'].get('reflection_system_mode') != 'standalone':
        problems.append('Use a standalone private reflection system, not the discussion prompt')
    path = spec.get('generator_instruction_file')
    if not path or not (Path(root) / path).is_file():
        problems.append('Missing Plan C target-turn generator instructions')
    base_path = spec.get('generator_base_prompt_file')
    if base_path is not None and (not isinstance(base_path, str) or not base_path
                                 or not (Path(root) / base_path).is_file()):
        problems.append('Missing Plan C generator base prompt file')
    home = (Path(root) / cfg['output_dir']).resolve()
    # Protect the active experiment and transferred source prompt artifacts.
    from src.online_rl.tree.experiment import load_config
    ours = load_config('configs/ours.json')
    protected = [(Path(root) / ours['output_dir']).resolve(),
                 (Path(root) / cfg['prompt_optimization']['output_dir']).resolve()]
    if any(home == p or home in p.parents or p in home.parents for p in protected):
        problems.append('Plan C output must not overlap current training or prompt-search outputs')
    if cfg['checkpoint_upload'].get('path_prefix') == ours['checkpoint_upload'].get('path_prefix'):
        problems.append('Plan C needs an independent HF checkpoint path prefix')
    return problems
