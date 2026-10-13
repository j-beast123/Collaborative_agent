"""Launch SHERPA training for one condition. Without --launch, or with any blocker, it only prints the blockers (and the
preflight when there are none). Resumes automatically from <output_dir>/<condition>/run_state.json."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.online_rl.integrated.conditions import condition_config, unresolved

DEFAULT_CONFIG = 'configs/online_rl/integrated_v1.json'


def file_sha(rel: str) -> str: return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def blockers(cfg: dict) -> list[str]:
    out = unresolved(cfg); rc, pu = cfg['rubric'], cfg['prompt_update']
    if rc['prompt_sha256'] not in ('TBD', file_sha(rc['prompt_file'])): out.append('P0 file differs from rubric.prompt_sha256')
    if rc.get('approval_file') and json.loads((ROOT / rc['approval_file']).read_text()).get('prompt_sha256') != file_sha(rc['prompt_file']):
        out.append('P0 hash does not match the approval record')
    for k in ('feedback', 'refiner'):
        if pu[f'{k}_prompt_sha256'] not in ('TBD', file_sha(pu[f'{k}_prompt_file'])): out.append(f'{k} prompt file differs from its approved hash')
    if not (ROOT / cfg['data']['train']).exists(): out.append(f"training split missing: {cfg['data']['train']}")
    elif not file_sha(cfg['data']['train']).startswith(cfg['data']['train_sha256']): out.append('training split hash mismatch')
    return out


def preflight(cfg: dict, config_path: str, max_steps) -> dict:
    """Settings, model roles, protected-data check and an upper-bound call estimate for max_steps steps."""
    from src.online_rl.cases import load_cases
    from src.online_rl.guards import assert_not_protected
    from src.online_rl.integrated.conditions import prompt_mode
    from src.online_rl.integrated.rubric import STATE_SCHEMA, STATE_SCHEMA_SHA256
    from src.prompt_opt import harness as H
    ro, rc, pu, se = cfg['rollout'], cfg['rubric'], cfg['prompt_update'], cfg['state_elicitation']
    cases = load_cases(ROOT / cfg['data']['train']); assert_not_protected(cases, 'integrated training cases')
    steps = max_steps or cfg['schedule']['stages'] * cfg['schedule']['steps_per_stage']; G, K, R = ro['group_size'], ro['outcome_continuations_K'], ro['rounds']
    conts = G * K + (K if cfg['effectiveness']['baseline_mode'] != 'off' else 0)
    return {'config': config_path, 'output_dir': cfg['output_dir'], 'condition': cfg['condition'], 'prompt_mode': prompt_mode(cfg),
            'train_cases': f"{cfg['data']['train']} ({len(cases)} cases, protected-data guard passed: ids + normalized-question sha256)",
            'models': {'doctor': ro['doctor_model'], 'agent_policy': cfg['policy']['base_model'] + ' + LoRA (current policy)',
                       'state_reconstruction': se['reconstructor_model'], 'state_q1q2_judgment': 'current policy, greedy (turn >= 2)',
                       'rubric_generator': H.SOL, 'gates_and_R_A': H.JEV['model']},
            'G': G, 'K': K, 'advantage_mode': cfg['advantage_normalization']['mode'], 'w_O': cfg['reward']['w_O'], 'w_A': cfg['reward']['w_A'],
            'learning_rate': cfg['optimizer']['learning_rate'], 'P0': {'file': rc['prompt_file'], 'sha256': file_sha(rc['prompt_file'])},
            'state_schema': {'version': STATE_SCHEMA['version'], 'sha256': STATE_SCHEMA_SHA256},
            'feedback_prompt_sha256': file_sha(pu['feedback_prompt_file']), 'refiner_prompt_sha256': file_sha(pu['refiner_prompt_file']),
            'baseline_mode': cfg['effectiveness']['baseline_mode'], 'prompt_evolution_expected': bool(pu['enabled'] and steps >= pu['every_n_steps']),
            'upper_bound_calls_per_step': {'doctor_api': (R + 1) + conts * (R + 1), 'reconstruction_api': 3, 'sol_generator': rc['max_attempts'],
                                           'jev': rc['max_attempts'] + G, 'local_agent_generations': R + 1 + G + conts * R + K},
            'steps': steps,
            'artifacts': ['run_manifest.json', 'run_state.json', 'checkpoints/step_000000', 'checkpoints/step_000001 (if the policy updates)',
                          'evidence/step_000000.json', 'continuations.jsonl', 'train_log.jsonl', 'prompts/<sha>.txt']}


def build(cfg: dict):
    from src.prompt_opt import harness as H
    from src.online_rl.cases import load_cases
    from src.online_rl.runtime import SharedPolicy
    from src.online_rl.integrated.continuation import adapter_sha256, protocol_sha256
    from src.online_rl.integrated.grpo import make_updater
    from src.online_rl.integrated.prompt_search import sol_backend
    from src.online_rl.integrated.rubric import real_components, real_state_elicitor
    from src.online_rl.integrated.trainer import IntegratedTrainer
    key = H.load_key(); client = H.jev(); pu, se = cfg['prompt_update'], cfg['state_elicitation']
    policy = SharedPolicy(cfg, key); updater = make_updater(policy, cfg)
    backend = sol_backend(key, (ROOT / pu['feedback_prompt_file']).read_text(), (ROOT / pu['refiner_prompt_file']).read_text(), *real_components(client, key))
    return IntegratedTrainer(cfg, out_dir=ROOT / cfg['output_dir'], cases=load_cases(ROOT / cfg['data']['train']), policy=policy, updater=updater,
                             policy_sha=lambda: adapter_sha256(policy.model), protocol_sha=protocol_sha256(cfg, ROOT),
                             outcome_fn=lambda case, opt: H.ST.final_outcome(case, opt), backend=backend,
                             initial_prompt=(ROOT / cfg['rubric']['prompt_file']).read_text(),
                             elicit_state=real_state_elicitor(key, policy, se['reconstructor_model'], se['reconstruction_max_tokens'],
                                                              (ROOT / se['judgment_system_prompt']).read_text(encoding='utf-8'), se['judgment_max_tokens']))


def main():
    a = argparse.ArgumentParser(); a.add_argument('--config', default=DEFAULT_CONFIG); a.add_argument('--condition', default=None)
    a.add_argument('--launch', action='store_true'); a.add_argument('--max-steps', type=int); x = a.parse_args()
    base = json.loads((ROOT / x.config).read_text()); cfg = condition_config(base, x.condition or base['condition']); todo = blockers(cfg)
    print(json.dumps({'blockers': todo, **({} if todo else {'preflight': preflight(cfg, x.config, x.max_steps)})}, indent=1))
    if todo or not x.launch: raise SystemExit('not launching' + (' (blockers above)' if todo else ' (pass --launch after approval)'))
    build(cfg).run(max_steps=x.max_steps)


if __name__ == '__main__': main()
