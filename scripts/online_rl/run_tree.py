"""SHERPA case-tree launcher. Without --launch, or with any blocker, it only prints the blockers; every check runs before
credentials, the policy or any API client is loaded. A smoke config (one with a "smoke" block) must pass the smoke size
contract and runs one phase per launch: --phase A = the training case at case_cursor 0, --phase B = the prompt event
that follows it (case_cursor 1 is never run). Resumes from <output_dir>/run_state.json and its call cache."""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.online_rl.integrated.conditions import unresolved

PROMPT_HASHES = (('rubric', 'generator_prompt_file', 'prompt_sha256'), ('feedback', 'prompt_file', 'prompt_sha256'),
                 ('prompt_update', 'feedback_prompt_file', 'feedback_prompt_sha256'), ('prompt_update', 'refiner_prompt_file', 'refiner_prompt_sha256'),
                 ('intervention', 'score0_template_file', 'score0_template_sha256'), ('intervention', 'score2_template_file', 'score2_template_sha256'))
SMOKE_DIR = 'outputs/online_rl/tree_smoke'
SMOKE_LIMITS = ((('schedule', 'cases'), 2), (('rollout', 'branching'), 2), (('rollout', 'rounds'), 2),
                (('prompt_update', 'candidate_count'), 1), (('prompt_update', 'retain_top_k'), 1), (('prompt_update', 'refinement_depth'), 1))


def file_sha(rel: str) -> str: return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def blockers(cfg: dict) -> list[str]:
    """Unresolved settings plus every approved prompt/template whose file differs from its pinned sha256 (as run_integrated)."""
    out = unresolved(cfg)
    for sec, f, h in PROMPT_HASHES:
        if cfg[sec][h] not in ('TBD', file_sha(cfg[sec][f])): out.append(f'{sec}.{f} differs from {sec}.{h}')
    return out


def smoke_problems(cfg: dict) -> list[str]:
    """The smoke launch contract; a config without a "smoke" block is not checked here."""
    if 'smoke' not in cfg: return []
    out = [f'smoke: {s}.{k}={cfg[s][k]!r} exceeds {lim}' for (s, k), lim in SMOKE_LIMITS if not isinstance(cfg[s][k], int) or cfg[s][k] > lim]
    if type(cfg['intervention']['reruns_per_intervention']) is not int or cfg['intervention']['reruns_per_intervention'] != 3:
        out.append('smoke: intervention.reruns_per_intervention must be exactly 3')
    if (ROOT / cfg['output_dir']).resolve() != (ROOT / SMOKE_DIR).resolve(): out.append(f'smoke: output_dir must be {SMOKE_DIR}')
    return out


def setup_problems(cfg: dict, config_path: str) -> list[str]:
    """Output collisions with any other config, training data, and the model/backend settings the wiring needs."""
    out, mine = [], (ROOT / cfg['output_dir']).resolve()
    for p in sorted((ROOT / 'configs/online_rl').glob('*.json')):
        if p.resolve() == (ROOT / config_path).resolve(): continue
        other = json.loads(p.read_text()).get('output_dir')
        if other and (ROOT / other).resolve() == mine: out.append(f'output_dir collides with {p.relative_to(ROOT)}')
    if not (ROOT / cfg['data']['train']).exists(): out.append(f"training split missing: {cfg['data']['train']}")
    elif not file_sha(cfg['data']['train']).startswith(cfg['data']['train_sha256']): out.append('training split hash mismatch')
    for sec, k in (('policy', 'base_model'), ('rollout', 'doctor_model'), ('rollout', 'max_tokens_doctor'), ('feedback', 'max_tokens'),
                   ('advantage_normalization', 'mode')):
        if sec == 'advantage_normalization' and cfg.get('pilot', {}).get('prompt_only'): continue
        if not (cfg.get(sec) or {}).get(k): out.append(f'missing {sec}.{k}')
    return out


def phase_problems(cfg: dict, phase) -> list[str]:
    smoke = 'smoke' in cfg
    if smoke and phase is None: return ['smoke config: --phase A or --phase B is required']
    if not smoke and phase is not None: return ['--phase is only defined for a smoke config']
    if phase is None: return []
    out = ROOT / cfg['output_dir']; sp = out / 'run_state.json'
    st = json.loads(sp.read_text()) if sp.exists() else None
    if phase == 'A':
        return ['phase A already completed (case_cursor >= 1)'] if st and st['case_cursor'] >= 1 else []
    if st is None or st['case_cursor'] < 1 or not (out / 'cases' / 'case_000000.json').exists(): return ['phase B requires a completed phase A in output_dir']
    if st['case_cursor'] > 1: return ['phase B: a training case after the event already ran; refusing']
    return ['phase B already completed (lineage_cursor >= 1)'] if st['lineage_cursor'] >= 1 else []


def load_key() -> str:
    from src.prompt_opt import harness as H
    return H.load_key()


def jev_client():
    from src.prompt_opt import harness as H
    return H.jev()


def judge_client(cfg: dict, key: str):
    settings = cfg.get('judge', {})
    if settings.get('backend', 'jev') == 'jev':
        return jev_client()
    if settings.get('backend') != 'openrouter_chat':
        raise ValueError('Unknown rubric judge backend')
    from src.judges.openrouter_choice import OpenRouterChoiceJudge
    return OpenRouterChoiceJudge(key, settings['model'], settings.get('max_tokens', 4096), settings.get('batch_size', 8), max_retries=settings.get('max_retries', 2),
                                 audit_path=ROOT / cfg['output_dir'] / 'judge_calls_raw.jsonl')


def make_policy(cfg: dict, key: str):
    if cfg.get('pilot', {}).get('backend') == 'ollama':
        from src.online_rl.tree.ollama_policy import OllamaPolicy
        return OllamaPolicy(cfg, key)
    from src.online_rl.runtime import SharedPolicy
    if cfg.get('pilot', {}).get('prompt_only'):
        import torch
        torch.manual_seed(cfg['schedule']['seed'])
    return SharedPolicy(cfg, key, trainable=not cfg.get('pilot', {}).get('prompt_only', False))


def build(cfg: dict):
    """Bind the existing components to the TreeTrainer interfaces. Policy rounds / Doctor model come from cfg['rollout']."""
    from src.collaboration.generator import final_answer_option, stop_reason
    from src.online_rl.cases import load_cases
    from src.online_rl.environment import final_outcome
    from src.online_rl.integrated.continuation import adapter_sha256, protocol_sha256
    from src.online_rl.integrated.grpo import make_updater
    from src.online_rl.integrated.prompt_search import sol_backend
    from src.online_rl.tree.backends import ApiDoctor, PolicyAgent, PolicyInterventions, api_feedback, api_rubric_generator, posthoc_gates, two_stage_scorer
    from src.online_rl.tree.team import team_strategies
    from src.online_rl.tree.trainer import TreeEnv, TreeTrainer
    text = lambda rel: (ROOT / rel).read_text()
    ro, pu, ar = cfg['rollout'], cfg['prompt_update'], cfg['agent_reward']
    key = load_key(); client = judge_client(cfg, key); policy = make_policy(cfg, key)
    if (policy.args.rounds, policy.args.doctor_model) != (ro['rounds'], ro['doctor_model']):
        raise RuntimeError('policy/harness rounds or Doctor model differ from rollout config')
    prompt_only = cfg.get('pilot', {}).get('prompt_only', False)
    updater = None if prompt_only else make_updater(policy, cfg)
    cases = load_cases(ROOT / cfg['data']['train'])
    doctor_call = None
    if cfg.get('doctor_backend', {}).get('backend') == 'ollama':
        from src.online_rl.tree.ollama_doctor import OllamaDoctorCall
        doctor_call = OllamaDoctorCall(cfg['doctor_backend'])
    elif cfg.get('doctor_backend', {}).get('backend', 'openrouter') != 'openrouter':
        raise ValueError('Unknown Doctor backend')
    doctor = ApiDoctor(key, ro['doctor_model'], policy.doctor_system, ro['max_tokens_doctor'], call=doctor_call)
    iv = PolicyInterventions(policy, text(cfg['intervention']['score0_template_file']), text(cfg['intervention']['score2_template_file']), doctor=doctor)
    if cfg.get('pilot', {}).get('backend') == 'ollama':
        from src.online_rl.tree.ollama_policy import OllamaAgent
        agent = OllamaAgent(policy)
    else:
        agent = PolicyAgent(policy)
    env = TreeEnv(agent=agent, doctor=doctor,
                  stop_rule=stop_reason, final_option=final_answer_option, outcome_fn=final_outcome,
                  feedback_fn=api_feedback(key, ro['doctor_model'], text(cfg['feedback']['prompt_file']), cfg['feedback']['max_tokens'], call=doctor_call),
                  generate_fn=api_rubric_generator(key, cfg.get('pilot', {}).get('prompt_model'), cfg.get('pilot', {}).get('generator_max_tokens')),  scorer=two_stage_scorer(client), writer=iv.write, continuation=iv.continue_,
                  gate_fn=posthoc_gates(client, {c['case_id']: c for c in cases}), behavior_logprobs=updater.behavior_logprobs if updater else None)
    # the tree prompt search uses only the reflection / refinement calls of the approved Sol backend
    backend = sol_backend(key, text(pu['feedback_prompt_file']), text(pu['refiner_prompt_file']), env.generate_fn, env.gate_fn, env.scorer, model=cfg.get('pilot', {}).get('prompt_model'), feedback_model=pu.get('feedback_model'), refiner_model=pu.get('refiner_model'))
    strategies = team_strategies(updater, ar['w_outcome'], ar['w_rubric'])
    if prompt_only:
        from dataclasses import replace
        strategies = replace(strategies, agent_update=lambda *args: {'optimizer_step': False, 'reason': 'prompt-only pilot; fixed Agent policy'})
    trainer = TreeTrainer(cfg, out_dir=ROOT / cfg['output_dir'], cases=cases, env=env, strategies=strategies,
                       search_backend=backend, initial_prompt=text(cfg['rubric']['generator_prompt_file']), policy_sha=lambda: policy.policy_sha() if hasattr(policy, 'policy_sha') else adapter_sha256(policy.model),
                       protocol_sha=protocol_sha256(cfg, ROOT), feedback_prompt_sha256=cfg['feedback']['prompt_sha256'], updater=updater)
    if hasattr(policy, 'policy_sha'):
        policy.cache = trainer.cache
    return trainer


def launch(cfg: dict, phase) -> dict:
    tr = build(cfg)
    if phase == 'A':
        if tr.state.case_cursor != 0: raise RuntimeError('phase A expects case_cursor 0')
        rec = tr.step()
        return {'phase': 'A', 'case_id': rec['case_id'], 'counts': rec['counts'], 'rollout_policy_sha256': rec['policy_sha256'],
                'policy_sha256_after': tr.state.policy_sha256, 'update': rec['update'], 'case_cursor': tr.state.case_cursor}
    if phase == 'B':
        before = tr.state.lineage_cursor; tr._maybe_prompt_event()
        if tr.state.lineage_cursor == before: raise RuntimeError('phase B: the prompt event did not fire (too few decided trajectories?)')
        ev = json.loads(tr._event_path(tr.state.lineage_cursor).read_text())
        return {'phase': 'B', 'event': ev['event'], 'decision': ev['decision'], 'rerun_policy_sha256': ev['rerun_policy_sha256'],
                'evidence_policy_sha256': ev['evidence_policy_sha256'], 'case_cursor': tr.state.case_cursor}
    tr.run(); return {'case_cursor': tr.state.case_cursor}


def main(argv=None):
    a = argparse.ArgumentParser(); a.add_argument('--config', default='configs/online_rl/tree_v1.json'); a.add_argument('--launch', action='store_true')
    a.add_argument('--phase', choices=['A', 'B']); x = a.parse_args(argv)
    cfg = json.loads((ROOT / x.config).read_text())
    todo = blockers(cfg) + smoke_problems(cfg) + setup_problems(cfg, x.config) + phase_problems(cfg, x.phase)
    usage_log = os.environ.get('API_USAGE_LOG') or str(ROOT / cfg['output_dir'] / 'api_usage.jsonl')
    print(json.dumps({'config': x.config, 'phase': x.phase, 'output_dir': cfg['output_dir'], 'max_agent_turns': cfg['rollout']['rounds'],
                      'branching': cfg['rollout']['branching'], 'api_usage_log': usage_log, 'blockers': todo}, indent=1), flush=True)
    if todo: raise SystemExit('not launching: ' + '; '.join(todo[:3]) + (' ...' if len(todo) > 3 else ''))
    if not x.launch: raise SystemExit('not launching (pass --launch after approval)')
    os.environ['API_USAGE_LOG'] = usage_log                                   # existing ledger: src/common/usage.py (no credentials)
    print(json.dumps(launch(cfg, x.phase), indent=1, default=str), flush=True)


if __name__ == '__main__': main()
