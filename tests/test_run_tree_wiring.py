"""scripts/online_rl/run_tree.py real wiring with fake network edges only (OpenRouter call, Jev client, key, 8B model ->
tiny CPU LoRA model). Everything between them is the production binding: backends, harness continue_from, TreeTrainer,
call cache, checkpoints and IntegratedGRPOUpdater.update_groups."""
import hashlib, importlib.util, json, re
from argparse import Namespace
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SECRET = 'test-placeholder-credential-0123456789'
_spec = importlib.util.spec_from_file_location('run_tree', ROOT / 'scripts/online_rl/run_tree.py')
assert _spec is not None and _spec.loader is not None
RT = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RT)
RESEARCH = json.loads((ROOT / 'configs/online_rl/tree_v1.json').read_text())


def smoke_config() -> dict:
    """The research config with the smoke contract sizes and placeholder values (a 'smoke' block enables the smoke checks)."""
    c = json.loads(json.dumps(RESEARCH))
    c['rollout'].update(branching=2, rounds=2); c['schedule']['cases'] = 2; c['optimizer']['learning_rate'] = 1e-5
    c['advantage_normalization'] = {'mode': 'mean_center'}; c['intervention']['reruns_per_intervention'] = 2
    c['selection'] = {'mode': 'threshold', 'min_R_O': 0.5}; c['agent_reward'] = {'w_outcome': 0.5, 'w_rubric': 0.5}
    c['prompt_update'].update(every_n_cases=1, minimum_trajectories=2, candidate_count=1, retain_top_k=1, refinement_depth=1)
    c['acceptance'] = {'rule': 'stratified_paired_bootstrap_J', 'delta': 0.0, 'p_min': 0.5, 'minimum_success_trajectories': 1,
                       'minimum_failure_trajectories': 1, 'bootstrap_n': 200, 'bootstrap_seed': 0}
    c.pop('evaluation')                                          # smoke phases never reach a milestone
    c['smoke'] = {'purpose': 'test'}
    # the legacy protocol (OpenRouter Doctor, per-trajectory rubrics, scorer v1); the new protocol has its own tests below
    for k in ('doctor_backend', 'halting', 'doctor_system_prompt_file', 'doctor_system_prompt_sha256', 'doctor_opening'): c['rollout'].pop(k, None)
    c['rollout']['doctor_model'] = 'google/gemma-3-12b-it'
    c['rubric'] = {k: v for k, v in c['rubric'].items() if k not in ('reuse', 'generation')}
    c['rubric'].update(generator_prompt_file='prompts/sherpa/trajectory_rubric_generator_P0.txt', prompt_id='P0',
                       prompt_sha256='9d9116275b08ba8e0c50e533aa3ac33b39be20ab6eb0d721860401fc25b948d0')
    c['judges'] = {k: v for k, v in c['judges'].items() if k != 'scorer_protocol'}
    c['judges'].update(scorer_prompt_file='prompts/sherpa/luna_scorer_v1.txt', scorer_prompt_sha256=hashlib.sha256((ROOT / 'prompts/sherpa/luna_scorer_v1.txt').read_bytes()).hexdigest())
    c['prompt_update'].update(feedback_prompt_file='prompts/sherpa/tree_prompt_feedback_v1.txt',
                              feedback_prompt_sha256=hashlib.sha256((ROOT / 'prompts/sherpa/tree_prompt_feedback_v1.txt').read_bytes()).hexdigest())
    return c


SMOKE = smoke_config()
needs_data = pytest.mark.skipif(not (ROOT / RESEARCH['data']['train']).exists(), reason='the training split is not redistributed')
text = lambda rel: (ROOT / rel).read_text()
h = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


class FakePolicy:
    """SharedPolicy surface used by the tree wiring, on a tiny CPU LoRA model; prompts and continuations are the harness code."""
    def __init__(self, cfg, key, events, rounds=None):
        import torch, transformers as tf, peft
        torch.manual_seed(0)
        base = tf.LlamaForCausalLM(tf.LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                                  num_key_value_heads=2, max_position_embeddings=64))
        self.model = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], lora_dropout=0.0))
        self.torch, self.cfg, self.key, self.events = torch, cfg, key, events; ro = cfg['rollout']
        self.args = Namespace(rounds=ro['rounds'] if rounds is None else rounds, doctor_model=ro['doctor_model'], max_tokens_doctor=ro['max_tokens_doctor'])
        self.doctor_system, self.agent_system = text('prompts/user_simulator.txt'), text('prompts/agent.txt')

    def _H(self):
        from src.prompt_opt import harness as H
        return H

    def agent_user_prompt(self, case, prefix, t): return self._H().Env.agent_user_prompt(self, case, prefix, t)

    def agent(self, user, seed):
        self.events.append(('agent', 'Score-2' if 'Score-2' in user else 'Score-0' if 'Score-0' in user else 'plain'))
        return f'agent reply {seed % 5}'

    def continue_from(self, case, prefix, t, fixed, seed, turn_cache=None):
        return self._H().Env.continue_from(self, case, prefix, t, fixed, seed, turn_cache=turn_cache)

    def generate(self, case, seed):
        self.torch.rand(3); self.events.append(('episode', case['case_id'], seed))
        return {'final_option': 'A', 'termination_reason': 'x', 'num_discussion_rounds': 2}

    def sample_completions(self, case, prefix, turn, seeds):
        user = self.agent_user_prompt(case, prefix, turn); pid = [1 + (h(user) + i) % 30 for i in range(4)]; self.events.append(('agent', 'node'))
        return [{'prompt_text': user, 'prompt_input_ids': pid, 'completion_ids': [2 + s % 30, 5, 7], 'old_policy_logprobs': [-1.0] * 3,
                 'completion_text': f'reply t{turn} s{s % 97}', 'seed': s, 'stage': f'A{turn}'} for s in seeds]


class FakeJev:
    """Fake hard-label judge (Luna-like): one allowed option per question, no probabilities; fail_at counts judge calls
    across every client sharing `events` (gate and scorer)."""
    def __init__(self, events, fail_at=None, model='openai/gpt-5.6-luna', system=''):
        self.events, self.fail_at, self.model, self.protocol_sha256, self.log = events, fail_at, model, f'proto:{system[:20]}', []

    def decide(self, state, questions, purpose='jev'):
        n = sum(e[0] == 'jev' for e in self.events) + 1
        if self.fail_at and n == self.fail_at: self.fail_at = None; raise RuntimeError('Jev transport failure')
        self.events.append(('jev', purpose, self.model)); out = {}
        for qid in questions:
            v = 'N' if purpose == 'tree_posthoc_rubric_gate' else 'Y' if purpose.endswith('_applicability') else '2'
            out[qid] = {'value': v, 'probabilities': None, 'confidence': None}
        rec = {'purpose': purpose, 'raw': [json.dumps({'answers': {q: o['value'] for q, o in out.items()}})], 'repaired': False}
        self.log.append(rec); return out, rec


RUBRIC = {'name': 'Ask for the reason', 'applicability_condition': 'the physician states a conclusion', 'required_behavior': 'ask what supports it',
          'scores': {'0': 'no question', '1': 'vague question', '2': 'specific question about the stated evidence'}}


def fake_call(events):
    from src.common import usage
    sys_kind = {text('prompts/user_simulator.txt'): 'doctor', text(SMOKE['feedback']['prompt_file']): 'feedback',
                text(SMOKE['prompt_update']['feedback_prompt_file']): 'reflect', text(SMOKE['prompt_update']['refiner_prompt_file']): 'refine'}

    def call(key, model, system, user, max_tokens=None, temperature=None):
        assert key == SECRET
        kind = sys_kind.get(system, 'generate')
        if kind == 'doctor': kind = 'final' if 'final decision' in user.lower() else 'doctor_turn'
        events.append(('api', kind, model, user)); usage.record('collaboration', model, {'prompt_tokens': 10, 'completion_tokens': 5, 'cost': 1e-6})
        if kind == 'final': return f'Final answer: option {"D" if h(user) % 2 else "B"}'
        if kind == 'doctor_turn': return 'What supports that answer?'
        if kind == 'feedback': return "The Agent's first reply helped me weigh the evidence."
        if kind == 'reflect':
            m = re.search(r'NUMBER OF DIRECTIONS: (\d+)', user); assert m; n = int(m.group(1))
            return json.dumps({'diagnosis': 'rubrics rarely change outcomes', 'directions': [f'focus on decisive reasoning {i}' for i in range(n)]})
        if kind == 'refine': return text(SMOKE['rubric']['generator_prompt_file']) + '\nPrefer behaviors that change how the physician weighs evidence.'
        return json.dumps(RUBRIC)
    return call


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Smoke config with its output in tmp; fake edges; real network refused."""
    import urllib.request
    import src.collaboration.generator as G
    from src.prompt_opt import harness as H
    out = tmp_path / 'tree_smoke'; cfg = {**SMOKE, 'output_dir': str(out)}; cfg_path = tmp_path / 'tree_smoke.json'; cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(RT, 'SMOKE_DIR', str(out)); monkeypatch.delenv('API_USAGE_LOG', raising=False)
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network call in a test')))
    events, made = [], []
    for mod in (G, H): monkeypatch.setattr(mod, 'doctor_call', fake_call(events))
    monkeypatch.setattr(RT, 'load_key', lambda: SECRET)
    monkeypatch.setattr(RT, 'judge_client', lambda model, key, system: FakeJev(events, model=model, system=system))
    monkeypatch.setattr(RT, 'make_policy', lambda c, k: made.append(c) or FakePolicy(c, k, events))
    from src.online_rl.integrated.grpo import IntegratedGRPOUpdater
    real = IntegratedGRPOUpdater.update_groups
    monkeypatch.setattr(IntegratedGRPOUpdater, 'update_groups', lambda self, g: events.append(('update', len(g))) or real(self, g))
    return Namespace(cfg=cfg, path=str(cfg_path), out=out, events=events, made=made)


def run(w, *args):
    return RT.main(['--config', w.path, *args])


def api(w, kind=None): return [e for e in w.events if e[0] == 'api' and (kind is None or e[1] == kind)]


@needs_data
def test_build_binds_the_real_tree_trainer(wired):
    from src.online_rl.tree.trainer import TreeTrainer
    from src.online_rl.tree import backends as B
    tr = RT.build(wired.cfg)
    assert isinstance(tr, TreeTrainer) and isinstance(tr.env.agent, B.PolicyAgent) and isinstance(tr.env.doctor, B.ApiDoctor)
    assert wired.made[0]['rollout']['rounds'] == 2 and tr.env.doctor.model == SMOKE['rollout']['doctor_model'] == 'google/gemma-3-12b-it'
    assert getattr(tr.env.writer, '__self__').policy.args.rounds == 2 and getattr(tr.env.behavior_logprobs, '__self__') is tr.updater
    assert tr.state.case_cursor == 0 and (wired.out / 'calls.jsonl').parent.exists()


@needs_data
def test_policy_rounds_mismatch_is_rejected(wired, monkeypatch):
    monkeypatch.setattr(RT, 'make_policy', lambda c, k: FakePolicy(c, k, [], rounds=7))
    with pytest.raises(RuntimeError, match='rounds or Doctor model'): RT.build(wired.cfg)


@needs_data
def test_phase_a_then_b(wired, capsys):
    run(wired, '--launch', '--phase', 'A')
    st = json.loads((wired.out / 'run_state.json').read_text()); rec = json.loads((wired.out / 'cases/case_000000.json').read_text())
    assert st['case_cursor'] == 1 and st['lineage_cursor'] == 0 and not (wired.out / 'prompt_events').exists()
    assert rec['counts']['agent_nodes'] == 6 and rec['counts']['leaves'] == 4 and len(api(wired, 'final')) >= 4
    turns = api(wired, 'doctor_turn')
    assert turns and all(e[2] == 'google/gemma-3-12b-it' for e in api(wired) if e[1] in ('doctor_turn', 'final'))
    HAIKU, GEMINI, JEV = 'anthropic/claude-4.5-haiku-20251001', 'google/gemini-3.8-flash-20260902', 'openai/gpt-5.6-luna'      # JEV = judge model
    assert {e[2] for e in api(wired, 'feedback')} == {HAIKU} and {e[2] for e in api(wired, 'generate')} == {GEMINI}       # role routing
    assert {e[2] for e in wired.events if e[0] == 'jev'} == {JEV}
    assert {e[1] for e in wired.events if e[0] == 'jev'} >= {'tree_posthoc_rubric_gate', 'tree_response_rubric_score_applicability'}
    roles = json.loads((wired.out / 'run_manifest.json').read_text())['model_roles']
    assert roles == {'doctor': 'google/gemma-3-12b-it', 'agent': 'Qwen/Qwen3-8B + LoRA (local)', 'feedback_generator': HAIKU, 'rubric_generator': GEMINI,
                     'rubric_gate': JEV, 'rubric_scorer': JEV, 'prompt_reflection': HAIKU, 'prompt_refiner': HAIKU}
    assert all('of at most 2' in e[3] for e in turns if 'turn ' in e[3] and 'of at most' in e[3]) and any('of at most 2' in e[3] for e in turns)
    assert {e[2] for e in api(wired, 'generate')} == {'google/gemini-3.8-flash-20260902'} and len(api(wired, 'generate')) == len(rec['trajectory_inputs'])     # one rubric each
    upd = [i for i, e in enumerate(wired.events) if e[0] == 'update']
    assert len(upd) == 1 and upd[0] == len(wired.events) - 1                                                  # update after every call of the case
    assert any(e == ('agent', 'Score-2') or e == ('agent', 'Score-0') for e in wired.events)
    assert (wired.out / 'checkpoints/checkpoint_000000').exists() and (wired.out / 'api_usage.jsonl').exists()
    n_a = len(wired.events)

    run(wired, '--launch', '--phase', 'B')
    st = json.loads((wired.out / 'run_state.json').read_text()); ev = json.loads((wired.out / 'prompt_events/event_0001.json').read_text())
    assert st['case_cursor'] == 1 and st['lineage_cursor'] == 1 and not (wired.out / 'cases/case_000001.json').exists()   # case 1 never runs
    new = wired.events[n_a:]
    assert [e[1] for e in new if e[0] == 'api' and e[1] in ('reflect', 'refine')] == ['reflect', 'refine'] and not any(e[0] == 'update' for e in new)
    assert {e[2] for e in new if e[0] == 'api' and e[1] in ('reflect', 'refine')} == {'anthropic/claude-4.5-haiku-20251001'}
    stages = {json.loads(l)['stage'] for l in (wired.out / 'api_usage.jsonl').read_text().splitlines()}
    assert stages == {'doctor', 'feedback', 'rubric_generator', 'doctor_rerun', 'prompt_reflection', 'prompt_refiner'}       # usage tagged by role
    assert not any(e[0] == 'api' and e[1] in ('doctor_turn',) and 'of at most 2' not in e[3] for e in new)
    assert ev['rerun_policy_sha256'] == st['policy_sha256'] and ev['evidence_policy_sha256'] == [rec['policy_sha256']]
    assert set(ev['decision']) >= {'accepted', 'reason'} and len(ev['search_leaf_ids']) == len(ev['acceptance_leaf_ids']) == 2
    assert 'initial_policy_sha256' in json.loads((wired.out / 'run_manifest.json').read_text())
    for p, msg in (('B', 'already completed'), ('A', 'already completed')):
        with pytest.raises(SystemExit, match=msg): run(wired, '--launch', '--phase', p)
    printed = capsys.readouterr().out + ''.join(p.read_text(errors='ignore') for p in wired.out.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.txt'))
    assert SECRET not in printed                                                                              # credentials never written
    rows = [json.loads(l) for l in (wired.out / 'api_usage.jsonl').read_text().splitlines()]
    assert rows and all(set(r) >= {'model', 'cost_usd'} for r in rows) and len(rows) == len(api(wired))


def test_phase_b_without_phase_a_fails_before_any_call(wired, monkeypatch):
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError('credentials or model loaded'))
    for name in ('load_key', 'judge_client', 'make_policy'): monkeypatch.setattr(RT, name, boom)
    with pytest.raises(SystemExit, match='phase B requires a completed phase A'): run(wired, '--launch', '--phase', 'B')
    with pytest.raises(SystemExit, match='--phase A or --phase B is required'): run(wired, '--launch')
    assert not wired.out.exists() and not wired.events


@pytest.mark.parametrize('change,msg', [({'rollout': {'branching': 4, 'rounds': 5}}, 'smoke: rollout.branching=4 exceeds 2'),
                                        ({'intervention': {'reruns_per_intervention': 3}}, 'reruns_per_intervention=3 exceeds 2'),
                                        ({'prompt_update': {'candidate_count': 4}}, 'candidate_count=4 exceeds 1'),
                                        ({'output_dir': RESEARCH['output_dir']}, 'collides with configs/online_rl/tree_v1.json'),
                                        ({'rubric': {'prompt_sha256': '0' * 64}}, 'rubric.generator_prompt_file differs'),
                                        ({'schedule': {'cases': 'CALIBRATION_REQUIRED'}}, 'CALIBRATION_REQUIRED')])
def test_pre_launch_validation_stops_before_any_call(wired, monkeypatch, tmp_path, change, msg):
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError('credentials or model loaded'))
    for name in ('load_key', 'judge_client', 'make_policy'): monkeypatch.setattr(RT, name, boom)
    cfg = json.loads(json.dumps(wired.cfg))
    for k, v in change.items(): cfg[k] = {**cfg[k], **v} if isinstance(v, dict) else v
    p = tmp_path / 'bad.json'; p.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match='not launching'): RT.main(['--config', str(p), '--launch', '--phase', 'A'])
    assert msg in ' '.join(RT.blockers(cfg) + RT.smoke_problems(cfg) + RT.setup_problems(cfg, str(p)))
    assert not wired.events


def test_smoke_limits_do_not_apply_to_the_research_config():
    research = json.loads((ROOT / 'configs/online_rl/tree_v1.json').read_text())
    assert RT.smoke_problems(research) == [] and research['rollout'] == {**research['rollout'], 'branching': 4, 'rounds': 5}
    assert research['advantage_normalization'] == {'mode': 'mean_center'} and research['optimizer']['learning_rate'] == 1e-5   # frozen 2026-10-07
    assert RT.phase_problems(research, None) == ['research config: --condition outcome_only|rubric_only|fixed|evolving is required']
    from src.online_rl.tree.conditions import arm_config
    assert RT.phase_problems(arm_config(research, 'fixed'), 'A') == ['--phase is only defined for a smoke config']
    assert RT.phase_problems(arm_config(research, 'fixed'), None) == []


@needs_data
def test_phase_a_resumes_from_the_call_cache(wired, tmp_path, monkeypatch):
    run(wired, '--launch', '--phase', 'A'); ref = len(api(wired)); ref_jev = sum(e[0] == 'jev' for e in wired.events)
    out2 = tmp_path / 'resume_smoke'; cfg = {**wired.cfg, 'output_dir': str(out2)}; p = tmp_path / 'resume.json'; p.write_text(json.dumps(cfg))
    monkeypatch.setattr(RT, 'SMOKE_DIR', str(out2)); wired.events.clear()
    monkeypatch.setattr(RT, 'judge_client', lambda model, key, system: FakeJev(wired.events, fail_at=15, model=model, system=system))
    with pytest.raises(RuntimeError, match='Jev transport failure'): RT.main(['--config', str(p), '--launch', '--phase', 'A'])
    st = json.loads((out2 / 'run_state.json').read_text()); assert st['case_cursor'] == 0 and not (out2 / 'cases/case_000000.json').exists()
    first = len(api(wired)); first_jev = sum(e[0] == 'jev' for e in wired.events)
    monkeypatch.setattr(RT, 'judge_client', lambda model, key, system: FakeJev(wired.events, model=model, system=system))
    RT.main(['--config', str(p), '--launch', '--phase', 'A'])
    assert len(api(wired)) == ref and sum(e[0] == 'jev' for e in wired.events) == ref_jev                    # nothing completed is repeated
    assert first < ref and first_jev == 14
    a, b = json.loads((wired.out / 'cases/case_000000.json').read_text()), json.loads((out2 / 'cases/case_000000.json').read_text())
    assert a['agent_targets'] == b['agent_targets'] and a['trajectory_rubrics'] == b['trajectory_rubrics']


@needs_data
def test_build_wires_the_milestone_evaluator(wired, monkeypatch):
    import src.online_rl.tree.evaluation as EV
    eval_set = [{**c, 'case_id': f'eval{i}'} for i, c in enumerate(json.loads(json.dumps(CASES_FOR_EVAL)))]
    monkeypatch.setattr(EV, 'eval_cases', lambda path, split, manifest: eval_set)
    cfg = {**wired.cfg, 'evaluation': {'checkpoint_every_cases': 2, 'split': 'val50', 'cases_file': 'local.jsonl', 'seed': 0}}
    tr = RT.build(cfg)
    assert isinstance(tr.evaluator, EV.MilestoneEvaluator) and tr.every == 2 and tr.evaluator.info['split'] == 'val50'
    import torch
    before = torch.get_rng_state().clone(); t = tr.evaluator.play(eval_set[0], 5)
    assert t['final_option'] == 'A' and torch.equal(torch.get_rng_state(), before)                       # evaluation leaves training RNG untouched
    assert ('episode', 'eval0', 5) in wired.events and not api(wired)


CASES_FOR_EVAL = [{'case_id': 'e', 'question': 'q?', 'options': {'A': 'a', 'B': 'b'}, 'ground_truth': {'correct_option': 'A'}, 'scenario': ''}]


def test_pilot_guard_caps_cases_and_output():
    from src.online_rl.tree.conditions import arm_config
    pilot = {**arm_config(RESEARCH, 'evolving'), 'output_dir': 'outputs/online_rl/tree_pilot/stage_x', 'pilot': {'max_cases': 1}}
    pilot['schedule'] = {**pilot['schedule'], 'cases': 1}
    assert RT.pilot_problems(pilot) == []
    assert 'must equal pilot.max_cases' in ' '.join(RT.pilot_problems({**pilot, 'schedule': {**pilot['schedule'], 'cases': 500}}))
    assert 'integer in 1..5' in ' '.join(RT.pilot_problems({**pilot, 'pilot': {'max_cases': 500}}))
    assert 'inside outputs/online_rl/tree_pilot' in ' '.join(RT.pilot_problems({**pilot, 'output_dir': RESEARCH['output_dir']}))
    assert RT.pilot_problems(RESEARCH) == []                                                              # the research config is not a pilot


def test_main_config_stays_calibration_blocked():
    from src.online_rl.tree.conditions import arm_config
    todo = RT.blockers(arm_config(RESEARCH, 'evolving'))
    assert len(todo) == 6 and all('CALIBRATION_REQUIRED' in t and t.startswith(('prompt_update.', 'acceptance.')) for t in todo)   # prompt events open
    assert RT.blockers(arm_config(RESEARCH, 'outcome_only')) == RT.blockers(arm_config(RESEARCH, 'rubric_only')) == RT.blockers(arm_config(RESEARCH, 'fixed')) == []
    assert arm_config(RESEARCH, 'rubric_only')['agent_reward'] == {'w_outcome': 0.0, 'w_rubric': 1.0} and arm_config(RESEARCH, 'outcome_only')['agent_reward'] == {'w_outcome': 1.0, 'w_rubric': 0.0}
    assert arm_config(RESEARCH, 'fixed')['agent_reward'] == arm_config(RESEARCH, 'evolving')['agent_reward'] == {'w_outcome': 0.5, 'w_rubric': 0.5}
    assert not any(k in ' '.join(todo) for k in ('schedule.cases', 'checkpoint_every_cases', 'cases_file', 'data.train'))
    assert RESEARCH['schedule']['cases'] == 500 and RESEARCH['feedback']['model'] != RESEARCH['rollout']['doctor_model']


# ---------- deterministic initial LoRA, checkpoint_000000 and checkpoint-0 evaluation reuse ----------
def tiny_base():
    import torch, transformers as tf
    torch.manual_seed(0)                                         # stands for the fixed pretrained base weights
    return tf.LlamaForCausalLM(tf.LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                              num_key_value_heads=2, max_position_embeddings=64))


def tiny_lc(cfg):
    """The config's LoRA section on the tiny model's module names (r / alpha / rslora / seed from the config)."""
    lc = cfg['policy']['lora']; return {**lc, 'r': 2, 'target_modules': ['q_proj', 'v_proj']}


class SeededFakePolicy(FakePolicy):
    """FakePolicy whose LoRA is created exactly as SharedPolicy creates it (runtime.seeded_lora with policy.lora.init_seed);
    `noise` global-RNG draws happen first, so any dependence on the global RNG would show."""
    def __init__(self, cfg, key, events, noise=0):
        import torch
        from src.online_rl.runtime import seeded_lora
        super().__init__(cfg, key, events); base = tiny_base(); torch.rand(noise + 1)
        self.model = seeded_lora(base, tiny_lc(cfg))


def test_seeded_lora_is_deterministic_and_leaves_the_global_rng_alone():
    import torch
    from src.online_rl.runtime import seeded_lora
    from src.online_rl.integrated.continuation import adapter_sha256
    lc = tiny_lc(RESEARCH); assert isinstance(lc['init_seed'], int)
    hashes = []
    for noise in (0, 7):
        base = tiny_base(); torch.rand(noise + 1); before = torch.get_rng_state().clone()
        m = seeded_lora(base, lc); hashes.append(adapter_sha256(m))
        assert torch.equal(torch.get_rng_state(), before)                                              # global RNG untouched
        assert all(float(p.abs().sum()) == 0 for n, p in m.named_parameters() if 'lora_B' in n)       # B = 0: behaves as the base
    assert hashes[0] == hashes[1]                                                                      # same seed -> same adapter SHA
    assert adapter_sha256(seeded_lora(tiny_base(), {**lc, 'init_seed': lc['init_seed'] + 1})) != hashes[0]   # different seed -> different
    with pytest.raises(ValueError, match='init_seed'): seeded_lora(tiny_base(), {**lc, 'init_seed': None})


def test_research_and_pilot_configs_pin_the_same_init_seed():
    seeds = {p.name: json.loads(p.read_text())['policy']['lora'].get('init_seed') for p in sorted((ROOT / 'configs/online_rl').glob('tree_*.json'))
             if p.name != 'tree_smoke.json'}
    assert seeds['tree_v1.json'] == 20261006 and set(seeds.values()) == {20261006}, seeds
    cfg = {**RESEARCH, 'policy': {**RESEARCH['policy'], 'lora': {k: v for k, v in RESEARCH['policy']['lora'].items() if k != 'init_seed'}}}
    assert any('init_seed' in p for p in RT.setup_problems(cfg, 'configs/online_rl/tree_v1.json'))
    assert not any('init_seed' in p for p in RT.setup_problems(RESEARCH, 'configs/online_rl/tree_v1.json'))


@needs_data
def test_all_arms_start_from_the_same_checkpoint_000000(wired, monkeypatch):
    """Outcome-only, Fixed and Evolving built in one process, each after different global-RNG use: same initial adapter."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from src.online_rl.integrated.continuation import adapter_sha256
    from src.online_rl.runtime import seeded_lora
    from src.online_rl.tree.conditions import ARMS, arm_config
    noise = iter(range(len(ARMS)))
    monkeypatch.setattr(RT, 'make_policy', lambda c, k: SeededFakePolicy(c, k, wired.events, noise=next(noise)))
    mans, loaded = {}, {}
    for arm in ARMS:
        cfg = arm_config(wired.cfg, arm); tr = RT.build(cfg)
        man = json.loads((Path(cfg['output_dir']) / 'run_manifest.json').read_text()); mans[arm] = man
        ck = Path(man['initial_checkpoint']); assert ck.name == 'checkpoint_000000' and tr.state.checkpoint == str(ck)
        other = seeded_lora(tiny_base(), {**tiny_lc(cfg), 'init_seed': 1})                            # load the committed adapter
        set_peft_model_state_dict(other, load_file(str(ck / 'adapter' / 'adapter_model.safetensors'))); loaded[arm] = adapter_sha256(other)
    assert len({m['initial_policy_sha256'] for m in mans.values()}) == 1 and set(loaded.values()) == {mans['evolving']['initial_policy_sha256']}
    assert {m['lora_init_seed'] for m in mans.values()} == {RESEARCH['policy']['lora']['init_seed']}
    assert {m['config']['rubric']['enabled'] for m in mans.values()} == {True, False}                 # the arms do differ in the condition


@needs_data
def test_resume_refuses_a_changed_checkpoint_000000(wired, monkeypatch):
    monkeypatch.setattr(RT, 'make_policy', lambda c, k: SeededFakePolicy(c, k, wired.events))
    RT.build(wired.cfg); man = json.loads((wired.out / 'run_manifest.json').read_text())
    assert man['initial_checkpoint_sha256'] and RT.build(wired.cfg).state.case_cursor == 0                # unchanged: resumes
    (Path(man['initial_checkpoint']) / 'adapter' / 'README.md').write_text('edited')
    with pytest.raises(RuntimeError, match='checkpoint_000000 changed'): RT.build(wired.cfg)


EVAL4 = [{**CASES_FOR_EVAL[0], 'case_id': f'v{i}'} for i in range(4)]


def eval_main(monkeypatch, tmp_path, events, cfg, out, *flags, fail_after=None):
    """scripts/online_rl/eval_checkpoint.py main() with the fake policy / key / eval split; play fails after `fail_after` episodes."""
    import sys
    import src.online_rl.tree.evaluation as EV
    spec = importlib.util.spec_from_file_location('eval_checkpoint', ROOT / 'scripts/online_rl/eval_checkpoint.py')
    EC = importlib.util.module_from_spec(spec); spec.loader.exec_module(EC)
    monkeypatch.setitem(sys.modules, 'run_tree', RT); monkeypatch.setattr(EV, 'eval_cases', lambda path, split, manifest: EVAL4)
    monkeypatch.setattr(RT, 'load_key', lambda: SECRET)
    played = []

    def make(c, k):
        p = SeededFakePolicy(c, k, events); real = p.generate

        def generate(case, seed):
            if fail_after is not None and len(played) >= fail_after: raise doctor_exhausted()
            played.append(case['case_id']); return real(case, seed)
        p.generate = generate; return p
    monkeypatch.setattr(RT, 'make_policy', make)
    path = tmp_path / f'eval_cfg_{cfg["policy"]["lora"]["init_seed"]}.json'; path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, 'argv', ['eval_checkpoint.py', '--config', str(path), '--out', str(out), *flags, '--launch'])
    EC.main(); return played


def doctor_exhausted():
    from src.common.retry import RetryExhausted
    return RetryExhausted('OpenRouter google/gemma-3-12b-it: FAILED (test)')


@needs_data
def test_checkpoint0_eval_resumes_reuses_and_rejects_another_adapter(wired, monkeypatch, tmp_path):
    from src.common.retry import RetryExhausted
    cfg = {**wired.cfg, 'evaluation': {'checkpoint_every_cases': 50, 'split': 'val50', 'cases_file': 'local.jsonl', 'seed': 0}}
    out = tmp_path / 'ck0_eval'
    with pytest.raises(RetryExhausted): eval_main(monkeypatch, tmp_path, wired.events, cfg, out, '--initial', fail_after=2)
    rows = [json.loads(l) for l in (out / 'cases.jsonl').read_text().splitlines()]
    assert [r['case_id'] for r in rows] == ['v0', 'v1'] and not (out / 'result.json').exists()        # partial, no result
    # a new launch (fresh process state, different global-RNG use) resumes: only the missing cases are played
    assert eval_main(monkeypatch, tmp_path, wired.events, cfg, out, '--initial') == ['v2', 'v3']
    r = json.loads((out / 'result.json').read_text())
    assert r['n_cases'] == 4 and r['checkpoint_sha256'] == rows[0]['checkpoint_sha256'] and r['lora_init_seed'] == 20261006
    # the run's committed checkpoint_000000 (same seed) is the same adapter: its evaluation is the stored one
    monkeypatch.setattr(RT, 'make_policy', lambda c, k: SeededFakePolicy(c, k, wired.events)); RT.build(wired.cfg)
    man = json.loads((wired.out / 'run_manifest.json').read_text())
    assert man['initial_policy_sha256'] == r['checkpoint_sha256']
    assert eval_main(monkeypatch, tmp_path, wired.events, cfg, out, '--checkpoint', man['initial_checkpoint']) == []
    # a changed seed is another adapter: the stored rows are not reused and the directory is not appended to
    other = {**cfg, 'policy': {**cfg['policy'], 'lora': {**cfg['policy']['lora'], 'init_seed': 7}}}
    before = (out / 'cases.jsonl').read_text()
    with pytest.raises(SystemExit, match='another checkpoint'): eval_main(monkeypatch, tmp_path, wired.events, other, out, '--initial')
    assert (out / 'cases.jsonl').read_text() == before
    assert eval_main(monkeypatch, tmp_path, wired.events, other, tmp_path / 'ck0_eval_seed7', '--initial') == ['v0', 'v1', 'v2', 'v3']


def test_same_seed_same_adapter_sha_across_independent_processes():
    """Two separate interpreter launches (different global-RNG use before the LoRA is made) give the same initial adapter."""
    import subprocess, sys
    code = ("import sys, json, torch; sys.path.insert(0, %r); sys.path.insert(0, %r); import test_run_tree_wiring as W; "
            "from src.online_rl.runtime import seeded_lora; from src.online_rl.integrated.continuation import adapter_sha256; "
            "b = W.tiny_base(); torch.rand(int(sys.argv[1])); print(adapter_sha256(seeded_lora(b, {**W.tiny_lc(W.RESEARCH), 'init_seed': int(sys.argv[2])})))"
            % (str(ROOT), str(ROOT / 'tests')))
    sha = lambda noise, seed: subprocess.run([sys.executable, '-c', code, str(noise), str(seed)], capture_output=True, text=True, check=True,
                                             cwd=ROOT).stdout.strip().splitlines()[-1]
    a, b = sha(1, 20261006), sha(50, 20261006)
    assert len(a) == 64 and a == b and sha(1, 20261007) != a


# ---------- Doctor protocol v2 (local Ollama Doctor, hidden status, adaptive halting) through the production wiring ----------
def v2_config(cfg):
    from test_doctor_v2 import BACKEND, HALT
    ro = {**{k: v for k, v in cfg['rollout'].items() if k != 'doctor_opening'}, 'doctor_backend': BACKEND, 'halting': HALT,
          'doctor_model': f"ollama/{BACKEND['model']}@{BACKEND['model_digest'][:12]}", 'rounds': 5}
    return {**cfg, 'rollout': ro}


class V2Policy(SeededFakePolicy):
    def __init__(self, cfg, key, events, http):
        from src.online_rl.doctor import make_doctor
        super().__init__(cfg, key, events); self.args.rounds = cfg['rollout']['rounds']
        self.doctor = make_doctor(cfg['rollout'], self.doctor_system); self.doctor.http = http


@needs_data
@pytest.mark.parametrize('arm', ['evolving', 'outcome_only'])
def test_v2_doctor_status_never_reaches_feedback_rubric_or_judges(wired, monkeypatch, arm):
    from test_doctor_v2 import FakeOllama
    from src.online_rl.tree.conditions import arm_config
    states = []

    class Judge(FakeJev):
        def decide(self, state, questions, purpose='jev'): states.append(state); return super().decide(state, questions, purpose)
    http = FakeOllama()
    monkeypatch.setattr(RT, 'judge_client', lambda model, key, system: Judge(wired.events, model=model, system=system))
    monkeypatch.setattr(RT, 'make_policy', lambda c, k: V2Policy(c, k, wired.events, http))
    cfg = arm_config(v2_config(wired.cfg), arm); rec = RT.launch(cfg, 'A')
    case = json.loads((Path(cfg['output_dir']) / 'cases/case_000000.json').read_text())
    assert {l['stop_reason'] for l in case['tree']['leaves']} == {'doctor_halt'} and rec['case_cursor'] == 1
    assert not api(wired, 'doctor_turn') and not api(wired, 'final') and http.chats                       # every Doctor call went to Ollama
    text = json.dumps([e[3] for e in api(wired)]) + json.dumps(states) + json.dumps(case['tree']['leaves'])
    assert 'STATUS' not in text and 'open_issue' not in text and 'collaborator_objection' not in text
    if arm == 'outcome_only': assert api(wired) == [] and states == []                                    # zero rubric-side calls
    else: assert api(wired, 'feedback') and api(wired, 'generate') and states
    man = json.loads((Path(cfg['output_dir']) / 'run_manifest.json').read_text())
    assert man['doctor_backend']['model_digest'] == 'a' * 64 and man['doctor_api_seeded'] and man['halting']['min_agent_turns'] == 2


@needs_data
def test_frozen_shared_d1_through_the_production_wiring(wired, monkeypatch, tmp_path):
    """build() binds the pinned training table: the tree opens with the frozen D1 (no D1 Ollama call), D2+ go to Ollama,
    and the manifest records mode / schema / table SHA256s."""
    from test_doctor_v2 import FakeOllama
    from src.online_rl import doctor_openings as DO
    from src.online_rl.cases import load_cases
    from src.online_rl.doctor import make_doctor
    from src.online_rl.tree.conditions import arm_config
    cfg = arm_config(v2_config(wired.cfg), 'outcome_only'); ro = cfg['rollout']
    gen = make_doctor(ro, (ROOT / 'prompts/user_simulator.txt').read_text()); gen.http = FakeOllama()
    gen_text = lambda c: {**gen.turn(c, [], 1, 5, DO.train_seed(cfg, c['case_id'])), 'visible': f"FROZEN OPENING {c['case_id']}"}
    cases = load_cases(ROOT / cfg['data']['train'])
    recs = [DO.make_record(DO.identity('train', c, gen, 5, DO.train_seed(cfg, c['case_id'])), gen_text(c), {'test': True}) for c in cases]
    sha = DO.write_table(tmp_path / 'train.json', 'train', gen, 5, recs, {'test': True})
    spec = {'mode': DO.MODE, 'schema': DO.SCHEMA, 'train_table': str(tmp_path / 'train.json'), 'train_table_sha256': sha,
            'eval_table': str(tmp_path / 'train.json'), 'eval_table_sha256': sha}
    cfg = {**cfg, 'rollout': {**ro, 'doctor_opening': spec}}
    http = FakeOllama()
    monkeypatch.setattr(RT, 'judge_client', lambda model, key, system: FakeJev(wired.events, model=model, system=system))
    monkeypatch.setattr(RT, 'make_policy', lambda c, k: V2Policy(c, k, wired.events, http))
    rec = RT.launch(cfg, 'A'); case = json.loads((Path(cfg['output_dir']) / 'cases/case_000000.json').read_text())
    d1 = {n['prefix'][0][3] for n in case['tree']['nodes']}
    assert d1 == {f"FROZEN OPENING {case['case_id']}"} and rec['case_cursor'] == 1
    assert http.chats and not any('discussion turn 1 of' in u for u, _ in http.chats)            # D1 never generated; D2+ were
    man = json.loads((Path(cfg['output_dir']) / 'run_manifest.json').read_text())
    assert man['doctor_opening']['doctor_opening_mode'] == DO.MODE and man['doctor_opening']['train_table_sha256'] == sha
