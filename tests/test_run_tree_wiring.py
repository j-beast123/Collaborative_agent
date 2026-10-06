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
    c['advantage_normalization'] = {'mode': 'mean_center'}; c['intervention']['reruns_per_intervention'] = 3
    c['selection'] = {'mode': 'threshold', 'min_R_O': 2 / 3}; c['agent_reward'] = {'w_outcome': 0.5, 'w_rubric': 0.5}
    c['prompt_update'].update(every_n_cases=1, minimum_trajectories=2, candidate_count=1, retain_top_k=1, refinement_depth=1)
    c['acceptance'] = {'delta': 0.0, 'p_min': 0.5, 'minimum_trajectories': 1, 'bootstrap_n': 200, 'bootstrap_seed': 0}
    c['smoke'] = {'purpose': 'test'}; return c


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

    def continue_from(self, case, prefix, t, fixed, seed): return self._H().Env.continue_from(self, case, prefix, t, fixed, seed)

    def sample_completions(self, case, prefix, turn, seeds):
        user = self.agent_user_prompt(case, prefix, turn); pid = [1 + (h(user) + i) % 30 for i in range(4)]; self.events.append(('agent', 'node'))
        return [{'prompt_text': user, 'prompt_input_ids': pid, 'completion_ids': [2 + s % 30, 5, 7], 'old_policy_logprobs': [-1.0] * 3,
                 'completion_text': f'reply t{turn} s{s % 97}', 'seed': s, 'stage': f'A{turn}'} for s in seeds]


class FakeJev:
    def __init__(self, events, fail_at=None): self.events, self.n, self.fail_at = events, 0, fail_at

    def decide(self, state, questions, purpose='jev'):
        self.n += 1
        if self.fail_at and self.n == self.fail_at: raise RuntimeError('Jev transport failure')
        self.events.append(('jev', purpose)); out = {}
        for qid in questions:
            v = 'N' if purpose == 'tree_posthoc_rubric_gate' else 'Y' if purpose.endswith('_applicability') else '2'
            out[qid] = {'value': v, 'probabilities': {'0': 0.0, '1': 0.0, '2': 1.0} if v == '2' else {'Y': float(v == 'Y'), 'N': float(v == 'N')}, 'confidence': 1.0}
        return out, {'provider': 'fake'}


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
    monkeypatch.setattr(RT, 'jev_client', lambda: FakeJev(events))
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
    assert turns and all(e[2] == 'google/gemma-3-12b-it' for e in api(wired) if e[1] in ('doctor_turn', 'final', 'feedback'))
    assert all('of at most 2' in e[3] for e in turns if 'turn ' in e[3] and 'of at most' in e[3]) and any('of at most 2' in e[3] for e in turns)
    from src.prompt_opt import harness as H
    assert {e[2] for e in api(wired, 'generate')} == {H.SOL} and len(api(wired, 'generate')) == len(rec['trajectory_inputs'])     # one rubric each
    upd = [i for i, e in enumerate(wired.events) if e[0] == 'update']
    assert len(upd) == 1 and upd[0] == len(wired.events) - 1                                                  # update after every call of the case
    assert any(e == ('agent', 'Score-2') or e == ('agent', 'Score-0') for e in wired.events)
    assert (wired.out / 'checkpoints/case_000000').exists() and (wired.out / 'api_usage.jsonl').exists()
    n_a = len(wired.events)

    run(wired, '--launch', '--phase', 'B')
    st = json.loads((wired.out / 'run_state.json').read_text()); ev = json.loads((wired.out / 'prompt_events/event_0001.json').read_text())
    assert st['case_cursor'] == 1 and st['lineage_cursor'] == 1 and not (wired.out / 'cases/case_000001.json').exists()   # case 1 never runs
    new = wired.events[n_a:]
    assert [e[1] for e in new if e[0] == 'api' and e[1] in ('reflect', 'refine')] == ['reflect', 'refine'] and not any(e[0] == 'update' for e in new)
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
    for name in ('load_key', 'jev_client', 'make_policy'): monkeypatch.setattr(RT, name, boom)
    with pytest.raises(SystemExit, match='phase B requires a completed phase A'): run(wired, '--launch', '--phase', 'B')
    with pytest.raises(SystemExit, match='--phase A or --phase B is required'): run(wired, '--launch')
    assert not wired.out.exists() and not wired.events


@pytest.mark.parametrize('change,msg', [({'rollout': {'branching': 4, 'rounds': 5}}, 'smoke: rollout.branching=4 exceeds 2'),
                                        ({'intervention': {'reruns_per_intervention': 4}}, 'reruns_per_intervention must be exactly 3'),
                                        ({'prompt_update': {'candidate_count': 4}}, 'candidate_count=4 exceeds 1'),
                                        ({'output_dir': json.loads((RT.ROOT / 'configs/online_rl/tree_v1.json').read_text())['output_dir']}, 'collides with configs/online_rl/tree_v1.json'),
                                        ({'rubric': {'prompt_sha256': '0' * 64}}, 'rubric.generator_prompt_file differs'),
                                        ({'schedule': {'cases': 'CALIBRATION_REQUIRED'}}, 'CALIBRATION_REQUIRED')])
def test_pre_launch_validation_stops_before_any_call(wired, monkeypatch, tmp_path, change, msg):
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError('credentials or model loaded'))
    for name in ('load_key', 'jev_client', 'make_policy'): monkeypatch.setattr(RT, name, boom)
    cfg = json.loads(json.dumps(wired.cfg))
    for k, v in change.items(): cfg[k] = {**cfg[k], **v} if isinstance(v, dict) else v
    p = tmp_path / 'bad.json'; p.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match='not launching'): RT.main(['--config', str(p), '--launch', '--phase', 'A'])
    assert msg in ' '.join(RT.blockers(cfg) + RT.smoke_problems(cfg) + RT.setup_problems(cfg, str(p)))
    assert not wired.events


def test_smoke_limits_do_not_apply_to_the_research_config():
    research = json.loads((ROOT / 'configs/online_rl/tree_v1.json').read_text())
    assert RT.smoke_problems(research) == [] and research['rollout'] == {**research['rollout'], 'branching': 4, 'rounds': 5}
    assert research['advantage_normalization'] == {'mode': 'CALIBRATION_REQUIRED'} and RT.phase_problems(research, 'A') == ['--phase is only defined for a smoke config']


@needs_data
def test_phase_a_resumes_from_the_call_cache(wired, tmp_path, monkeypatch):
    run(wired, '--launch', '--phase', 'A'); ref = len(api(wired)); ref_jev = sum(e[0] == 'jev' for e in wired.events)
    out2 = tmp_path / 'resume_smoke'; cfg = {**wired.cfg, 'output_dir': str(out2)}; p = tmp_path / 'resume.json'; p.write_text(json.dumps(cfg))
    monkeypatch.setattr(RT, 'SMOKE_DIR', str(out2)); wired.events.clear()
    monkeypatch.setattr(RT, 'jev_client', lambda: FakeJev(wired.events, fail_at=15))
    with pytest.raises(RuntimeError, match='Jev transport failure'): RT.main(['--config', str(p), '--launch', '--phase', 'A'])
    st = json.loads((out2 / 'run_state.json').read_text()); assert st['case_cursor'] == 0 and not (out2 / 'cases/case_000000.json').exists()
    first = len(api(wired)); first_jev = sum(e[0] == 'jev' for e in wired.events)
    monkeypatch.setattr(RT, 'jev_client', lambda: FakeJev(wired.events))
    RT.main(['--config', str(p), '--launch', '--phase', 'A'])
    assert len(api(wired)) == ref and sum(e[0] == 'jev' for e in wired.events) == ref_jev                    # nothing completed is repeated
    assert first < ref and first_jev == 14
    a, b = json.loads((wired.out / 'cases/case_000000.json').read_text()), json.loads((out2 / 'cases/case_000000.json').read_text())
    assert a['agent_targets'] == b['agent_targets'] and a['trajectory_rubrics'] == b['trajectory_rubrics']
