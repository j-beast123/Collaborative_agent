import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest
from src.online_rl.tree import b200_runner as R
from src.online_rl.tree.records import AgentNode, Leaf, Tree
from src.online_rl.tree.team import node_credit
from src.online_rl.tree.reward import leaf_outcomes
from src.online_rl.integrated.records import atomic_write_json


def records(policy='before', no_scores=False):
    result = []
    for i in range(5):
        cid = f'case-{i}'
        nodes = tuple(AgentNode(f'{cid}/{j}', None, cid, 1, (('doctor', 'D1', 1, 'question'),), 'reply',
            {'prompt_text': 'same', 'prompt_input_ids': [1], 'completion_ids': [2]}, policy, j, 'max_agent_turns', True, None) for j in range(4))
        leaves = tuple(Leaf(n.node_id, cid, (n.node_id,), (), '', 'A', float(j % 2), 'max_agent_turns') for j, n in enumerate(nodes))
        tree = Tree(cid, policy, 1, 4, nodes, leaves)
        ra = {} if no_scores else {n.node_id: j / 3 for j, n in enumerate(nodes)}
        targets = node_credit(0, 1)(tree, leaf_outcomes(tree), ra)
        result.append({'case_cursor': i, 'case_id': cid, 'policy_sha256': policy, 'tree': asdict(tree), 'agent_targets': targets})
    return result


class Updater:
    def __init__(self): self.steps = 0; self.policy_hash = 'before'; self.groups = []
    def update_groups(self, groups):
        self.groups = groups
        if not any(r['reward'] is not None for g in groups for r in g):
            return {'optimizer_step': False, 'skip_reason': 'no_group_with_nonzero_advantage', 'optimizer_steps': self.steps}
        self.steps += 1; self.policy_hash = 'after'
        return {'optimizer_step': True, 'optimizer_steps': self.steps}


def state():
    return {'case_cursor': 0, 'batch_cursor': 0, 'config_sha256': 'cfg', 'source_sha256': 'code', 'policy_sha256': 'before'}


def save(up, path, proof):
    atomic_write_json(path / 'COMPLETE.json', proof)
    atomic_write_json(path / 'fake_optimizer.json', {'steps': up.steps, 'policy_hash': up.policy_hash})


def load(up, path):
    saved = json.loads((path / 'fake_optimizer.json').read_text())
    up.steps = saved['steps']; up.policy_hash = saved['policy_hash']


def test_full_five_case_boundary_and_resume_after_checkpoint_before_state(tmp_path, monkeypatch):
    up = Updater(); recs = records(); saved = state()
    real_write = R.atomic_write_json
    def interrupted(path, value):
        if path.name == 'run_state.json': raise RuntimeError('interrupted after checkpoint')
        real_write(path, value)
    monkeypatch.setattr(R, 'atomic_write_json', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        R.commit_batch({}, tmp_path, saved, recs, up, lambda: up.policy_hash, save=save, load=load)
    assert up.steps == 1 and len(up.groups) == 5 and all(len(g) == 4 for g in up.groups)
    monkeypatch.setattr(R, 'atomic_write_json', real_write)
    restarted = Updater()
    result = R.commit_batch({}, tmp_path, saved, recs, restarted, lambda: restarted.policy_hash, save=save, load=load)
    assert restarted.steps == 1 and restarted.groups == []
    assert restarted.policy_hash == 'after'
    assert result['case_cursor'] == 5 and result['batch_cursor'] == 1
    assert json.loads((tmp_path / 'run_state.json').read_text()) == result


def test_refuses_partial_batch_or_stale_policy(tmp_path):
    up = Updater()
    with pytest.raises(RuntimeError, match='five'):
        R.commit_batch({}, tmp_path, state(), records()[:4], up, lambda: up.policy_hash, save=save, load=load)
    with pytest.raises(RuntimeError, match='Mixed source'):
        R.commit_batch({}, tmp_path, state(), records('stale'), up, lambda: up.policy_hash, save=save, load=load)
    assert up.steps == 0


def test_rubric_only_has_no_outcome_fallback_and_na_batch_commits_skip(tmp_path):
    recs = records(no_scores=True); up = Updater()
    assert all(n['reward'] is None for r in recs for n in r['agent_targets']['nodes'].values())
    result = R.commit_batch({}, tmp_path, state(), recs, up, lambda: up.policy_hash, save=save, load=load)
    assert result['case_cursor'] == 5 and up.steps == 0
    diag = json.loads((tmp_path / 'diagnostics/batch_000001.json').read_text())
    assert diag['actual_optimizer_steps_this_batch'] == 0
    assert diag['optimizer_update']['skip_reason']


def test_outcomes_do_not_change_rubric_reward_or_advantage():
    from src.online_rl.integrated.records import from_dict
    from src.online_rl.integrated.metrics import advantages
    rec = records()[0]; tree = from_dict(Tree, rec['tree'])
    opposite = replace(tree, leaves=tuple(replace(l, outcome=1 - l.outcome) for l in tree.leaves))
    ra = {n: row['R_A'] for n, row in rec['agent_targets']['nodes'].items()}
    a = node_credit(0, 1)(tree, leaf_outcomes(tree), ra)
    b = node_credit(0, 1)(opposite, leaf_outcomes(opposite), ra)
    rewards = lambda t: [r['reward'] for r in t['nodes'].values()]
    assert rewards(a) == rewards(b)
    assert advantages(rewards(a), 'mean_center') == advantages(rewards(b), 'mean_center')


def test_schedule_is_24_ordered_batches_of_five():
    batches = [R.batch_slots(i, 120, interval=30, workers=1, lanes=5) for i in range(0, 120, 5)]
    assert len(batches) == 24 and all(len(b) == 5 for b in batches)
    assert [s['index'] for b in batches for s in b] == list(range(120))


def test_evaluation_rejects_incomplete_training(tmp_path):
    with pytest.raises(RuntimeError, match='completed final'):
        R.final_evaluation(R.load_config(), tmp_path, {'case_cursor': 115}, None, 'protocol', None)


def test_real_cpu_optimizer_checkpoint_restores_policy_rng_and_step(tmp_path, monkeypatch):
    from test_agent_precommit import tiny, joint_trace
    from src.online_rl.integrated.continuation import adapter_sha256
    torch, model, up = tiny()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    traces = [joint_trace(i) for i in range(4)]
    for trace, logprob in zip(traces, up.behavior_logprobs(traces)): trace['behavior_logprobs'] = logprob
    initial = adapter_sha256(model)
    group = [{'trajectory_id': f'n{i}', 'case_id': 'g', 'reward': i % 2,
              'rollout_policy_sha256': initial, 'agent_traces': [trace]} for i, trace in enumerate(traces)]
    up.update_groups([group])
    expected = adapter_sha256(model); path = tmp_path / 'checkpoint'
    R.save_checkpoint(up, path, {'policy_sha256': expected})
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        next(p for p in model.parameters() if p.requires_grad).add_(1)
    up.steps = 99; torch.manual_seed(9)
    R.load_checkpoint(up, path)
    assert adapter_sha256(model) == expected and up.steps == 1
    assert torch.equal(torch.get_rng_state(), rng)
    assert up.optimizer.state_dict()['state']


def test_final_evaluation_is_single_final_job_622_ordered_seeds_and_resume(tmp_path, monkeypatch):
    cases = [{'case_id': f'test-{i}'} for i in range(622)]
    monkeypatch.setattr('src.online_rl.cases.load_cases', lambda path: cases)
    calls = []
    def generate(case, seed):
        calls.append((case['case_id'], seed))
        return {'dialogue': [], 'final_option': None if seed == 621 else 'A', 'num_discussion_rounds': 2,
                'private_needs': [{'need_id': f'need-{seed}'}]}
    policy = SimpleNamespace(generate=generate)
    saved = {**state(), 'case_cursor': 120, 'checkpoint': str(tmp_path / 'checkpoints/checkpoint_000120')}
    outcome = lambda case, option: None if option is None else 1
    R.final_evaluation(R.load_config(), tmp_path, saved, policy, 'proto', outcome)
    assert sorted(calls, key=lambda c: c[1]) == [(c['case_id'], i) for i, c in enumerate(cases)]
    result = json.loads((tmp_path / 'final_accuracy.json').read_text())
    assert result['correct'] == 621 and result['unparsed'] == 1 and result['accuracy'] == 621 / 622
    R.final_evaluation(R.load_config(), tmp_path, saved, policy, 'proto', outcome)
    assert len(calls) == 622
    with pytest.raises(RuntimeError, match='identity changed'):
        R.final_evaluation(R.load_config(), tmp_path, {**saved, 'policy_sha256': 'another'}, policy, 'proto', outcome)


def test_http_attempt_audit_keeps_usage_but_never_credentials_or_prompt(tmp_path, monkeypatch):
    import io
    import urllib.request
    from src.common.http_audit import request_json
    from src.common import usage
    path = tmp_path / 'attempts.jsonl'
    monkeypatch.setenv('SHERPA_HTTP_AUDIT', str(path))
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(json.dumps(
        {'id': 'req-test', 'model': 'fake', 'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}).encode()))
    request = urllib.request.Request('http://test.invalid', data=b'private-prompt', headers={'Authorization': 'Bearer TEST-SECRET'})
    with usage.case('case-a'): request_json(request, 1)
    data = path.read_text()
    assert 'TEST-SECRET' not in data and 'private-prompt' not in data
    row = json.loads(data)
    assert row['case_id'] == 'case-a' and row['input_tokens'] == 3 and row['output_tokens'] == 2


@pytest.fixture
def lr_environment(monkeypatch):
    monkeypatch.delenv('SHERPA_LEARNING_RATE', raising=False)
    monkeypatch.delenv('SHERPA_DOCTOR_RECEIPT', raising=False)
    monkeypatch.setattr('src.online_rl.tree.b200_doctor_setup.receipt_path', lambda: R.ROOT / '__no_test_receipt__')


def test_lr_default_env_cli_precedence_and_separate_paths(lr_environment, monkeypatch):
    default = R.load_config()
    assert default['optimizer']['learning_rate'] == 1e-5
    assert R.load_config(learning_rate='0.00001') == default
    monkeypatch.setenv('SHERPA_LEARNING_RATE', '5e-6')
    changed = R.load_config()
    assert changed['optimizer']['learning_rate'] == 5e-6
    assert changed['output_dir'] == default['output_dir'] + '/lr_5e-06'
    assert changed['experiment']['output_root'] == changed['output_dir']
    assert R.load_config(learning_rate='1e-5') == default
    assert R.load_config(learning_rate='0.000005') == changed
    assert changed['optimizer'] == {**default['optimizer'], 'learning_rate': 5e-6}
    for field in ('agent_reward', 'rollout', 'verification', 'policy', 'two_gpu', 'rubric'):
        assert changed[field] == default[field]


@pytest.mark.parametrize('bad', ['0', '-1e-5', 'nan', 'inf', '-inf', '', 'oops', True, '1e-999'])
def test_invalid_learning_rate_rejected_before_loading_models(lr_environment, bad):
    with pytest.raises(ValueError, match='finite positive'):
        R.load_config(learning_rate=bad)


def test_same_lr_resume_identity_and_changed_lr_refusal(lr_environment):
    from src.online_rl.integrated.records import sha256_json
    cfg = R.load_config(learning_rate='5e-6')
    saved = {'config_sha256': sha256_json(cfg), 'source_sha256': 'code'}
    R.validate_resume_identity(saved, R.load_config(learning_rate='0.000005'), 'code')
    with pytest.raises(RuntimeError, match='learning rate'):
        R.validate_resume_identity(saved, R.load_config(learning_rate='1e-5'), 'code')


def test_lr_override_reaches_real_cpu_optimizer(lr_environment):
    import torch
    from src.online_rl.integrated.grpo import make_updater
    cfg = R.load_config(learning_rate='5e-6')
    policy = SimpleNamespace(torch=torch, model=torch.nn.Linear(2, 2))
    updater = make_updater(policy, cfg)
    assert all(group['lr'] == 5e-6 for group in updater.optimizer.param_groups)
    assert updater.identity()['learning_rate'] == 5e-6


def test_lr_cli_preflight_and_progress_report(lr_environment, monkeypatch, capsys):
    import runpy, sys
    from scripts.online_rl import run_b200
    monkeypatch.setattr(R, 'physical_devices', lambda: ['0', '1'])
    monkeypatch.setattr(R, 'preflight', lambda cfg, **kw: [])
    monkeypatch.setenv('SHERPA_LEARNING_RATE', '8e-6')
    monkeypatch.setattr(sys, 'argv', ['run_b200.py', '--action', 'preflight', '--cpu-only', '--learning-rate', '5e-6'])
    run_b200.main()
    report = json.loads(capsys.readouterr().out)
    assert report['learning_rate'] == 5e-6 and report['output'].endswith('/lr_5e-06')
    monkeypatch.setattr(sys, 'argv', ['report_b200.py', '--learning-rate', '5e-6'])
    runpy.run_path(str(R.ROOT / 'scripts/online_rl/report_b200.py'), run_name='__main__')
    assert json.loads(capsys.readouterr().out)['output'] == report['output']
