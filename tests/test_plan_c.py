"""Fake policies/Doctors only: state isolation, turn credit and durable training."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.online_rl.sherpa import plan_c_rollouts as ro
from src.online_rl.sherpa.plan_c_backend import (
    TurnChecklistBackend, generation_examples, scoring_request, generator_prompt,
)
from src.online_rl.sherpa import plan_c_training as training
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.experiment import load_config

ROOT = Path(__file__).resolve().parents[1]


class Doctor:
    model = 'fake-doctor'
    protocol = {'protocol': 'fake-v2'}
    min_agent_turns = 2
    def __init__(self, halt_at=None):
        self.calls, self.halts, self.halt_at = [], [], halt_at
    def turn(self, case, dialogue, round_, rounds, seed):
        self.calls.append(('turn', round_, deepcopy(dialogue)))
        return {'visible': f'Doctor question {round_}', 'status': {'round': round_},
                'status_error': None, 'seed': seed, 'private_state': {'hidden': round_}}
    def final(self, case, dialogue, seed):
        self.calls.append(('final', 0, deepcopy(dialogue)))
        return 'Final answer: A' if seed % 2 else 'Final answer: B'
    def halt(self, **kw):
        self.halts.append(deepcopy(kw))
        return 'doctor_halt' if self.halt_at and kw['agent_turns'] >= self.halt_at else None


def fake_policy(monkeypatch, halt_at=None):
    monkeypatch.setattr('src.online_rl.sherpa.rollouts.adapter_sha256', lambda model: 'policy0')
    doctor = Doctor(halt_at)
    samples = []
    def sample(case, history, turn, seeds):
        samples.append((turn, deepcopy(history), seeds[0]))
        return [{'prompt_text': json.dumps(history), 'completion_text': f'Consider the reasoning {seeds[0]}',
                 'completion_ids': [turn], 'old_policy_logprobs': [-1.0], 'seed': seeds[0]}]
    return SimpleNamespace(model=object(), doctor=doctor, sample_completions=sample), samples


def setup():
    cfg = load_config('configs/plan_c.json')
    cfg['plan_c']['responses_per_turn'] = 3
    cfg['execution'].update(doctor_workers=2, agent_workers=2)
    case = {'case_id': 'synthetic', 'question': 'Synthetic question',
            'options': {'A': 'alpha', 'B': 'beta'}, 'ground_truth': {'correct_option': 'A'}}
    return cfg, case


def test_spine_histories_unchanged_target_only_and_cache(monkeypatch, tmp_path):
    cfg, case = setup()
    policy, samples = fake_policy(monkeypatch)
    cache = CallCache(tmp_path / 'calls.jsonl', enabled=True)
    ready = []
    result = ro.collect_turn_groups(case, cfg, policy, cache, 1, on_group=ready.append)
    assert len(result['groups']) == len(ready) == 4
    assert sum(kind == 'turn' and turn == 1 for kind, turn, _ in policy.doctor.calls) == 1
    assert len(samples) == 4 + 2 * (4 + 3 + 2 + 1)
    original = result['main_trajectory']['dialogue']
    for group in result['groups']:
        turn = group['target_turn']
        assert group['state']['doctor_private_state'] == {'hidden': turn}
        assert group['state']['doctor_status'] == {'round': turn}
        candidates = group['candidates']
        assert len(candidates) == 3 and sum(c['is_main'] for c in candidates) == 1
        assert candidates[0]['dialogue'] == original
        assert all(c['shared_history'] == group['state']['history'] for c in candidates)
        for c in candidates:
            prefix = c['dialogue'][:len(c['shared_history'])]
            assert prefix == c['shared_history']
            action = c['dialogue'][len(prefix)]
            assert action['round'] == turn and action['text'] == c['target_response']
            assert c['agent_turns'] == 4 and c['stop_reason'] == 'max_agent_turns'
        rows = ro.training_records(group, [
            {'trajectory_id': c['trajectory_id'], 'reward': i / 2}
            for i, c in enumerate(candidates)])
        assert all(len(r['agent_traces']) == 1 for r in rows)
        assert all(r['agent_traces'][0]['completion_ids'] == [turn] for r in rows)
        assert len({r['case_id'] for r in rows}) == 1
    # Neither the opening nor completed continuations repeat after restart.
    count = len(samples), len(policy.doctor.calls)
    resumed = ro.collect_turn_groups(case, cfg, policy, CallCache(tmp_path / 'calls.jsonl'), 1)
    assert count == (len(samples), len(policy.doctor.calls))
    assert [g['candidates'] for g in resumed['groups']] == [g['candidates'] for g in result['groups']]
    # A new step generates fresh main and counterfactual observations.
    ro.collect_turn_groups(case, cfg, policy, cache, 2)
    assert len(samples) == count[0] * 2


def test_early_doctor_halt_uses_absolute_turn_and_stored_status(monkeypatch):
    cfg, case = setup()
    policy, _ = fake_policy(monkeypatch, halt_at=2)
    result = ro.collect_turn_groups(case, cfg, policy, CallCache(None), 1)
    assert [g['target_turn'] for g in result['groups']] == [1, 2]
    assert all(c['agent_turns'] == 2 and c['stop_reason'] == 'doctor_halt'
               for g in result['groups'] for c in g['candidates'])
    assert all(h['prev_status'] == {'round': h['agent_turns']} for h in policy.doctor.halts)
    assert all(row['round'] <= 3 for g in result['groups'] for c in g['candidates']
               for row in c['dialogue'] if row['stage'] != 'FINAL')


def test_group_rejects_mismatched_prompt_and_reward_ids(monkeypatch):
    cfg, case = setup()
    policy, _ = fake_policy(monkeypatch)
    group = ro.collect_turn_groups(case, cfg, policy, CallCache(None), 1)['groups'][0]
    scores = [{'trajectory_id': c['trajectory_id'], 'reward': 1} for c in group['candidates']]
    with pytest.raises(ValueError, match='aligned'):
        ro.training_records(group, scores[:-1])
    wrong = deepcopy(scores); wrong[0]['trajectory_id'] = 'foreign'
    with pytest.raises(ValueError, match='IDs'):
        ro.training_records(group, wrong)
    wrong_group = deepcopy(group); wrong_group['candidates'][0]['target_trace']['prompt_text'] = 'another history'
    with pytest.raises(RuntimeError, match='mixes'):
        ro.training_records(wrong_group, scores)


def test_blind_judge_and_generator_payloads(monkeypatch):
    cfg, case = setup()
    policy, _ = fake_policy(monkeypatch)
    candidate = ro.collect_turn_groups(case, cfg, policy, CallCache(None), 1)['groups'][0]['candidates'][0]
    rubrics = [{'name': 'Check', 'situation': 'always', 'behavior': 'Checks a premise'}]
    request = scoring_request(case, candidate, rubrics)
    assert set(request) == {'case', 'target_turn', 'shared_history', 'target_response', 'rubrics'}
    assert 'ground_truth' not in request['case']
    assert not any(row.get('stage') == 'FINAL' for row in request['shared_history'])
    changed = deepcopy(candidate)
    changed.update(outcome=1-candidate['outcome'], final_option='UNKNOWN', dialogue=[{'text':'FUTURE_SECRET'}])
    assert scoring_request(case, changed, rubrics) == request
    examples = generation_examples([{'trajectory': candidate, 'doctor_reflection': {'text':'private opinion'}}])
    assert 'target_trace' not in json.dumps(examples)
    assert examples[0]['continuation'][0] == candidate['dialogue'][2]
    assert examples[0]['doctor_reflection'] == 'private opinion'


def test_target_reflection_uses_standalone_system_and_exact_evidence(monkeypatch):
    cfg, case = setup()
    backend = TurnChecklistBackend.__new__(TurnChecklistBackend)
    backend.cfg = cfg
    backend.reflection_prompt = 'PRIVATE TARGET REFLECTION'
    seen = []
    structured = {k: {'evidence_id':'target','reflection':'I wanted a premise check.'}
                  for k in ('helpful','hindering','wished_for')}
    def chat(user, seed, max_tokens, system, format_schema=None):
        seen.append((json.loads(user), system, format_schema))
        return json.dumps(structured)
    backend.policy = SimpleNamespace(doctor=SimpleNamespace(system='Discussion: FINAL', _chat=chat))
    candidate = {'trajectory_id':'c', 'target_turn':1, 'shared_history':[{'text':'opening'}],
                 'target_response':'actual whole target', 'dialogue':[{}, {}, {'text':'FINAL'}], 'outcome':0}
    result = backend.reflect(case, candidate, 4)
    assert seen[0][1] == 'PRIVATE TARGET REFLECTION'
    assert set(result['evidence'].values()) == {'actual whole target'}
    assert seen[0][0]['continuation'] == [{'text':'FINAL'}]
    assert seen[0][2]['properties']['helpful']['properties']['evidence_id']['enum'] == ['target','none']


def test_config_and_transferred_prompt_do_not_modify_ours():
    from src.online_rl.sherpa.plan_c_config import config_problems
    ours_before = (ROOT/'configs/ours.json').read_bytes()
    cfg = load_config('configs/plan_c.json')
    assert not config_problems(cfg, ROOT)
    assert cfg['rollout']['branching'] == 1 and cfg['plan_c']['responses_per_turn'] == 32
    assert cfg['advantage_normalization']['mode'] == 'group_mean'
    assert cfg['output_dir'] != load_config('configs/ours.json')['output_dir']
    text, provenance = generator_prompt(cfg, ROOT)
    assert 'PLAN C TARGET-TURN CONTRACT' in text and provenance['optimized_for_plan_c'] is False
    assert text.startswith((ROOT/'prompts/sherpa/trajectory_checklist_generator_v4.txt').read_text())
    assert provenance['base_prompt_source'] == 'explicit_user_revision'
    assert provenance['source_prompt_sha256'] != provenance['source_optimized_prompt_sha256']
    judge = (ROOT/cfg['rubric_pipeline']['judge_prompt_file']).read_text()
    assert 'Evidence rule:' in judge and 'Evaluate ONLY target_response' in judge
    assert 'occurs anywhere IN target_response' in judge
    assert provenance['source_prompt_sha256'] != provenance['prompt_sha256']
    assert (ROOT/'configs/ours.json').read_bytes() == ours_before
    cfg['output_dir'] = load_config('configs/ours.json')['output_dir']
    assert any('overlap' in p for p in config_problems(cfg, ROOT))


def test_occupied_gpu_aborts_before_model_loading(tmp_path, monkeypatch):
    from scripts.online_rl import run_plan_c as runner
    import fcntl
    cfg = load_config('configs/plan_c.json')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr('src.online_rl.tree.experiment.load_config', lambda path:cfg)
    monkeypatch.setattr('src.online_rl.sherpa.plan_c_config.config_problems', lambda *a:[])
    monkeypatch.setattr('src.online_rl.sherpa.plan_c_backend.generator_prompt', lambda *a:('', {}))
    monkeypatch.setattr('src.online_rl.cases.load_cases', lambda *a:[])
    monkeypatch.setattr('src.online_rl.sherpa.rollouts.schedule', lambda *a:[])
    from src.online_rl import runtime
    monkeypatch.setattr(runtime, 'SharedPolicy', lambda *a:pytest.fail('A model was loaded on an occupied GPU'))
    calls = []
    def flock(fd, mode):
        calls.append(mode)
        if len(calls) == 2:
            raise BlockingIOError('fake occupied GPU')
    monkeypatch.setattr(fcntl, 'flock', flock)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit) as error:
        runner.main(['--launch'])
    assert error.value.code == 2 and len(calls) == 2


def test_turn_groups_one_update_and_resume(tmp_path, monkeypatch):
    cfg = {'output_dir':'plan_c', 'schedule':{'seed':0}, 'plan_c':{'responses_per_turn':3},
           'advantage_normalization':{'mode':'group_mean'},
           'training_execution':{'case_workers':4}}
    cases = [{'case_id':str(i)} for i in range(4)]
    policy = SimpleNamespace(model=SimpleNamespace(version=0, eval=lambda:None))
    synced, updated, collected = [], [], []
    policy.sync_replicas = lambda expected_sha256: synced.append(expected_sha256)
    monkeypatch.setattr(training, 'adapter_sha256', lambda m:str(m.version))
    monkeypatch.setattr(training, 'generator_prompt', lambda *a,**k:('TURN PROMPT', {'prompt_sha256':'frozen'}))
    monkeypatch.setattr(training, 'schedule', lambda *a:[cases]*90)
    def collect(case, cfg, policy, cache, step, on_group):
        collected.append((step,case['case_id']))
        for turn in (1,2):
            on_group({'target_turn':turn, 'case_id':case['case_id'], 'group_id':f'{case["case_id"]}|{turn}'})
        return {'spine':{}, 'main_trajectory':{}, 'seconds':1}
    monkeypatch.setattr(training, 'collect_turn_groups', collect)
    def assess(case, group, cfg, backend, cache, prompt, step, expected, folder):
        return {**group, 'metrics':{'zero_advantage':False,'auc':.7}, 'training_records':[
            {'case_id':group['group_id'],'trajectory_id':f'{group["group_id"]}/{i}',
             'rollout_policy_sha256':expected,'reward':i/2,
             'agent_traces':[{'prompt_text':group['group_id'], 'completion_ids':[group['target_turn']]}]}
            for i in range(3)]}
    monkeypatch.setattr(training, 'assess_turn', assess)
    class Updater:
        def update_groups(self, groups):
            assert len(groups) == 8 and all(len(g)==3 for g in groups)
            assert all(len(r['agent_traces']) == 1 for g in groups for r in g)
            assert len(collected) == 4*(len(updated)+1)
            updated.append(groups); policy.model.version += 1
            return {'optimizer_step':True}
    monkeypatch.setattr(training, 'make_updater', lambda *a:Updater())
    def save(updater, path, receipt):
        path.mkdir(parents=True); (path/'COMPLETE.json').write_text(json.dumps(receipt)); return str(path)
    monkeypatch.setattr(training, 'save_checkpoint', save)
    monkeypatch.setattr(training, 'load_checkpoint', lambda updater,path:
                        setattr(policy.model, 'version', int(json.loads((Path(path)/'COMPLETE.json').read_text())['policy_sha256'])))
    state = training.train(cfg, policy, None, cases, tmp_path, max_steps=2)
    assert state['completed_steps'] == 2 and synced == ['0','1']
    state = training.train(cfg, policy, None, cases, tmp_path, max_steps=1)
    assert state['completed_steps'] == 3 and synced == ['0','1','2']
    metrics = json.loads((tmp_path/'plan_c/steps/step_000003/metrics.json').read_text())
    assert metrics['turn_groups'] == 8 and metrics['trajectories'] == 24
    assert metrics['training_scope'] == 'target_action_only'
    assert metrics['advantage_mode'] == 'group_mean'
    assert json.loads((tmp_path/'plan_c/config.json').read_text())['advantage_normalization']['mode'] == 'group_mean'
    assert (tmp_path/'plan_c/steps/step_000003/learning_signal.json').is_file()
    enqueued = []
    monkeypatch.setattr('src.common.checkpoint_hub.enqueue_safely',
                        lambda cfg,home,state:enqueued.append(state['completed_steps']))
    state = training.train(cfg, policy, None, cases, tmp_path, max_steps=7)
    assert state['completed_steps'] == 10 and enqueued == [10]
    assert (tmp_path/'plan_c/checkpoints/checkpoint_000010/COMPLETE.json').is_file()
    assert not (tmp_path/'plan_c/recovery/step_000009').exists()
    cfg['plan_c']['responses_per_turn'] = 4
    with pytest.raises(RuntimeError, match='changed'):
        training.train(cfg, policy, None, cases, tmp_path, max_steps=1)


def test_plan_c_targets_have_behavior_evidence_and_mean_only_groups(monkeypatch):
    from src.online_rl.integrated.grpo import prepare_advantage_groups, validate_raw_sampling, teacher_forced_behavior_records
    from src.online_rl.integrated.metrics import advantages
    cfg, case = setup()
    cfg['plan_c']['responses_per_turn'] = 32
    policy, _ = fake_policy(monkeypatch)
    group = ro.collect_turn_groups(case, cfg, policy, CallCache(None), 1)['groups'][1]
    rewards = [.49]*15 + [.51]*15 + [.5]*2
    for candidate in group['candidates']:
        trace = candidate['target_trace']
        trace.update(sampling_logprobs=list(trace['old_policy_logprobs']),
                     sampling_temperature=1., sampling_top_p=1., sampling_top_k=0,
                     logprob_semantics='raw_policy_before_sampling_warpers_v1')
    rows = ro.training_records(group,[{'trajectory_id':c['trajectory_id'],'reward':r}
                                      for c,r in zip(group['candidates'],rewards)])
    for row in rows:
        trace = row['agent_traces'][0]
        validate_raw_sampling(trace,recorded=True,label='Plan C target')
        assert trace['behavior_logprobs'] == trace['old_policy_logprobs']
        assert trace['completion_ids'] == [2]  # No A1 prefix or A3/A4 suffix tokens.
    used, adv, skipped, details = prepare_advantage_groups([rows],cfg['advantage_normalization']['mode'])
    assert len(used) == 30 and adv == pytest.approx([-.01]*15+[.01]*15)
    assert details[0]['advantages'] == advantages(rewards, 'group_mean')
    assert len({r['case_id'] for r in used}) == 1
    assert all('outcome' in r for r in used)
    monkeypatch.setattr('src.online_rl.integrated.grpo.behavior_logprobs',
                        lambda model,traces,chunk:[[-.9] for trace in traces])
    aligned, audit = teacher_forced_behavior_records(None,used,16)
    assert audit['tokens'] == 30 and aligned[0]['agent_traces'][0]['behavior_logprobs'] == [-.9]
    assert used[0]['agent_traces'][0]['behavior_logprobs'] == [-1.]
