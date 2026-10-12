"""Training barriers, reward wiring and restart behavior; no model/API work."""
from contextlib import nullcontext
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from src.online_rl.sherpa import training
from src.online_rl.tree.experiment import load_config


def test_ours_config_matches_frozen_prompt_experiment():
    cfg = load_config('configs/ours.json')
    prompt = load_config('configs/sherpa_qwen_doctor.json')
    for key in ('rollout', 'policy', 'data', 'schedule', 'rubric', 'judges', 'rubric_pipeline', 'prompt_optimization'):
        assert cfg[key] == prompt[key]
    assert cfg['output_dir'] != prompt['output_dir']
    assert cfg['execution']['agent_workers'] == 128
    assert cfg['training_execution']['assistant_max_inflight'] == 128
    assert cfg['checkpoint_every_steps'] == 10
    assert cfg['rubric']['generator_model'] == cfg['judges']['scorer_model'] == 'gpt-6-luna'
    assert cfg['rubric']['generation']['api_provider'] == cfg['judges']['api_provider'] == 'openai_direct'
    assert cfg['execution']['api_limits']['luna'] == 100
    assert cfg['execution']['judge_workers'] == 100


def test_four_groups_one_update_and_resume(tmp_path, monkeypatch):
    cases = [{'case_id': str(i)} for i in range(4)]
    cfg = {'output_dir': 'ours', 'method': 'sherpa', 'schedule': {'seed': 0},
           'training_execution': {'case_workers': 4}}
    calls, synced, updates = [], [], []
    policy = SimpleNamespace(model=SimpleNamespace(version=0, eval=lambda: None))
    def sync(expected_sha256):
        synced.append(expected_sha256)
        return {'adapter_sha256': [expected_sha256]*2}
    policy.sync_replicas = sync
    monkeypatch.setattr(training, 'adapter_sha256', lambda model: str(model.version))
    monkeypatch.setattr(training, 'frozen_prompt', lambda *a, **k: ('OPTIMIZED', {'prompt_sha256':'fixed'}))
    monkeypatch.setattr(training, 'schedule', lambda *a: [cases]*90)
    monkeypatch.setattr(training, 'to_dict', lambda tree: {})
    monkeypatch.setattr(training, 'emit', lambda *a, **k: None)
    monkeypatch.setattr(training.usage, 'case', lambda *a: nullcontext())
    barrier = threading.Barrier(4)
    def collect(case, cfg, policy, cache, step):
        barrier.wait(timeout=5)
        calls.append((step, case['case_id']))
        return {}, [{'trajectory_id':str(i), 'outcome':1} for i in range(32)]
    monkeypatch.setattr(training, 'collect', collect)
    monkeypatch.setattr(training, 'reflections', lambda *a: [])
    def assess(case, trajectories, examples, prompt, *args):
        assert prompt == 'OPTIMIZED'
        return {'valid':True,'scores':[{'reward':i%2} for i in range(32)]}
    monkeypatch.setattr(training, 'assess', assess)
    monkeypatch.setattr(training, 'training_records', lambda tree, rows, scores: [
        {'reward':s['reward'],'agent_traces':[{'completion_ids':[1,2]}]} for s in scores])
    monkeypatch.setattr(training, 'case_metrics', lambda *a: {'zero_advantage':False,'auc':0.5})
    class Updater:
        def update_groups(self, groups):
            assert len(groups) == 4 and all(len(g)==32 for g in groups)
            assert sum(r['reward'] for g in groups for r in g) == 64
            updates.append(groups)
            policy.model.version += 1
            return {'optimizer_step':True}
    monkeypatch.setattr(training, 'make_updater', lambda *a: Updater())
    def save(updater, path, receipt):
        path.mkdir(parents=True)
        (path/'COMPLETE.json').write_text(json.dumps(receipt))
        return str(path)
    monkeypatch.setattr(training, 'save_checkpoint', save)
    def load(updater, path):
        policy.model.version = int(json.loads((Path(path)/'COMPLETE.json').read_text())['policy_sha256'])
    monkeypatch.setattr(training, 'load_checkpoint', load)
    first = training.train(cfg, policy, None, cases, tmp_path, max_steps=2)
    assert first['completed_steps'] == 2 and len(updates) == 2 and len(calls) == 8
    resumed = training.train(cfg, policy, None, cases, tmp_path, max_steps=1)
    assert resumed['completed_steps'] == 3 and len(updates) == 3 and len(calls) == 12
    assert synced == ['0','1','2']
    assert (tmp_path/'ours/frozen_generator_prompt.txt').read_text() == 'OPTIMIZED'


def test_step_ten_checkpoint_upload_is_durable(tmp_path, monkeypatch):
    from src.common import checkpoint_hub as hub
    out = tmp_path/'ours'; cp = out/'checkpoints/checkpoint_000010'
    (cp/'adapter').mkdir(parents=True)
    receipt = {'step':10,'fingerprint':'run','policy_sha256':'weights'}
    (cp/'COMPLETE.json').write_text(json.dumps(receipt))
    (cp/'trainer_state.pt').write_bytes(b'optimizer')
    (cp/'adapter/adapter_config.json').write_text('{}')
    (cp/'adapter/adapter_model.safetensors').write_bytes(b'weights')
    cfg = {'checkpoint_upload':{'enabled':True,'repo_id':'example/ours','path_prefix':'ours'}}
    (out/'run_manifest.json').write_text(json.dumps({'config':cfg}))
    (out/'frozen_generator_prompt.txt').write_text('optimized prompt')
    state = {**receipt,'completed_steps':10,'last_curve_checkpoint':str(cp)}
    jobs = []
    monkeypatch.setattr(hub, 'start_job', jobs.append)
    path = hub.enqueue_checkpoint(cfg, out, state)
    job = json.loads(path.read_text())
    assert job['completed_cases'] == 40 and job['batch'] == 10
    assert job['path_in_repo'].endswith('/step_000010')
    assert (Path(job['payload'])/'artifacts/prompts/frozen_generator_prompt.txt').read_text() == 'optimized prompt'
    assert hub.audit_bundle(job['payload'])
    job['status']='uploaded'; path.write_text(json.dumps(job))
    hub.enqueue_checkpoint(cfg, out, state)
    assert jobs == [path]
