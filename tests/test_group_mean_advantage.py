"""Mean-only math, loss masking, per-trajectory token credit and signal alerts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.online_rl.integrated.metrics import advantages
from src.online_rl.integrated import grpo
from src.online_rl.sherpa import pipeline
from src.online_rl.sherpa.learning_signal import signal_checks, save_step_signal
from src.online_rl.tree.experiment import load_config


def group(case, rewards, outcomes):
    return [{'case_id':case,'trajectory_id':f'{case}/{i}', 'reward':r, 'outcome':y,
             'rollout_policy_sha256':'fixed', 'agent_traces':[
                 {'completion_ids':[1,2], 'prompt_input_ids':[0], 'behavior_logprobs':[0.,0.]},
                 {'completion_ids':[3], 'prompt_input_ids':[0,1], 'behavior_logprobs':[0.]}]}
            for i,(r,y) in enumerate(zip(rewards,outcomes))]


def test_mean_only_preserves_small_reward_differences_and_zeros():
    values = [.49]*15 + [.51]*15 + [.5]*2
    assert advantages(values, 'group_mean') == pytest.approx([-.01]*15 + [.01]*15 + [0.]*2)
    assert advantages(values, 'mean_center') == advantages(values, 'group_mean')
    assert max(advantages(values, 'group_zscore')) > 1
    assert advantages([.7]*32, 'group_mean') == [0.]*32
    assert advantages([None,.4,.6], 'group_mean') == pytest.approx([None,-.1,.1])
    with pytest.raises(ValueError, match='finite'):
        advantages([0.,float('nan')], 'group_mean')


def test_full_group_mean_before_masking_zero_and_preserve_legacy():
    records = group('a', [0.,.5,.5,1.], [0,1,1,1])
    original = deepcopy(records)
    used, adv, skipped, details = grpo.prepare_advantage_groups([records], 'group_mean')
    assert adv == [-.5,.5] and [r['trajectory_id'] for r in used] == ['a/0','a/3']
    assert records == original and details[0]['advantages'] == [-.5,0.,0.,.5]
    assert details[0]['used_assistant_tokens'] == 6
    assert details[0]['zero_advantage_trajectories'] == 2 and details[0]['group_type'] == 'mixed'
    legacy, legacy_adv, _, _ = grpo.prepare_advantage_groups([records], 'group_zscore')
    assert len(legacy) == 4 and legacy_adv[1:3] == [0.,0.]
    flat = group('b',[.4]*32,[1]*32)
    used, adv, skipped, details = grpo.prepare_advantage_groups([flat], 'group_mean')
    assert not used and not adv and skipped == ['zero_advantage_group']
    assert details[0]['used_assistant_tokens'] == 0


def test_group_types_signal_mass_and_shares():
    groups = [group('m',[.3]*16+[.7]*16,[0]*16+[1]*16),
              group('s',[.49]*15+[.51]*15+[.5]*2,[1]*32),
              group('f',[.499]*16+[.501]*16,[0]*32), group('flat',[1.]*32,[1]*32)]
    _, _, _, details = grpo.prepare_advantage_groups(groups, 'group_mean')
    by_type = grpo.signal_by_group_type(details)
    assert by_type['mixed']['sum_abs_advantage'] == pytest.approx(6.4)
    assert by_type['all_success']['sum_abs_advantage'] == pytest.approx(.3)
    assert by_type['all_failure']['sum_abs_advantage'] == pytest.approx(.032)
    assert by_type['all_success']['groups_total'] == 2 and by_type['all_success']['groups_skipped'] == 1
    assert sum(r['share_abs_advantage'] for r in by_type.values()) == pytest.approx(1.)
    assert by_type['all_success']['used_trajectories'] == 30
    _,_,_, flat_details = grpo.prepare_advantage_groups([groups[-1]], 'group_mean')
    assert all(r['share_abs_advantage'] is None for r in grpo.signal_by_group_type(flat_details).values())


def test_updater_excludes_zero_records_before_policy_and_kl_loss(monkeypatch):
    import torch
    model = torch.nn.Linear(1,1,bias=False)
    model.config = SimpleNamespace(use_cache=True)
    updater = grpo.IntegratedGRPOUpdater.__new__(grpo.IntegratedGRPOUpdater)
    updater.policy = SimpleNamespace(torch=torch, model=model)
    updater.mode, updater.steps = 'group_mean', 0
    updater.trainable = list(model.parameters())
    updater.optimizer = torch.optim.SGD(updater.trainable,lr=.01)
    updater.o = {'beta':.04,'clip_epsilon':.2,'logprob_token_chunk_size':16,
                 'pre_update_ratio_tolerance':.001,'grad_clip':1.}
    monkeypatch.setattr(grpo, 'adapter_sha256', lambda m:'fixed')
    seen = []
    def loss(m, records, adv, beta, clip, chunk):
        seen.append(([r['trajectory_id'] for r in records],adv))
        m.weight.sum().backward()
        return {'ratio':{'max_abs_deviation':0},'kl_loss':.01}
    monkeypatch.setattr(grpo, 'grpo_loss', loss)
    result = updater.update_groups([group('a',[0.,.5,.5,1.],[0,1,1,1]), group('flat',[1.]*32,[1]*32)])
    assert seen == [(['a/0','a/3'],[-.5,.5])]
    assert result['groups_skipped'] == 1 and result['used_trajectories'] == 2
    assert result['kl_scope'] == 'nonzero_advantage_trajectories_only'
    assert result['grad_norm'] > 0
    flat = updater.update_groups([group('flat',[1.]*32,[0]*32)])
    assert not flat['optimizer_step'] and len(seen) == 1


def test_one_trajectory_advantage_applies_to_all_turn_tokens(monkeypatch):
    import torch
    parameter = torch.tensor(0., requires_grad=True)
    monkeypatch.setattr(grpo, 'sequence_logprobs', lambda model,p,c,**k:parameter*torch.tensor(c,dtype=torch.float32))
    monkeypatch.setattr(grpo, 'reference_logprobs', lambda model,p,c,**k:torch.zeros(len(c)))
    records = group('c',[0.,1.],[0,1])
    records[1]['agent_traces'] = [{'prompt_input_ids':[0], 'completion_ids':[4,5], 'behavior_logprobs':[0.,0.]}]
    # Average tokens within each trajectory, then trajectories. All three tokens
    # in the first trajectory (across two turns) receive the same +0.1 advantage.
    grpo.grpo_loss(None, records, [.1,-.1], .04, .2, 16)
    assert parameter.grad.item() == pytest.approx(.125)


def test_new_config_changes_only_mode_and_run_metadata():
    old, new = load_config('configs/ours.json'), load_config('configs/ours_group_mean.json')
    for key in old:
        if key not in ('advantage_normalization','output_dir','checkpoint_upload','rubric_pipeline'):
            assert new[key] == old[key], key
    assert {k:v for k,v in new['rubric_pipeline'].items()
            if k not in ('generator_prompt_file','judge_prompt_file')} == {
                k:v for k,v in old['rubric_pipeline'].items() if k != 'judge_prompt_file'}
    assert old['advantage_normalization']['mode'] == 'group_zscore'
    assert new['advantage_normalization']['mode'] == 'group_mean'
    assert load_config('configs/plan_c.json')['advantage_normalization']['mode'] == 'group_mean'
    from src.online_rl.sherpa.config import config_problems
    assert not config_problems(new, Path(__file__).resolve().parents[1])


def test_metrics_use_selected_mode():
    trajectories = [{'outcome':i%2,'agent_turns':2} for i in range(32)]
    scores = [{'reward':.49 if i%2 else .51} for i in range(32)]
    result = pipeline.case_metrics(trajectories, scores, 'group_mean')
    assert result['advantage_mode'] == 'group_mean'
    assert max(result['advantages']) == pytest.approx(.01)


def test_gradient_and_persistent_kl_alerts_are_read_only(tmp_path):
    reference, home = tmp_path/'reference', tmp_path/'new'
    cfg = {'advantage_normalization':{'mode':'group_mean'},
           'learning_signal_monitor':{'reference_output_dir':'reference','kl_patience_steps':3}}
    (reference/'steps/step_000003').mkdir(parents=True)
    (reference/'run_manifest.json').write_text(json.dumps({'config':{'advantage_normalization':{'mode':'group_zscore'}},
                                                        'frozen_prompt':'same'}))
    reference_metrics = reference/'steps/step_000003/metrics.json'
    reference_metrics.write_text(json.dumps({'kl':.01,'update':{'grad_norm':2.}}))
    original = reference_metrics.read_bytes()
    for step in (1,2):
        folder = home/'steps'/f'step_{step:06d}';folder.mkdir(parents=True)
        (folder/'metrics.json').write_text(json.dumps({'update':{'optimizer_step':True,'kl_loss':0.}}))
    update = {'optimizer_step':True,'grad_norm':.1,'kl_loss':0., 'groups_total':4,'groups_skipped':1,
              'advantage_diagnostics':{'all':{'mean_abs':.02,'max_abs':.1},'by_group_type':{
                  'mixed':{'sum_abs_advantage':1.,'share_abs_advantage':1.}}}}
    signal = signal_checks(cfg, home, tmp_path, 3, update, 'same')
    assert signal['reference']['gradient_norm_ratio'] == .05 and len(signal['alerts']) == 2
    assert signal['skipped_group_fraction'] == .25
    assert reference_metrics.read_bytes() == original
    folder = home/'steps/step_000003';folder.mkdir()
    save_step_signal(folder, 3, signal)
    assert (folder/'signal_by_group_type.csv').is_file()
    early = signal_checks(cfg, home, tmp_path, 1, update, 'same')
    assert not any(a['kind'].startswith('kl_') for a in early['alerts'])


def test_report_reconstructs_historical_types_without_changing_logs(tmp_path):
    from scripts.online_rl.analyze_advantage_signal import analyze_run
    from src.online_rl.sherpa import PROTOCOL
    cfg = {'advantage_normalization':{'mode':'group_zscore'}}
    (tmp_path/'run_manifest.json').write_text(json.dumps({'protocol':PROTOCOL,'config':cfg}))
    (tmp_path/'run_state.json').write_text(json.dumps({'completed_steps':1}))
    groups = [group('mixed',[.3]*16+[.7]*16,[0]*16+[1]*16),
              group('success',[.49]*16+[.51]*16,[1]*32),
              group('failure',[.49]*16+[.51]*16,[0]*32),group('flat',[1.]*32,[1]*32)]
    _,_,_, details = grpo.prepare_advantage_groups(groups, 'group_zscore')
    legacy = [{k:g[k] for k in ('group','trajectory_ids','rewards','advantages','used')}
              for g in details]
    for rows in groups:
        folder = tmp_path/'steps/step_000001'/rows[0]['case_id'];folder.mkdir(parents=True)
        (folder/'record.json').write_text(json.dumps({'training_records':rows,'trajectories':[
            {'trajectory_id':r['trajectory_id'],'outcome':r['outcome']} for r in rows]}))
    path = tmp_path/'steps/step_000001/metrics.json'
    path.write_text(json.dumps({'step':1,'kl':0.,'case_auc':{'mixed':.8},
        'update':{'grad_norm':1.,'advantage_diagnostics':{'mode':'group_zscore','groups':legacy}}}))
    original = path.read_bytes()
    report = analyze_run(tmp_path, 30)
    assert report['analyzed_steps'] == 1 and len(report['group_rows']) == 4
    assert report['summary'][0]['mean_auc'] == .8
    assert report['summary'][1]['groups_skipped'] == 1
    assert sum(row['share_abs_advantage'] for row in report['summary']) == pytest.approx(1.)
    assert path.read_bytes() == original


def test_report_supports_plan_c_dynamic_turn_groups(tmp_path):
    from scripts.online_rl.analyze_advantage_signal import analyze_run
    from src.online_rl.sherpa.plan_c_rollouts import PROTOCOL
    cfg = {'advantage_normalization':{'mode':'group_mean'}, 'rollout':{'rounds':2},
           'plan_c':{'responses_per_turn':3}}
    (tmp_path/'run_manifest.json').write_text(json.dumps({'protocol':PROTOCOL,'config':cfg}))
    (tmp_path/'run_state.json').write_text(json.dumps({'completed_steps':1}))
    groups = [group(f'case_{case}|turn_{turn}|state',[.49,.5,.51],[0,1,1])
              for case in range(4) for turn in (1,2)]
    _,_,_,details = grpo.prepare_advantage_groups(groups, 'group_mean')
    folder = tmp_path/'steps/step_000001';folder.mkdir(parents=True)
    (folder/'metrics.json').write_text(json.dumps({'step':1,'kl':0.,
        'turn_auc':{g['group']:.75 for g in details},
        'update':{'grad_norm':.1,'advantage_diagnostics':{'mode':'group_mean','groups':details}}}))
    report = analyze_run(tmp_path,30)
    assert report['group_scope'] == 'case_turn_shared_history'
    assert len(report['group_rows']) == 8 and report['summary'][0]['mean_auc'] == .75
    assert report['summary'][0]['sum_abs_advantage'] == pytest.approx(.16)
    assert report['summary'][0]['used_assistant_tokens'] == 48
