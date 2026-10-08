import copy
import json

import pytest

from src.online_rl.integrated.records import sha256_json
import scripts.diagnostics.transition_halt4 as migration


@pytest.fixture
def migration_source(tmp_path,monkeypatch):
    root=migration.ROOT
    old=json.loads((root/'configs/online_rl/tree_v21_five_gpu_shared5.json').read_text())
    for key,default in (('agent_system_prompt_file','prompts/agent.txt'),
                        ('doctor_system_prompt_file','prompts/user_simulator.txt')):
        old['rollout'][key]=str(root/old['rollout'].get(key,default))
    old['output_dir']='source'
    new=copy.deepcopy(old);new['output_dir']='target'
    new['rollout']['halting']['stable_choice_no_objection_after_turns']=4
    for name,cfg in (('old.json',old),('new.json',new)):(tmp_path/name).write_text(json.dumps(cfg))
    source=tmp_path/'source';checkpoint=source/'checkpoints/recovery_000001';checkpoint.mkdir(parents=True)
    proof={'completed_cases':5,'policy_sha256':'theta1','protocol_sha256':'old-protocol','update':{'steps':1}}
    (checkpoint/'COMPLETE.json').write_text(json.dumps(proof))
    (checkpoint/'optimizer.bin').write_bytes(b'optimizer-moments')
    (checkpoint/'cuda_rng.pt').write_bytes(b'cuda-rng')
    state={'case_cursor':5,'batch_cursor':1,'config_sha256':sha256_json(old),'protocol_sha256':'old-protocol',
           'policy_sha256':'theta1','checkpoint':str(checkpoint),'prompt_id':'winner','prompt_sha256':'winner-sha'}
    (source/'run_state.json').write_text(json.dumps(state))
    (source/'cases').mkdir()
    for i in range(5):(source/'cases'/f'case_{i:06d}.json').write_text(json.dumps({'case_cursor':i,'protocol_sha256':'old-protocol'}))
    for name in ('prompts','prompt_events','openings'):
        (source/name).mkdir();(source/name/'evidence.json').write_text('{}')
    for number in (1,2):
        batch=source/'batches'/f'batch_{number:06d}';batch.mkdir(parents=True)
        (batch/'manifest.json').write_text('{}')
    (source/'doctor_servers').mkdir();(source/'doctor_servers/gpu0.pid').write_text('123')
    (source/'launch.json').write_text(json.dumps({'launched_at':1000}))
    (source/'api_usage.jsonl').write_text('{"cost_usd":1}\n')
    monkeypatch.setattr(migration,'ROOT',tmp_path)
    return source,old,new,state


def test_migration_preserves_optimizer_prompt_evidence_and_cursors(migration_source):
    source,old,new,state=migration_source
    target=migration.transition('old.json','new.json')
    resumed=json.loads((target/'run_state.json').read_text())
    assert (resumed['case_cursor'],resumed['batch_cursor'],resumed['policy_sha256'])==(5,1,'theta1')
    assert resumed['prompt_id']==state['prompt_id'] and resumed['prompt_sha256']==state['prompt_sha256']
    assert resumed['config_sha256']==sha256_json(new) and resumed['protocol_sha256']!='old-protocol'
    assert (target/'checkpoints/recovery_000001/optimizer.bin').read_bytes()==b'optimizer-moments'
    assert (target/'checkpoints/recovery_000001/cuda_rng.pt').read_bytes()==b'cuda-rng'
    assert (target/'cases/case_000004.json').read_bytes()==(source/'cases/case_000004.json').read_bytes()
    assert (target/'batches/batch_000001/manifest.json').exists()
    assert not (target/'batches/batch_000002').exists()
    assert (source/'batches/batch_000002/manifest.json').exists()
    assert json.loads((source/'run_state.json').read_text())==state
    assert json.loads((source/'checkpoints/recovery_000001/COMPLETE.json').read_text())['protocol_sha256']=='old-protocol'
    assert (target/'doctor_servers/gpu0.pid').read_text()=='123'
    assert json.loads((target/'launch.json').read_text())['launched_at']==1000
    with pytest.raises(RuntimeError,match='fresh'):migration.transition('old.json','new.json')


def test_migration_refuses_uncommitted_batch_and_other_changes(migration_source):
    source,old,new,state=migration_source
    (source/'run_state.json').write_text(json.dumps({**state,'case_cursor':4}))
    with pytest.raises(RuntimeError,match='committed five-case'):migration.transition('old.json','new.json')
    (source/'run_state.json').write_text(json.dumps(state))
    new['rollout']['rounds']=4
    (migration.ROOT/'new.json').write_text(json.dumps(new))
    with pytest.raises(RuntimeError,match='Only the fourth-turn'):migration.transition('old.json','new.json')
    assert not (source.parent/'target').exists()


def test_controller_dry_run_never_reads_credentials_signals_or_launches(migration_source,monkeypatch,capsys):
    import scripts.diagnostics.after_batch_halt4 as controller
    from pathlib import Path
    source,_,_,state=migration_source
    monkeypatch.setattr(controller,'ROOT',migration.ROOT)
    monkeypatch.setattr(controller,'info',lambda pid:{'cmd':[b'scripts/online_rl/run_five_gpu.py']})
    monkeypatch.setattr(controller.os,'readlink',lambda path:str(migration.ROOT))
    import scripts.online_rl.run_five_gpu as runner
    monkeypatch.setattr(runner,'preflight',lambda cfg:[])
    def forbidden(*args,**kwargs):pytest.fail('Dry run attempted a side effect or credential access')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    monkeypatch.setattr(controller.os,'kill',forbidden)
    monkeypatch.setattr(controller.subprocess,'Popen',forbidden)
    monkeypatch.setattr(controller,'atomic_write_json',forbidden)
    monkeypatch.setattr(controller.sys,'argv',['after_batch_halt4','--pid','42','--old-config','old.json',
                                             '--new-config','new.json','--dry-run'])
    controller.main()
    assert json.loads(capsys.readouterr().out)['launch'] is False
    assert json.loads((source/'run_state.json').read_text())==state
    assert not (migration.ROOT/'target').exists()


def test_combined_transition_preserves_evidence_and_records_visible_pipeline_protocol(migration_source):
    import hashlib
    source,old,new,state=migration_source
    contract=migration.ROOT/'prompts/sherpa/observed_context_behavior_v1.txt'
    contract.parent.mkdir(parents=True);contract.write_text('visible User history, earlier Agent behavior, three behavioral levels')
    new['rubric'].update(schema='observed_context_behavior_v1',state_addendum_file='prompts/sherpa/observed_context_behavior_v1.txt',
                         state_addendum_sha256=hashlib.sha256(contract.read_bytes()).hexdigest())
    new['prompt_update']['candidate_seed']='first_completed_P0_v1'
    new.setdefault('execution', {})['member_workers']=2
    (migration.ROOT/'new.json').write_text(json.dumps(new))
    target=migration.transition('old.json','new.json')
    audit=json.loads((target/'protocol_transition.json').read_text())
    assert audit['candidate_seed']=='first_completed_P0_v1'
    assert audit['selection_scope']=='all_batch_trajectories_v1'
    assert audit['rubric_schema']=='observed_context_behavior_v1'
    assert audit['completed_cases_with_original_rubrics']==5
    assert audit['member_workers']==2
    assert 'all same-ID members' in audit['member_parallelism']
    assert json.loads((target/'run_state.json').read_text())['prompt_sha256']==state['prompt_sha256']
    assert (target/'cases/case_000000.json').read_bytes()==(source/'cases/case_000000.json').read_bytes()


def test_transition_rejects_unapproved_member_parallelism(migration_source):
    _,_,new,_=migration_source
    new.setdefault('execution', {})['member_workers']=3
    (migration.ROOT/'new.json').write_text(json.dumps(new))
    with pytest.raises(RuntimeError,match='limited to the approved value 2'):
        migration.validate_config_change(json.loads((migration.ROOT/'old.json').read_text()),new)


def test_combined_transition_rejects_silent_belief_or_training_change(migration_source):
    source,old,new,state=migration_source
    new['rollout'].pop('agent_protocol')
    (migration.ROOT/'new.json').write_text(json.dumps(new))
    with pytest.raises(RuntimeError,match='Only the fourth-turn'):migration.transition('old.json','new.json')


def test_random10_transition_preserves_source_cases_and_records_protocol_boundary(migration_source):
    from pathlib import Path
    source,old,new,state=migration_source
    new['output_dir']='target'
    new['rollout']['halting']['stable_choice_no_objection_after_turns']=4
    repo=Path(__file__).resolve().parents[1]
    random_cfg=json.loads((repo/'configs/online_rl/tree_v21_five_gpu_shared5_halt4_visible_pipeline_random10.json').read_text())
    new['verification']=random_cfg['verification']; new.setdefault('notes',{})['verification']=random_cfg['notes']['verification']
    new['prompt_update']['aggregation']='mean_sampled_witness_effect_per_direction_product_v1'
    (migration.ROOT/'new.json').write_text(json.dumps(new))
    target=migration.transition('old.json','new.json')
    audit=json.loads((target/'protocol_transition.json').read_text())
    assert audit['training_verification_transition']=={
        'source_protocol':'all_members_witness_v1','destination_protocol':'random10_witness_v1',
        'cases_before_transition_keep_source_protocol':5,'cases_after_transition_use_destination_protocol':True,
        'prompt_search_effectiveness_reuses_witness_sample':True,
        'prompt_search_additional_member_reruns':0}
    assert json.loads((target/'run_state.json').read_text())['case_cursor']==5
    assert (target/'cases/case_000004.json').exists()


def test_random10_transition_rejects_wrong_sample_limit(migration_source):
    from pathlib import Path
    _,old,new,_=migration_source
    repo=Path(__file__).resolve().parents[1]
    random_cfg=json.loads((repo/'configs/online_rl/tree_v21_five_gpu_shared5_halt4_visible_pipeline_random10.json').read_text())
    new['verification']=random_cfg['verification']; new['verification']['max_valid_supports']=9
    new.setdefault('notes',{})['verification']=random_cfg['notes']['verification']
    with pytest.raises(RuntimeError,match='random10 witness protocol'):
        migration.validate_config_change(old,new)
