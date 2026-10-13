"""Scheduler, checkpoint replay and real CPU LoRA restoration without API/GPU calls."""
import json
from concurrent.futures import Future
from types import SimpleNamespace
from pathlib import Path
import pytest
from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.five_gpu import (batch_slots, EvaluationQueue, cache_openings,
    evaluation_adapter, save_checkpoint, load_checkpoint, wait_with_evaluation, update_batch)


def test_four_case_batch_one_case_per_gpu_and_exact_boundaries():
    cursor = 0; sizes=[]
    while cursor < 100:
        slots = batch_slots(cursor,100,interval=30)
        sizes.append(len(slots))
        assert len({(s['worker'],s['lane']) for s in slots}) == len(slots)
        assert all(0 <= s['worker'] < 4 and s['lane'] == 0 for s in slots)
        assert len({s['worker'] for s in slots}) == len(slots)
        cursor += len(slots)
    assert sizes == ([4]*7+[2])*3+[4,4,2]
    assert [s['worker'] for s in batch_slots(24,100,interval=30)] == [0,1,2,3]
    assert [s['worker'] for s in batch_slots(28,100,interval=30)] == [0,1]


def test_updates_take_priority_at_case_boundary():
    a,b = Future(),Future(); calls=[]
    def evaluate():
        calls.append('eval')
        a.set_result('GPU0'); b.set_result('GPU1')
        return True
    assert wait_with_evaluation([a,b],evaluate) == ['GPU0','GPU1']
    assert calls == ['eval']
    assert wait_with_evaluation([a,b],lambda:pytest.fail('Ready batch must update')) == ['GPU0','GPU1']


def test_worker_failure_propagates_before_more_evaluation():
    a,b=Future(),Future();a.set_exception(RuntimeError('rollout failed'))
    with pytest.raises(RuntimeError,match='rollout failed'):
        wait_with_evaluation([a,b],lambda:pytest.fail('Failed batch'))


def test_durable_queue_restart_and_frozen_identity(tmp_path):
    cases=[{'case_id':f'c{i}'} for i in range(3)]
    settings={'seed':7,'data_sha256':'data','protocol_sha256':'protocol'}
    q=EvaluationQueue(tmp_path,cases,settings)
    q.register('/checkpoint0',0,'weights0');q.register('/checkpoint30',30,'weights30')
    played=[]
    def play(job,case,seed):
        played.append((job['completed_cases'],case['case_id'],seed))
        return {'outcome':None if case['case_id']=='c1' else 1,'num_discussion_rounds':3,'dialogue':[]}
    q.step(play)
    q=EvaluationQueue(tmp_path,cases,settings)
    while q.step(play):pass
    assert played == [(n,f'c{i}',7+i) for n in (0,30) for i in range(3)]
    summary=json.loads((tmp_path/'checkpoint_000000/summary.json').read_text())
    assert summary['status']=='complete' and summary['unparsed']==1 and summary['correct']==2
    assert summary['mean_agent_turns']==3
    with pytest.raises(RuntimeError,match='identity'):
        q.register('/checkpoint0',0,'different')
    with pytest.raises(RuntimeError,match='protocol changed'):
        EvaluationQueue(tmp_path,cases,{**settings,'seed':9}).next_case()


def test_failed_eval_case_is_retried(tmp_path):
    q=EvaluationQueue(tmp_path,[{'case_id':'a'}],{'seed':0})
    q.register('/checkpoint',0,'weights')
    with pytest.raises(RuntimeError):q.step(lambda *args:(_ for _ in ()).throw(RuntimeError('API error')))
    assert q.next_case()[2]==0


def test_parallel_eval_drains_five_way_and_resumes_missing_cases(tmp_path):
    cases=[{'case_id':f'c{i}'} for i in range(11)]
    q=EvaluationQueue(tmp_path,cases,{'seed':4})
    q.register('/checkpoint30',30,'weights30')
    q.register('/checkpoint60',60,'weights60')
    import threading
    lock=threading.Lock(); active=0; peak=0; seen=[]
    def play(job,case,seed):
        nonlocal active,peak
        with lock:
            active+=1; peak=max(peak,active); seen.append((case['case_id'],seed))
        try:
            import time; time.sleep(0.02)
            return {'outcome':1,'num_discussion_rounds':2,'dialogue':[]}
        finally:
            with lock: active-=1
    snapshots=[]
    q.run_all_parallel(play,workers=5,before_job=lambda manifest:snapshots.append(manifest['policy_sha256']))
    assert peak==5
    assert sorted(seen,key=lambda row:(int(row[0][1:]),row[1]))==[
        pair for i in range(11) for pair in ((f'c{i}',4+i),(f'c{i}',4+i))]
    assert snapshots==['weights30','weights60']
    assert json.loads((tmp_path/'checkpoint_000030/summary.json').read_text())['status']=='complete'
    q.run_all_parallel(lambda *args:pytest.fail('Completed evaluations must not rerun'),workers=5)


def test_openings_ignore_policy_and_call_seed_and_exclude_gold(tmp_path):
    calls=[]
    doctor=SimpleNamespace(turn=lambda case,dialogue,r,rounds,seed:calls.append(seed) or {'visible':str(seed)})
    cache_openings(doctor,tmp_path/'D1.jsonl',{'model':'pinned'})
    case={'case_id':'a','question':'Q','options':{'A':'a'},'ground_truth':{'correct_option':'A'}}
    a=doctor.turn(case,[],1,5,123)
    b=doctor.turn({**case,'ground_truth':{'correct_option':'B'}},[],1,5,456)
    assert a==b and len(calls)==1
    fresh=SimpleNamespace(turn=lambda *args:pytest.fail('D1 must survive restart'))
    cache_openings(fresh,tmp_path/'D1.jsonl',{'model':'pinned'})
    assert fresh.turn(case,[],1,5,789)==a
    doctor.turn(case,[],2,5,999)
    assert calls[-1]==999
    assert 'ground_truth' not in (tmp_path/'D1.jsonl').read_text()


def tiny_policy():
    import torch,transformers as tf,peft
    import threading
    from src.online_rl.integrated.continuation import adapter_sha256
    torch.manual_seed(0)
    model=peft.get_peft_model(tf.LlamaForCausalLM(tf.LlamaConfig(vocab_size=40,hidden_size=16,
        intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)),
        peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj']))
    return SimpleNamespace(model=model,torch=torch,mark_dirty=lambda:None,gen_lock=threading.RLock())


@pytest.mark.parametrize('fail',[False,True])
def test_real_adapter_and_rng_restored_after_eval(tmp_path,fail):
    from src.online_rl.integrated.continuation import adapter_sha256
    p=tiny_policy(); torch=p.torch
    for param in p.model.parameters():
        if param.requires_grad:param.data.add_(0.1)
    p.model.save_pretrained(tmp_path/'adapter'); eval_sha=adapter_sha256(p.model)
    for param in p.model.parameters():
        if param.requires_grad:param.data.add_(0.2)
    sha=adapter_sha256(p.model);p.model.train();p.model.config.use_cache=False
    rng=torch.get_rng_state().clone()
    try:
        with evaluation_adapter(p,tmp_path):
            assert adapter_sha256(p.model)==eval_sha
            assert not p.model.training
            torch.rand(4);p.model.config.use_cache=True
            if fail:raise RuntimeError('evaluation failure')
    except RuntimeError as e:
        assert fail and str(e)=='evaluation failure'
    assert adapter_sha256(p.model)==sha and p.model.training and not p.model.config.use_cache
    assert torch.equal(torch.get_rng_state(),rng)


def test_atomic_checkpoint_receipt_and_optimizer_resume(tmp_path):
    from src.online_rl.runtime import GRPOUpdater
    from src.online_rl.integrated.continuation import adapter_sha256
    policy=tiny_policy();updater=GRPOUpdater(policy,{'optimizer':{'learning_rate':1e-5}})
    receipt={'policy_sha256':adapter_sha256(policy.model),'completed_cases':30}
    path=tmp_path/'checkpoint_000030'
    save_checkpoint(updater,path,receipt)
    assert json.loads((path/'COMPLETE.json').read_text())==receipt
    assert not path.with_name(path.name+'.pending').exists()
    for p in policy.model.parameters():
        if p.requires_grad:p.data.add_(1)
    updater.steps=99;load_checkpoint(updater,path)
    assert adapter_sha256(policy.model)==receipt['policy_sha256'] and updater.steps==0
    with pytest.raises(RuntimeError,match='receipt differs'):
        save_checkpoint(updater,path,{**receipt,'completed_cases':60})


def test_batch_keeps_case_groups_and_na(tmp_path):
    def record(case):
        nodes=[{'node_id':f'{case}/{i}','parent_id':None,'case_id':case,'depth':1,
                'prefix':[],'response':'x','trace':{'prompt_input_ids':[1]},'policy_sha256':'source',
                'seed':i,'stop_reason':None,'terminal':True,'next_doctor':None} for i in range(3)]
        return {'policy_sha256':'source','tree':{'case_id':case,'policy_sha256':'source',
            'max_agent_turns':3,'branching':3,'nodes':nodes,'leaves':[]},
            'agent_targets':{'groups':{'root':[n['node_id'] for n in nodes]},
                            'nodes':{n['node_id']:{'reward':r} for n,r in zip(nodes,[None,0,1])}}}
    seen=[]
    updater=SimpleNamespace(update_groups=lambda groups:seen.extend(groups) or {'optimizer_step':True})
    update_batch(updater,[record('a'),record('b')],'source',lambda:'source')
    assert len(seen)==2 and [{r['case_id'] for r in g} for g in seen]==[{'a|root'},{'b|root'}]
    assert [r['reward'] for r in seen[0]]==[None,0,1]
    with pytest.raises(RuntimeError,match='Mixed source'):
        update_batch(updater,[{**record('a'),'policy_sha256':'other'}],'source',lambda:'source')


@pytest.mark.parametrize('shared,evolving,first_seed',[(False,False,False),(False,True,False),(True,False,False),(True,True,False),(True,True,True)])
def test_runner_commits_30_resumes_and_drains_eval_queue(tmp_path,monkeypatch,evolving,shared,first_seed):
    """Execute the production coordinator with CPU weights and immediate rollout futures."""
    import torch
    import scripts.online_rl.run_five_gpu as runner
    import scripts.online_rl.run_tree as tree_runner
    import src.online_rl.runtime as runtime
    import src.online_rl.integrated.grpo as grpo
    from src.online_rl.runtime import GRPOUpdater
    policy=tiny_policy()
    policy.doctor=SimpleNamespace(verify=lambda:{'problems':[]},
                                  turn=lambda *a:{'visible':'D1'},http=lambda *a:{})
    policy.generate=lambda case,seed:{'dialogue':[{'speaker':'agent','text':'a'}],
                                     'final_option':'A','num_discussion_rounds':3}
    monkeypatch.setattr(runtime,'SharedPolicy',lambda *a,**k:policy)
    monkeypatch.setattr(tree_runner,'load_key',lambda:'dummy')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:5)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda state:None)
    # This CPU runner fixture must not contend with a live physical-GPU owner.
    path_open=Path.open
    def test_open(path,*args,**kwargs):
        if str(path)=='/tmp/sherpa_tree_physical_gpu4.lock':path=tmp_path/'gpu4.lock'
        return path_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',test_open)
    updates=[]
    class Updater(GRPOUpdater):
        def update_groups(self,groups):
            updates.append(len(groups));self.steps+=1
            for p in self.policy.model.parameters():
                if p.requires_grad:p.data.add_(0.01)
            return {'optimizer_step':True,'steps':self.steps}
    monkeypatch.setattr(grpo,'make_updater',Updater)
    def record(job,sha):
        cid=job['case']['case_id']
        nodes=[{'node_id':f'{cid}/{i}','parent_id':None,'case_id':cid,'depth':1,'prefix':[],
                'response':'x','trace':{'prompt_input_ids':[1]},'policy_sha256':sha,'seed':i,
                'stop_reason':None,'terminal':True,'next_doctor':None} for i in range(3)]
        return {'case_cursor':job['index'],'case_id':cid,'policy_sha256':sha,
                'tree':{'case_id':cid,'policy_sha256':sha,'max_agent_turns':3,
                        'branching':3,'nodes':nodes,'leaves':[]},
                'agent_targets':{'groups':{'root':[n['node_id'] for n in nodes]},
                                'nodes':{n['node_id']:{'reward':r} for n,r in zip(nodes,[None,0,1])}}}
    phases=[]; seen_parents=[]; originals={}
    class Executor:
        def __init__(self,**kwargs):pass
        def submit(self,fn,*args):
            if fn.__name__=='prepare':
                future=Future();future.set_result(args[1]);return future
            jobs,checkpoint,sha,*extra=args
            phase=extra[0] if extra else 'fixed';prompt=extra[1] if extra else None
            phases.append(phase); records=[]
            for j in jobs:
                r=record(j,sha)
                if phase!='fixed':
                    cid=r['case_id']; keys={cid+'/success':1.,cid+'/failure':0.}
                    r['prompt_sha256']=prompt['sha256']
                    r['prompt_evaluation']={'per_trajectory':{k:(.2 if phase=='P0' else .8) for k in keys},
                        'source_outcome':keys,'rerun_policy_sha256':sha}
                    if phase=='P0':originals[cid]=r;seen_parents.append(prompt['sha256'])
                    if phase=='union':r=originals[cid]
                records.append(r)
            future=Future();future.set_result(records);return future
        def shutdown(self,**kwargs):pass
    monkeypatch.setattr(runner,'ProcessPoolExecutor',Executor)
    config='tree_v21_five_gpu_shared5.json' if shared else 'tree_v21_five_gpu.json'
    cfg=json.loads((runner.ROOT/'configs/online_rl'/config).read_text())
    size=5 if shared else 4
    if shared:
        from contextlib import contextmanager
        from src.online_rl.tree import five_gpu_worker as worker, gpu_pool
        class Actor:
            @contextmanager
            def changing_policy(self):yield
            def close(self):pass
        def initialize(*args,existing_policy=None):existing_policy.pool_actor=Actor()
        monkeypatch.setattr(worker,'initialize',initialize)
        monkeypatch.setattr(runner,'ThreadPoolExecutor',Executor)
        monkeypatch.setattr(runner.multiprocessing,'get_context',lambda *args:SimpleNamespace(Manager=lambda:SimpleNamespace(shutdown=lambda:None)))
        directory=tmp_path/'pool';directory.mkdir()
        monkeypatch.setattr(gpu_pool,'create_pool',lambda *args:{'leases':{},'directory':str(directory),'lock':__import__('threading').RLock()})
        policy.generate=lambda *args:pytest.fail('Idle GPU evaluation must be disabled')
    cfg['prompt_update']['enabled']=evolving
    if first_seed:cfg['prompt_update']['candidate_seed']='first_completed_P0_v1'
    cfg['five_gpu']['base_evaluation']['enabled']=False  # BASE mode has independent evaluation tests
    if evolving:
        from src.online_rl.tree import batch_prompt as BP
        def propose(rows,text,store,*args):
            candidate=text+'\nRefine once.'
            return {'text':candidate,'sha256':store.put(candidate),'proposal':{},'valid_candidate':True}
        monkeypatch.setattr(BP,'propose_candidate',propose)
    cfg['five_gpu'].pop('train_order_file',None)
    cfg['output_dir']=str(tmp_path/'run');cfg['schedule']['cases']=30
    cfg['five_gpu']['checkpoint_every_cases']=30  # exercise generic non-divisible interval recovery
    cfg['schedule']['calibration_only_cursors']=[]
    train=tmp_path/'train.jsonl';test=tmp_path/'test.jsonl'
    def case(i):return {'case_id':f'c{i}','question':f'question{i}',
                       'options':{'A':'alpha','B':'beta'},'correct_option':'A'}
    train.write_text(''.join(json.dumps(case(i))+'\n' for i in range(30)))
    test.write_text(''.join(json.dumps(case(i))+'\n' for i in range(30,33)))
    cfg['data']['train']=str(train);cfg['five_gpu']['eval_file']=str(test)
    cfg['five_gpu']['eval_cases']=3;cfg['five_gpu']['eval_sha256']='fixture'
    runner.run(cfg,max_batches=1)
    state=json.loads((tmp_path/'run/run_state.json').read_text())
    assert state['case_cursor']==size and updates==[size]
    if first_seed:
        seed=json.loads((tmp_path/'run/batches/batch_000001/prompt_seed.json').read_text())
        assert seed['seed_mode']=='first_completed_P0_v1'
        event=json.loads((tmp_path/'run/prompt_events/batch_000001.json').read_text())
        assert event['selection_scope']=='all_batch_trajectories_v1' and event['candidate_seed']==seed
    if evolving:
        assert phases==['P0']*size+['P1']*size+['union']*size
        assert state['prompt_sha256']!=cfg['rubric']['prompt_sha256']
    real_write=runner.atomic_write_json
    def crash_before_state(path,value):
        if Path(path).name=='run_state.json' and value.get('case_cursor')==30:
            raise RuntimeError('simulated crash after checkpoint publication')
        return real_write(path,value)
    monkeypatch.setattr(runner,'atomic_write_json',crash_before_state)
    with pytest.raises(RuntimeError,match='simulated crash'):
        runner.run(cfg)
    expected_updates=[5]*6 if shared else [4]*7+[2]
    assert updates==expected_updates
    monkeypatch.setattr(runner,'atomic_write_json',real_write)
    runner.run(cfg)
    state=json.loads((tmp_path/'run/run_state.json').read_text())
    assert state['case_cursor']==30 and updates==expected_updates
    if evolving:
        events=sorted((tmp_path/'run/prompt_events').glob('batch_*.json'))
        assert len(events)==(6 if shared else 8)
        assert json.loads(events[-1].read_text())['next_prompt']['sha256']==state['prompt_sha256']
        assert seen_parents[:size]==[cfg['rubric']['prompt_sha256']]*size
    checkpoints=list((tmp_path/'run/checkpoints').iterdir())
    assert {p.name for p in checkpoints}=={'checkpoint_000000','checkpoint_000030'}
    assert len(list((tmp_path/'run/cases').glob('case_*.json')))==30
    if shared:
        assert not (tmp_path/'run/evaluation').exists()
        with pytest.raises(RuntimeError,match='disabled'):runner.run(cfg,eval_only=True)
        runner.run(cfg)
        assert updates==expected_updates
        return
    for n in (0,30):
        summary=json.loads((tmp_path/f'run/evaluation/checkpoint_{n:06d}/summary.json').read_text())
        assert summary['status']=='complete' and summary['evaluated']==3 and summary['accuracy']==1
    runner.run(cfg,eval_only=True)
    assert updates==[4]*7+[2]  # A completed resume never repeats optimizer updates.


def test_complete_queue_repairs_interrupted_summary(tmp_path):
    q=EvaluationQueue(tmp_path,[{'case_id':'a'}],{'seed':0})
    job=q.register('/checkpoint',0,'weights')
    q.step(lambda *args:{'outcome':1,'num_discussion_rounds':4,'dialogue':[]})
    (job/'summary.json').unlink()
    assert q.next_case() is None
    assert json.loads((job/'summary.json').read_text())['status']=='complete'


def test_worker_runs_one_case_with_one_policy(tmp_path,monkeypatch):
    import threading
    import src.online_rl.tree.five_gpu_worker as worker
    from src.online_rl.integrated.continuation import adapter_sha256
    policy=tiny_policy();policy.model.save_pretrained(tmp_path/'adapter')
    monkeypatch.setattr(worker,'_POLICY',policy)
    monkeypatch.setattr(worker,'_CFG',{'five_gpu':{'lanes_per_gpu':1}})
    barrier=threading.Barrier(1)
    policies=[]
    def case(job, phase='fixed', prompt=None):
        policies.append(id(worker._POLICY))
        barrier.wait(timeout=5)
        return job['index']
    monkeypatch.setattr(worker,'run_case',case)
    assert worker.run_many([{'index':0}],tmp_path,adapter_sha256(policy.model))==[0]
    assert len(set(policies))==1


def test_training_order_pinned_unique_and_calibration_excluded(tmp_path):
    from src.online_rl.tree.five_gpu import training_order
    from src.online_rl.cases import file_sha256
    path=tmp_path/'order.json';atomic_write_json(path,['c','a','b'])
    cfg={'data':{'excluded_calibration_case_ids':['reserved']},'schedule':{'cases':3,'seed':0},
         'five_gpu':{'train_order_file':str(path),'train_order_sha256':file_sha256(path)}}
    cases=[{'case_id':cid} for cid in ('a','b','c')]
    assert training_order(cfg,cases,tmp_path)==['c','a','b']
    with pytest.raises(ValueError,match='Reserved calibration'):
        training_order(cfg,cases+[{'case_id':'reserved'}],tmp_path)
    atomic_write_json(path,['c','a','a'])
    with pytest.raises(ValueError,match='hash differs'):training_order(cfg,cases,tmp_path)
    cfg['five_gpu']['train_order_sha256']=file_sha256(path)
    with pytest.raises(ValueError,match='exactly once'):training_order(cfg,cases,tmp_path)


def test_500_case_batches_end_at_exact_snapshot_boundaries():
    cursor=0;counts=[]
    while cursor<500:
        cursor+=len(batch_slots(cursor,500));counts.append(cursor)
    assert len(counts)==125 and counts[-1]==500
    assert all(b-a==4 for a,b in zip([0]+counts,counts))
    assert set(range(24,481,24)) <= set(counts)


def test_four_case_update_waits_for_last_gpu():
    futures=[Future() for _ in range(4)]
    rounds=[]
    def evaluate():
        rounds.append(1)
        if len(rounds)==1:
            for i in range(3):futures[i].set_result(i)
        else:futures[3].set_result(3)
        return True
    assert wait_with_evaluation(futures,evaluate)==[0,1,2,3]
    assert len(rounds)==2


def test_production_config_is_four_cases_total_and_500_training_cases():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/online_rl/tree_v21_five_gpu.json').read_text())
    fg=cfg['five_gpu']
    assert fg['rollout_gpus']==[0,1,2,3] and fg['lanes_per_gpu']==1
    assert fg['batch_cases']==len(fg['rollout_gpus'])*fg['lanes_per_gpu']==4
    assert fg['update_gpu']==4 and cfg['schedule']['cases']==500 and fg['checkpoint_every_cases']==24
    assert fg['eval_cases']==622
    assert cfg['verification']['mode']=='all_members_witness_v1' and cfg['verification']['max_valid_supports'] is None
    assert fg['base_evaluation']['mode']=='original_base_no_precommit_v1'
