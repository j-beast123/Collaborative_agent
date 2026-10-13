"""Real local RPC: seeds/provenance, capacity, helper dispatch and update barriers."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import json
from pathlib import Path
import queue
import shutil
import threading
from types import SimpleNamespace
import pytest

from src.online_rl.tree.gpu_pool import Actor, PoolClient, assert_idle, create_pool, attach
from src.online_rl.tree.five_gpu import batch_slots

_child_actor = _child_client = None


def child_start(spec,gpu,root):
    global _child_actor, _child_client
    _child_actor=Actor(spec,gpu,{'agent':lambda prompt,n,seed,segment:seed},root)
    _child_actor.expected='theta0'
    _child_client=PoolClient(spec,'theta0',gpu,timeout=15)


def child_ready():
    return _child_actor.gpu


def child_requests(start):
    _child_client.context={'case_id':f'case{start}','phase':'P0'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(lambda seed:_child_client.call('agent','prompt',8,seed,'agent'),range(start,start+8)))


@pytest.fixture
def actors(tmp_path):
    manager = SimpleNamespace(RLock=threading.RLock,dict=dict,Queue=queue.Queue)
    spec = create_pool(manager,list(range(5)),agent_capacity=4,doctor_capacity=1)
    servers=[]
    for gpu in range(5):
        handlers={'agent':lambda prompt,n,seed,segment: {'seed':seed,'prompt':prompt,'tokens':[seed%n]},
                  'doctor':lambda prompt,seed:f'{prompt}/{seed}'}
        actor=Actor(spec,gpu,handlers,tmp_path)
        actor.expected='theta0'
        servers.append(actor)
    yield spec, servers, tmp_path
    for actor in servers:actor.close()
    shutil.rmtree(spec['directory'])


def test_ready_requests_help_original_case_with_exact_seeds_and_no_duplicates(actors):
    spec,servers,root=actors
    client=PoolClient(spec,'theta0',0)
    client.context={'case_id':'original-case','case_index':0,'phase':'P0'}
    with ThreadPoolExecutor(max_workers=20) as pool:
        results=list(pool.map(lambda seed:client.call('agent','same-prompt',11,seed,'private'),range(40)))
    assert [r['seed'] for r in results]==list(range(40))
    rows=[json.loads(line) for file in root.glob('pool_gpu*.jsonl') for line in file.read_text().splitlines()]
    assert len(rows)==len({r['task_id'] for r in rows})==40
    assert {r['gpu'] for r in rows}==set(range(5))
    assert all(r['source_gpu']==0 and r['context']['case_id']=='original-case' for r in rows)
    assert {r['seed'] for r in rows}==set(range(40))
    assert_idle(spec)


def test_slow_gpu_does_not_block_idle_gpu_and_update_requires_barrier(actors):
    spec,servers,root=actors
    started,release=threading.Event(),threading.Event()
    def slow(prompt,seed):
        started.set()
        assert release.wait(10)
        return prompt
    servers[0].handlers['doctor']=slow
    client=PoolClient(spec,'theta0',0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(client.call,'doctor','pending',12)
        assert started.wait(5)
        with pytest.raises(RuntimeError,match='in flight'):assert_idle(spec)
        with pytest.raises(RuntimeError,match='active'):
            with servers[0].changing_policy():pass
        second=pool.submit(client.call,'doctor','ready',13)
        assert second.result(timeout=5)=='ready/13'
        assert not first.done()
        release.set()
        assert first.result(timeout=5)=='pending'
    assert_idle(spec)
    assert json.loads((root/'pool_gpu1.jsonl').read_text())['source_gpu']==0
    with servers[0].changing_policy():servers[0].expected='theta1'
    assert servers[0].expected=='theta1'


def test_wrong_policy_or_handler_error_releases_completed_lease(actors):
    spec,servers,_=actors
    with pytest.raises(RuntimeError,match='actor failed'):
        PoolClient(spec,'stale',0).call('doctor','prompt',1)
    assert_idle(spec)
    def failed(*args):raise ValueError('private credential text')
    servers[1].handlers['doctor']=failed
    with pytest.raises(RuntimeError,match='ValueError') as error:
        PoolClient(spec,'theta0',0).call('doctor','prompt',2)
    assert 'private' not in str(error.value)
    assert_idle(spec)
    assert spec['doctor'].qsize()==5


def test_transport_failure_quarantines_uncertain_request(actors):
    spec,_,_=actors
    original=spec['addresses'][0]
    spec['addresses'][0]='/tmp/nonexistent_sherpa_actor.sock'
    with pytest.raises(OSError):PoolClient(spec,'theta0',0).call('doctor','prompt',1)
    assert spec['doctor'].qsize()==4
    with pytest.raises(RuntimeError,match='in flight'):assert_idle(spec)
    spec['addresses'][0]=original


def test_five_case_batches_align_with_30_case_snapshots_and_final500():
    cursor=0;milestones=[];updates=0
    while cursor<500:
        slots=batch_slots(cursor,500,interval=30,workers=5)
        assert len(slots)==5 and [s['worker'] for s in slots]==list(range(5))
        cursor+=len(slots);updates+=1
        if cursor%30==0 or cursor==500:milestones.append(cursor)
    assert updates==100
    assert milestones==list(range(30,500,30))+[500]


def test_one_case_fills_five_doctor_actors_without_increasing_native_capacity(actors):
    spec,servers,root=actors
    assert spec['request_workers']=={'agent_workers':20,'doctor_workers':5}
    together=threading.Barrier(5,timeout=10)
    lock=threading.Lock();active={gpu:0 for gpu in range(5)};peak=dict(active)
    def handler(gpu):
        def run(prompt,seed):
            with lock:
                active[gpu]+=1;peak[gpu]=max(peak[gpu],active[gpu])
            try:
                together.wait()
                return seed
            finally:
                with lock:active[gpu]-=1
        return run
    for gpu,server in enumerate(servers):server.handlers['doctor']=handler(gpu)
    client=PoolClient(spec,'theta0',1,timeout=15)
    client.context={'case_id':'one-large-case','phase':'P0'}
    with ThreadPoolExecutor(max_workers=spec['request_workers']['doctor_workers']) as pool:
        results=list(pool.map(lambda seed:client.call('doctor','prompt',seed),range(10)))
    assert results==list(range(10))
    assert peak=={gpu:1 for gpu in range(5)}
    rows=[json.loads(line) for path in root.glob('pool_gpu*.jsonl') for line in path.read_text().splitlines()]
    assert len(rows)==len({r['task_id'] for r in rows})==10
    assert all(r['source_gpu']==1 and r['context']['case_id']=='one-large-case' for r in rows)
    assert_idle(spec)


def test_runtime_submission_windows_do_not_change_resume_config(tmp_path,monkeypatch):
    import copy
    from scripts.online_rl.run_tree import runtime_execution
    cfg={'output_dir':str(tmp_path),'rollout':{'agent_sampling_protocol':'per_row_rng_batch_v1'},
         'five_gpu':{'agent_batch_size':4},'execution':{'doctor_workers':1,'agent_workers':4}}
    original=copy.deepcopy(cfg)
    monkeypatch.setenv('SHERPA_AGENT_BATCH_SIZE','4')
    native=SimpleNamespace(n_replicas=1)
    assert runtime_execution(cfg,native)['agent_workers']==4
    native.pool_execution={'agent_workers':20,'doctor_workers':5}
    runtime=runtime_execution(cfg,native)
    assert runtime['agent_workers']==20 and runtime['doctor_workers']==5
    assert cfg==original


def test_shared_config_allows_checkpoint_barrier_evaluation(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    from scripts.online_rl.run_five_gpu import ROOT, preflight
    cfg=json.loads((ROOT/'configs/online_rl/tree_v21_five_gpu_shared5.json').read_text())
    assert not preflight(cfg)
    assert cfg['five_gpu']['batch_cases']==5
    assert not cfg['five_gpu']['base_evaluation']['enabled']
    assert not cfg['five_gpu']['checkpoint_evaluation_enabled']
    assert cfg['prompt_update']['every_n_cases']==5
    cfg['five_gpu']['checkpoint_evaluation_enabled']=True
    assert not preflight(cfg)


def test_spawned_coordinators_share_manager_capacity_with_parent_actor(tmp_path):
    """Production process topology, including calls served by one's own actor."""
    context=multiprocessing.get_context('spawn')
    with context.Manager() as manager:
        spec=create_pool(manager,list(range(5)),4,1)
        parent=Actor(spec,4,{'agent':lambda prompt,n,seed,segment:seed},tmp_path)
        parent.expected='theta0'
        executors=[]
        try:
            for gpu in range(4):
                executors.append(ProcessPoolExecutor(max_workers=1,mp_context=context,
                    initializer=child_start,initargs=(spec,gpu,str(tmp_path))))
            assert [p.submit(child_ready).result(timeout=20) for p in executors]==list(range(4))
            futures=[p.submit(child_requests,gpu*10) for gpu,p in enumerate(executors)]
            client=PoolClient(spec,'theta0',4,timeout=15)
            client.context={'case_id':'parent-case','phase':'P0'}
            with ThreadPoolExecutor(max_workers=4) as pool:
                result=list(pool.map(lambda seed:client.call('agent','prompt',8,seed,'agent'),range(40,48)))
            assert result==list(range(40,48))
            for gpu,future in enumerate(futures):assert future.result(timeout=20)==list(range(gpu*10,gpu*10+8))
            assert_idle(spec)
            rows=[json.loads(line) for file in tmp_path.glob('pool_gpu*.jsonl') for line in file.read_text().splitlines()]
            assert len(rows)==len({r['task_id'] for r in rows})==40
            assert {r['source_gpu'] for r in rows}==set(range(5))
            assert {r['gpu'] for r in rows}==set(range(5))
        finally:
            for p in executors:p.shutdown(wait=True)
            parent.close()
            shutil.rmtree(spec['directory'])


def test_attached_policy_uses_native_handlers_without_recursive_dispatch(tmp_path,monkeypatch):
    import src.online_rl.integrated.grpo as grpo
    monkeypatch.setattr(grpo,'behavior_logprobs',lambda model,traces,chunk:[[float(chunk)] for trace in traces])
    manager=SimpleNamespace(RLock=threading.RLock,dict=dict,Queue=queue.Queue)
    spec=create_pool(manager,[4],4,1)
    policy=SimpleNamespace(_generate=lambda prompt,n,seed,segment='agent':{'prompt':prompt,'n':n,'seed':seed,'segment':segment},
        doctor=SimpleNamespace(_chat=lambda prompt,seed,max_tokens=None:f'{prompt}/{seed}/{max_tokens}'),
        gen_lock=threading.RLock(),model=object())
    actor=attach(policy,{'output_dir':str(tmp_path),'optimizer':{'logprob_token_chunk_size':32}},spec,4)
    actor.expected=policy.pool_client.expected='theta0'
    try:
        assert policy.pool_execution=={'agent_workers':4,'doctor_workers':1}
        assert policy._generate('prompt',max_new_tokens=8,seed=12,segment='private')=={
            'prompt':'prompt','n':8,'seed':12,'segment':'private'}
        assert policy.doctor._chat('doctor',13)=='doctor/13/None'
        assert policy.doctor._chat('final',14,32)=='final/14/32'
        assert policy.pool_behavior_logprobs([{'completion_ids':[1]}])==[[32.]]
        assert_idle(spec)
    finally:
        actor.close();shutil.rmtree(spec['directory'])
