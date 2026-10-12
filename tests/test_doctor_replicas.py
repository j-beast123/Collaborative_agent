import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from src.online_rl.tree.experiment import load_config
from src.online_rl.tree.doctor_replicas import endpoint_groups, bind_backend, bind_queue
from src.online_rl.doctor import OllamaDoctor


def test_independent_replica_queues_use_both_servers_without_changing_doctor_identity():
    cfg=load_config('configs/online_doctor_replicas2.json')
    local=copy.deepcopy(cfg)
    bind_backend(local,0)
    barrier=threading.Barrier(2);seen=[];lock=threading.Lock()
    def http(url,payload,timeout):
        with lock:seen.append(url)
        barrier.wait(timeout=3)
        return {'message':{'content':'Option A.'},'prompt_eval_count':30,'eval_count':5}
    doctor=OllamaDoctor(local['rollout']['doctor_backend'],local['rollout']['halting'],'system',256,http=http)
    baseline=OllamaDoctor(cfg['rollout']['doctor_backend'],cfg['rollout']['halting'],'system',256)
    assert doctor.protocol==baseline.protocol
    bind_queue(doctor,local,0)
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(doctor._chat,'Discuss.',seed) for seed in [1,2]]
        assert [f.result() for f in futures]==['Option A.','Option A.']
    assert set(seen)=={'http://127.0.0.1:11955/api/chat','http://127.0.0.1:11956/api/chat'}
    assert doctor._free.qsize()==2
    assert len(endpoint_groups(cfg))==2


def test_single_server_parallelism_remains_supported():
    cfg=load_config('configs/online_doctor2.json')
    assert all(len(group)==1 for group in endpoint_groups(cfg))
    doctor=type('Doctor',(),{})()
    bind_queue(doctor,cfg,1)
    assert doctor._free.qsize()==2
    assert doctor._free.get()==doctor._free.get()


@pytest.mark.parametrize('failure', ['duplicate','capacity','remote'])
def test_invalid_replica_topology_is_rejected(failure):
    cfg=load_config('configs/online_doctor_replicas2.json')
    if failure=='duplicate':cfg['five_gpu']['doctor_replica_urls'][0][0]=cfg['five_gpu']['doctor_urls'][0]
    if failure=='capacity':cfg['five_gpu']['doctor_concurrency']=1
    if failure=='remote':cfg['five_gpu']['doctor_replica_urls'][0][0]='http://example.com:11956'
    with pytest.raises(ValueError):endpoint_groups(cfg)
