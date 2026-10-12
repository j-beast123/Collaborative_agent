from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS
from src.online_rl.tree.overlap_cases import run
from src.online_rl.tree.gpu_pool import case_policy_view, PoolClient


def test_next_rollout_starts_before_previous_bank_finishes_and_join_waits():
    release_bank=Event(); second=Event(); rollout_end=Event()
    def execute(job,slot):
        if job==0:
            rollout_end.set();slot.release()
            assert release_bank.wait(3)
        else:
            assert rollout_end.is_set();second.set();slot.release()
        return job
    with ThreadPoolExecutor(1) as executor:
        future=executor.submit(run,[0,1],execute)
        assert second.wait(3) and not future.done()
        release_bank.set();assert future.result()==[0,1]


def test_next_case_cannot_start_until_rollout_slot_is_released():
    release=Event();first=Event();second=Event()
    def execute(job,slot):
        if job==0:first.set();assert release.wait(3)
        else:second.set()
        return job
    with ThreadPoolExecutor(1) as executor:
        future=executor.submit(run,[0,1],execute)
        assert first.wait(3) and not second.wait(.05)
        release.set();assert future.result()==[0,1]


def test_case_views_share_model_but_isolate_nested_rpc_context(monkeypatch):
    calls=[]
    monkeypatch.setattr(PoolClient,'call',lambda self,op,*args:calls.append((dict(self.context),op,args)))
    parent=NS(spec={},expected='policy',source_gpu=0,timeout=1,context={'case_id':'parent'})
    policy=NS(model=object(),doctor=NS(model='doctor',_chat=None),pool_client=parent)
    a=case_policy_view(policy,{'case_id':'a'});b=case_policy_view(policy,{'case_id':'b'})
    a._generate('p',4,0);b.doctor._chat('q',0);a.pool_behavior_logprobs([])
    assert a.model is b.model is policy.model
    assert [r[0]['case_id'] for r in calls]==['a','b','a']
    assert policy.pool_client.context=={'case_id':'parent'} and policy.doctor._chat is None
