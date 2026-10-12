from concurrent.futures import ThreadPoolExecutor
import json
import threading
import pytest
from src.online_rl.tree.case_pipeline import ready_cases


def jobs(tmp_path):
    return [{'index':i,'worker':i%2,'out':str(tmp_path/f'case_{i}')} for i in range(5)]


def test_bank_starts_before_other_rollout_finishes_and_coordinator_does_not_overlap(tmp_path):
    release=threading.Event(); started=threading.Event(); submitted=[]
    def run(job):
        if job['index']==0:
            started.set(); assert release.wait(3)
        return [{'case_cursor':job['index']}]
    with ThreadPoolExecutor(2) as pool:
        def submit(job):
            submitted.append(job['index']); return pool.submit(run,job)
        ready=ready_cases(jobs(tmp_path),submit,tmp_path/'order.json')
        first=next(ready)
        assert started.wait(1)
        assert first['case_cursor']==1
        assert 3 in submitted  # worker 1's next rollout is submitted BEFORE this bank transaction
        assert not release.is_set()
        release.set()
        records=[first,*ready]
    assert sorted(r['case_cursor'] for r in records)==list(range(5))
    assert json.loads((tmp_path/'order.json').read_text())==[r['case_cursor'] for r in records]


def test_resume_replays_persisted_order_even_when_completion_order_differs(tmp_path):
    path=tmp_path/'order.json';path.write_text('[1, 3, 0]')
    with ThreadPoolExecutor(2) as pool:
        records=list(ready_cases(jobs(tmp_path),lambda j:pool.submit(lambda:[{'case_cursor':j['index']}]),path))
    assert [r['case_cursor'] for r in records][:3]==[1,3,0]


def test_close_drains_other_rollout(tmp_path):
    release=threading.Event();finished=threading.Event()
    def run(job):
        if job['index']==0:
            assert release.wait(3);finished.set()
        return [{'case_cursor':job['index']}]
    with ThreadPoolExecutor(2) as pool:
        ready=ready_cases(jobs(tmp_path),lambda j:pool.submit(run,j),tmp_path/'order.json')
        assert next(ready)['case_cursor']==1
        release.set();ready.close()
        assert finished.is_set()


@pytest.mark.parametrize('order',['[1,1]','[9]','[true]'])
def test_invalid_resume_order_rejected(tmp_path,order):
    path=tmp_path/'order.json';path.write_text(order)
    with pytest.raises(RuntimeError):
        next(ready_cases(jobs(tmp_path),lambda j:None,path))


def test_round_records_uses_ready_order_latest_bank_and_scores_after_all_cases(tmp_path,monkeypatch):
    from src.online_rl.tree import cumulative_run as cr
    from src.online_rl.tree.cumulative_bank import empty_snapshot
    snapshot=empty_snapshot(); policy='policy';order=[1,0,3,2,4];events=[]
    monkeypatch.setattr(cr,'committed_snapshot',lambda *a:snapshot)
    monkeypatch.setattr(cr,'publish_snapshot',lambda *a:None)
    monkeypatch.setattr(cr,'emit_stage',lambda *a:None)
    (tmp_path/'prompt.txt').write_text('prompt')
    cfg={'five_gpu':{'batch_cases':5},'rubric':{'generator_prompt_file':'prompt.txt',
         'prompt_id':'p','prompt_sha256':'hash','cumulative_bank':{
         'working_bank_protocol':cr.OFFLINE_CHUNK_PROTOCOL,'scoring_scope':'all'}}}
    def row(i):
        return {'case_cursor':i,'case_id':str(i),'policy_sha256':policy,
                'cumulative_bank_input':snapshot,'cumulative_bank_output':snapshot}
    def ready(prompt):
        for i in order:
            events.append(('rollout',i));yield row(i)
    def collect(phase,prompt,selected=None,barrier=True):
        assert phase!='bank_rollout'
        if phase=='bank_online':
            assert not barrier
            assert prompt['snapshot']==snapshot
            events.append(('bank',selected[0]));return [row(selected[0])]
        assert phase=='bank_score'
        assert events==[event for i in order for event in [('rollout',i),('bank',i)]]
        return [{**row(i),'cumulative_bank_scoring':{'snapshot_sha256':snapshot['sha256'],'scoring_scope':'all'}} for i in range(5)]
    collect.rollout_ready=ready
    scored,after=cr.round_records(cfg,tmp_path,tmp_path,{'policy_sha256':policy,'case_cursor':0},1,tmp_path/'batch',collect)
    assert len(scored)==5 and after==snapshot


def test_other_coordinator_finishes_next_case_while_bank_caller_is_paused(tmp_path):
    release=threading.Event();third_done=threading.Event()
    def run(job):
        if job['index']==0:
            assert release.wait(3)
        if job['index']==3:
            third_done.set()
        return [{'case_cursor':job['index']}]
    with ThreadPoolExecutor(1) as p0, ThreadPoolExecutor(1) as p1:
        pools=[p0,p1]
        ready=ready_cases(jobs(tmp_path),lambda j:pools[j['worker']].submit(run,j),tmp_path/'order.json')
        assert next(ready)['case_cursor']==1
        # Simulate slow rubric API: caller does not resume the iterator yet.
        assert third_done.wait(2)
        release.set()
        records=list(ready)
    assert sorted(r['case_cursor'] for r in records)==[0,2,3,4]
