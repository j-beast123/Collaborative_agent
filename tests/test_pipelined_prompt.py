import json
from concurrent.futures import Future
from threading import Thread

import pytest

from src.online_rl.tree.pipelined_prompt import evaluate
from src.online_rl.tree.batch_prompt import choose
from src.online_rl.integrated.records import sha256_text


def record(i, child=False):
    values={f'case{i}/success': .8 if child and i else .1, f'case{i}/failure': .8 if child and i else .1}
    if not child: values={k:.5 for k in values}
    return {'case_id':f'case{i}','policy_sha256':'frozen','tree':{'case_id':f'case{i}'},
            'prompt_evaluation':{'per_trajectory':values,'source_outcome':{f'case{i}/success':1.,f'case{i}/failure':0.},
                                 'rerun_policy_sha256':'frozen'}}


def test_first_finisher_p1_overlaps_remaining_p0_but_winner_uses_all_cases(tmp_path):
    jobs=[{'case':{'case_id':f'case{i}'}} for i in range(5)]
    parent={'sha256':sha256_text('parent'),'text':'parent','id':'P0'}
    unfinished=Future(); events=[]; proposals=[]
    def submit(phase,job,prompt):
        i=int(job['case']['case_id'][-1]);events.append((phase,i,prompt['sha256']))
        if phase=='P0' and i==0:return unfinished
        f=Future();f.set_result(record(i,phase=='P1'))
        if phase=='P1' and i==1:
            assert not unfinished.done() # P1 starts BEFORE last P0 is allowed to finish
            unfinished.set_result(record(0))
        return f
    def propose(rows):
        assert len(rows)==1 and rows[0]['case_id']=='case1'
        proposals.append(rows[0]['case_id'])
        return {'text':'child','sha256':sha256_text('child'),'proposal':{},'valid_candidate':True}
    p0,p1,candidate,seed=evaluate(jobs,parent,submit,propose,tmp_path,'frozen')
    assert seed['seed_case_id']=='case1' and proposals==['case1']
    assert len([x for x in events if x[0]=='P1'])==5
    assert len({x[2] for x in events if x[0]=='P1'})==1
    assert choose([p0[0]],[p1[0]])['winner']=='P0' # this case alone would reject
    assert choose(p0,p1)['winner']=='P1' # full batch chooses differently
    assert json.loads((tmp_path/'prompt_seed.json').read_text())==seed
    # Cached P0 now completes in a different order. Neither seed nor P1 changes.
    def resumed_submit(phase,job,prompt):
        f=Future();f.set_result(record(int(job['case']['case_id'][-1]),phase=='P1'));return f
    def forbidden(*args):pytest.fail('Resume regenerated a fixed P1')
    again=evaluate(jobs,parent,resumed_submit,forbidden,tmp_path,'frozen')
    assert again[2]==candidate and again[3]==seed


def test_saved_seed_cannot_be_replaced_or_used_with_changed_policy(tmp_path):
    jobs=[{'case':{'case_id':'case0'}}];parent={'sha256':'p','text':'p','id':'P0'}
    def submit(*args):
        f=Future();f.set_result(record(0));return f
    def fail(*args):raise RuntimeError('API interrupted')
    with pytest.raises(RuntimeError,match='API interrupted'):evaluate(jobs,parent,submit,fail,tmp_path,'frozen')
    assert (tmp_path/'prompt_seed.json').exists() and not (tmp_path/'prompt_candidate.json').exists()
    with pytest.raises(RuntimeError,match='differs'):evaluate(jobs,parent,submit,fail,tmp_path,'changed')


def test_ready_case_union_scoring_runs_before_other_case_p0_and_p1(tmp_path):
    jobs=[{'case':{'case_id':f'case{i}'}} for i in range(5)]
    parent={'sha256':sha256_text('parent'),'text':'parent','id':'P0'}
    slow_p0=Future();slow_p1=Future();events=[]
    def submit(phase,job,prompt):
        i=int(job['case']['case_id'][-1]);events.append((phase,i))
        if phase=='P0' and i==0:return slow_p0
        if phase=='P1' and i==0:return slow_p1
        f=Future()
        if phase=='union':
            assert prompt['parent_sha256']==parent['sha256']
            f.set_result({'case_id':f'case{i}','union':True})
            if i==1:
                assert not slow_p0.done() and not slow_p1.done()
                slow_p0.set_result(record(0))
            if i==2:slow_p1.set_result(record(0,True))
        else:f.set_result(record(i,phase=='P1'))
        return f
    def propose(rows):return {'text':'child','sha256':sha256_text('child'),'proposal':{},'valid_candidate':True}
    p0,p1,candidate,seed,unions=evaluate(jobs,parent,submit,propose,tmp_path,'frozen',eager_union=True)
    assert len(unions)==5 and len({r['case_id'] for r in unions})==5
    assert events.index(('union',1))<events.index(('P1',0))
    assert choose(p0,p1)['winner']=='P1'
