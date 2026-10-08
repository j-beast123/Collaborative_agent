import json
from concurrent.futures import Future
from threading import Thread

import pytest

from src.online_rl.tree.pipelined_prompt import evaluate, SEED_MODE
from src.online_rl.tree.batch_prompt import choose
from src.online_rl.integrated.records import sha256_text, sha256_json, atomic_write_json


def record(i, child=False):
    values={f'case{i}/success': .8 if child and i else .1, f'case{i}/failure': .8 if child and i else .1}
    if not child: values={k:.5 for k in values}
    return {'case_id':f'case{i}','policy_sha256':'frozen','tree':{'case_id':f'case{i}'},
            'prompt_evaluation':{'per_trajectory':values,'source_outcome':{f'case{i}/success':1.,f'case{i}/failure':0.},
                                 'rerun_policy_sha256':'frozen'}}


def test_p1_waits_for_all_p0_and_proposal_uses_all_cases(tmp_path):
    jobs=[{'case':{'case_id':f'case{i}'}} for i in range(5)]
    parent={'sha256':sha256_text('parent'),'text':'parent','id':'P0'}
    unfinished=Future(); events=[]; proposals=[]
    def submit(phase,job,prompt):
        i=int(job['case']['case_id'][-1]);events.append((phase,i,prompt['sha256']))
        if phase=='P0' and i==0:return unfinished
        f=Future();f.set_result(record(i,phase=='P1'))
        if phase=='P0' and i==4:
            assert not any(p=='P1' for p,_,_ in events)
            unfinished.set_result(record(0))
        return f
    def propose(rows):
        assert len(rows)==5 and {row['case_id'] for row in rows}=={f'case{i}' for i in range(5)}
        proposals.extend(row['case_id'] for row in rows)
        return {'text':'child','sha256':sha256_text('child'),'proposal':{},'valid_candidate':True}
    p0,p1,candidate,seed=evaluate(jobs,parent,submit,propose,tmp_path,'frozen')
    assert seed['seed_mode']==SEED_MODE and len(seed['p0_record_sha256'])==5
    assert set(proposals)=={f'case{i}' for i in range(5)}
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


def test_resume_revalidates_saved_invalid_fallback_candidate(tmp_path):
    jobs=[{'case':{'case_id':'case0'}}]
    parent={'sha256':sha256_text('parent'),'text':'parent','id':'P0'}
    source=record(0)
    seed={'parent_prompt_sha256':parent['sha256'],'policy_sha256':'frozen','case_ids':['case0'],
          'seed_mode':SEED_MODE,'p0_record_sha256':[sha256_json(source)],
          'p0_evidence_sha256':sha256_json({'record_sha256':[sha256_json(source)]})}
    atomic_write_json(tmp_path/'prompt_seed.json',seed)
    atomic_write_json(tmp_path/'prompt_candidate.json',{'text':'parent','sha256':parent['sha256'],
          'valid_candidate':False,'id':'P1-parent','seed_sha256':sha256_json(seed)})
    phases=[]
    def submit(phase,job,prompt):
        phases.append((phase,prompt['sha256']))
        f=Future();f.set_result(record(0,phase=='P1'));return f
    calls=[]
    def propose(rows):
        calls.append([row['case_id'] for row in rows])
        return {'text':'new child','sha256':sha256_text('new child'),'proposal':{},'valid_candidate':True}
    _,p1,candidate,returned_seed=evaluate(jobs,parent,submit,propose,tmp_path,'frozen')
    assert calls==[['case0']] and candidate['valid_candidate']
    assert returned_seed==seed and p1[0]['case_id']=='case0'
    assert ('P1',sha256_text('new child')) in phases


def test_union_phase_waits_until_all_p0_and_p1_rubric_sets_are_final(tmp_path):
    jobs=[{'case':{'case_id':f'case{i}'}} for i in range(5)]
    parent={'sha256':sha256_text('parent'),'text':'parent','id':'P0'}
    events=[]
    def submit(phase,job,prompt):
        i=int(job['case']['case_id'][-1]);events.append((phase,i))
        f=Future()
        if phase in ('P0','P1'):
            f.set_result(record(i,phase=='P1'))
        else:
            assert phase=='union' and prompt['parent_sha256']==parent['sha256']
            f.set_result({'case_id':f'case{i}','union':True})
        return f
    def propose(rows):return {'text':'child','sha256':sha256_text('child'),'proposal':{},'valid_candidate':True}
    p0,p1,candidate,seed=evaluate(jobs,parent,submit,propose,tmp_path,'frozen')
    assert len(p0)==len(p1)==5
    assert all(('P1',i) in events for i in range(5))
    assert not any(phase=='union' for phase,_ in events)
    assert choose(p0,p1)['winner']=='P1'
    # The caller starts union scoring only after evaluate has returned all P0/P1 records.
    for job in jobs:
        submit('union',job,{'parent_sha256':parent['sha256']})
    last_prompt_phase=max(i for i,(phase,_) in enumerate(events) if phase in ('P0','P1'))
    first_union_phase=min(i for i,(phase,_) in enumerate(events) if phase=='union')
    assert last_prompt_phase < first_union_phase
