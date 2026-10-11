import dataclasses,json
import pytest
import test_rubric_reuse as R
from src.online_rl.tree import rubric_bank as B,backends as A,posthoc as P
from src.online_rl.tree.cache import CallCache


def many():
    _,ls,ts=R.tree();leaf=ls[0]
    leaves=[dataclasses.replace(leaf,leaf_id=f'L{i:03d}',path=(f'L{i:03d}',)) for i in range(33)]
    tins={l.leaf_id:P.freeze_trajectory_input(R.CASE,l,dataclasses.replace(R.DoctorFeedback(l.leaf_id,l.outcome,'same reason','h'),target_reply_id=1),True) for l in leaves}
    return leaves,tins


def test_33_same_reason_members_use_four_api_calls_and_resume_without_calls(tmp_path):
    leaves,tins=many();f=R.Fakes();calls=[]
    def match(package,items,bank):
        calls.append(([t.leaf_id for t in items],B.bank_view(bank)))
        assert len(items)<=8
        return {'decisions':{t.leaf_id:{'decision':'REUSE_EXISTING','rubric_id':bank[0].rubric_id,'checks':{k:True for k in B.CHECKS},'rationale':f'locus {t.target_reply_id}'} for t in items}}
    def single(*args):pytest.fail('Used individual match API')
    single.batch=match;single.batch_size=8
    cache=CallCache(tmp_path/'calls.jsonl')
    canon,assign=B.build_banks(R.CASE,leaves,tins,R.PROMPT,'p',f.generate_fn,single,f.gate_fn,cache)
    assert len(canon)==1 and len(calls)==4 and len(assign)==33
    assert assign[0].assignment=='CREATED_NEW' and all(a.assignment=='REUSED_EXISTING' for a in assign[1:])
    assert [a.order for a in assign]==list(range(33))
    assert cache.calls['rubric_match_batch']==4
    B.build_banks(R.CASE,leaves,tins,R.PROMPT,'p',f.generate_fn,single,f.gate_fn,CallCache(tmp_path/'calls.jsonl'))
    assert len(calls)==4 and len(f.calls['generate'])==1


def test_new_members_rechecked_against_real_new_rubric_not_blindly_clustered():
    leaves,tins=many();leaves=leaves[:9];calls=[];f=R.Fakes()
    def single(*args):pytest.fail('Used individual match API')
    def match(package,items,bank):
        calls.append((len(items),len(bank)))
        return {'decisions':{t.leaf_id:{'decision':'CREATE_NEW' if len(bank)==1 else 'REUSE_EXISTING',
           'rubric_id':bank[-1].rubric_id if len(bank)>1 else '', 'checks':{k:True for k in B.CHECKS},'rationale':'new construct'} for t in items}}
    single.batch=match;single.batch_size=8
    canon,asg=B.build_banks(R.CASE,leaves,tins,R.PROMPT,'p',f.generate_fn,single,f.gate_fn,CallCache(None))
    assert calls==[(8,1),(7,2)] and len(canon)==2
    assert asg[1].assignment=='CREATED_NEW' and all(a.canonical_rubric_id==asg[1].canonical_rubric_id for a in asg[2:])


def test_gate_passed_saved_first_candidates_anchor_batch_match_without_regeneration():
    leaves,tins=R.tree()[1:]
    generated=R.Fakes(reuse=R.SAME)
    original,_=B.build_banks(R.CASE,leaves,tins,R.PROMPT,'p',generated.generate_fn,
                             generated.assign_fn,generated.gate_fn,CallCache(None))
    assert len(original)==2
    original={rid:dataclasses.replace(r,rubric={**r.rubric,'source_provenance':{'source_case_id':R.CASE['case_id']}})
              for rid,r in original.items()}
    calls=[]
    def single(*args):pytest.fail('Seeded bank must use the configured batch matcher')
    def batch(package,items,bank):
        calls.append((len(items),tuple(r.rubric_id for r in bank)))
        return {'decisions':{t.leaf_id:{'decision':'REUSE_EXISTING','rubric_id':bank[0].rubric_id,
            'checks':{k:True for k in B.CHECKS},'rationale':'same criterion'} for t in items}}
    single.batch=batch;single.batch_size=8
    fresh=R.Fakes()
    canon,assign=B.build_banks(R.CASE,leaves,tins,R.PROMPT,'p',fresh.generate_fn,single,
        fresh.gate_fn,CallCache(None),initial_candidates=list(original.values()))
    assert fresh.calls['generate']==[]
    assert sum(a.assignment=='CREATED_NEW' for a in assign)==2
    assert sum(a.assignment=='REUSED_EXISTING' for a in assign)==3
    assert len(canon)==2 and len(assign)==5
    assert calls==[(2,(next(r.rubric_id for r in original.values() if r.leaf_id.startswith('L0')),)),
                   (1,(next(r.rubric_id for r in original.values() if r.leaf_id.startswith('L2')),))]


def test_all_generated_candidates_are_consolidated_in_order_not_preseeded_as_bank():
    leaves,tins=many();leaves=leaves[:9];tins={l.leaf_id:tins[l.leaf_id] for l in leaves}
    old=R.Fakes();saved=[]
    for leaf in leaves:
        candidate=P.generate_rubric(R.PROMPT,'p',tins[leaf.leaf_id],old.generate_fn,CallCache(None),old.gate_fn)
        candidate=dataclasses.replace(candidate,rubric={**candidate.rubric,
            'source_provenance':{'source_case_id':R.CASE['case_id']}})
        saved.append(candidate)
    calls=[]
    def single(*args):pytest.fail('Saved full-generation candidates must use batch consolidation')
    def batch(package,items,bank):
        calls.append((tuple(t.leaf_id for t in items),tuple(r.leaf_id for r in bank)))
        return {'decisions':{t.leaf_id:{'decision':'REUSE_EXISTING','rubric_id':bank[0].rubric_id,
            'checks':{k:True for k in B.CHECKS},'rationale':'same reason'} for t in items}}
    single.batch=batch;single.batch_size=8
    fresh=R.Fakes()
    canon,assign=B.build_banks(R.CASE,leaves,tins,R.PROMPT,'p',fresh.generate_fn,single,
        fresh.gate_fn,CallCache(None),initial_candidates=saved)
    assert fresh.calls['generate']==[]
    assert len(canon)==1
    assert assign[0].assignment=='CREATED_NEW'
    assert all(a.assignment=='REUSED_EXISTING' for a in assign[1:])
    assert calls==[(tuple(l.leaf_id for l in leaves[1:]),(leaves[0].leaf_id,))]


@pytest.mark.parametrize('bad',['missing','duplicate','unknown','bad_bool'])
def test_malformed_batch_response_raises_and_is_not_cached(monkeypatch,bad):
    leaves,tins=many();items=[tins[l.leaf_id] for l in leaves[:2]]
    rows=[{'trajectory_id':t.leaf_id,'decision':'CREATE_NEW','rubric_id':'','checks':{k:False for k in A.MATCH_CHECKS},'rationale':'new'} for t in items]
    if bad=='missing':rows.pop()
    if bad=='duplicate':rows[1]['trajectory_id']=rows[0]['trajectory_id']
    if bad=='unknown':rows[1]['trajectory_id']='elsewhere'
    if bad=='bad_bool':rows[0]['checks']['same_construct']='true'
    monkeypatch.setattr(A,'_openrouter_json',lambda *args:(json.dumps({'decisions':rows}),{}))
    matcher=A.api_rubric_matcher('fake','fake',{},match_batch_size=8);cache=CallCache(None)
    with pytest.raises(A.RubricOutputIncomplete):cache.call('batch',{},lambda:matcher.batch('prompt',items,[]))
    assert not cache.store
