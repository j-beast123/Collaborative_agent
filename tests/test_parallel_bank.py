"""Concurrency/merge/recovery tests with no models, servers or API requests."""
from dataclasses import dataclass
from threading import RLock,Event,Barrier
from types import SimpleNamespace as NS
from copy import deepcopy
from pathlib import Path
import json
import pytest
from src.online_rl.tree import trajectory_stream as S, bank_delta as D, cumulative_bank as CB
from src.online_rl.tree.records import TrajectoryRubric,SelectionDecision
from src.online_rl.integrated.records import sha256_json,to_dict
from tests.test_trajectory_stream import State,Tin,item


def extend(snapshot, records):
    entries={e['content_sha256']:deepcopy(e) for e in snapshot['entries']}
    for record in records:
        canon={r['rubric_id']:r for r in record['trajectory_rubrics']}
        for choice in record['selection']:
            if not choice['retained']:continue
            raw=canon[choice['rubric_id']];key=CB.content_hash(raw['rubric'])
            if key not in entries:entries[key]={'bank_id':'bank_'+key,'content_sha256':key,'rubric':raw,'sources':[]}
            proof={'case_id':record['case_id'],'proof_sha':sha256_json(record)}
            if proof not in entries[key]['sources']:entries[key]['sources'].append(proof)
    return CB._seal({'protocol':CB.VERSION,'version':snapshot['version']+1,
                     'parent_sha256':snapshot['sha256'],'entries':[entries[k] for k in sorted(entries)]})


def fake_built(kw, rule):
    ids=[l.leaf_id for l in kw['ruled']];rid='r_'+ids[0]
    rubric=TrajectoryRubric(rid,ids[0],'prompt','{}',{'rubric_id':rid,'rule':rule},True,(),{'passed':True})
    choice=SelectionDecision(rid,ids[0],True,'witness','verified')
    record={'case_id':kw['case']['case_id'],'case_cursor':kw['state'].case_cursor,
            'policy_sha256':'policy','prompt_sha256':'prompt','trajectory_rubrics':[to_dict(rubric)],
            'selection':[to_dict(choice)],'rubric_bank':{'witness':[]}}
    before=kw['snapshot'];after=extend(before,[record]);tail=before['sha256'];decisions=[]
    for i,tid in enumerate(ids):
        next_sha=after['sha256'] if i==0 else tail
        decisions.append({'case_id':kw['case']['case_id'],'case_cursor':kw['state'].case_cursor,'leaf_id':tid,
                          'action':'GENERATED_KEEP' if i==0 else 'GENERATED_REMOVE',
                          'input_bank_sha256':tail,'output_bank_sha256':next_sha})
        tail=next_sha
    report={'chunk_index':kw['chunk_offset'],'trajectory_ids':ids,
            'generator_api_requests':1,'generator_correction_requests':0,'peer_matching_calls':0,
            'bank_before_sha256':before['sha256'],'bank_after_sha256':after['sha256'],
            'bank_before_size':len(before['entries']),'bank_after_size':len(after['entries'])}
    return {'working':after,'canon':{rid:rubric},'assignments':[],'selection':[choice],
            'witness':[],'witness_objects':{},'cumulative_reuse':[],'decisions':decisions,
            'summary':{'chunks':[report],'uncovered_ids':[]},'order_next':{'success':len(ids),'failure':0}}


def setup(monkeypatch,tmp_path,process):
    monkeypatch.setattr(CB,'validate_snapshot',lambda v:v)
    monkeypatch.setattr(CB,'content_hash',lambda r:sha256_json(r['rule']))
    monkeypatch.setattr(CB,'extend_snapshot',extend)
    monkeypatch.setattr(S,'freeze_trajectory_input',lambda c,l,f,**kw:Tin(l.leaf_id))
    monkeypatch.setattr(S.OCB,'process_case',process)
    monkeypatch.setattr(D,'compare',lambda *args:{'aliases':{},'status':'distinct'})
    before=CB.empty_snapshot();root=S.initialize(tmp_path/'bank',before,{'policy':'policy'},protocol=D.VERSION)
    lock=RLock()
    def make(cursor):
        out=tmp_path/f'case_{cursor}';out.mkdir(exist_ok=True)
        trainer=NS(out=out,cumulative_bank={'discovery_trajectory_cap':None,'parallel_bank_processing':True},
                   source_consistency=False,cfg={'rollout':{'rounds':4,'branching':4}})
        return S.StreamingCase(trainer,{'case_id':f'c{cursor}'},State(),'p',before,
                              {'root':root,'lock':lock,'case_cursor':cursor})
    return make,Path(root),before


def push(stream,cursor):
    for i in range(8):stream.push(*item(i,cursor))


def test_two_chunks_verify_concurrently_and_latest_merge_loses_nothing(monkeypatch,tmp_path):
    barrier=Barrier(2);inputs=[]
    def process(trainer,**kw):
        inputs.append(kw['snapshot']['sha256']);barrier.wait(timeout=3)
        return fake_built(kw,kw['case']['case_id'])
    make,root,before=setup(monkeypatch,tmp_path,process)
    with make(0) as a,make(1) as b:
        push(a,0);push(b,1);a.finish();b.finish()
    after,decisions=S.audit(root,before)
    assert inputs==[before['sha256']]*2
    assert len(after['entries'])==2 and len(decisions)==16
    receipts=[S.read(root/'receipts'/(key+'.json')) for key in S.read(root/'state.json')['transactions']]
    assert receipts[1]['commit_input_bank_sha256']!=receipts[1]['input_bank_sha256']


def test_exact_duplicate_race_registers_once_and_keeps_both_proofs(monkeypatch,tmp_path):
    barrier=Barrier(2)
    def process(trainer,**kw):barrier.wait(timeout=3);return fake_built(kw,'same')
    make,root,before=setup(monkeypatch,tmp_path,process)
    with make(0) as a,make(1) as b:
        push(a,0);push(b,1);a.finish();b.finish()
    after,_=S.audit(root,before)
    assert len(after['entries'])==1 and len(after['entries'][0]['sources'])==2


def test_semantic_duplicate_checks_only_delta_and_keeps_first(monkeypatch,tmp_path):
    both=Barrier(2);allow_b=Event();comparisons=[]
    def process(trainer,**kw):
        both.wait(timeout=3)
        if kw['case']['case_id']=='c1':assert allow_b.wait(3)
        return fake_built(kw,'first wording' if kw['case']['case_id']=='c0' else 'other wording')
    make,root,before=setup(monkeypatch,tmp_path,process)
    def compare(trainer,pending,delta):
        comparisons.append((pending,delta));target=delta[0];key=next(iter(pending))
        return {'status':'compared','aliases':{key:{'protocol':D.VERSION,'candidate_content_sha256':key,
            'bank_id':target['bank_id'],'bank_content_sha256':target['content_sha256'],
            'checks':{k:True for k in D.CHECKS},'rationale':'equivalent','comparison_sha256':'proof'}}}
    monkeypatch.setattr(D,'compare',compare)
    with make(0) as a,make(1) as b:
        push(a,0);push(b,1);a.finish();allow_b.set();result=b.finish()
    after,_=S.audit(root,before)
    assert len(after['entries'])==1 and after['entries'][0]['rubric']['rubric']['rule']=='first wording'
    assert len(comparisons)==1 and len(comparisons[0][1])==1
    assert result['summary']['semantic_aliases']
    assert result['summary']['chunks'][0]['commit_added_unique']==0


def test_append_during_delta_comparison_gets_another_delta_pass(monkeypatch,tmp_path):
    ready=Barrier(3);allow_b=Event();allow_c=Event();comparing=Event();resume_compare=Event();deltas=[]
    def process(trainer,**kw):
        ready.wait(timeout=3)
        cid=kw['case']['case_id']
        if cid=='c1':assert allow_b.wait(3)
        if cid=='c2':assert allow_c.wait(3)
        return fake_built(kw,cid)
    make,root,before=setup(monkeypatch,tmp_path,process)
    def compare(trainer,pending,delta):
        if trainer.out.name=='case_1':
            deltas.append([e['rubric']['rubric']['rule'] for e in delta])
            if len(deltas)==1:comparing.set();assert resume_compare.wait(3)
        return {'aliases':{},'status':'distinct'}
    monkeypatch.setattr(D,'compare',compare)
    with make(0) as a,make(1) as b,make(2) as c:
        push(a,0);push(b,1);push(c,2);a.finish();allow_b.set()
        assert comparing.wait(3)
        allow_c.set();c.finish();resume_compare.set();b.finish()
    assert deltas==[['c0'],['c2']]
    assert len(S.audit(root,before)[0]['entries'])==3


def test_resume_uses_verified_draft_and_commits_once(monkeypatch,tmp_path):
    calls=[]
    def process(trainer,**kw):calls.append(kw);return fake_built(kw,'rule')
    make,root,before=setup(monkeypatch,tmp_path,process)
    with make(0) as a:push(a,0);first=a.finish()
    journal=S.read(root/'state.json');journal.update(snapshot=before,transactions=[])
    S.atomic_write_json(root/'state.json',journal)
    with make(0) as a:push(a,0);second=a.finish()
    assert len(calls)==1 and first['summary']==second['summary']
    assert len(S.audit(root,before)[0]['entries'])==1


def test_delta_failure_keeps_verified_candidates_distinct(monkeypatch,tmp_path):
    def process(trainer,**kw):return fake_built(kw,kw['case']['case_id'])
    make,root,before=setup(monkeypatch,tmp_path,process)
    # Pin B's discovery before A's commit, but let A commit first.
    both=Barrier(2);allow=Event()
    def parallel_process(trainer,**kw):
        both.wait(timeout=3)
        if kw['case']['case_id']=='c1':assert allow.wait(3)
        return fake_built(kw,kw['case']['case_id'])
    monkeypatch.setattr(S.OCB,'process_case',parallel_process)
    monkeypatch.setattr(D,'compare',lambda *args:{'aliases':{},'status':'execution_unresolved'})
    with make(0) as a,make(1) as b:
        push(a,0);push(b,1);a.finish();allow.set();b.finish()
    assert len(S.audit(root,before)[0]['entries'])==2


def test_config_declares_parallel_protocol_and_separate_output():
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.tree.cumulative_run import config_identity
    root=Path(__file__).resolve().parents[1]
    cfg=load_config('configs/online_parallel_bank.json')
    assert config_identity(cfg,root)['parallel_bank_processing']==D.VERSION
    assert cfg['five_gpu']['overlap_rollout_with_bank']
    assert cfg['agent_reward']['w_outcome']==cfg['agent_reward']['w_rubric']==.5
    other=load_config('configs/online_rubric_only_parallel_bank.json')
    assert other['agent_reward']['w_outcome']==0 and other['agent_reward']['w_rubric']==1


def test_full_bank_scoring_accepts_only_a_bound_semantic_alias(monkeypatch):
    from src.online_rl.tree.trainer import TreeTrainer
    from src.online_rl.tree import trainer as T,scoring
    from src.online_rl.tree.records import Tree
    monkeypatch.setattr(CB,'validate_snapshot',lambda v:v)
    monkeypatch.setattr(CB,'content_hash',lambda r:r['hash'])
    monkeypatch.setattr(CB,'scoring_rubrics',lambda v:[])
    monkeypatch.setattr(scoring,'score_cumulative_nodes',lambda *a,**kw:([],{}))
    monkeypatch.setattr(T,'agent_targets',lambda *a:{})
    monkeypatch.setattr(T,'sibling_group_diagnostics',lambda *a:{})
    monkeypatch.setattr(T,'interaction_diagnostics',lambda *a:{})
    tr=object.__new__(TreeTrainer);tr.cumulative_bank={'scoring_scope':'all'}
    tr.validate_member_interventions=lambda record:None;tr.policy_sha=lambda:'policy'
    tr.cases={'c':{'case_id':'c'}};tr.env=NS(scorer=None);tr.cache=None;tr.api_workers=1;tr.st=None
    alias={'protocol':D.VERSION,'candidate_content_sha256':'local','bank_id':'bank_existing',
           'bank_content_sha256':'existing','checks':{k:True for k in D.CHECKS},
           'rationale':'equivalent','comparison_sha256':'proof'}
    record={'case_id':'c','rubric_bank':{},'cumulative_bank_input':{'entries':[]},
            'trajectory_rubrics':[{'rubric_id':'r','rubric':{'hash':'local'}}],
            'selection':[{'rubric_id':'r','retained':True}],
            'offline_chunk_bank':{'parallel_bank_protocol':D.VERSION,'semantic_aliases':{'local':alias}},
            'tree':to_dict(Tree('c','policy',4,4,(),()))}
    snapshot={'sha256':'final','entries':[{'bank_id':'bank_existing','content_sha256':'existing'}]}
    assert tr.score_cumulative(record,snapshot)['cumulative_bank_scoring']['snapshot_sha256']=='final'
    record['offline_chunk_bank']['semantic_aliases']={}
    with pytest.raises(RuntimeError,match='omits'):tr.score_cumulative(record,snapshot)


def test_real_public_bank_proofs_survive_concurrent_exact_merge_and_resume(monkeypatch,tmp_path):
    from src.online_rl.tree.records import RubricAssignment
    from src.online_rl.tree.witness import WitnessVerification
    real_validate,real_hash,real_extend=CB.validate_snapshot,CB.content_hash,CB.extend_snapshot
    barrier=Barrier(2);calls=[]
    def process(trainer,**kw):
        calls.append(kw['case']['case_id'])
        if len(calls)<=2:barrier.wait(timeout=3)
        ids=[l.leaf_id for l in kw['ruled']];rid='r_'+ids[0]
        app={'user_side_context':'A requested comparison','agent_side_state':'No extra state required'}
        rubric={'rubric_id':rid,'name':'comparison','applicability':app,'required_behavior':'Compare the requested alternatives',
                'minimum_performance':'Identify an alternative','completion_condition':'Explain the comparison',
                'criterion_version':'public_behavior_app_alignment_v5',
                'scores':{str(i):{**app,'behavior':f'Observable behavior at level {i}'} for i in range(3)},
                'source_provenance':{'source_case_id':kw['case']['case_id'],'source_trajectory_id':ids[0],'source_outcome':'success'}}
        gate={'protocol':'public_mc_gate_v3','passed':True,'observable':True,'score_boundaries_clear':True,
              'source_anchor':{'matches':True},'criterion_sha256':CB.content_hash(rubric),
              'public_criterion_version':rubric['criterion_version']}
        raw=TrajectoryRubric(rid,ids[0],'prompt','{}',rubric,True,(),gate,{'status':'PASS','first':{'applicable':True}})
        tested={'leaf_id':ids[0],'result':'PASS','stage1_applicable':True,'outcomes':[0.,0.],'reruns':2,
                'intervention_audit':{'protocol':'intervention_instruction_audit_v1','passed':True,
                    'answer':{'instruction_aligned':True,'unsupported_context_premise':False,'conclusion_prescribed':False}},
                'rerun_records':[{'outcome':0.,'rubric_id':rid,'leaf_id':ids[0],'continuation_policy_sha256':'policy',
                    'level':'0','seed':i,'agent_seed':i,'intervention_spec':{'rubric_sha256':sha256_json(rubric)}} for i in range(2)]}
        wit=WitnessVerification(rid,'success',1,(ids[0],),(ids[0],),(tested,),1,2,ids[0],'KEEP','witness_found')
        result={'canon':{rid:raw},'selection':[SelectionDecision(rid,ids[0],True,'witness','verified')],
                'assignments':[RubricAssignment(ids[0],kw['case']['case_id'],'success',0,rid,'CREATED_NEW',None,'','f')],
                'witness':[to_dict(wit)],'witness_objects':{rid:wit},'cumulative_reuse':[],'order_next':{'success':1,'failure':0}}
        after=CB.extend_snapshot(kw['snapshot'],D.merge_records(result,kw['case'],kw['state'],{}))
        result['working']=after;tail=kw['snapshot']['sha256'];rows=[]
        for i,tid in enumerate(ids):
            output=after['sha256'] if i==0 else tail
            rows.append({'case_id':kw['case']['case_id'],'leaf_id':tid,'action':'GENERATED_KEEP' if i==0 else 'GENERATED_REMOVE',
                         'input_bank_sha256':tail,'output_bank_sha256':output});tail=output
        result['decisions']=rows
        result['summary']={'uncovered_ids':[],'chunks':[{'chunk_index':0,'trajectory_ids':ids,
              'generator_api_requests':1,'generator_correction_requests':0,'peer_matching_calls':0,
              'bank_before_sha256':kw['snapshot']['sha256'],'bank_after_sha256':after['sha256'],
              'bank_before_size':len(kw['snapshot']['entries']),'bank_after_size':len(after['entries'])}]}
        return result
    make,root,before=setup(monkeypatch,tmp_path,process)
    monkeypatch.setattr(CB,'validate_snapshot',real_validate)
    monkeypatch.setattr(CB,'content_hash',real_hash)
    monkeypatch.setattr(CB,'extend_snapshot',real_extend)
    with make(0) as a,make(1) as b:
        push(a,0);push(b,1);a.finish();b.finish()
    after,_=S.audit(root,before)
    assert len(real_validate(after)['entries'])==1 and len(after['entries'][0]['sources'])==2
    with make(0) as a:push(a,0);a.finish()
    assert len(calls)==2 and S.audit(root,before)[0]==after
