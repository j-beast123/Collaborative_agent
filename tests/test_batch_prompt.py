"""Batch prompt inheritance, frozen comparisons and union rewards; no API/GPU."""
import copy
import dataclasses
import json
from types import SimpleNamespace
import pytest
import test_witness_trainer as WT
from src.online_rl.tree import batch_prompt as B
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.team import team_strategies
from src.online_rl.integrated.records import sha256_json, sha256_text


def record(cid, success, failure, sha='p'):
    values={}; outcomes={}
    for name,xs,o in (('s',success,1.0),('f',failure,0.0)):
        for i,v in enumerate(xs):values[f'{cid}/{name}{i}']=v;outcomes[f'{cid}/{name}{i}']=o
    return {'case_id':cid,'policy_sha256':'frozen','prompt_sha256':sha,'tree':{'case_id':cid},
            'prompt_evaluation':{'per_trajectory':values,'source_outcome':outcomes,'rerun_policy_sha256':'frozen'}}


def test_pool_trajectories_across_four_cases_not_average_case_products():
    p=[record('a',[1]*9,[0]),record('b',[0],[1]*9),record('c',[],[]),record('d',[],[])]
    assert B.aggregate(p)['J']==pytest.approx(.81)
    assert B.aggregate(p)['n_success']==10


@pytest.mark.parametrize('s,f,winner,reason', [([.6],[.6],'P1','higher_product'),([.5],[.5],'P0','tie_keep_parent'),
                                             ([.2],[.9],'P0','parent_higher_product'),([], [.8],'P0','missing_direction')])
def test_product_winner_tie_and_missing_direction(s,f,winner,reason):
    parent=record('a',[.5] if s else [],[.5])
    decision=B.choose([parent],[record('a',s,f)])
    assert decision['winner']==winner and decision['reason']==reason


def test_reject_changed_tree_policy_or_feedback_mask():
    p=record('a',[1],[1]);c=copy.deepcopy(p);c['tree']['changed']=True
    with pytest.raises(RuntimeError,match='exact tree'):B.choose([p],[c])
    c=copy.deepcopy(p);c['policy_sha256']='updated'
    with pytest.raises(RuntimeError,match='frozen Agent'):B.choose([p],[c])
    c=record('a',[1,1],[1]);c['prompt_evaluation']['source_outcome']['a/s0']=0.0
    with pytest.raises(RuntimeError,match='sampled trajectory outcomes'):B.choose([p],[c])


def assessments(tmp_path,eager_score=False):
    tr,log,seen=WT.make(tmp_path,'rubric_only')
    tr.st=team_strategies(None,0,1)
    generate=tr.env.generate_fn
    def always_applicable(prompt,tin):
        value=generate(prompt,tin)
        value['rubric']['applicability_condition']='any turn'
        return value
    tr.env=dataclasses.replace(tr.env,generate_fn=always_applicable,
        continuation=lambda case,prefix,turn,text,seed:{'final_option':'A' if '[2]' in text else 'B',
                                                       'outcome':float('[2]' in text)})
    p0=tr.step(prompt_text=WT.P0R,prompt_id='P0',score_responses=eager_score,commit=False,apply_update=False)
    # P1 must not call any original rollout/final/feedback function.
    def forbidden(*args,**kwargs):pytest.fail('P1 regenerated the original dialogue/feedback')
    tr.env=dataclasses.replace(tr.env,agent=forbidden,doctor=forbidden,feedback_fn=forbidden)
    p1=tr.step(frozen_record=p0,prompt_text=WT.P0R+'\nBe precise.\n',prompt_id='P1',score_responses=eager_score,commit=False,apply_update=False)
    p0['prompt_evaluation']=B.evaluate_prompt_record(tr,p0)
    p1['prompt_evaluation']=B.evaluate_prompt_record(tr,p1)
    return tr,p0,p1,seen


def test_real_pipeline_freezes_dialogue_feedback_and_full_member_effect(tmp_path):
    tr,p0,p1,seen=assessments(tmp_path)
    assert p0['tree']==p1['tree'] and p0['feedback']==p1['feedback']
    assert p0['policy_sha256']==p1['policy_sha256']=='policy-0'
    assert tr.state.case_cursor==0 and not seen
    assert p0['response_scores']==p1['response_scores']==[]
    assert p0['prompt_evaluation']['effectiveness']
    assert p0['prompt_evaluation']['source_outcome']==p1['prompt_evaluation']['source_outcome']
    assert p0['prompt_evaluation']['n_success']>0 and p0['prompt_evaluation']['n_failure']>0
    assert B.choose([p0],[p1])['winner'] in ('P0','P1')


def test_losing_prompt_rubrics_still_score_once_and_na_remains_none(tmp_path):
    tr,p0,p1,_=assessments(tmp_path)
    assert p0['response_scores']==[] and p1['response_scores']==[]
    merged=B.score_union(tr,p0,p1,WT.P0R)
    ids=set(merged['rubric_weights'])
    assert ids and all(w==1 for w in merged['rubric_weights'].values())
    assert {r['side'] for r in merged['prompt_comparison']['union_lineage']}=={'P0','P1'}
    # Same behavior from both prompts collapses despite different generated IDs.
    assert len(ids)<len(p0['rubric_weights'])+len(p1['rubric_weights'])
    pairs=[(s['response_node_id'],s['rubric_id']) for s in merged['response_scores']]
    assert len(pairs)==len(set(pairs))
    assert merged['update']['deferred_to_gpu4']
    empty=copy.deepcopy(p1);empty['rubric_weights']={}
    no_rub=copy.deepcopy(p0);no_rub['rubric_weights']={}
    out=B.score_union(tr,no_rub,empty,WT.P0R)
    assert all(n['reward'] is None for n in out['agent_targets']['nodes'].values())


def test_eager_p0_p1_scores_are_reused_for_final_union(tmp_path):
    tr,p0,p1,seen=assessments(tmp_path,eager_score=True)
    assert p0['response_scores'] and p1['response_scores']
    seen.clear()
    merged=B.score_union(tr,p0,p1,WT.P0R)
    assert not seen  # final union did not issue a third response-scoring request
    assert merged['response_scores']
    assert {n['reward'] for n in merged['agent_targets']['nodes'].values()} <= {0.,.5,1.,None}


def test_union_retains_distinct_candidate_rubric_and_only_kept(tmp_path):
    tr,p0,p1,_=assessments(tmp_path)
    from src.online_rl.tree.rubric_bank import CHECKS
    tr.env=dataclasses.replace(tr.env,assign_fn=lambda *a:{'decision':'CREATE_NEW'})
    p1=copy.deepcopy(p1)
    kept=set(p1['rubric_weights'])
    for r in p1['trajectory_rubrics']:
        if r['rubric_id'] in kept:r['rubric']['required_behavior']+=' and identify uncertainty'
    pool,lineage=B.union_kept(tr,p0,p1,WT.P0R)
    assert set(r.rubric_id for r in pool)>=kept
    assert all(r.valid for r in pool)


def test_content_dedup_ignores_id_whitespace_and_case():
    r={'name':'Reason','applicability_condition':'When asked','required_behavior':'Explain why',
       'scores':{'0':'None','1':'Some','2':'Full'},'rubric_id':'one'}
    other={**r,'name':'  REASON  ','required_behavior':'Explain   why','rubric_id':'two'}
    assert B.content_key(r)==B.content_key(other)


def test_candidate_must_preserve_match_or_create_section_except_trailing_whitespace(tmp_path,monkeypatch):
    import types
    import src.online_rl.integrated.prompt_search as search
    from src.online_rl.integrated.prompt_search import PromptStore
    parent='CREATE rubric instructions\n\n[Assign or create]\nKeep all six checks.'
    monkeypatch.setattr(search,'_propose',lambda *args:{'proposals':[{'direction':'rewrite','text':
        'Revised CREATE instructions\n\n[Assign or create]\nA shortened reuse section.'}]})
    store=PromptStore(tmp_path/'prompts')
    backend=types.SimpleNamespace(feedback=lambda *a:None,refine=lambda *a:None,
                                  generate=lambda *a:None,gate=lambda *a:None,score=lambda *a:None)
    out=B.propose_candidate([],parent,store,backend,CallCache(None),[],[])
    assert not out['valid_candidate'] and out['text']==parent
    assert 'changed [Assign or create]' in out['proposal']['proposals'][0]['guard_violation']


def test_candidate_with_only_missing_final_newline_preserves_reuse_section():
    parent='CREATE rubric instructions\n\n[Assign or create]\nKeep all six checks.\n'
    candidate=parent.rstrip('\n')
    assert B._candidate_section_problem(candidate,parent,'Improve the creation instruction') is None


def test_candidate_cannot_edit_reuse_section_for_unrelated_direction():
    parent='CREATE rubric instructions\n\n[Assign or create]\nKeep all six checks.\n'
    candidate=parent.replace('all six','five')
    assert 'changed [Assign or create]' in B._candidate_section_problem(candidate,parent,'Improve creation')


def test_candidate_may_edit_reuse_section_when_direction_targets_reuse():
    parent=('CREATE rubric instructions\n\n[Assign or create]\n- same_construct: same behavior\n'
            '- applicability_visible: visible\n- same_required_behavior: required\n- same_direction: same direction\n'
            '- score_semantics_apply: applies\n- no_new_premise: none\n'
            'REUSE_EXISTING CREATE_NEW "decision" "rubric_id" "checks" "rationale"')
    candidate=parent.replace('same behavior','same behavior directly supported by target reply')
    assert B._candidate_section_problem(candidate,parent,'Tighten rubric creation and reuse') is None


def test_reflection_context_is_bounded_but_effect_score_uses_every_trajectory():
    rows=[]
    for i in range(4):
        r=record(str(i),[.2]*10,[.8]*10)
        r['prompt_evaluation'].update(S_success=.2,S_failure=.8,J=.16,n_success=10,n_failure=10,invalid=0,no_match=0)
        r['prompt_evaluation']['reflection']=[{'trajectory':k,'outcome':'success' if o else 'failure','R_O':.2 if o else .8}
                                               for k,o in r['prompt_evaluation']['source_outcome'].items()]
        r['rubric_bank']={'telemetry':{}}
        rows.append(r)
    view=B.batch_reflection_view(rows,max_examples=16)
    assert len(view)==17 and view[0]['total_trajectories']==80
    assert view[0]['batch_summary']['n_success']==40
    assert view[0]['batch_summary']['J']==pytest.approx(.16)
    assert len({(r['trajectory'].split('/')[0],r['outcome']) for r in view[1:]})==8


def test_one_common_candidate_then_union_before_update(tmp_path,monkeypatch):
    from scripts.online_rl import run_five_gpu as runner
    from src.online_rl.integrated.prompt_search import PromptStore
    store=PromptStore(tmp_path/'prompts'); sha=store.put('parent')
    state={'prompt_sha256':sha,'prompt_id':'P0','policy_sha256':'frozen'}
    phases=[]; parents=[record(str(i),[.2],[.2]) for i in range(4)]
    children=[record(str(i),[.8],[.8]) for i in range(4)]
    def collect(phase,prompt):
        phases.append((phase,prompt['sha256']))
        return parents if phase=='P0' else children if phase=='P1' else [{'union':i} for i in range(4)]
    def propose(*args):
        assert args[0]==parents
        return {'text':'child','sha256':store.put('child'),'proposal':{},'valid_candidate':True}
    monkeypatch.setattr(B,'propose_candidate',propose)
    rows,event=runner.prompt_round(1,[{'case':{'case_id':str(i)}} for i in range(4)],state,collect,store,None,None,[])
    assert [p[0] for p in phases]==['P0','P1','union']
    assert event['next_prompt']['sha256']==sha256_text('child') and len(rows)==4
    assert event['decision']['accepted'] and state['prompt_sha256']==sha


@pytest.mark.parametrize('mutate,expected',[
    (lambda c:c['prompt_update'].update(candidate_count=4),'one P1 candidate'),
    (lambda c:c['acceptance'].update(rule='stratified_paired_bootstrap_J'),'strictly higher'),
    (lambda c:c['five_gpu']['opening_identity'].update(system_prompt_sha256='old'),'current Doctor prompt'),
])
def test_preflight_rejects_legacy_search_and_stale_doctor_identity(monkeypatch,mutate,expected):
    from scripts.online_rl import run_five_gpu as runner
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    cfg=json.loads((runner.ROOT/'configs/online_rl/tree_v21_five_gpu.json').read_text())
    mutate(cfg)
    assert any(expected in e for e in runner.preflight(cfg))
