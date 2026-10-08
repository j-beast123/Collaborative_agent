import json
import pytest
from src.online_rl.tree.batch import batch_slots,update_batch,optimize_case
from src.online_rl.integrated.records import to_dict,sha256_text
from src.online_rl.integrated.prompt_search import PromptStore
from src.online_rl.tree.prompt_search import ScoringDeps
from src.online_rl.tree.posthoc import doctor_feedback,freeze_trajectory_input
from src.online_rl.tree.rollout import build_tree
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.team import team_strategies
from test_tree_v1 import CASES,P0,Log,mock_env,cfg


def test_assignment_twelve_cases_and_short_tail():
    slots=batch_slots(12,500)
    assert len(slots)==12 and [s.worker for s in slots]==[0]*3+[1]*3+[2]*3+[3]*3
    assert [s.lane for s in slots]==[0,1,2]*4
    assert len(batch_slots(24,6))==6
    assert len(batch_slots(498,2))==2


def fixture_tree(case,log,w):
    env=mock_env(log,w,k_conds=('any turn',))
    tree=build_tree(case,max_agent_turns=2,branching=2,run_seed=4,policy_sha256='fixed',protocol_sha256='test',
        agent=env.agent,doctor=env.doctor,stop_rule=env.stop_rule,final_option=env.final_option,outcome_fn=env.outcome_fn,cache=CallCache(None))
    return env,tree


def test_batch_update_once_and_groups_stay_case_local():
    records=[]
    for case in CASES[:3]:
        _,tree=fixture_tree(case,Log(),{'v':0})
        nodes={n.node_id:{'reward':float(n.seed%2)} for n in tree.nodes};groups={}
        for n in tree.nodes:groups.setdefault(n.parent_id or 'root',[]).append(n.node_id)
        records.append({'tree':to_dict(tree),'agent_targets':{'nodes':nodes,'groups':groups}})
    calls=[]
    class Updater:
        def update_groups(self,groups):calls.append(groups);return {'optimizer_step':True}
    assert update_batch(Updater(),records,'fixed',lambda:'fixed')['optimizer_step']
    assert len(calls)==1 and len(calls[0])==9
    assert all(len({r['case_id'] for r in g})==1 for g in calls[0])
    with pytest.raises(RuntimeError):update_batch(Updater(),records,'fixed',lambda:'new')
    assert len(calls)==1


def test_case_optimization_scores_parent_and_candidate_without_policy_update(tmp_path,monkeypatch):
    import src.online_rl.tree.batch as module
    from src.online_rl.integrated.records import PromptVersion
    log=Log();w={'v':0};env,tree=fixture_tree(CASES[0],log,w)
    cache=CallCache(tmp_path/'calls.jsonl');nodes={n.node_id:n for n in tree.nodes}
    inputs=[freeze_trajectory_input(CASES[0],leaf,doctor_feedback(CASES[0],leaf,env.feedback_fn,'fb',cache),nodes) for leaf in tree.leaves]
    store=PromptStore(tmp_path/'prompts');candidate=P0.replace('VARIANT=0','VARIANT=1');csha=store.put(candidate)
    def event(*args,**kwargs):
        return {'versions':[PromptVersion('P1','P0',csha,1,1,0,'accepted','refiner',None,None,(),(),{},None,None)],
                'decision':{'accepted':True,'candidate_sha256':csha}}
    monkeypatch.setattr(module,'run_event',event)
    c=cfg()
    def continuation(case,prefix,turn,text,seed):
        return {'final_option':'A' if '[2]' in text else 'B','outcome':float('[2]' in text),'status':'complete'}
    deps=ScoringDeps(3,4,env.generate_fn,env.gate_fn,env.scorer,env.writer,continuation,'fixed',team_strategies(None,0,1),cache)
    result=optimize_case(case=CASES[0],tree=tree,inputs=inputs,prompt_id='P0',prompt_text=P0,store=store,backend=None,deps=deps,
                         cfg=c,event=1,case_index=0,history=[],out_dir=tmp_path)
    assert result['prompt_id']=='P1' and result['next_prompt_text']==candidate and w['v']==0
    assert len(result['candidate_summary'])==2
    for sha in [sha256_text(P0),csha]:
        record=json.loads((tmp_path/'prompt_evaluations'/f'{sha}.json').read_text())
        assert record['response_scores']
        assert {s['response_node_id'] for s in record['response_scores']}==set(nodes)


def test_frozen_lora_replica_has_same_adapter_hash():
    import torch
    from src.online_rl.integrated.continuation import adapter_sha256
    model=torch.nn.Module();model.register_parameter('lora_A',torch.nn.Parameter(torch.tensor([1.,2.])))
    before=adapter_sha256(model)
    model.requires_grad_(False)
    assert adapter_sha256(model)==before
    model.lora_A.data.add_(1)
    assert adapter_sha256(model)!=before


def test_real_search_keeps_parent_on_tie_and_scores_candidates(tmp_path):
    from test_tree_v1 import search_backend
    from src.online_rl.tree.batch import optimize_case
    log=Log();env=mock_env(log,{'v':0},k_conds=('any turn',));cache=CallCache(tmp_path/'calls.jsonl')
    tree=build_tree(CASES[0],max_agent_turns=5,branching=2,run_seed=4,policy_sha256='fixed',protocol_sha256='test',
        agent=env.agent,doctor=env.doctor,stop_rule=env.stop_rule,final_option=env.final_option,outcome_fn=env.outcome_fn,cache=cache)
    nodes={n.node_id:n for n in tree.nodes}
    inputs=[freeze_trajectory_input(CASES[0],leaf,doctor_feedback(CASES[0],leaf,env.feedback_fn,'fb',cache),nodes) for leaf in tree.leaves]
    def continuation(case,prefix,turn,text,seed):
        return {'final_option':'A' if '[2]' in text else 'B','outcome':float('[2]' in text),'status':'complete'}
    c=cfg();c['prompt_update']['candidate_count']=2;c['prompt_update']['refinement_depth']=1
    deps=ScoringDeps(3,4,env.generate_fn,env.gate_fn,env.scorer,env.writer,continuation,'fixed',team_strategies(None,0,1),cache)
    result=optimize_case(case=CASES[0],tree=tree,inputs=inputs,prompt_id='P0',prompt_text=P0,store=PromptStore(tmp_path/'prompts'),
          backend=search_backend(log),deps=deps,cfg=c,event=1,case_index=0,history=[],out_dir=tmp_path)
    assert not result['prompt_event']['decision']['accepted']
    assert result['prompt_id']=='P0'
    assert len(result['candidate_summary'])>=2
    assert all(p['objective']['J']==1 for p in result['candidate_summary'])
