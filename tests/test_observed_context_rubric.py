import copy
import json
from pathlib import Path

import pytest
import test_source_consistency as S

from src.online_rl.tree import backends as B, posthoc as P
from src.online_rl.tree.agent_state import rubric_problems, SCHEMA
from src.online_rl.tree.batch_prompt import content_key
from src.online_rl.tree.cache import CallCache


def rubric():
    a={'user_side_context':'The physician repeats a request for a different explanation.',
       'agent_side_state':'The Agent previously repeated the same explanation.'}
    return {'rubric_id':'r','name':'Adapt explanation','applicability':a,'required_behavior':'Respond to the repeated request with a different explanation.',
            'scores':{k:{**a,'behavior':v} for k,v in {'0':'Repeat the same explanation again.',
                      '1':'Acknowledge the request but provide little new explanation.',
                      '2':'Address the request with a different, relevant explanation.'}.items()}}


class Client:
    model='fake';protocol_sha256='fake';log=[]
    def __init__(self,applicable=True):self.calls=[];self.applicable=applicable
    def judge(self,stage,state,items,fields,purpose):
        self.calls.append((stage,copy.deepcopy(state),items))
        return ({rid:({'applicable':self.applicable,'evidence_basis':'Visible earlier messages','missing_requirement':''}
                       if stage=='applicability' else {'level':'2','anchor_basis':'Changed explanation'}) for rid in items},None)


def test_visible_history_only_applicability_then_behavior_no_extra_belief_call():
    client=Client();scorer=B.two_stage_structured_scorer(client,observed_context=True)
    view={'case_id':'x','question':'Question','options':{'A':'alpha','B':'beta'},'ground_truth':{'correct_option':'A'}}
    prefix=[{'speaker':'doctor','text':'Explain differently.'},{'speaker':'agent','text':'Same explanation.'},
            {'speaker':'doctor','text':'I asked for a different explanation again.'}]
    target='NEW explanation TARGET_SENTINEL'
    assert scorer(view,prefix,[rubric()],target)['r']['hard']==2
    assert [s for s,_,_ in client.calls]==['applicability','level']
    stage1=json.dumps(client.calls[0]);stage2=json.dumps(client.calls[1])
    assert target not in stage1 and target in stage2
    assert 'Same explanation.' in stage1 and 'agent_side_state' in stage1
    for _, state, _ in client.calls:
        text = json.dumps(state)
        assert 'ground_truth' not in text and 'private_assessment' not in text and 'outcome' not in text
    assert 'required_behavior' not in json.dumps(client.calls[0][2]) and 'score_anchors' not in json.dumps(client.calls[0][2])
    stage1_item = next(iter(client.calls[0][2].values()))
    assert 'Evaluate the supplied Agent-side state condition as written' in stage1_item['instructions']
    assert 'Both User-side context and earlier visible Agent behavior must hold' not in stage1_item['instructions']
    client=Client(False);scorer=B.two_stage_structured_scorer(client,observed_context=True)
    result=scorer(view,prefix,[rubric()],target)['r']
    assert result['hard'] is None and not result['applicable'] and len(client.calls)==1


def test_generator_and_matcher_use_same_mandatory_contract_and_no_trace_input(monkeypatch):
    contract=Path('prompts/sherpa/observed_context_behavior_v1.txt').read_text();calls=[]
    def fake(key,model,system,user,settings,schema,name,role):
        calls.append((system,user,schema))
        r={k:v for k,v in rubric().items() if k!='rubric_id'}
        if role=='rubric_matcher':r={'decision':'CREATE_NEW','rubric_id':'','checks':{k:False for k in B.MATCH_CHECKS},'rationale':'different history'}
        return json.dumps(r),{}
    monkeypatch.setattr(B,'_openrouter_json',fake)
    gen=B.api_rubric_generator('fake','fake',context_addendum=contract)
    assert not rubric_problems(gen('old flat P0 prompt',S.tin())['rubric'])
    assert calls[0][0].endswith(contract) and calls[0][2]==B.OBSERVED_RUBRIC_JSON_SCHEMA
    assert 'Conversation before the locus reply:' in calls[0][1] and 'private_assessment' not in calls[0][1]
    matcher=B.api_rubric_matcher('fake','fake',{},contract);matcher('P1 package','Visible conversation')
    assert calls[1][0].endswith(contract) and calls[1][2]==B.MATCH_JSON_SCHEMA
    assert gen.settings['context_contract_sha256']==matcher.settings['context_contract_sha256']


def test_source_repair_changes_contexts_but_preserves_each_behavior():
    original=rubric();new=copy.deepcopy(original)
    new['applicability']={k:'Repaired '+v for k,v in new['applicability'].items()}
    new['name']='malicious changed construct'
    for k in new['scores']:new['scores'][k]['behavior']='wrong changed behavior'
    def generate(*args):return {'rubric':original,'raw':'original'}
    generate.repair=lambda *args:{'rubric':new,'raw':'repaired'}
    decisions=iter([False,True])
    def stage1(r):return {'applicable':next(decisions),'stage1':{},'node_id':'n'}
    result=P.generate_rubric('P','p'*64,S.tin(),generate,CallCache(None),lambda *args:{'passed':True},stage1)
    assert result.valid and result.source_check['status']=='REPAIRED_PASS'
    assert result.rubric['name']==original['name'] and not rubric_problems(result.rubric)
    for k in '012':
        assert result.rubric['scores'][k]['behavior']==original['scores'][k]['behavior']
        assert result.rubric['scores'][k]['agent_side_state']==new['applicability']['agent_side_state']


def test_context_changes_prevent_exact_dedup_and_invalid_levels_are_rejected():
    r=rubric();other=copy.deepcopy(r);other['applicability']['agent_side_state']='The Agent has not replied yet.'
    for k in '012':other['scores'][k]['agent_side_state']=other['applicability']['agent_side_state']
    assert content_key(r)!=content_key(other)
    invalid=copy.deepcopy(r);invalid['scores']['0']['agent_side_state']='different state'
    assert rubric_problems(invalid)


def test_guided_reply_uses_observed_schema_without_private_assessment():
    class Policy:
        def agent_user_prompt(self,*args):return 'visible prefix'
        def agent(self,user,seed):self.user=user;return 'guided reply'
    policy=Policy();writer=B.PolicyInterventions(policy,'Score 0 {rubric_json}','Score 2 {rubric_json}')
    r=rubric();r['analysis']={'internal_note':'must stay hidden'}
    writer.write({},[],2,r,'2',42)
    assert 'agent_side_state' in policy.user and 'private_assessment' not in policy.user
    assert 'internal_note' not in policy.user and 'analysis' not in policy.user
    assert '"2": {"behavior": "Address the request with a different, relevant explanation."}' in policy.user
    assert '"0":' not in policy.user and '"1":' not in policy.user and 'required_behavior' not in policy.user
