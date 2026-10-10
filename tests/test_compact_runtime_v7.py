"""Synthetic CPU-only wiring checks, not semantic or clinical validation."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import pytest
from scripts.online_rl.run_tree import make_scorer, make_gate, evidence_gate_identity
from src.online_rl.tree.experiment import load_config
from src.online_rl.tree.luna import LunaDecisionClient
from src.online_rl.tree.applicability import state_applicability
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.scoring import score_nodes
from src.common.config import IntegrationUnavailable
from tests.test_scoring_batch import CASE, KEPT, RUBRIC, node

CFG = 'configs/experiments/train120_v7_fresh.json'

class Transport:
    def __init__(self, mode='normal'):
        self.calls=[]; self.mode=mode
    def __call__(self,url,payload,key,timeout):
        self.calls.append(copy.deepcopy(payload)); p=json.loads(payload['messages'][1]['content'])
        assert 'correct_option' not in json.dumps(p)
        if 'responses' not in p:
            assert 'target_agent_reply' not in json.dumps(p)
            assert 'minimum_performance' not in json.dumps(p['items'])
            ans={}
            for rid,item in p['items'].items():
                prefix=p['state']['dialogue_before_target_reply'];t=prefix[0]
                ans[rid]={
                  'doctor_condition':[{'clause':item['doctor_condition'],'status':'supported','evidence':[{'turn_id':t['turn_id'],'quote':t['text']}],'inspected':[],'note':'Synthetic support.'}],
                  'agent_condition':[{'clause':item['agent_condition'],'status':'not_required','evidence':[],'inspected':[],'note':'No extra state required.'}]}
        else:
            ans={}
            for response in p['responses']:
                reply=response['reply'];level=reply[-1] if reply[-1] in '012' else '2'
                ans[response['response_id']]={}
                for rid,r in p['rubric_items'].items():
                    refs=['score_anchors.0','score_anchors.1','score_anchors.2']
                    checks=[{'ref':ref,'met':str(i)==level,'cites':[0] if str(i)==level and level!='0' else [],'why':'Synthetic match.'} for i,ref in enumerate(refs)]
                    ans[response['response_id']][rid]={'level':level,'quotes':[reply],'checks':checks,'gap':'' if level=='2' else 'Written completion missing.','defect':''}
        return {'choices':[{'finish_reason':'length' if self.mode=='truncated' else 'stop','message':{'content':json.dumps({'answers':ans})}}], 'usage':{}}

def setup(mode='normal'):
    cfg=load_config(CFG); t=Transport(mode)
    client=LunaDecisionClient(model=cfg['judges']['scorer_model'],key='OFFLINE',system=Path(cfg['judges']['scorer_prompt_file']).read_text(),max_tokens=4096,post=t)
    return cfg,t,client,make_scorer(cfg,client)

def test_sibling_alias_shared_app_and_actual_messages():
    _,t,_,s=setup();cache=CallCache(None)
    a=state_applicability(CASE,node(0),RUBRIC,s,cache)
    b=state_applicability(CASE,node(1),{**RUBRIC,'rubric_id':'bank_alias'},s,cache)
    assert a==b and len(t.calls)==1
    assert 'Target reply' not in json.dumps(t.calls[0])
    c=state_applicability({**CASE,'question':'Different question'},node(1),RUBRIC,s,cache)
    assert len(t.calls)==2 and a['provenance']!=c['provenance']

def test_native_scoring_keeps_zero_and_sibling_vector():
    _,t,_,s=setup();nodes=[replace(node(i),response=f'Observed response {v}') for i,v in enumerate([0,1,2,2])]
    rows=score_nodes(CASE,nodes,[KEPT],s,CallCache(None))
    assert [r.score for r in rows]==[0,0.5,1,1]
    assert sum('responses' not in json.loads(x['messages'][1]['content']) for x in t.calls)==1
    for x in t.calls:
        kind='behavior' if 'responses' in json.loads(x['messages'][1]['content']) else 'applicability'
        assert Path(f'prompts/sherpa/compact_{kind}_semantics_v2.txt').read_text() in x['messages'][0]['content']
        assert 'intervention_instruction' not in x['messages'][1]['content']
    from src.online_rl.integrated.metrics import advantages
    rewards=[0.5*r.score+0.5*o for r,o in zip(rows,[0,1,0,1])]
    assert rewards==[0,0.75,0.5,1.0]
    assert advantages(rewards, 'mean_center')==[-0.5625,0.1875,-0.0625,0.4375]

def test_truncation_is_error_never_score_or_na():
    _,_,_,s=setup('truncated')
    with pytest.raises(IntegrationUnavailable) as e:s.review_state(CASE,[{'speaker':'doctor','text':'Please compare.'}],[RUBRIC])
    assert e.value.audit['validation']['status']=='execution_incomplete'

def test_gate_uses_pinned_prompt_and_blind_compact_scorer():
    cfg,_,client,_=setup();client.system=Path(cfg['judges']['gate_prompt_file']).read_text()
    gate=make_gate(cfg,client,[CASE])
    assert gate.settings['blind_source_protocol']
    assert gate.settings['contract_sha256']!=make_gate({**cfg,'judges':{**cfg['judges'],'compact_runtime':None}},client,[CASE]).settings['contract_sha256']
    assert evidence_gate_identity(cfg)['contract_sha256']==gate.settings['contract_sha256']
