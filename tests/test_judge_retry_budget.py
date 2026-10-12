from types import SimpleNamespace
import time
from src.online_rl.tree import compact_runtime as cr
from src.online_rl.tree.experiment import load_config


def test_all_online_arms_use_three_total_attempts():
    for name in ('online','online_doctor_replicas2','online_doctor_v2','online_rubric_only'):
        assert load_config(f'configs/{name}.json')['judges']['execution_retry_attempts']==3


def test_invalid_behavior_is_excluded_after_three_attempts(monkeypatch):
    calls=[]
    def chat(*args,**kwargs):
        calls.append(1)
        return {'choices':[{'message':{'content':'{}'}}]}
    client=SimpleNamespace(model='fake',max_tokens=100,system='system',_chat=chat)
    monkeypatch.setattr(cr,'upgrade',lambda payload,**kw:{**payload,'response_format':{'json_schema':{'schema':{}}}})
    monkeypatch.setattr(cr,'validate_response',lambda *a:{'status':'execution_error','kind':'evidence_contract','error':'C cannot be complete without M'})
    monkeypatch.setattr(cr,'condition_state',lambda *a:{})
    monkeypatch.setattr(cr,'public_behavior_items',lambda rubrics,*a:rubrics)
    monkeypatch.setattr(cr,'public_client_identity',lambda client:{'model':client.model})
    monkeypatch.setattr(time,'sleep',lambda *a:None)
    scorer=cr.compact_scorer(client,client,revision='v7_3',execution_retry_attempts=3)
    result=scorer.score_behavior_batch({},[],[{'rubric_id':'r'}],[{'response_id':'n','reply':'reply'}])['n']['r']
    assert len(calls)==3
    assert result['scoring_status']=='execution_unresolved'
    assert result['hard'] is None and result['probabilities'] is None
    assert result['total_attempts']==3 and len(result['reasons_by_attempt'])==3


def test_contract_repair_tells_judge_error_without_changing_inputs_or_budget(monkeypatch):
    import copy
    calls=[]
    def chat(messages,*args,**kwargs):
        calls.append(copy.deepcopy(messages))
        return {'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
    client=SimpleNamespace(model='fake',max_tokens=100,system='system',_chat=chat)
    monkeypatch.setattr(cr,'upgrade',lambda payload,**kw:{**payload,'response_format':{'json_schema':{'schema':{}}}})
    monkeypatch.setattr(cr,'validate_response',lambda *a:{'status':'execution_error','kind':'evidence_contract','error':'C cannot be complete without M'})
    monkeypatch.setattr(cr,'condition_state',lambda *a:{})
    monkeypatch.setattr(cr,'public_behavior_items',lambda rubrics,*a:rubrics)
    monkeypatch.setattr(cr,'public_client_identity',lambda client:{'model':client.model})
    monkeypatch.setattr(time,'sleep',lambda *a:None)
    scorer=cr.compact_scorer(client,client,revision='v7_3',execution_retry_attempts=3,execution_contract_repair=True)
    result=scorer.score_behavior_batch({},[],[{'rubric_id':'r'}],[{'response_id':'n','reply':'reply'}])['n']['r']
    assert len(calls)==3 and len(calls[0])==2 and len(calls[1])==3
    assert 'C cannot be complete without M' in calls[1][-1]['content']
    assert all(messages[:2]==calls[0] for messages in calls)
    assert result['hard'] is None and result['scoring_status']=='execution_unresolved'
