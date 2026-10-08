import json
from types import SimpleNamespace
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree import belief
from src.collaboration.generator import RECONSTRUCTION_FIELD_ORDER


def policy_and_rec():
    strings={'focal_claim','clinician_current_position','agent_current_position'}
    rec={key:'claim' if key in strings else [] for key in RECONSTRUCTION_FIELD_ORDER}
    rec['parse_status']='valid'
    calls=[]
    def agent(prompt,seed,**kwargs):
        calls.append((prompt,seed,kwargs))
        return json.dumps({'q1':{'answer':'aligned','reason':'Same prior position'},'q2':{'answer':'N','reason':'Unresolved support'}})
    cfg={'belief':{'reconstructor_model':'mock','max_tokens':100,'judgment_max_tokens':100}}
    policy=SimpleNamespace(cfg=cfg,key='test',cache=CallCache(None),args=SimpleNamespace(rounds=5),
                           policy_sha=lambda:'fixed',agent=agent,judgment_system='judge')
    return policy,rec,calls

CASE={'case_id':'belief-test','question':'question','options':{'A':'first','B':'second'},'ground_truth':{'correct_option':'A'}}


def test_initial_turn_does_not_invent_agent_belief(monkeypatch):
    policy,rec,calls=policy_and_rec()
    monkeypatch.setattr(belief,'reconstruct_call',lambda *args:dict(rec))
    _,packet=belief.prepare(policy,CASE,[{'speaker':'doctor','text':'claim'}],1)
    assert packet['judgment'] is None and not calls
    assert packet['state']['status']=='elicited_initial_turn'
    assert packet['state']['agent_side_state']['agent_current_position']==''


def test_greedy_prefix_measurement_is_shared_and_blind(monkeypatch):
    policy,rec,calls=policy_and_rec(); reconstruct_calls=[]
    def reconstruct(*args):
        reconstruct_calls.append(args)
        assert 'ground_truth' not in args[2]
        return dict(rec)
    monkeypatch.setattr(belief,'reconstruct_call',reconstruct)
    prefix=[{'speaker':'doctor','text':'claim'},{'speaker':'agent','text':'my position'},{'speaker':'doctor','text':'support?'}]
    prompt,packet=belief.prepare(policy,CASE,prefix,2)
    assert belief.prepare(policy,CASE,prefix,2)==(prompt,packet)
    assert len(calls)==len(reconstruct_calls)==1
    assert calls[0][1]==0 and calls[0][2]['greedy'] is True
    assert 'ground_truth' not in calls[0][0]
    assert packet['state']['agent_side_state']['better_supported_position']=='aligned'
    assert packet['state']['agent_side_state']['evidence_sufficient'] is False
    assert 'do not present the focal decision as settled' in prompt
