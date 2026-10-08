import pytest
from src.online_rl.tree.backends import ApiDoctor
from src.collaboration.generator import DoctorProtocolError
CASE={'question':'Choose','options':{'A':'one','B':'two'}}


def test_final_repairs_discussion_without_gold_or_new_turn():
    outputs=iter(['Can you verify?\nDISCUSSION_STATUS: CONTINUE','I choose B. Final answer: option B - two'])
    calls=[]
    def call(key,model,system,user,tokens):calls.append(user);return next(outputs)
    result=ApiDoctor('','local','SYS',512,call=call).final(CASE,[{'speaker':'agent','text':'Evidence for B'}])
    assert 'Final answer: option B' in result and len(calls)==2
    assert 'Evidence for B' in calls[1] and 'Previous output' in calls[1]
    assert all('ground_truth' not in c for c in calls)


def test_unrepaired_final_is_technical_incomplete():
    with pytest.raises(DoctorProtocolError):
        ApiDoctor('','local','SYS',512,call=lambda *a:'Discuss.\nDISCUSSION_STATUS: CONTINUE').final(CASE,[])


def test_incomplete_continuation_recovery_preserves_agent_reply_cache():
    from tests.test_tree_selected_protocol import source, CASE
    from src.online_rl.tree.effectiveness import run_interventions
    from src.online_rl.tree.cache import CallCache
    n,leaf,_,rubric=source();cache=CallCache(None);writes=[];finals=[]
    def writer(case,prefix,turn,rubric,level,seed):writes.append(seed);return 'Fresh intervention'
    def continuation(case,prefix,turn,text,seed):
        finals.append((text,seed))
        return {'status':'incomplete','final_option':'','outcome':None} if len(finals)==1 else {'status':'complete','final_option':'B','outcome':0.0}
    args=dict(reruns=3,run_seed=0,policy_sha256=n.policy_sha256,writer=writer,continuation=continuation,cache=cache)
    runs=run_interventions(CASE,leaf,n,rubric,**args)
    assert len(writes)==3 and len(finals)==4 and finals[0]==finals[1]
    assert all(r.status=='complete' for r in runs)
    run_interventions(CASE,leaf,n,rubric,**args)
    assert len(writes)==3 and len(finals)==4
    assert any(r['kind']=='rerun' and r['value']['status']=='incomplete' for r in cache.store.values())
