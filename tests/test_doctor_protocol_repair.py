import pytest
from src.collaboration.generator import checked_doctor_turn, DoctorProtocolError


def scripted(outputs):
    calls = []
    answers = iter(outputs)
    def call(system, prompt):
        calls.append((system, prompt))
        return next(answers)
    return call, calls


import json


def decision(need=False, answered=False, resolvable=True, evidence=None, unavailable=False):
    return json.dumps({'needs_reply':need,'unresolved_issue':'Compare alternatives' if need else None,
        'why_reply_could_change_choice':'May favor B' if need else None,
        'issue_already_answered':answered,'resolvable_from_available_context':resolvable,
        'requires_unavailable_information':unavailable,'new_evidence':evidence,'end_reason':None if need else 'No further resolvable issue'})


def test_status_is_derived_from_decision_not_speech():
    call,calls=scripted([decision(),json.dumps({'assessment':'I choose B; further discussion cannot resolve the limitation.','question':None})])
    assert checked_doctor_turn(call,'SYS','PREFIX').endswith('DISCUSSION_STATUS: END')
    assert len(calls)==2 and 'Fixed Doctor decision' in calls[1][1]


@pytest.mark.parametrize('answered,resolvable',[(True,True),(False,False)])
def test_answered_or_unresolvable_issue_ends(answered,resolvable):
    call,_=scripted([decision(True,answered,resolvable),json.dumps({'assessment':'I choose B with the remaining limitation.','question':None})])
    assert checked_doctor_turn(call,'SYS','PREFIX').endswith('END')


def test_new_evidence_can_reopen_answered_issue():
    call,_=scripted([decision(True,True,True,'New contradictory statement'),json.dumps({'assessment':'The new premise changes the comparison.','question':'Could you reconcile the new premise?'})])
    assert checked_doctor_turn(call,'SYS','PREFIX').endswith('CONTINUE')


def test_invalid_decision_is_technical_error_after_bounded_retry():
    call,calls=scripted(['invalid','invalid'])
    with pytest.raises(DoctorProtocolError):checked_doctor_turn(call,'SYS','PREFIX')
    assert len(calls)==2


def test_end_question_is_repaired_without_changing_decision():
    call,_=scripted([decision(),json.dumps({'assessment':'I choose B.','question':'Please check?'}),json.dumps({'assessment':'I choose B.','question':None})])
    assert checked_doctor_turn(call,'SYS','PREFIX').endswith('END')


def test_structured_end_prevents_unnecessary_sibling_expansion(tmp_path):
    from src.online_rl.tree.backends import ApiDoctor
    from src.online_rl.tree.rollout import build_tree
    from src.online_rl.tree.cache import CallCache
    from src.collaboration.generator import stop_reason
    def completion(key,model,system,prompt,max_tokens):
        if 'STRUCTURED DISCUSSION DECISION' in system:
            return decision('agent:' not in prompt.lower())
        if 'VERBALIZE FIXED DECISION' in system:
            needs='"effective_needs_reply": true' in prompt
            return json.dumps({'assessment':'I favor A based on the supplied premise.', 'question':'Could you compare A and B?' if needs else None})
        return 'Final answer: option A'
    class Agent:
        def sample(self,*args):return {'completion_text':'A follows from the supplied premise.'}
    tree=build_tree({'case_id':'case','question':'Q','options':{'A':'one','B':'two'}},
        max_agent_turns=3,branching=4,run_seed=0,policy_sha256='p',protocol_sha256='v15',
        agent=Agent(),doctor=ApiDoctor('','local','SYS',512,call=completion),stop_rule=stop_reason,
        final_option=lambda t:'A',outcome_fn=lambda *a:1.,cache=CallCache(tmp_path/'calls.jsonl'))
    assert len(tree.nodes)==4 and len(tree.leaves)==4
    assert all(leaf.stop_reason=='doctor_conclusion' for leaf in tree.leaves)


def test_protocol_error_audit_does_not_cache_outcome(tmp_path):
    import json
    from src.online_rl.tree.cache import CallCache
    cache=CallCache(tmp_path/'calls.jsonl')
    def fail():raise DoctorProtocolError('doctor_protocol_error: unresolved')
    with pytest.raises(DoctorProtocolError):cache.call('doctor_turn',{'path':'0'},fail)
    assert not cache.store
    error=json.loads((tmp_path/'calls_protocol_errors.jsonl').read_text())
    assert error['status']=='technical_incomplete' and 'outcome' not in error


def test_missing_case_information_cannot_extend_discussion():
    call,_=scripted([decision(True,resolvable=True,unavailable=True),json.dumps({'assessment':'The missing duration cannot be obtained here; I choose the best-supported option.','question':None})])
    assert checked_doctor_turn(call,'SYS','PREFIX').endswith('END')
