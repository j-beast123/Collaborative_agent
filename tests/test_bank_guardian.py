import copy
import pytest

from src.online_rl.tree.bank_guardian import CHECKS, decide, validate_response, request


def rubric(name='r', suffix=''):
    app={'user_side_context':'Doctor asks for a comparison'+suffix,
         'agent_side_state':'No additional recorded Agent state is required.'}
    return {'rubric_id':name,'name':name,'applicability':app,
            'required_behavior':'Compare the two stated alternatives'+suffix,
            'minimum_performance':'State one concrete difference'+suffix,
            'completion_condition':'Connect that difference to the decision'+suffix,
            'criterion_version':'public_behavior_app_alignment_v5',
            'scores':{str(i):{**app,'behavior':f'level {i}{suffix}'} for i in range(3)}}


def row(cid, suffix=''):
    return {'candidate_id':cid,'rubric':rubric(cid,suffix)}


def test_exact_duplicates_never_call_model():
    existing=[{'representative_id':'bank-a','rubric':rubric('old')}]
    result=decide(existing,[row('new')],lambda _: (_ for _ in ()).throw(AssertionError()))
    assert result['aliases']=={'new':'bank-a'} and result['accepted']==[]


def test_semantic_same_requires_every_check_and_earlier_representative():
    existing=[{'representative_id':'bank-a','rubric':rubric('old')}]
    candidate=row('new',' wording')
    def model(payload):
        return {'decisions':[{'candidate_id':'new','decision':'SAME_MEANING','representative_id':'bank-a',
            'checks':{key:True for key in CHECKS},'reason':'All public behavior fields are equivalent.'}]}
    result=decide(existing,[candidate],model)
    assert result['aliases']=={'new':'bank-a'} and result['accepted']==[]


@pytest.mark.parametrize('decision,representative,checks',[
    ('SAME_MEANING','missing',{key:True for key in CHECKS}),
    ('SAME_MEANING','bank-a',{**{key:True for key in CHECKS},'same_score_boundaries':False}),
    ('DISTINCT','bank-a',{key:False for key in CHECKS}),
])
def test_invalid_or_unsafe_merges_fail_closed(decision,representative,checks):
    payload=request([{'representative_id':'bank-a','rubric':rubric('old')}],[row('new',' x')])
    value={'decisions':[{'candidate_id':'new','decision':decision,'representative_id':representative,
                        'checks':checks,'reason':'test reason'}]}
    with pytest.raises(ValueError): validate_response(value,payload)


def test_uncertain_and_distinct_are_kept_separate():
    candidates=[row('a',' a'),row('b',' b')]
    def model(payload):
        return {'decisions':[
            {'candidate_id':'a','decision':'UNCERTAIN','representative_id':None,
             'checks':{key:False for key in CHECKS},'reason':'Insufficient evidence.'},
            {'candidate_id':'b','decision':'DISTINCT','representative_id':None,
             'checks':{key:False for key in CHECKS},'reason':'Completion differs.'}]}
    result=decide([],candidates,model)
    assert result['accepted']==['a','b'] and result['aliases']=={}
