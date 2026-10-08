import pytest
from src.judges.applicability_review import review_state,validate_review


def test_review_payload_has_no_target_or_future_fields():
    prefix=[{'speaker':'doctor','text':'Please compare the claims.','outcome':1,'target_response':'hidden'}]
    state=review_state(prefix,{'user_side_context':'A comparison requested','agent_side_state':'No additional requirement','scores':{'2':'hidden'}})
    assert set(state)=={'applicability','prefix'}
    assert set(state['prefix'][0])=={'turn_index','speaker','text'}
    assert 'scores' not in state['applicability']


def test_exact_evidence_is_required():
    state=review_state([{'speaker':'doctor','text':'Please compare the claims.'}],{'user_side_context':'Comparison','agent_side_state':'None'})
    obj={'applicable':True,'reason':'A live request','evidence':[{'turn_index':0,'speaker':'doctor','quote':'compare the claims','supports':'user_side_context'}]}
    assert validate_review(obj,state)==obj
    obj['evidence'][0]['quote']='invented'
    with pytest.raises(ValueError,match='quote'):validate_review(obj,state)
    assert validate_review({'applicable':False,'evidence':[],'reason':'Resolved'},state)['applicable'] is False
