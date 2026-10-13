import pytest
from tests.test_tree_selected_protocol import source
from src.online_rl.tree.reuse_bank import packet,validate_reuse


def test_reuse_is_posthoc_but_does_not_expose_gold_or_old_analysis():
    _,_,tin,r=source();state=packet(tin,[r])
    assert 'ground_truth' not in state and 'full_dialogue' in state
    assert set(state['candidates'][0])=={'rubric_id','candidate_rubric'}
    result={'eligible':True,'reuse':True,'rubric_id':r.rubric_id,'agent_turn_id':tin.agent_node_ids[0],
            'doctor_quote':'Please explain','agent_quote':'Original explanation','reason':'Same required explanation'}
    assert validate_reuse(result,state)==result
    with pytest.raises(ValueError):validate_reuse({**result,'agent_turn_id':'another-branch'},state)
    with pytest.raises(ValueError):validate_reuse({**result,'doctor_quote':'Final answer: A'},state)


def test_unknown_bank_rubric_rejected():
    _,_,tin,_=source();state=packet(tin,[])
    with pytest.raises(ValueError):validate_reuse({'eligible':True,'reuse':True,'rubric_id':'foreign','reason':'same'},state)
