from src.collaboration.generator import stop_reason


def text(choice):
    return f'I choose option {choice}. The evidence resolves the issue. No remaining uncertainty. I am ready to conclude.'


def test_shared_choice_can_stop_only_from_second_turn():
    assert stop_reason(1,text('C'),text('C')) is None
    assert stop_reason(2,text('C'),text('C'))=='mutual_agreement'


def test_different_choices_and_unilateral_final_do_not_stop():
    assert stop_reason(2,text('B'),text('C')) is None
    assert stop_reason(2,'Please compare the evidence.','Final answer: option C') is None
    assert stop_reason(2,'I am ready to conclude.','I am ready to conclude.') is None


def test_uncertainty_question_and_negated_readiness_prevent_stop():
    for suffix in ['However, I am uncertain about the definition.','Could you confirm the assumption?','I am not ready to conclude.','Unless the terminology means something different.']:
        assert stop_reason(2,text('C'),text('C')+' '+suffix) is None


def test_mentioned_alternative_is_not_a_shared_recommendation():
    assert stop_reason(2,text('C'),'I choose option B. Unlike option C, B has stronger support. I am ready to conclude.') is None
