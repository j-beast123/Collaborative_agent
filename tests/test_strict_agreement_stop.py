from src.collaboration.generator import stop_reason


def test_doctor_end_alone_controls_termination():
    for turn in (2,5):
        assert stop_reason(turn,'I choose B.\nDISCUSSION_STATUS: END','I choose C. I am uncertain.')=='doctor_conclusion'


def test_continue_and_plain_readiness_do_not_stop():
    for text in ['I am ready to conclude.', 'Could you verify?\nDISCUSSION_STATUS: CONTINUE', 'DISCUSSION_STATUS: END\nDISCUSSION_STATUS: CONTINUE', 'The literal marker DISCUSSION_STATUS: END is an example.']:
        assert stop_reason(2,text,'I agree.') is None


def test_end_must_be_final_standalone_line():
    assert stop_reason(2,'No further reply needed.\n\nDISCUSSION_STATUS: END\n','I disagree.')=='doctor_conclusion'
    assert stop_reason(2,'DISCUSSION_STATUS: END\nPlease clarify.','I agree.') is None
    assert stop_reason(2,'Please explain.','DISCUSSION_STATUS: END') is None


def test_rhetorical_question_and_residual_uncertainty_can_end():
    assert stop_reason(2,'Why favor B? Its mechanism addresses the stated finding. Some uncertainty remains, but another reply would not resolve it.\nDISCUSSION_STATUS: END','I choose C.')=='doctor_conclusion'


def test_multiple_status_lines_are_not_an_unambiguous_end():
    assert stop_reason(2,'DISCUSSION_STATUS: CONTINUE\nDISCUSSION_STATUS: END','I agree.') is None


def test_doctor_discussion_prompt_does_not_force_final_turn():
    from src.online_rl.environment import discussion_prompt
    case={'question':'Example','options':{'A':'first','B':'second'}}
    early=discussion_prompt(case,[],1,5);last=discussion_prompt(case,[],5,5)
    assert early==last
    assert 'final allowed' not in last and 'turn 5' not in last
