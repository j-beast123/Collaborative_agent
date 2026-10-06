from src.collaboration.generator import repeated_agent_response, stop_reason
from src.online_rl.tree.rollout import build_tree
from src.online_rl.tree.cache import CallCache

TEXT='I favor option C because the provided evidence supports this choice without requiring additional assumptions.'


def test_repeat_ignores_punctuation_and_brief_agreement():
    history=[{'speaker':'agent','text':TEXT},{'speaker':'doctor','text':'A new question'}]
    assert repeated_agent_response(history,'Yes, '+TEXT.upper())
    assert stop_reason(2,'A new question',TEXT,history=history)=='repeated_agent_response'
    assert not repeated_agent_response([],TEXT)
    assert not repeated_agent_response(history,TEXT+' However, the new test result favors option A.')
    assert not repeated_agent_response(history,TEXT.replace('supports','does not support'))
    assert not repeated_agent_response(history,'I favor option C because a different mechanism explains the new evidence.')
    assert not repeated_agent_response([{'speaker':'agent','text':'I agree.'}],'I agree.')


def test_repeat_stops_even_if_doctor_requested_continuation():
    class Agent:
        def sample(self,*args):return {'completion_text':TEXT}
    class Doctor:
        def turn(self,*args):return 'Discuss the evidence.'
        def final(self,*args):return 'Final answer: option C'
    tree=build_tree({'case_id':'case'},max_agent_turns=3,branching=2,run_seed=0,
        policy_sha256='policy',protocol_sha256='protocol',agent=Agent(),doctor=Doctor(),
        stop_rule=lambda *args:None,final_option=lambda text:'C',outcome_fn=lambda *args:1.0,cache=CallCache(None))
    assert len(tree.nodes)==6 and len(tree.leaves)==4
    assert all(len(leaf.path)==2 and leaf.stop_reason=='repeated_agent_response' for leaf in tree.leaves)
    assert all(leaf.dialogue[-1][1]=='FINAL' for leaf in tree.leaves)
