from src.online_rl.tree.rollout import build_tree
from src.online_rl.tree.cache import CallCache
from src.collaboration.generator import stop_reason


def test_doctor_declaration_skips_next_agent_response():
    calls=[]
    class Agent:
        def sample(self,case,prefix,turn,seed):
            calls.append(turn);return {'completion_text':'I favor A.'}
    class Doctor:
        def turn(self,case,dialogue,r,rounds):
            return 'Please explain.' if r==1 else 'I choose B.\nDISCUSSION_STATUS: END'
        def final(self,case,dialogue):
            assert dialogue[-1]['speaker']=='doctor'
            return 'Final answer: option B'
    tree=build_tree({'case_id':'test'},max_agent_turns=5,branching=1,run_seed=0,policy_sha256='p',protocol_sha256='x',agent=Agent(),doctor=Doctor(),stop_rule=stop_reason,final_option=lambda t:'B',outcome_fn=lambda c,o:1.,cache=CallCache(None))
    assert calls==[1]
    assert len(tree.leaves[0].path)==1
    assert tree.leaves[0].stop_reason=='doctor_conclusion'


def test_opening_end_cannot_skip_review_of_new_agent_alternative():
    calls=[]
    class Agent:
        def sample(self,case,prefix,turn,seed):
            calls.append(('agent',turn))
            return {'completion_text':f'Consider B instead of A, using comparison {turn}.'}
    class Doctor:
        def turn(self,case,dialogue,r,rounds):
            calls.append(('doctor',r))
            if r==1:return 'I choose A.\nDISCUSSION_STATUS: END'
            if r==2:
                assert dialogue[-1]['speaker']=='agent' and 'B instead' in dialogue[-1]['text']
                return 'The new alternative needs a comparison.\nDISCUSSION_STATUS: CONTINUE'
            return 'I have compared the new alternative and choose B.\nDISCUSSION_STATUS: END'
        def final(self,case,dialogue):return 'Final answer: option B'
    tree=build_tree({'case_id':'test'},max_agent_turns=5,branching=1,run_seed=0,
        policy_sha256='p',protocol_sha256='v17',agent=Agent(),doctor=Doctor(),stop_rule=stop_reason,
        final_option=lambda t:'B',outcome_fn=lambda c,o:1.,cache=CallCache(None))
    assert calls==[('doctor',1),('agent',1),('doctor',2),('agent',2),('doctor',3)]
    assert len(tree.leaves[0].path)==2 and tree.leaves[0].stop_reason=='doctor_conclusion'
