from dataclasses import replace
from tests.test_scoring_batch import CASE, KEPT, node
from src.online_rl.tree.scoring import score_nodes
from src.online_rl.tree.cache import CallCache


def test_four_siblings_share_matching_and_review_before_scoring():
    n,r=node(0),KEPT;nodes=[replace(n,node_id=f'synthetic/{i}',response=f'reply {i}') for i in range(4)]
    calls=[]
    def scorer(*args):raise AssertionError('combined scoring must not run')
    def match(view,prefix,rubrics):
        assert 'ground_truth' not in view and 'target_turn' not in view
        calls.append('match');return {r.rubric_id:{'value':'Y'}}
    def review(view,prefix,rubrics):
        calls.append('review');return {r.rubric_id:{'applicable':True}}
    def behavior(view,prefix,rubrics,reply,matches):
        calls.append(reply);return {r.rubric_id:{'applicable':True,'hard':2,'probabilities':{'0':0,'1':0,'2':1}}}
    scorer.match_state=match;scorer.review_state=review;scorer.score_behavior=behavior
    scorer.judge_identity={'model':'luna'};scorer.review_identity={'prompt':'v1'}
    cache=CallCache(None)
    result=score_nodes(CASE,nodes,[r],scorer,cache)
    assert calls==['match','review','reply 0','reply 1','reply 2','reply 3']
    assert len(result)==4 and all(s.score==1 for s in result)
    score_nodes(CASE,nodes,[r],scorer,cache)
    assert len(calls)==6
    # Hidden belief still partitions behavior/masking, but cannot change the
    # applicability decision for identical visible judge input.
    changed=replace(nodes[0],trace={'pre_response_belief':{'state':{'belief':'different'}}})
    score_nodes(CASE,[changed],[r],scorer,cache)
    assert calls[-2:]==['match','reply 0'] and calls.count('review') == 1


def test_review_na_excludes_behavior_score():
    n,r=node(0),KEPT
    from src.online_rl.tree.backends import two_stage_structured_scorer
    from src.online_rl.tree.luna import LunaDecisionClient
    import json
    calls=[]
    def post(url,payload,key,timeout):
        data=json.loads(payload['messages'][1]['content']);calls.append(data)
        assert data['stage']=='applicability'
        assert 'target_agent_reply' not in data['state']
        answer={'answers':{rid:{'applicable':False,'evidence_basis':'Resolved before response',
                                'missing_requirement':'No pending request'} for rid in data['items']}}
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(answer)}}]}
    judge=LunaDecisionClient(key='test',model='openai/gpt-6-luna',system='Score.',post=post)
    result=score_nodes(CASE,[n],[r],two_stage_structured_scorer(judge),CallCache(None))
    assert len(calls)==1 and result[0].score is None and not result[0].match
    assert 'target_response' not in str(calls)


def test_ready_state_is_scored_before_reviewing_the_next_state():
    n,r=node(0),KEPT;calls=[]
    nodes=[n,replace(n,node_id='synthetic/1',prefix=(('doctor','D1',1,'A different request'),))]
    def scorer(*args):raise AssertionError()
    def match(*args):calls.append('match');return {r.rubric_id:{'value':'Y'}}
    def review(*args):calls.append('review');return {r.rubric_id:{'applicable':True}}
    def behavior(*args):
        calls.append('behavior');return {r.rubric_id:{'applicable':True,'hard':1,'probabilities':{'0':0,'1':1,'2':0}}}
    scorer.match_state=match;scorer.review_state=review;scorer.score_behavior=behavior
    score_nodes(CASE,nodes,[r],scorer,CallCache(None))
    assert calls==['match','review','behavior','match','review','behavior']
