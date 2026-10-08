from dataclasses import replace
from tests.test_tree_selected_protocol import CASE, source
from src.online_rl.tree.scoring import score_nodes
from src.online_rl.tree.cache import CallCache


def test_four_siblings_share_matching_and_review_before_scoring():
    n,_,_,r=source();nodes=[replace(n,node_id=f'synthetic/{i}',response=f'reply {i}') for i in range(4)]
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
    # Different recorded belief is a different state, even for the same prefix.
    changed=replace(nodes[0],trace={'pre_response_belief':{'state':{'belief':'different'}}})
    score_nodes(CASE,[changed],[r],scorer,cache)
    assert calls[-3:]==['match','review','reply 0']


def test_review_na_excludes_behavior_score():
    n,_,_,r=source()
    from src.online_rl.tree.backends import two_stage_scorer
    from src.judges.openrouter_choice import OpenRouterChoiceJudge
    import json
    import tempfile
    calls=[]
    def post(url,payload,key,timeout):
        data=json.loads(payload['messages'][1]['content']);calls.append(data)
        if 'conditions' in data:
            answer={'reviews':{rid:{'applicable':False,'reason':'Resolved before response','evidence':[]} for rid in data['conditions']}}
        else:
            assert 'target_turn' not in data['state']
            answer={'answers':{rid:'Y' for rid in data['questions']}}
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(answer)}}]}
    with tempfile.TemporaryDirectory() as d:
        judge=OpenRouterChoiceJudge('test',model='openai/gpt-6-luna',post=post,audit_path=d+'/calls.jsonl')
        result=score_nodes(CASE,[n],[r],two_stage_scorer(judge),CallCache(None))
    assert len(calls)==2 and result[0].score is None and not result[0].match
    assert 'target_response' not in str(calls)


def test_all_state_reviews_finish_before_any_behavior_call():
    n,_,_,r=source();calls=[]
    nodes=[n,replace(n,node_id='synthetic/1',prefix=(('doctor','D1',1,'A different request'),))]
    def scorer(*args):raise AssertionError()
    def match(*args):calls.append('match');return {r.rubric_id:{'value':'Y'}}
    def review(*args):calls.append('review');return {r.rubric_id:{'applicable':True}}
    def behavior(*args):
        calls.append('behavior');return {r.rubric_id:{'applicable':True,'hard':1,'probabilities':{'0':0,'1':1,'2':0}}}
    scorer.match_state=match;scorer.review_state=review;scorer.score_behavior=behavior
    score_nodes(CASE,nodes,[r],scorer,CallCache(None))
    assert calls==['match','review','match','review','behavior','behavior']
