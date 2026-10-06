import json
import pytest
from src.judges.openrouter_choice import OpenRouterChoiceJudge


def test_batched_hard_choices_and_payload():
    calls = []
    def post(url, payload, key, timeout):
        data = json.loads(payload['messages'][1]['content'])
        calls.append(data)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'answers': {qid: next(iter(q['criteria'])) for qid, q in data['questions'].items()}})}}]}
    judge = OpenRouterChoiceJudge('test-key', batch_size=1, post=post)
    questions = {'a': {'type': 'choice', 'criteria': {'Y': 'yes', 'N': 'no'}},
                 'b': {'type': 'choice', 'criteria': {'2': 'complete', '0': 'absent', '1': 'partial'}}}
    answers, metadata = judge.decide('{"dialogue": []}', questions)
    assert len(calls) == 2
    assert answers['b']['probabilities'] == {'2': 1., '0': 0., '1': 0.}
    assert answers['a']['confidence'] is None
    assert metadata['score_semantics'] == 'hard_label_one_hot'
    assert 'test-key' not in json.dumps(metadata)


@pytest.mark.parametrize('labels,finish', [({}, 'stop'), ({'a':'invalid'}, 'stop'), ({'a':'Y'}, 'length')])
def test_missing_invalid_and_truncated_outputs_fail(labels, finish):
    def post(*args):
        return {'choices':[{'finish_reason':finish, 'message':{'content':json.dumps({'answers':labels})}}]}
    judge = OpenRouterChoiceJudge('test-key', post=post)
    with pytest.raises(ValueError):
        judge.decide('{}', {'a':{'type':'choice','criteria':{'Y':'yes','N':'no'}}})


def test_two_stage_sonnet_excludes_reply_from_matching():
    from src.online_rl.tree.backends import two_stage_scorer
    payloads = []
    def post(url, payload, key, timeout):
        data = json.loads(payload['messages'][1]['content'])
        payloads.append(data)
        label = 'Y' if 'target_turn' not in data['state'] else '2'
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answers':{qid:label for qid in data['questions']}})}}]}
    scorer = two_stage_scorer(OpenRouterChoiceJudge('test-key', post=post))
    rubric = {'rubric_id':'r', 'name':'Clarify', 'applicability_condition':'Doctor asks for clarification',
              'required_behavior':'Explain', 'scores':{'0':'Absent','1':'Partial','2':'Explain'}}
    result = scorer({'case_id':'c','question':'question','options':['a','b']},
                   [{'speaker':'doctor','text':'Explain?'}], [rubric], 'Agent explanation')
    assert result['r']['applicable'] is True
    assert result['r']['hard'] == 2
    assert 'target_turn' not in payloads[0]['state']
    assert payloads[1]['state']['target_turn']['text'] == 'Agent explanation'
    assert scorer.judge_identity['model'] == 'anthropic/claude-sonnet-5.5'


def test_launcher_selects_configured_sonnet():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/online_rl/run_tree.py'
    spec = importlib.util.spec_from_file_location('sonnet_launcher_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((path.parents[2] / 'configs/online_rl/tree_v1.json').read_text())
    client = module.judge_client(config, 'test-key')
    assert isinstance(client, OpenRouterChoiceJudge)
    assert client.model == 'google/gemini-3.8-flash'


def test_fenced_judge_output_is_parsed_and_audited(tmp_path):
    def post(*args):
        return {'choices':[{'finish_reason':'stop','message':{'content':'```json\n{"answers":{"a":"Y"}}\n```'}}]}
    path=tmp_path/'raw.jsonl'
    judge=OpenRouterChoiceJudge('private-test-key',post=post,audit_path=path)
    result,_=judge.decide('{}',{'a':{'type':'choice','criteria':{'Y':'yes','N':'no'}}})
    assert result['a']['value']=='Y'
    assert json.loads(path.read_text())['error'] is None
    assert 'private-test-key' not in path.read_text()


def test_judge_retries_missing_json_and_keeps_error_raw(tmp_path):
    responses=iter(['', '{"answers":{"a":"N"}}'])
    def post(*args):
        return {'choices':[{'finish_reason':'stop','message':{'content':next(responses)}}]}
    path=tmp_path/'raw.jsonl'
    judge=OpenRouterChoiceJudge('test',post=post,audit_path=path,sleep=lambda *args:None)
    answers,_=judge.decide('{}',{'a':{'type':'choice','criteria':{'Y':'yes','N':'no'}}})
    assert answers['a']['value']=='N'
    records=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(records)==2 and records[0]['error'] and records[1]['error'] is None


def test_explanation_before_json_still_validates_labels():
    def post(*args):
        return {'choices':[{'finish_reason':'stop','message':{'content':'This response meets level 1.\n```json\n{"answers":{"a":"1"}}\n```'}}]}
    judge=OpenRouterChoiceJudge('test',model='google/gemini-3.8-flash',post=post)
    answers,_=judge.decide('{}',{'a':{'type':'choice','criteria':{'0':'no','1':'partial','2':'yes'}}})
    assert answers['a']['value']=='1'


def test_multiple_answer_objects_are_rejected():
    from src.judges.openrouter_choice import parse_judge_object
    with pytest.raises(ValueError,match='unambiguous'):
        parse_judge_object('{"answers":{"a":"Y"}} {"answers":{"a":"N"}}')


def test_shared_instructions_hoisted_without_changing_criteria():
    from src.judges.rubric_scoring import applicability_questions
    payloads=[]
    def post(url,payload,key,timeout):
        data=json.loads(payload['messages'][1]['content']);payloads.append(data)
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answers':{qid:'Y' for qid in data['questions']}})}}]}
    rubrics=[{'rubric_id':f'r{i}','name':f'Rubric{i}','applicability_condition':f'condition{i}'} for i in range(2)]
    questions=applicability_questions(rubrics)
    OpenRouterChoiceJudge('test',post=post).decide('{}',questions)
    packed=payloads[0]
    for rid,q in questions.items():
        assert packed['shared_question_instructions']+packed['questions'][rid]['instructions']==q['instructions']
        assert packed['questions'][rid]['criteria']==q['criteria']
