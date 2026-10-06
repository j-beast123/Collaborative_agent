from scripts.evaluation.run_casimedicos_base import run_case, shard


def test_four_shards_cover_every_case_once():
    cases=[{'case_id':str(i)} for i in range(622)]
    groups=[shard(cases,rank,4) for rank in range(4)]
    assert [len(g) for g in groups]==[156,156,155,155]
    assert sorted(i for g in groups for i,c in g)==list(range(622))


def test_plain_five_rounds_no_gold_and_resume(tmp_path):
    seen=[]
    class Doctor:
        def turn(self,view,dialogue,turn,rounds):
            assert 'ground_truth' not in view and 'metadata' not in view
            seen.append(('doctor',turn));return 'I agree; ready to conclude.'
        def final(self,view,dialogue):
            assert len(dialogue)==10 and 'ground_truth' not in view
            seen.append(('final',None));return 'Final answer: option E - five'
    class Agent:
        def agent(self,prompt,seed):
            seen.append(('agent',seed));return 'I agree; ready to conclude.'
    case={'case_id':'test','question':'Choose an option','options':{'A':'one','E':'five'},
          'ground_truth':{'correct_option':'E'},'metadata':{'source_split':'test'}}
    checkpoint=tmp_path/'progress.json'
    result=run_case(case,Doctor(),Agent(),5,42,checkpoint)
    assert result['outcome']==1.0 and result['status']=='complete'
    assert result['agent_turns']==5 and len(result['dialogue'])==11
    assert len(seen)==11
    assert run_case(case,Doctor(),Agent(),5,42,checkpoint)==result
    assert len(seen)==11
