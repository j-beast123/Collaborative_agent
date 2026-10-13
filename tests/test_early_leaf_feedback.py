import dataclasses
from threading import Event

import test_witness_trainer as W
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.rollout import build_tree
import test_doctor_v2 as D


def test_feedback_api_starts_before_case_tree_finishes_and_records_keep_leaf_order(tmp_path):
    trainer,log,seen=W.make(tmp_path,'rubric_only',cases=1)
    trainer.cfg['five_gpu']={'execution_mode':'shared_ready_tasks_v1'}
    started=Event();calls=[]
    feedback=trainer.env.feedback_fn
    def early(view,dialogue,outcome):
        calls.append(dialogue);started.set()
        return feedback(view,dialogue,outcome)
    trainer.env=dataclasses.replace(trainer.env,feedback_fn=early)
    original=trainer.env.doctor.final;count=[0]
    def final(*args):
        count[0]+=1
        if count[0]==2:
            assert started.wait(2), 'Feedback waited for the whole tree instead of the first completed leaf'
        return original(*args)
    trainer.env.doctor.final=final
    record=trainer.step(commit=False)
    decided=[l for l in record['tree']['leaves'] if l['outcome'] is not None]
    assert len(calls)==len(decided)>1
    assert [f['leaf_id'] for f in record['feedback']]==[l['leaf_id'] for l in decided]
    assert trainer.state.case_cursor==0
    # Cache replay calls the leaf callback again but never duplicates the API request.
    record2=trainer.step(commit=False)
    assert len(calls)==len(decided) and record2['feedback']==record['feedback']


def test_parallel_final_callback_does_not_wait_for_another_final():
    ready=Event();base=D.doctor(halting={'enabled':False,'min_agent_turns':2})
    class Agent:
        def sample(self,case,prefix,turn,seed):return {'completion_text':'natural reply','seed':seed}
    calls=[0]
    original=base.final
    def final(case,dialogue,seed):
        calls[0]+=1
        if calls[0]==2:assert ready.wait(2), 'Leaf callback waited for all FINALs'
        return original(case,dialogue,seed)
    base.final=final
    callbacks=[]
    def on_leaf(leaf):callbacks.append(leaf);ready.set()
    tree=build_tree(D.CASE,max_agent_turns=1,branching=2,run_seed=3,policy_sha256='p',protocol_sha256='proto',
                    agent=Agent(),doctor=base,stop_rule=lambda *args:None,final_option=D.final_answer_option,
                    outcome_fn=lambda case,opt:float(opt=='B'),cache=CallCache(None),workers=2,on_leaf=on_leaf)
    assert {l.leaf_id for l in callbacks}=={l.leaf_id for l in tree.leaves}
