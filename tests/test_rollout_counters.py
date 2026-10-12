import json
from src.common.rollout_monitor import RolloutCounters


def record(kind,path='0.1',key='key'):
    return {'key':key,'kind':kind,'identity':{'case':'c','policy':'p','protocol':'x','path':path}}


def test_incremental_counts_handle_partial_tail_and_duplicate_calls(tmp_path):
    p=tmp_path/'calls.jsonl';counter=RolloutCounters()
    agent=record('agent',key='a');final=record('final',key='f')
    p.write_text(json.dumps(agent)+'\n'+json.dumps(final)[:20])
    assert counter.read(p)['agent_nodes']==1
    assert counter.read(p)['trajectories']==0
    with p.open('a') as stream:stream.write(json.dumps(final)[20:]+'\n'+json.dumps(final)+'\n')
    result=counter.read(p)
    assert result['trajectories']==1 and result['leaf_depths']=={2:1}
    assert counter.read(p)==result


def test_reruns_do_not_count_as_original_rollout_and_reflection_deduplicates(tmp_path):
    p=tmp_path/'calls.jsonl';counter=RolloutCounters()
    rows=[record('final',key='f'),{'kind':'final','key':'rerun','identity':{'intervention':'i','path':'0.1'}},
          {'kind':'feedback','key':'fb1','identity':{'leaf':'c/0.1'}},
          {'kind':'feedback','key':'fb2','identity':{'leaf':'c/0.1'}}]
    p.write_text('\n'.join(map(json.dumps,rows))+'\n')
    result=counter.read(p)
    assert result['trajectories']==1 and result['reflections']==1
    p.write_text('')
    assert counter.read(p)['trajectories']==0
