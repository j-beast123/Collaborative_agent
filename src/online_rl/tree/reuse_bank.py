"""Post-hoc reuse inspection, separate from prefix-only training applicability."""
from pathlib import Path
PROMPT=Path(__file__).resolve().parents[3]/'prompts/sherpa/rubric_reuse_v1.txt'
PROTOCOL='case_outcome_prompt_separate_behavior_reuse_v1'


def packet(tin, candidates):
    nodes=iter(tin.agent_node_ids);dialogue=[]
    for speaker,stage,turn,text in tin.dialogue:
        row={'speaker':speaker,'stage':stage,'round':turn,'text':text}
        if speaker=='agent':row['node_id']=next(nodes)
        dialogue.append(row)
    return {'case_id':tin.case_id,'trajectory_id':tin.leaf_id,'source_outcome':tin.outcome,
        'question':tin.question,'options':list(tin.options),'full_dialogue':dialogue,
        'doctor_feedback':tin.feedback,'agent_pre_response_states':list(tin.agent_pre_response_states),
        'candidates':[{'rubric_id':r.rubric_id,'candidate_rubric':r.rubric['candidate_rubric']} for r in candidates]}


def validate_reuse(result, state):
    if not isinstance(result,dict) or any(type(result.get(k)) is not bool for k in ('eligible','reuse')):
        raise ValueError('Reuse flags must be boolean')
    if not isinstance(result.get('reason'),str) or not result['reason'].strip():raise ValueError('Reuse reason missing')
    if not result['reuse']:
        if any(result.get(k) is not None for k in ('rubric_id','agent_turn_id','doctor_quote','agent_quote')):
            raise ValueError('Nonreuse cannot assert a reused evidence target')
        return result
    if not result['eligible'] or result.get('rubric_id') not in {r['rubric_id'] for r in state['candidates']}:
        raise ValueError('Unknown reuse candidate')
    positions=[i for i,r in enumerate(state['full_dialogue']) if r.get('node_id')==result.get('agent_turn_id')]
    if len(positions)!=1:raise ValueError('Reuse target outside source trajectory')
    i=positions[0];d=state['full_dialogue'];aq=result.get('agent_quote');dq=result.get('doctor_quote')
    if not isinstance(aq,str) or not aq.strip() or aq not in d[i]['text']:raise ValueError('Agent quote not exact')
    if not isinstance(dq,str) or not dq.strip() or not any(r['speaker']=='doctor' and dq in r['text'] for r in d[:i]):
        raise ValueError('Doctor quote must precede target response')
    return result
