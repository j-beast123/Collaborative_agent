"""Independent prefix-only matching review with auditable exact evidence."""
import json,time
from pathlib import Path
from src.common import usage

PROMPT=Path(__file__).resolve().parents[2]/'prompts/sherpa/applicability_review_v1.txt'


def review_state(prefix,applicability,recorded_state=None):
    # Construct an allowlisted payload; never forward source envelopes or replies.
    state={'applicability':{k:applicability[k] for k in ('user_side_context','agent_side_state')},
           'prefix':[{'turn_index':i,'speaker':d['speaker'],'text':d['text']} for i,d in enumerate(prefix)]}
    if recorded_state is not None:state['recorded_pre_response_agent_state']=recorded_state
    return state


def validate_review(obj,state):
    if not isinstance(obj,dict) or type(obj.get('applicable')) is not bool:raise ValueError('Review applicability must be boolean')
    if not isinstance(obj.get('reason'),str) or not obj['reason'].strip():raise ValueError('Review reason missing')
    evidence=obj.get('evidence')
    if not isinstance(evidence,list) or (obj['applicable'] and not evidence):raise ValueError('Applicable review needs exact evidence')
    for e in evidence:
        if not isinstance(e,dict):raise ValueError('Evidence must be an object')
        index=e.get('turn_index')
        if type(index) is not int or not 0<=index<len(state['prefix']):raise ValueError('Invalid evidence turn index')
        d=state['prefix'][index];q=e.get('quote')
        if not isinstance(q,str) or not q.strip() or q not in d['text']:raise ValueError('Evidence quote not present in prefix')
        if e.get('speaker')!=d['speaker'] or e.get('supports') not in state['applicability']:raise ValueError('Evidence attribution invalid')
    return obj


class ApplicabilityReviewer:
    def __init__(self,client,audit_path):
        self.client=client;self.audit_path=Path(audit_path);self.system=PROMPT.read_text()
    def review(self,prefix,applicability,recorded_state=None):
        state=review_state(prefix,applicability,recorded_state)
        payload={'model':self.client.model,'temperature':0,'max_tokens':4096,'reasoning':{'effort':'low'},'response_format':{'type':'json_object'},
                 'messages':[{'role':'system','content':self.system},{'role':'user','content':json.dumps(state,ensure_ascii=False)}]}
        for attempt in range(self.client.max_retries+1):
            body=None;error=None
            try:
                body=self.client.post('https://openrouter.ai/api/v1/chat/completions',payload,self.client.key,180)
                usage.record('independent_applicability_review',body.get('model',self.client.model),body.get('usage') or {},attempt+1)
                choice=body['choices'][0]
                if choice.get('finish_reason')!='stop':raise ValueError('Review completion truncated')
                result=validate_review(json.loads(choice['message']['content']),state)
            except (ValueError,KeyError,TypeError,IndexError,OSError) as exc:error=f'{type(exc).__name__}: {exc}'
            self.audit_path.parent.mkdir(parents=True,exist_ok=True)
            with self.audit_path.open('a') as f:f.write(json.dumps({'state':state,'system':self.system,'model':self.client.model,'raw_response':body,'error':error},ensure_ascii=False)+'\n')
            if error is None:return result
            if attempt==self.client.max_retries:raise ValueError(error)
            time.sleep(min(2**attempt,10))

    def review_many(self, prefix, conditions, recorded_state=None):
        """Independent reviews batched per state; no initial labels or replies."""
        if not conditions:
            return {}
        states = {rid: review_state(prefix, app, recorded_state) for rid, app in conditions.items()}
        result = {}
        names = list(states)
        for start in range(0, len(names), self.client.batch_size):
            ids = names[start:start + self.client.batch_size]
            packet = {'prefix': next(iter(states.values()))['prefix'],
                      'conditions': {rid: states[rid]['applicability'] for rid in ids}}
            if recorded_state is not None:
                packet['recorded_pre_response_agent_state'] = recorded_state
            system = self.system + '\nFor this batch return {"reviews": {"rubric_id": review_object, ...}}. ' \
                     'Return every requested rubric id and no other id. Apply the same rules independently to each condition.'
            payload = {'model': self.client.model, 'temperature': 0, 'max_tokens': self.client.max_tokens,
                       'reasoning': {'effort': 'low'}, 'response_format': {'type': 'json_object'},
                       'messages': [{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]}
            for attempt in range(self.client.max_retries + 1):
                body, error = None, None
                try:
                    body = self.client.post('https://openrouter.ai/api/v1/chat/completions', payload, self.client.key, 180)
                    usage.record('state_applicability_review', body.get('model', self.client.model), body.get('usage') or {}, attempt+1)
                    choice = body['choices'][0]
                    if choice.get('finish_reason') != 'stop':
                        raise ValueError('Review completion truncated')
                    reviews = json.loads(choice['message']['content'])['reviews']
                    if set(reviews) != set(ids):
                        raise ValueError('Incomplete review ids')
                    validated = {rid: validate_review(reviews[rid], states[rid]) for rid in ids}
                except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
                    error = f'{type(exc).__name__}: {exc}'
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open('a') as f:
                    f.write(json.dumps({'state': packet, 'system': system, 'model': self.client.model,
                                        'raw_response': body, 'error': error}, ensure_ascii=False)+'\n')
                if error is None:
                    result.update(validated)
                    break
                if attempt == self.client.max_retries:
                    raise ValueError(error)
                self.client.sleep(min(2**attempt, 10))
        return result
