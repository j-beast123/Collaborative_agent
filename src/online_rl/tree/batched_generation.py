"""Short request queue + actual batched KV-cache decoding, with per-row RNG.
Padding and finished-row filler tokens never enter stored training traces.
"""
from concurrent.futures import Future
from dataclasses import dataclass
import queue
import threading
import time

VERSION='left_padded_per_row_rng_kv_decode_v1'


@dataclass(frozen=True)
class Request:
    prompt: str
    seed: int
    max_tokens: int
    greedy: bool=False


class GenerationBatcher:
    def __init__(self,generate,max_batch=3,wait_ms=15):
        if max_batch<1 or wait_ms<0:raise ValueError('Invalid generation batch limits')
        self.generate=generate;self.max_batch=max_batch;self.wait_s=wait_ms/1000
        self.queue=queue.Queue();self.closed=False;self.guard=threading.Lock()
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()

    def submit(self,request):
        future=Future()
        with self.guard:
            if self.closed:raise RuntimeError('Generation batcher closed')
            self.queue.put((request,future))
        return future.result()

    def _loop(self):
        while True:
            first=self.queue.get()
            if first is None:return
            pending=[first];deadline=time.monotonic()+self.wait_s
            while len(pending)<self.max_batch:
                try:item=self.queue.get(timeout=max(0,deadline-time.monotonic()))
                except queue.Empty:break
                if item is None:
                    self.queue.put(None);break
                pending.append(item)
            try:
                output=self.generate([r for r,_ in pending])
                if len(output)!=len(pending):raise RuntimeError('Batch output count mismatch')
            except Exception as error:
                for _,future in pending:future.set_exception(error)
            else:
                for (_,future),value in zip(pending,output):future.set_result(value)

    def close(self):
        with self.guard:
            if not self.closed:self.closed=True;self.queue.put(None)
        self.thread.join()


def decode_batch(model,tokenizer,requests,*,temperature,top_p,top_k):
    import torch
    from transformers import TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
    if not requests:return []
    device=next(model.parameters()).device
    if any(r.max_tokens<1 for r in requests):raise ValueError('Positive generation length required')
    # Encode individually: do not mutate the tokenizer's shared padding settings.
    prompts=[tokenizer(r.prompt,return_tensors='pt',truncation=True)['input_ids'][0].tolist() for r in requests]
    width=max(map(len,prompts));pad=tokenizer.pad_token_id
    if pad is None:pad=tokenizer.eos_token_id
    ids=torch.tensor([[pad]*(width-len(p))+p for p in prompts],device=device)
    mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts],device=device)
    eos=getattr(getattr(model,'generation_config',None),'eos_token_id',None)
    if eos is None:eos=tokenizer.eos_token_id
    eos=set(eos if isinstance(eos,(tuple,list)) else [eos])
    generators=[torch.Generator(device=device).manual_seed(r.seed) for r in requests]
    traces=[{'prompt_input_ids':p,'prompt_attention_mask':[1]*len(p),'completion_ids':[],
             'old_policy_logprobs':[],'sampling_logprobs':[],'seed':r.seed,
             'logprob_semantics':'raw_policy_before_sampling_warpers_v1','generation_protocol':VERSION,
             'generation_batch_size':len(requests),'sampling_temperature':0 if r.greedy else temperature,
             'sampling_top_p':top_p,'sampling_top_k':top_k} for p,r in zip(prompts,requests)]
    active=[True]*len(requests);past=None;warpers=[]
    if any(not r.greedy for r in requests):
        if temperature<=0:raise ValueError('Sampling temperature must be positive')
        if temperature!=1:warpers.append(TemperatureLogitsWarper(temperature))
        if top_k:warpers.append(TopKLogitsWarper(top_k))
        if top_p<1:warpers.append(TopPLogitsWarper(top_p))
    with torch.inference_mode():
        for step in range(max(r.max_tokens for r in requests)):
            positions=mask.long().cumsum(-1)-1;positions.masked_fill_(mask==0,0)
            kwargs={'input_ids':ids,'attention_mask':mask,'position_ids':positions if past is None else positions[:,-1:],
                    'past_key_values':past,'use_cache':True}
            try:output=model(**kwargs,logits_to_keep=1)
            except TypeError as error:
                if 'logits_to_keep' not in str(error):raise
                output=model(**kwargs)
            logits=output.logits[:,-1,:].float();raw_lp=logits.log_softmax(-1)
            sampling=logits
            for warper in warpers:sampling=warper(None,sampling)
            sampling_lp=sampling.log_softmax(-1);next_ids=[];new_mask=[]
            for row,(request,trace) in enumerate(zip(requests,traces)):
                was_active=active[row];new_mask.append(int(was_active))
                if not was_active:next_ids.append(pad);continue
                token=int(logits[row].argmax()) if request.greedy else int(torch.multinomial(sampling_lp[row].exp(),1,generator=generators[row]))
                next_ids.append(token);trace['completion_ids'].append(token)
                trace['old_policy_logprobs'].append(float(raw_lp[row,token]))
                trace['sampling_logprobs'].append(float(raw_lp[row,token] if request.greedy else sampling_lp[row,token]))
                if token in eos or step+1>=request.max_tokens:active[row]=False
            past=output.past_key_values
            if not any(active):break
            ids=torch.tensor(next_ids,device=device).unsqueeze(-1)
            mask=torch.cat([mask,torch.tensor(new_mask,device=device).unsqueeze(-1)],-1)
    for trace in traces:
        trace['completion_text']=tokenizer.decode(trace['completion_ids'],skip_special_tokens=True).strip()
        trace['finish_reason']='eos' if trace['completion_ids'][-1] in eos else 'length'
    return traces
