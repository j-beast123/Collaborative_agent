"""Single-model batched decoding with independent per-request RNG and raw-policy traces.
Sampling itself uses a row-private CUDA generator; HF handles KV caches/EOS/padding.
The resulting sampler is explicitly pinned as a new execution protocol.
"""
from concurrent.futures import Future
import queue
import threading
import time

PROTOCOL='per_row_rng_batch_v1'


def generate_batch(model,tokenizer,prompts,seeds,max_new_tokens,temperature,top_p,top_k):
    import copy
    import torch
    import torch.nn.functional as F
    from transformers import LogitsProcessor,LogitsProcessorList,TopKLogitsWarper,TopPLogitsWarper
    from src.online_rl.environment import POLICY_LOGPROB_SEMANTICS
    if len(prompts)!=len(seeds) or not prompts:raise ValueError('Nonempty prompts/seeds must align')
    if temperature<=0:raise ValueError('Sampling temperature must be positive')
    device=next(model.parameters()).device
    old_padding=tokenizer.padding_side;tokenizer.padding_side='left'
    try:encoded=tokenizer(prompts,return_tensors='pt',padding=True,truncation=True)
    finally:tokenizer.padding_side=old_padding
    encoded={k:v.to(device) for k,v in encoded.items()};width=encoded['input_ids'].shape[1]
    generators=[torch.Generator(device=device).manual_seed(s) for s in seeds]
    sampling=[]
    class RowSampler(LogitsProcessor):
        def __call__(self,input_ids,scores):
            warped=scores.float()/temperature
            if top_k:warped=TopKLogitsWarper(top_k)(input_ids,warped)
            if top_p<1:warped=TopPLogitsWarper(top_p)(input_ids,warped)
            logp=F.log_softmax(warped,dim=-1)
            # Each row draws from its own RNG; scheduling/batch position cannot borrow another row's stream.
            tokens=torch.stack([torch.multinomial(logp[i].exp(),1,generator=g).squeeze(0) for i,g in enumerate(generators)])
            sampling.append(logp.gather(1,tokens[:,None]).squeeze(1))
            forced=torch.full_like(scores,float('-inf'));forced.scatter_(1,tokens[:,None],0.)
            return forced
    gc=copy.deepcopy(model.generation_config)
    gc.do_sample=False
    # HF only advances the selected token and KV cache. RowSampler owns all
    # stochastic sampling; leave HF's inactive sampling flags unset.
    gc.temperature=None;gc.top_p=None;gc.top_k=None
    # Preserve the pinned model's normal EOS handling and maximum generation length.
    with torch.inference_mode():
        generated=model.generate(**encoded,generation_config=gc,max_new_tokens=max_new_tokens,
                                 use_model_defaults=False,
                                 pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
                                 logits_processor=LogitsProcessorList([RowSampler()]),return_dict_in_generate=True,
                                 output_scores=False,output_logits=True)
    if generated.logits is None:raise RuntimeError('Raw policy logits missing')
    completions=generated.sequences[:,width:]
    raw_logp=torch.stack([F.log_softmax(logits.float(),dim=-1).gather(1,completions[:,i,None]).squeeze(1)
                         for i,logits in enumerate(generated.logits)],dim=1)
    sample_logp=torch.stack(sampling,dim=1)
    ids=completions.cpu().tolist();raw=raw_logp.cpu().tolist();sample=sample_logp.cpu().tolist()
    eos=tokenizer.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
    outputs=[]
    for i,completion in enumerate(ids):
        end=next((j+1 for j,t in enumerate(completion) if t in eos),len(completion));completion=completion[:end]
        prompt_ids=encoded['input_ids'][i][encoded['attention_mask'][i].bool()].cpu().tolist()
        outputs.append({'prompt_input_ids':prompt_ids,'prompt_attention_mask':[1]*len(prompt_ids),'completion_ids':completion,
                        'old_policy_logprobs':raw[i][:end],'sampling_logprobs':sample[i][:end],
                        'logprob_semantics':POLICY_LOGPROB_SEMANTICS,'sampling_temperature':temperature,
                        'sampling_top_p':top_p,'sampling_top_k':top_k,'completion_text':tokenizer.decode(completion,skip_special_tokens=True).strip(),
                        'batch_sampling_protocol':PROTOCOL})
    return outputs


class Batcher:
    """Coalesce concurrent traced requests; execute one compatible segment batch under the model lock."""
    def __init__(self,policy,capacity):
        self.policy=policy;self.capacity=capacity;self.queue=queue.Queue()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def submit(self,prompt,max_new,seed):
        future=Future();self.queue.put((prompt,max_new,seed,future));return future.result()
    def close(self):
        self.queue.put(None);self.thread.join(timeout=30)
        if self.thread.is_alive():raise RuntimeError('Agent batch worker did not exit')
    def run(self):
        pending=[]
        while True:
            first=pending.pop(0) if pending else self.queue.get();batch=[first];deadline=time.monotonic()+.01
            if first is None:return
            while len(batch)<self.capacity:
                try:item=self.queue.get(timeout=max(0,deadline-time.monotonic()))
                except queue.Empty:break
                if item[1]==first[1]:batch.append(item)
                else:pending.append(item)
                if time.monotonic()>=deadline:break
            try:
                p=self.policy;a=p.args
                from src.common.live_gpu import activity
                with p.gen_lock,activity('agent_batch_generate'):
                    p.model.eval()
                    outputs=generate_batch(p.model,p.tok,[x[0] for x in batch],[x[2] for x in batch],first[1],a.generation_temperature,a.top_p,a.top_k)
                for item,result in zip(batch,outputs):item[3].set_result(result)
            except BaseException as e:
                for item in batch:item[3].set_exception(e)
