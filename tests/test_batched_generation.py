from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from src.online_rl.tree.batched_generation import Request,GenerationBatcher,decode_batch


def test_queue_really_combines_requests_and_delivers_each_result():
    seen=[]
    def generate(requests):
        seen.append(len(requests));return [r.seed for r in requests]
    batcher=GenerationBatcher(generate,max_batch=3,wait_ms=50)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(lambda n:batcher.submit(Request('text',n,2)),range(3)))
        assert results==[0,1,2] and seen==[3]
    finally:batcher.close()


class Tokenizer:
    pad_token_id=0;eos_token_id=22
    def __call__(self,text,**kwargs):
        import torch
        return {'input_ids':torch.tensor([[int(n) for n in text.split()]])}
    def decode(self,ids,**kwargs):return ' '.join(str(n) for n in ids if n not in [0,22])


def model():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(1)
    config=Qwen3Config(vocab_size=23,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                      num_attention_heads=2,num_key_value_heads=2,head_dim=8,pad_token_id=0,eos_token_id=22)
    config._attn_implementation='eager'
    return Qwen3ForCausalLM(config).eval()


def test_padding_seeds_and_raw_logprobs_match_single_row_and_teacher_forcing():
    from src.online_rl.integrated.grpo import sequence_logprobs
    m=model();tok=Tokenizer();requests=[Request('1 2 3',11,4),Request('4',22,3),Request('5 6',33,2)]
    args={'temperature':.7,'top_p':.9,'top_k':7}
    batched=decode_batch(m,tok,requests,**args)
    for request,trace in zip(requests,batched):
        single=decode_batch(m,tok,[request],**args)[0]
        assert trace['completion_ids']==single['completion_ids']
        assert trace['prompt_input_ids']==[int(n) for n in request.prompt.split()]
        assert trace['prompt_attention_mask']==[1]*len(trace['prompt_input_ids'])
        teacher=sequence_logprobs(m,trace['prompt_input_ids'],trace['completion_ids'],grad=False,token_chunk_size=2).tolist()
        assert trace['old_policy_logprobs']==pytest.approx(teacher,abs=1e-5)
        assert len(trace['completion_ids'])<=request.max_tokens


def test_finished_rows_have_no_filler_tokens_in_trace():
    import torch
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.generation_config=SimpleNamespace(eos_token_id=22)
        def forward(self,input_ids,**kwargs):
            logits=torch.full((input_ids.shape[0],1,23),-100.)
            for i,row in enumerate(input_ids):logits[i,0,22 if int(row[-1]) in [1,2] else 2]=100
            return SimpleNamespace(logits=logits,past_key_values='cache')
    traces=decode_batch(Toy(),Tokenizer(),[Request('3 1',1,4,True),Request('4',2,4,True)],temperature=.7,top_p=.9,top_k=7)
    assert traces[0]['completion_ids']==[22]
    assert traces[1]['completion_ids']==[2,22]
    assert all(t['finish_reason']=='eos' for t in traces)
