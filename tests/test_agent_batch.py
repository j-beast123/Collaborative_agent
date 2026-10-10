import torch
from transformers import GPT2Config,GPT2LMHeadModel
from src.online_rl.agent_batch import generate_batch


class Tokenizer:
    padding_side='right';pad_token_id=0;eos_token_id=2
    def __call__(self,prompts,**kwargs):
        ids=[[int(x) for x in p.split()] for p in prompts];width=max(map(len,ids))
        return {'input_ids':torch.tensor([[0]*(width-len(r))+r for r in ids]),
                'attention_mask':torch.tensor([[0]*(width-len(r))+[1]*len(r) for r in ids])}
    def decode(self,ids,**kwargs):return ' '.join(str(x) for x in ids if x not in (0,2))


def model():
    torch.manual_seed(1)
    m=GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=32,n_embd=16,n_layer=1,n_head=2,bos_token_id=1,eos_token_id=2,pad_token_id=0))
    m.eval();return m


def test_batch_rng_follows_seed_not_position_and_padding():
    m=model();tok=Tokenizer();prompts=['3 4','5 6 7'];seeds=[55,66]
    a=generate_batch(m,tok,prompts,seeds,4,.8,.9,8)
    b=generate_batch(m,tok,prompts[::-1],seeds[::-1],4,.8,.9,8)[::-1]
    assert tok.padding_side=='right'
    for x,y in zip(a,b):
        assert x['completion_ids']==y['completion_ids']
        assert x['prompt_input_ids']==y['prompt_input_ids']
        assert len(x['old_policy_logprobs'])==len(x['completion_ids'])
        assert torch.allclose(torch.tensor(x['old_policy_logprobs']),torch.tensor(y['old_policy_logprobs']),atol=1e-5)


def test_batch_raw_logprobs_match_teacher_forced_path():
    m=model();tok=Tokenizer();traces=generate_batch(m,tok,['3 4','5 6 7'],[55,66],4,.8,.9,8)
    for t in traces:
        prompt=t['prompt_input_ids'];completion=t['completion_ids'];ids=torch.tensor([prompt+completion])
        with torch.no_grad():logits=m(ids).logits[0,len(prompt)-1:len(prompt)+len(completion)-1]
        raw=logits.float().log_softmax(-1).gather(1,torch.tensor(completion)[:,None]).squeeze(1)
        assert torch.allclose(raw,torch.tensor(t['old_policy_logprobs']),atol=1e-5)
        assert t['logprob_semantics']=='raw_policy_before_sampling_warpers_v1'


def test_model_defaults_notice_does_not_change_explicit_row_sampling():
    m=model();tok=Tokenizer()
    baseline=generate_batch(m,tok,['3 4'],[55],4,.7,.9,0)
    m.generation_config.transformers_version='4.53.3'
    m.generation_config.do_sample=True;m.generation_config.temperature=.6;m.generation_config.top_p=.95
    actual=generate_batch(m,tok,['3 4'],[55],4,.7,.9,0)
    assert actual==baseline
