"""Generation trace keeps exact raw-policy probabilities after removing per-token CPU sync."""
from types import SimpleNamespace
import pytest
import torch
from src.online_rl.environment import generate_with_trace


@pytest.mark.parametrize('size',[0,1,7])
def test_trace_matches_previous_raw_policy_formula(size):
    logits=[torch.tensor([[0.25,-2.0,1.5,0.0]])+i/7 for i in range(size)]
    completion=torch.tensor([[i%4 for i in range(size)]],dtype=torch.long)
    prompt=torch.tensor([[1,2]],dtype=torch.long)
    generated=SimpleNamespace(sequences=torch.cat([prompt,completion],dim=1),
                              logits=tuple(logits),scores=tuple(logits))
    class Model:
        def parameters(self):yield torch.zeros(1)
        def generate(self,**kwargs):return generated
        def compute_transition_scores(self,*args,**kwargs):return torch.zeros((1,size))
    class Tokenizer:
        pad_token_id=0;eos_token_id=3
        def __call__(self,*args,**kwargs):return {'input_ids':prompt,'attention_mask':torch.ones_like(prompt)}
        def decode(self,*args,**kwargs):return 'visible reply'
    previous=[float(torch.nn.functional.log_softmax(row.float(),dim=-1)[0,tok])
              for row,tok in zip(logits,completion[0])]
    result=generate_with_trace(Model(),Tokenizer(),'prompt',max_new_tokens=7,temperature=.7,top_p=.9,top_k=0)
    assert result['old_policy_logprobs']==previous
    assert result['completion_ids']==completion[0].tolist()
    assert result['logprob_semantics']=='raw_policy_before_sampling_warpers_v1'
