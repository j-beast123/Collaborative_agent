"""Exercise actual HF decoding with a tiny CPU model; no CUDA, checkpoint or API needed."""
import copy
import torch
import pytest
from transformers import Qwen2Config, Qwen2ForCausalLM
from src.online_rl.agent_batch import generate_batch


class Tokenizer:
    padding_side = 'right'
    pad_token_id = 0
    eos_token_id = None
    def __call__(self, prompts, **kwargs):
        ids=torch.tensor([[1,2,3] for _ in prompts])
        return {'input_ids':ids, 'attention_mask':torch.ones_like(ids)}
    def decode(self, ids, **kwargs): return ' '.join(map(str,ids))


def test_raw_policy_sampling_preserves_seeds_and_has_no_generation_warnings(caplog):
    torch.manual_seed(13)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                                     num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=2, eos_token_id=None)).eval()
    model.generation_config.do_sample=True
    model.generation_config.temperature=.6
    model.generation_config.top_p=.95
    model.generation_config.top_k=20
    original=copy.deepcopy(model.generation_config.to_dict())
    args=(model,Tokenizer(),['a','b'],[11,23],4,1.,1.,0)
    first=generate_batch(*args)
    second=generate_batch(*args)
    assert first == second
    assert model.generation_config.to_dict() == original
    for trace in first:
        assert trace['sampling_temperature'] == 1.
        assert trace['sampling_top_p'] == 1.
        assert trace['sampling_top_k'] == 0
        assert torch.allclose(torch.tensor(trace['old_policy_logprobs']),
                              torch.tensor(trace['sampling_logprobs']),atol=1e-6)
    assert 'default values have been modified' not in caplog.text
    assert 'generation flags are not valid' not in caplog.text


@pytest.mark.parametrize('temperature,top_p,top_k', [(1., 1., 0), (.7, .8, 4)])
def test_untraced_base_generation_keeps_seeded_tokens_without_retaining_logits(monkeypatch, temperature, top_p, top_k):
    torch.manual_seed(13)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                                     num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=2, eos_token_id=None)).eval()
    captured=[]
    generate=model.generate
    def spy(*args, **kwargs):
        result=generate(*args, **kwargs)
        captured.append((kwargs['output_logits'], result.logits))
        return result
    monkeypatch.setattr(model, 'generate', spy)
    args=(model,Tokenizer(),['a','b'],[11,23],8,temperature,top_p,top_k)
    traced=generate_batch(*args)
    compact=generate_batch(*args,collect_training_traces=False)
    assert captured[0][0] is True
    assert captured[1] == (False, None)
    for full, small in zip(traced,compact):
        assert small == {key:full[key] for key in small}
        assert small['completion_ids'] == full['completion_ids']
        assert 'old_policy_logprobs' not in small
        assert 'sampling_logprobs' not in small
        assert 'prompt_input_ids' not in small
