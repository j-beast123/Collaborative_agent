"""CPU-only Doctor batching and isolated sampling checks."""
from concurrent.futures import ThreadPoolExecutor
import json
import pytest
from src.online_rl.doctor_batch import decode, Request, GenerationBatcher
from test_hf_doctor import local_model
from src.online_rl.tree import b200_runner as R


def tiny_model():
    import torch
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration
    torch.manual_seed(7)
    cfg = Gemma3Config(text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        sliding_window=16, sliding_window_pattern=2, max_position_embeddings=128,
        pad_token_id=0, eos_token_id=31, bos_token_id=1),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, image_size=16, patch_size=8), mm_tokens_per_image=4)
    cfg._attn_implementation='sdpa'
    model=Gemma3ForConditionalGeneration(cfg).eval()
    model.generation_config.eos_token_id=31;model.generation_config.pad_token_id=0
    return model


class Tok:
    pad_token_id=0
    def __call__(self, text, **kwargs):
        import torch
        assert kwargs == {'add_special_tokens': False, 'return_tensors': 'pt'}
        return {'input_ids':torch.tensor([[int(x) for x in text.split()]])}
    def decode(self, tokens, **kwargs): return ' '.join(str(x) for x in tokens)


def test_real_tiny_gemma_batch_seeds_budgets_and_global_rng():
    import torch
    model=tiny_model(); tok=Tok()
    requests=[Request('1 3 4',11,5),Request('1 5',22,2),Request('1 7 8 9',33,4)]
    before=torch.get_rng_state().clone()
    result=decode(model,tok,requests,context=64)
    assert torch.equal(before,torch.get_rng_state())
    assert all(r['batch_size']==3 for r in result)
    reverse=decode(model,tok,list(reversed(requests)),context=64)
    assert [r['body'] for r in result]==[r['body'] for r in reversed(reverse)]
    for req,row in zip(requests,result):
        single=decode(model,tok,[req],context=64)[0]
        assert row['body']==single['body']
        assert row['output_tokens']<=req.max_tokens
        assert row['input_tokens']==len(req.prompt.split())
    bounded=decode(model,tok,requests,context=64,token_budget=12)
    assert all(r['batch_size']==1 for r in bounded)
    with pytest.raises(ValueError,match='refusing truncation'):
        decode(model,tok,requests,context=5)


def test_queue_composes_requests_and_propagates_failure():
    batches=[]
    def generate(rs):
        batches.append(len(rs))
        if any(r.seed<0 for r in rs):raise ValueError('synthetic OOM')
        return [r.seed for r in rs]
    queue=GenerationBatcher(generate,4,50)
    try:
        with ThreadPoolExecutor(4) as pool:
            assert list(pool.map(lambda n:queue.submit(Request('1',n,1)),range(4)))==[0,1,2,3]
        assert batches==[4]
        with pytest.raises(ValueError,match='synthetic OOM'):queue.submit(Request('1',-1,1))
    finally:queue.close()
    with pytest.raises(RuntimeError,match='closed'):queue.submit(Request('1',1,1))


def test_new_config_is_separate_hf_protocol_and_old_run_refuses_it(local_model, monkeypatch):
    monkeypatch.setattr(R, "physical_devices", lambda: ["0", "1"])
    old=R.load_config('configs/experiments/train120_max4_v7_rubric_only_b200.json',run_name='same')
    new=R.load_config('configs/experiments/train120_max4_v7_doctor_batch_b200.json',run_name='same')
    assert R.preflight(new,resources=False)==[]
    assert new['rollout']['doctor_backend']['sampling_protocol']=='hf_doctor_batched_per_request_rng_v1'
    assert old['five_gpu']['opening_identity']!=new['five_gpu']['opening_identity']
    for field in ('policy','optimizer','agent_reward','schedule','verification','intervention','doctor_need','fixed_accuracy'):
        assert old[field]==new[field]
    assert new['two_gpu']['batch_cases']==5 and new['rollout']['rounds']==4
    assert new['execution']['global_api_max_in_flight']==50
    with pytest.raises(RuntimeError,match='Config/code changed'):
        R.validate_resume_identity({'config_sha256':R.sha256_json(old),'source_sha256':'old'},new,'new')


def test_hf_chat_shares_loaded_model_and_batches_all_doctor_roles(local_model, monkeypatch, tmp_path):
    from src.online_rl.doctor import make_doctor
    from src.online_rl.tree.backends import with_role
    cfg=R.load_config('configs/experiments/train120_max4_v7_doctor_batch_b200.json')
    cfg['rollout']['doctor_backend']['batching'].update(max_batch_size=3,wait_ms=100)
    doctor=make_doctor(cfg['rollout'],'ordinary physician')
    net=tiny_model()
    class ChatTok(Tok):
        def apply_chat_template(self,messages,**kw):
            assert len(messages)==1 and messages[0]['role']=='user'
            assert 'public history' in messages[0]['content']
            return '1 3 5'
    doctor._net=net;doctor._tok=ChatTok();doctor._verified=True
    monkeypatch.setenv('SHERPA_HF_DOCTOR_AUDIT',str(tmp_path/'audit.jsonl'))
    roles=['rollout','doctor_reflection','verification']
    def call(i):
        return with_role(roles[i],lambda:doctor._chat('public history',11+i,3))
    try:
        with ThreadPoolExecutor(3) as pool: values=list(pool.map(call,range(3)))
        rows=[json.loads(x) for x in (tmp_path/'audit.jsonl').read_text().splitlines()]
        assert len(values)==len(rows)==3 and all(r['batch_size']==3 for r in rows)
        assert {r['seed'] for r in rows}=={11,12,13}
        assert {r['stage'] for r in rows}==set(roles)
        assert doctor._net is net and all(r['queue_wait_seconds']>=0 for r in rows)
        assert all(r['error'] is None and r['input_tokens']==3 for r in rows)
    finally:doctor.close()


def test_batched_chat_failure_never_returns_empty_or_zero(local_model, monkeypatch):
    from src.online_rl.doctor import make_doctor,DoctorUnavailable
    cfg=R.load_config('configs/experiments/train120_max4_v7_doctor_batch_b200.json')
    d=make_doctor(cfg['rollout'],'system')
    class Queue:
        def submit(self,req):raise RuntimeError('synthetic out of memory')
    d._batcher=Queue();d._net=object()
    class Tok:
        def apply_chat_template(self,*a,**kw):return '1 3'
    d._tok=Tok()
    with pytest.raises(DoctorUnavailable,match='out of memory'):d._chat('history',1,32)
    assert d._chat_metadata.last['error'] and d._chat_metadata.last.get('output_tokens') is None
