"""CPU-only fake Gemma generation and byte-pinned local Doctor wiring."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.online_rl import hf_doctor as H
from src.online_rl.doctor import make_doctor, OllamaDoctor, DoctorUnavailable
from src.online_rl.tree import hf_doctor_setup as S, b200_runner as R


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    root = tmp_path/'gemma3'; root.mkdir()
    cfg = {'model_type': 'gemma3', 'architectures': ['Gemma3ForConditionalGeneration'],
           'text_config': {'hidden_size': 3840, 'num_hidden_layers': 48}}
    (root/'config.json').write_text(json.dumps(cfg))
    (root/'tokenizer_config.json').write_text('{"chat_template": "fake"}')
    (root/'tokenizer.json').write_text('{}')
    (root/'model-01.safetensors').write_bytes(b'fake model shard')
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'model-01.safetensors'}}))
    monkeypatch.setenv('SHERPA_DOCTOR_BACKEND', 'hf_local')
    monkeypatch.setenv('SHERPA_DOCTOR_MODEL', str(root))
    monkeypatch.setenv('SHERPA_HF_DOCTOR_RECEIPT', str(tmp_path/'receipt.json'))
    monkeypatch.delenv('SHERPA_RUN_NAME', raising=False)
    monkeypatch.delenv('SHERPA_LEARNING_RATE', raising=False)
    S.pin(root)
    return root


def doctor():
    cfg = R.load_config()
    return make_doctor(cfg['rollout'], 'normal physician role'), cfg


def test_pin_is_idempotent_and_changed_weights_or_tokenizer_refused(local_model):
    receipt = S.receipt_path().read_bytes()
    S.pin(local_model)
    assert S.receipt_path().read_bytes() == receipt
    (local_model/'model-01.safetensors').write_bytes(b'changed model shard')
    with pytest.raises(ValueError, match='changed'): S.pin(local_model)
    assert S.receipt_path().read_bytes() == receipt
    (local_model/'model-01.safetensors').write_bytes(b'fake model shard')
    (local_model/'tokenizer.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='changed'): S.verify_receipt(S.read_receipt())


def test_pin_rejects_wrong_architecture_quantization_and_missing_shard(local_model):
    p = local_model/'config.json'; original = json.loads(p.read_text())
    changed = copy.deepcopy(original); changed['text_config']['hidden_size'] = 4096
    p.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='architecture'): S.inventory(local_model)
    p.write_text(json.dumps({**original, 'quantization_config': {'bits': 8}}))
    with pytest.raises(ValueError, match='unquantized'): S.inventory(local_model)
    p.write_text(json.dumps(original)); (local_model/'model-01.safetensors').unlink()
    with pytest.raises(ValueError, match='shards are incomplete'): S.inventory(local_model)


def test_receipt_tampering_and_missing_pin_fail_without_fallback(local_model):
    p = S.receipt_path(); row = json.loads(p.read_text()); row['model_digest'] = '0'*64
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='integrity'): R.load_config()
    p.unlink()
    with pytest.raises(ValueError, match='HF Doctor receipt'): R.load_config()


def test_backend_switch_ignores_ollama_and_preserves_science(local_model, monkeypatch):
    monkeypatch.setattr(R, 'physical_devices', lambda: ['0', '1'])
    monkeypatch.setenv('SHERPA_DOCTOR_RECEIPT', '/missing/old/ollama/receipt')
    monkeypatch.setenv('SHERPA_OLLAMA_BINARY', '/old/ollama')
    hf = R.load_config(); raw = json.loads((R.ROOT/R.CONFIG).read_text())
    for field in ('policy', 'optimizer', 'agent_reward', 'schedule', 'verification', 'intervention', 'doctor_need', 'fixed_accuracy', 'rubric', 'two_gpu'):
        assert hf[field] == raw[field]
    assert hf['rollout']['doctor_backend']['options'] == raw['rollout']['doctor_backend']['options']
    assert hf['five_gpu']['doctor_urls'] == [] and hf['five_gpu']['ollama_binary'] is None
    assert hf['feedback']['model'] == hf['rollout']['doctor_model'] == hf['experiment']['feedback_model']
    assert 'hf_doctor_' in hf['output_dir']
    assert hf['five_gpu']['opening_identity'] != raw['five_gpu']['opening_identity']
    for field in ('branching', 'rounds', 'agent_protocol', 'doctor_system_prompt_sha256', 'agent_system_prompt_sha256'):
        assert hf['rollout'][field] == raw['rollout'][field]
    assert R.preflight(hf, resources=False) == []


def test_protocol_factory_and_reflection_model_labels_without_loading(local_model, monkeypatch):
    monkeypatch.setattr(H.HFDoctor, '_load', lambda *_: pytest.fail('must not load'))
    d, cfg = doctor()
    from src.online_rl.integrated.continuation import protocol_sha256
    from src.online_rl.tree.trainer import model_roles
    assert protocol_sha256(cfg, R.ROOT)
    assert 'hf_local' in model_roles(cfg)['doctor']
    assert d._net is None and d.model == cfg['feedback']['model']
    assert d.final_response['max_tokens'] == 32 and d.min_agent_turns == 2
    assert H.HFDoctor.turn is OllamaDoctor.turn and H.HFDoctor.halt is OllamaDoctor.halt
    assert H.HFDoctor.final is OllamaDoctor.final


def fake_loaded(d, monkeypatch, reply='visible reply', input_len=10):
    import torch
    captured = []
    class Batch(dict):
        def to(self, device): captured.append(('device', device)); return self
    class Tokenizer:
        pad_token_id = 0
        def apply_chat_template(self, messages, **kwargs): captured.append(('messages', messages)); return 'formatted'
        def __call__(self, prompt, **kwargs):
            assert kwargs == {'add_special_tokens': False, 'return_tensors': 'pt'}
            return Batch(input_ids=torch.ones((1, input_len), dtype=torch.long))
        def decode(self, tokens, **kwargs): return reply
    class Model:
        generation_config = SimpleNamespace(bos_token_id=2, eos_token_id=[1, 107], pad_token_id=0)
        def generate(self, **kwargs):
            captured.append(('generate', kwargs))
            return torch.cat([kwargs['input_ids'], torch.tensor([[7, 1]])], dim=1)
    @contextmanager
    def rng(torch_arg, device, seed):
        captured.append(('seed', (device, seed))); yield
    monkeypatch.setattr(H, 'doctor_rng', rng)
    d._net, d._tok, d._verified = Model(), Tokenizer(), True
    return captured


def test_explicit_decoding_chat_template_override_and_metadata(local_model, monkeypatch, tmp_path):
    d, _ = doctor(); calls = fake_loaded(d, monkeypatch)
    monkeypatch.setenv('SHERPA_HF_DOCTOR_AUDIT', str(tmp_path/'audit.jsonl'))
    assert d._chat('public history', 123, 32, 'reflection-only system') == 'visible reply'
    assert ('seed', ('cuda:0', 123)) in calls
    assert ('messages', [{'role': 'user', 'content': 'reflection-only system\n\npublic history'}]) in calls
    options = next(v for k, v in calls if k == 'generate')['generation_config']
    assert next(v for k, v in calls if k == 'generate')['use_model_defaults'] is False
    assert options.do_sample and options.temperature == .4 and options.top_p == 1. and options.top_k == 0
    assert options.repetition_penalty == 1. and options.max_new_tokens == 32 and options.eos_token_id == [1, 107]
    assert d._chat_metadata.last['input_tokens'] == 10 and d._chat_metadata.last['output_tokens'] == 2
    assert json.loads((tmp_path/'audit.jsonl').read_text())['seed'] == 123
    d._chat('next public turn', 124)
    assert ('messages', [{'role': 'user', 'content': 'normal physician role\n\nnext public turn'}]) in calls


def test_context_and_empty_generation_are_errors_not_outcomes(local_model, monkeypatch, tmp_path):
    d, _ = doctor(); calls = fake_loaded(d, monkeypatch, input_len=8190)
    monkeypatch.setenv('SHERPA_HF_DOCTOR_AUDIT', str(tmp_path/'audit.jsonl'))
    with pytest.raises(DoctorUnavailable, match='context budget'): d._chat('history', 1)
    assert not any(k == 'generate' for k, v in calls)
    fake_loaded(d, monkeypatch, reply='')
    with pytest.raises(DoctorUnavailable, match='empty reply'): d._chat('history', 1)
    rows = [json.loads(s) for s in (tmp_path/'audit.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and all(r['error'] for r in rows)


def test_rng_only_seeds_doctor_gpu_and_restores_on_failure():
    events = []
    @contextmanager
    def device(index): assert index == 0; yield
    cuda = SimpleNamespace(device=device,
        get_rng_state=lambda index: 'old-state', set_rng_state=lambda state, index: events.append(('restore', state, index)),
        default_generators=[SimpleNamespace(manual_seed=lambda seed: events.append(('seed', seed))), None])
    torch = SimpleNamespace(cuda=cuda, device=lambda d: SimpleNamespace(index=0))
    with pytest.raises(RuntimeError):
        with H.doctor_rng(torch, 'cuda:0', 72): raise RuntimeError('synthetic generation failure')
    assert events == [('seed', 72), ('restore', 'old-state', 0)]


def test_need_is_shared_protocol_with_no_gold_or_private_leakage(local_model, monkeypatch):
    d, cfg = doctor()
    from src.online_rl.tree.doctor_need import attach
    raw = json.dumps({'public_reply': 'I need evidence for the options.',
        'private_need': {'status': 'present', 'current_uncertainty': 'Option distinction',
          'information_needed': 'Evidence supporting or opposing alternatives',
          'public_evidence': [{'turn_id': 'D1', 'quote': 'evidence for the options'}]}})
    calls = fake_loaded(d, monkeypatch, reply=raw); attach(d, cfg)
    case = {'case_id': 'case', 'question': 'Question', 'options': {'A': 'a', 'B': 'b'}, 'answer': 'SECRET_GOLD'}
    result = d.turn(case, [], 1, 4, 17)
    assert result['visible'] == 'I need evidence for the options.' and result['private_need']['need_id']
    assert 'SECRET_GOLD' not in str(calls)
    assert 'current_uncertainty' not in result['visible']
    from src.online_rl.tree.five_gpu import cache_openings
    cache_openings(d, local_model/'openings.jsonl', cfg['five_gpu']['opening_identity'])
    first = d.turn(case, [], 1, 4, 17); before = len(calls)
    assert d.turn(case, [], 1, 4, 19) == first and len(calls) == before


def test_shared_name_cannot_resume_from_other_doctor(local_model, monkeypatch):
    cfg = R.load_config(run_name='lr1e5')
    saved = {'config_sha256': R.sha256_json(cfg), 'source_sha256': 'code'}
    R.validate_resume_identity(saved, R.load_config(run_name='lr1e5'), 'code')
    monkeypatch.setenv('SHERPA_DOCTOR_BACKEND', 'ollama')
    monkeypatch.delenv('SHERPA_DOCTOR_RECEIPT', raising=False)
    monkeypatch.setattr('src.online_rl.tree.b200_doctor_setup.receipt_path', lambda: local_model/'absent')
    with pytest.raises(RuntimeError, match='Config/code changed'):
        R.validate_resume_identity(saved, R.load_config(run_name='lr1e5'), 'code')


def test_loader_is_local_bf16_frozen_and_dedicated_to_gpu0(local_model, monkeypatch):
    from contextlib import nullcontext
    import torch
    import transformers
    d, _ = doctor(); d._verified = True
    calls = []
    net = SimpleNamespace(eval=lambda: calls.append('eval'), requires_grad_=lambda v: calls.append(('grad', v)))
    tok = SimpleNamespace(get_chat_template=lambda: 'template')
    def load(path, **kwargs): calls.append(('model', path, kwargs)); return net
    monkeypatch.setattr(transformers.Gemma3ForConditionalGeneration, 'from_pretrained', load)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda path, **kw: calls.append(('tokenizer', kw)) or tok)
    monkeypatch.setattr(torch.random, 'fork_rng', lambda **kw: calls.append(('fork_rng', kw)) or nullcontext())
    d._load(); d._load()
    model_calls = [v for v in calls if isinstance(v, tuple) and v[0] == 'model']
    assert len(model_calls) == 1
    assert model_calls[0][2] == {'local_files_only': True, 'torch_dtype': torch.bfloat16,
                                'device_map': {'': 'cuda:0'}, 'attn_implementation': 'sdpa'}
    assert ('grad', False) in calls and 'eval' in calls
    assert ('fork_rng', {'devices': [0]}) in calls


def test_doctor_runtime_version_and_byte_identity_are_enforced(local_model, monkeypatch):
    from importlib import metadata
    d, _ = doctor()
    monkeypatch.setattr(metadata, 'version', lambda _: '4.53.3')
    assert d.verify()['problems'] == []
    monkeypatch.setattr(metadata, 'version', lambda _: '4.51.1')
    assert 'Transformers version' in d.verify()['problems'][0]
    monkeypatch.setattr(metadata, 'version', lambda _: '4.53.3')
    (local_model/'model-01.safetensors').write_bytes(b'mutated')
    assert 'file identity changed' in d.verify()['problems'][0]
