import io
import json
import urllib.error
from pathlib import Path
import pytest

from src.online_rl.doctor import make_doctor, DoctorUnavailable
from src.online_rl import local_doctor
from src.online_rl.sherpa.config import config_problems


def doctor_config():
    cfg = json.loads(Path('configs/ours.json').read_text())
    return cfg, make_doctor(cfg['rollout'], Path(cfg['rollout']['doctor_system_prompt_file']).read_text())


def test_local_payload_and_unchanged_identity(monkeypatch):
    cfg, doctor = doctor_config()
    assert not config_problems(cfg, Path.cwd())
    assert doctor.model == 'Qwen/Qwen3-8B'
    assert doctor.identity['engine'] == 'vllm'
    original = doctor.protocol
    doctor.local_endpoint = 'http://127.0.0.1:1234'
    captured = []
    def send(request, **kwargs):
        captured.append(json.loads(request.data))
        assert 'Authorization' not in request.headers
        return io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"A"}}]}')
    monkeypatch.setattr(local_doctor.urllib.request, 'urlopen', send)
    schema = {'type':'object','properties':{}}
    assert doctor._chat('user', 10, 32, 'override', schema) == 'A'
    payload = captured[0]
    assert payload['seed'] == 10 and payload['temperature'] == 0.4
    assert payload['chat_template_kwargs'] == {'enable_thinking':False}
    assert payload['response_format']['json_schema']['schema'] == schema
    assert payload['messages'][0]['content'] == 'override'
    assert doctor.protocol == original


def test_local_rate_limit_is_not_retried(monkeypatch):
    _, doctor = doctor_config()
    doctor.local_endpoint = 'http://127.0.0.1:1234'
    calls = []
    def send(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('http://localhost',429,'limit',{},io.BytesIO())
    monkeypatch.setattr(local_doctor.urllib.request, 'urlopen', send)
    with pytest.raises(urllib.error.HTTPError): doctor._chat('user',1)
    assert len(calls) == 1


def test_local_incomplete_is_not_accepted(monkeypatch):
    _, doctor = doctor_config()
    doctor.local_endpoint = 'http://127.0.0.1:1234'
    monkeypatch.setattr(local_doctor.urllib.request, 'urlopen',lambda *a, **k:
        io.BytesIO(b'{"choices":[{"finish_reason":"length","message":{"content":"partial"}}]}'))
    with pytest.raises(DoctorUnavailable) as caught: doctor._chat('user',1)
    assert caught.value.audit['raw'] == 'partial'
    assert caught.value.audit['message']['content'] == 'partial'
    assert caught.value.audit['max_tokens'] == 256
    assert caught.value.audit['seed'] == 1
    assert 'finish_reason=length' in str(caught.value)


def test_new_prompt_experiment_is_isolated():
    cfg, doctor = doctor_config()
    old = json.loads(Path('configs/sherpa.json').read_text())
    assert old['rollout']['doctor_model'] == 'gpt-4o-mini'
    assert cfg['prompt_optimization']['final_artifact'] != old['prompt_optimization']['final_artifact']
    assert cfg['training_execution']['rollout_replicas'] == 1
    assert cfg['doctor_execution']['visible_gpu_index'] == 1
    assert cfg['training_execution']['assistant_max_inflight'] == 128
    assert cfg['execution']['doctor_workers'] == 128
    assert cfg['execution']['api_limits']['doctor'] == 128


def test_only_doctor_server_uses_compact_json(monkeypatch, tmp_path):
    import src.online_rl.vllm_policy as module
    cfg, doctor = doctor_config()
    original = json.loads(json.dumps(cfg))
    captured = []
    def server(server_cfg, root):
        captured.append(server_cfg)
        from types import SimpleNamespace
        return SimpleNamespace(urls=['http://localhost:1234'])
    monkeypatch.setattr(module, 'VLLMPolicy', server)
    local_doctor.start_server(cfg, Path.cwd(), tmp_path, doctor, '1')
    assert cfg == original
    spec = captured[0]['prompt_optimization']
    assert spec['assistant_max_inflight'] == 128
    assert spec['vllm']['max_num_seqs'] == 128
    command = module.server_command('python', '/weights', 1234, spec)
    settings = json.loads(command[command.index('--structured-outputs-config') + 1])
    assert settings == {'backend': 'xgrammar', 'disable_any_whitespace': True}
    assert doctor.identity['structured_outputs'] == settings
    assert 'structured_outputs' not in cfg['prompt_optimization']['vllm']
