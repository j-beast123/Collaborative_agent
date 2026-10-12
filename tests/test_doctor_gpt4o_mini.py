import io
import json
from pathlib import Path
import pytest

from src.online_rl.tree import openai_direct
from src.online_rl.doctor import make_doctor


@pytest.mark.parametrize('effort', [None, 'none'])
def test_gpt4o_mini_omits_reasoning_and_keeps_temperature(monkeypatch, effort):
    requests = []
    def send(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'A'}]}]}).encode())
    monkeypatch.setattr(openai_direct.urllib.request, 'urlopen', send)
    monkeypatch.setattr(openai_direct.api_usage, 'record', lambda *a, **k: None)
    raw, _ = openai_direct.responses_json('fake', 'gpt-4o-mini', 'system', 'user',
        {'max_tokens': 32, 'temperature': 0.4, 'reasoning_effort': effort}, None, 'doctor', 'doctor')
    assert raw == 'A'
    assert requests[0]['model'] == 'gpt-4o-mini'
    assert 'reasoning' not in requests[0]
    assert requests[0]['temperature'] == 0.4


def test_new_doctor_has_distinct_protocol_and_experiment_directory():
    import copy
    cfg = json.loads(Path('configs/sherpa.json').read_text())
    ro = cfg['rollout']
    system = Path(ro['doctor_system_prompt_file']).read_text()
    current = make_doctor(ro, system)
    legacy = copy.deepcopy(ro)
    legacy['doctor_model'] = legacy['doctor_backend']['model'] = 'gpt-6-luna'
    legacy['doctor_backend']['reasoning_effort'] = 'none'
    assert current.protocol != make_doctor(legacy, system).protocol
    assert current.model == 'gpt-4o-mini'
    assert cfg['prompt_optimization']['output_dir'] != 'outputs/sherpa_prompt_auc_v1'
    assert cfg['judges']['scorer_model'] == 'gpt-6-luna'
