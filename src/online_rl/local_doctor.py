"""Frozen Qwen Doctor, isolated from the trainable assistant's LoRA weights."""
import copy
import json
import threading
import urllib.request

_LOCAL_DOCTOR_CAPACITY = threading.BoundedSemaphore(128)


def chat(doctor, user, seed, max_tokens=None, system_override=None, format_schema=None):
    from src.online_rl.doctor import DoctorUnavailable
    from src.common.api_slots import lease
    from src.common.live_gpu import activity
    endpoint = getattr(doctor, 'local_endpoint', None)
    if not endpoint:
        raise DoctorUnavailable('Local Qwen Doctor server has not been started')
    payload = {'model': 'sherpa-base', 'messages': [
        {'role': 'system', 'content': doctor.system if system_override is None else system_override},
        {'role': 'user', 'content': user}],
        'temperature': doctor.options['temperature'], 'top_p': 1.0, 'top_k': -1,
        'seed': seed, 'max_tokens': doctor.max_tokens if max_tokens is None else max_tokens,
        'chat_template_kwargs': {'enable_thinking': False}}
    if format_schema is not None:
        payload['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'doctor_response', 'strict': True, 'schema': format_schema}}
    request = urllib.request.Request(endpoint + '/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    # No transport retry, including 429. Format validation remains in Doctor.turn.
    with _LOCAL_DOCTOR_CAPACITY, lease(doctor.model, slot_name='doctor'), activity('local_doctor_generate'):
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.load(response)
    choice = body['choices'][0]
    text = choice.get('message', {}).get('content')
    if choice.get('finish_reason') != 'stop' or not isinstance(text, str) or not text.strip():
        error = DoctorUnavailable(
            f"Local Qwen Doctor returned an incomplete response "
            f"(finish_reason={choice.get('finish_reason')}, max_tokens={payload['max_tokens']})")
        error.audit = {'engine': 'vllm', 'model': doctor.model, 'response_id': body.get('id'),
                       'finish_reason': choice.get('finish_reason'), 'usage': body.get('usage'),
                       'max_tokens': payload['max_tokens'], 'seed': seed,
                       'raw': text, 'message': choice.get('message')}
        raise error
    return text.strip()


def start_server(cfg, root, home, doctor, physical_device):
    from src.online_rl.vllm_policy import VLLMPolicy
    spec = cfg['doctor_execution']
    server_cfg = copy.deepcopy(cfg)
    server_cfg['doctor_execution']['vllm']['structured_outputs'] = {
        'backend': 'xgrammar', 'disable_any_whitespace': True}
    server_cfg['policy']['local_model_path'] = cfg['rollout']['doctor_backend']['local_model_path']
    server_cfg['prompt_optimization'].update(
        output_dir=str(home / 'doctor_server'), rollout_replicas=1,
        device_ids=[physical_device], assistant_max_inflight=cfg['execution']['doctor_workers'],
        vllm=server_cfg['doctor_execution']['vllm'])
    server = VLLMPolicy(server_cfg, root)
    doctor.local_endpoint = server.urls[0]
    return server
