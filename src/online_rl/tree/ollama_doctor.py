"""Local Doctor transport shared by source dialogue, feedback and intervention FINAL."""
import json
import os
from pathlib import Path
import urllib.request


class OllamaDoctorCall:
    def __init__(self, settings):
        self.settings = dict(settings)
        self.endpoint = settings['url'].rstrip('/')
        self.model = settings['model']
        with urllib.request.urlopen(self.endpoint + '/api/tags', timeout=10) as response:
            models = json.load(response)['models']
        actual = next((m for m in models if m['name'] == self.model), None)
        if actual is None or actual['digest'] != settings['digest']:
            raise ValueError('Local Doctor model missing or digest differs from pinned config')

    def __call__(self, key, model, system, user, max_tokens, temperature=0.4):
        if model != 'ollama/' + self.model:
            raise ValueError('Doctor model identifier differs from local transport')
        # Gemma supports one user role; fold the system instructions into the
        # user message explicitly rather than relying on server role rewriting.
        payload = {'model': self.model, 'stream': False,
                   'messages': [{'role': 'user', 'content': system + '\n\n' + user}],
                   'options': {'temperature': temperature, 'num_predict': max_tokens,
                               'num_ctx': self.settings.get('num_ctx', 16384), 'num_gpu': -1},
                   'keep_alive': '30m'}
        request = urllib.request.Request(self.endpoint + '/api/chat', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
        text = result.get('message', {}).get('content', '').strip()
        complete = result.get('done') is True and result.get('done_reason') != 'length' and bool(text)
        log = os.environ.get('API_USAGE_LOG')
        if log:
            path = Path(log).parent / 'doctor_calls_local.jsonl'
            row = {'model': model, 'digest': self.settings['digest'], 'complete': complete,
                   **{k: result.get(k) for k in ('done_reason', 'total_duration', 'load_duration',
                                                'prompt_eval_count', 'eval_count', 'eval_duration')}}
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as out:
                out.write(json.dumps(row) + '\n')
        if not complete:
            raise ValueError('Local Doctor returned an empty, incomplete or truncated response')
        return text
