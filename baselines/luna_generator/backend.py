"""Isolated Luna-generator transport; old running backend/source hashes stay unchanged.

Retains original bounded retry/audit semantics, adding strict parameter support
and incomplete-response checks for this new profile only.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from src.judges.llm import ChatResult, OpenAICompatibleBackend, RETRYABLE_HTTP, _retry_delay
from src.online_rl.integrated.records import sha256_json
from baselines.parallel_api.credentials import read_api_key
from baselines.parallel_api.io import append_json


class ModelIdentityError(ValueError):
    pass


class LunaGeneratorBackend(OpenAICompatibleBackend):
    def __init__(self, *, audit_path, request_model=None, accepted_response_models=None, **kwargs):
        super().__init__(**kwargs)
        if not self.revision:
            raise ValueError('A configured model identity is required before requests')
        self.accepted_response_models=accepted_response_models or [self.revision]
        self.audit_path = Path(audit_path)
        self.request_model = request_model or self.revision

    def identity(self):
        return {**super().identity(), 'request_model': self.request_model, 'accepted_response_models': self.accepted_response_models,
                'provider_parameters': {'allow_fallbacks': False, 'require_parameters': True}}

    def complete(self, request):
        key = read_api_key(variable=self.api_key_env)
        params = dict(request.params)
        if 'openrouter.ai' in self.base_url:
            params.update(provider={'allow_fallbacks': False, 'require_parameters': True}, usage={'include': True})
        # Explicit routing ID and exact expected response ID are both bound.
        # An alias is permitted only when deliberately configured; no substitution.
        payload = {'model': self.request_model, 'messages': [{'role':'system','content':request.system},
                   {'role':'user','content':request.user}], **params}
        began = time.time()
        for attempt in range(1, self.max_retries + 2):
            started = time.time()
            row = {'started_at': started, 'request_sha256': sha256_json(payload), 'kind': request.prompt_version,
                   'provider': self.provider, 'base_url': self.base_url, 'requested_model': self.request_model, 'model_family': self.model, 'declared_revision': self.revision, 'attempt': attempt,
                   'cost_usd': None, 'usage': None, 'request_id': None}
            delay = 0
            try:
                req = urllib.request.Request(self.base_url + '/chat/completions', data=json.dumps(payload).encode(),
                    headers={'Authorization':'Bearer ' + key.strip(), 'Content-Type':'application/json'}, method='POST')
                with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                    body = json.load(response)
                    row['request_id'] = body.get('id') or response.headers.get('x-request-id')
                row.update(response_model=body.get('model'), usage=body.get('usage'),
                           cost_usd=(body.get('usage') or {}).get('cost'))
                if body.get('model') not in self.accepted_response_models:
                    raise ModelIdentityError('Returned model does not match the configured accepted model IDs')
                details = (body.get('usage') or {}).get('completion_tokens_details') or {}
                if (details.get('reasoning_tokens') or 0) > 0:
                    raise ModelIdentityError('Provider returned reasoning tokens despite requested effort none')
                choice = body['choices'][0]
                if choice.get('finish_reason') not in (None, 'stop'):
                    raise ValueError('Incomplete generator response: ' + str(choice.get('finish_reason')))
                content = choice['message']['content']
                if not isinstance(content, str):
                    raise ValueError('Provider returned no text')
                row['status'] = 'ok'
            except urllib.error.HTTPError as exc:
                row.update(status='failed', error=f'HTTP {exc.code}', request_id=exc.headers.get('x-request-id'))
                # Retain bounded error response; API keys are never included in the request log.
                row['error_response'] = exc.read(8192).decode(errors='replace').replace(key, '[REDACTED]')
                delay = _retry_delay(exc.headers.get('Retry-After'), attempt) if exc.code in RETRYABLE_HTTP else None
            except ModelIdentityError as exc:
                row.update(status='failed', error=str(exc))
                delay = None
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                delay = _retry_delay(None, attempt)
            row['latency_s'] = time.time() - started
            append_json(self.audit_path, row)
            if row['status'] == 'ok':
                result = ChatResult('ok', content, self.provider, row.get('response_model') or self.model,
                    self.revision, row.get('usage') or {}, time.time()-began, attempt, started_at=began)
                return result
            if delay is None or attempt > self.max_retries:
                return ChatResult('failed', None, self.provider, self.model, self.revision, {}, time.time()-began,
                                  attempt, error=row['error'], started_at=began)
            self.sleep(min(delay, 60))

