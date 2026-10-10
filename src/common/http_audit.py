"""Optional metadata-only ledger for every physical JSON HTTP attempt."""
import json
import os
from pathlib import Path
import time
import urllib.request
from src.common import usage


def request_json(request, timeout):
    started = time.monotonic()
    row = {'case_id': usage._case.get(), 'role': usage.current_stage(), 'http_status': None,
           'input_tokens': None, 'output_tokens': None, 'cost_usd': None}
    try:
        from src.common.api_limiter import request_slot
        with request_slot() as slot:
            row.update(slot)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                row['http_status'] = getattr(response, 'status', 200)
                body = json.loads(response.read().decode())
        if isinstance(body, dict):
            meta = body.get('usage') or {}
            row.update(request_id=body.get('id'), model=body.get('model'), provider_error=bool(body.get('error')),
                       input_tokens=meta.get('prompt_tokens', body.get('prompt_eval_count')),
                       output_tokens=meta.get('completion_tokens', body.get('eval_count')), cost_usd=meta.get('cost'))
        return body
    except Exception as exc:
        row.update(exception_type=type(exc).__name__, http_status=getattr(exc, 'code', row['http_status']))
        raise
    finally:
        path = os.environ.get('SHERPA_HTTP_AUDIT')
        if path:
            row.update(seconds=time.monotonic() - started, unix_time=time.time())
            # Never serialize request headers, API keys, prompts or error bodies.
            with usage._lock:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'a') as stream: stream.write(json.dumps(row) + '\n')
