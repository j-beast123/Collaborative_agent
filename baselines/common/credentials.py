"""Environment-only credentials. Never print, persist, hash, or probe key values."""
import json
import os
import sys

API_KEY_ENV='OPENROUTER_API_KEY'


def read_api_key(*,variable=API_KEY_ENV,environ=None):
    env=os.environ if environ is None else environ
    value=env.get(variable)
    if not isinstance(value,str) or not value.strip():
        # Do not interpolate a configured variable name: a mistaken config could
        # contain a credential value rather than its environment-variable name.
        raise RuntimeError('OPENROUTER_API_KEY is missing or blank in this Python process; export it in the terminal that launches preflight/train')
    return value.strip()


def credential_status(environ=None):
    env=os.environ if environ is None else environ
    value=env.get(API_KEY_ENV)
    return {'source':'process_environment','variable':API_KEY_ENV,
        'present':API_KEY_ENV in env,'nonempty':isinstance(value,str) and bool(value.strip()),
        'provider_validated':False}


if __name__=='__main__':
    status=credential_status()
    print(json.dumps({'python_executable':sys.executable,'credentials':status},indent=2))
    raise SystemExit(0 if status['nonempty'] else 1)
