"""Only the two named provider environment variables; never credential files."""
import os

ALLOWED = ('OPENROUTER_API_KEY', 'OPENAI_API_KEY')


def read_api_key(*, variable):
    if variable not in ALLOWED:
        raise ValueError('Unsupported credential environment variable')
    value = os.environ.get(variable)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(variable + ' is missing or blank in this Python process; export it in the launching terminal')
    return value.strip()


def status():
    return {name: {'present':name in os.environ, 'nonempty':bool(os.environ.get(name, '').strip()),
                   'source':'process_environment', 'provider_validated':False} for name in ALLOWED}
