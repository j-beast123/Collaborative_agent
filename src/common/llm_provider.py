"""Explicit API routing. Keys are read from the current environment only."""
import os
import re

ENDPOINTS = {
    'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
    'openai': 'https://api.openai.com/v1/chat/completions',
}
KEY_ENVS = {'openrouter': 'OPENROUTER_API_KEY', 'openai': 'OPENAI_API_KEY'}


def key_env(provider):
    if provider not in ENDPOINTS:
        raise ValueError(f'Unsupported API provider: {provider!r}')
    return KEY_ENVS[provider]


def load_key(provider):
    name = key_env(provider)
    value = os.environ.get(name, '').strip()
    if not value:
        raise ValueError(f'{name} is missing in this terminal')
    return value


def model_matches(requested, returned, provider):
    # The direct API may resolve an alias to a dated snapshot of that same model.
    return returned == requested or (provider == 'openai' and isinstance(returned, str)
        and re.fullmatch(re.escape(requested) + r'-\d{4}-\d{2}-\d{2}', returned) is not None)
