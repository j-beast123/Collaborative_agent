"""Decode a whole JSON response, optionally inside one exact literal wrapper.

Never executes shell text, extracts arbitrary braces, or repairs JSON contents.
The caller must still apply the original rubric/grade schema validation.
"""
import json
import re

FENCE = re.compile(r'\A```(?:json)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*\Z', re.DOTALL | re.IGNORECASE)
HEREDOC = re.compile(r'''\Acat[ \t]+<<[ \t]*(['"]?)([A-Za-z_][A-Za-z0-9_]*)\1[ \t]*\r?\n(.*?)\r?\n\2[ \t]*\Z''', re.DOTALL)


def decode_json_response(text):
    if not isinstance(text, str):
        raise ValueError('Response is not text')
    text = text.strip(); normalization = 'none'
    fence = FENCE.fullmatch(text)
    heredoc = HEREDOC.fullmatch(text)
    if fence:
        text = fence.group(1); normalization = 'outer_json_fence_removed'
    elif heredoc:
        text = heredoc.group(3); normalization = 'outer_literal_heredoc_removed'
    return json.loads(text), normalization
