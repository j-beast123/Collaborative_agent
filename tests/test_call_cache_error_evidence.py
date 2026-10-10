import json
import pytest

from src.common.config import IntegrationUnavailable
from src.online_rl.tree.cache import CallCache


def test_gate_failure_audit_is_saved_separately_and_never_cached_as_a_result(tmp_path):
    path = tmp_path / 'calls.jsonl'
    cache = CallCache(path)
    error = IntegrationUnavailable('invalid quote after bounded correction')
    error.audit = {'requests': [{'messages': [{'role': 'user', 'content': 'visible input'}]}],
                   'decisions': [{'raw': ['not a literal quote']}],
                   'validation_errors': ['quote mismatch']}
    def fail():
        raise error
    with pytest.raises(IntegrationUnavailable) as caught:
        cache.call('posthoc_gate', {'rubric': 'r'}, fail)
    assert caught.value is error
    assert not cache.store and not path.exists()
    records = [json.loads(line) for line in path.with_suffix('.errors.jsonl').read_text().splitlines()]
    assert len(records) == 1 and records[0]['audit'] == error.audit
    assert records[0]['kind'] == 'posthoc_gate'
    assert records[0]['exception_type'] == 'IntegrationUnavailable'
    # Retrying repeats the operation; it cannot retrieve a failed pseudo-decision.
    assert cache.call('posthoc_gate', {'rubric': 'r'}, lambda: {'passed': True}) == {'passed': True}
    assert len(path.read_text().splitlines()) == 1
    assert len(cache.store) == 1


def test_audit_write_failure_preserves_the_original_exception_and_releases_single_flight(tmp_path):
    path = tmp_path / 'calls.jsonl'
    path.with_suffix('.errors.jsonl').mkdir()
    cache = CallCache(path)
    error = IntegrationUnavailable('original gate failure')
    error.audit = {'attempts': 2}
    def fail():
        raise error
    with pytest.raises(IntegrationUnavailable) as caught:
        cache.call('posthoc_gate', {}, fail)
    assert caught.value is error and not cache._inflight and not cache.store
    assert any('Failed to persist error audit' in note for note in error.__notes__)
