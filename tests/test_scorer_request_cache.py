"""Opt-in scorer request cache (judges.scorer_request_cache = canonical_request_sha256_v1): identical full requests share one
sampled answer, any difference in the request (including the reply) is a separate request, and frozen runs are unchanged."""
import json
import threading
import time
from pathlib import Path

import pytest

from src.online_rl.tree import experiment as X
from src.online_rl.tree.backends import two_stage_structured_scorer
from src.online_rl.tree.luna import REQUEST_CACHE, LunaDecisionClient, request_key

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = json.loads((ROOT / 'tests/fixtures/dev_opt_visible_rubric.json').read_text())
VIEW = {'question': 'q', 'options': {'A': 'a', 'B': 'b'}}
PREFIX = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'I think A.'}]


def flipping_post(calls):
    """A nondeterministic scorer: applicability alternates True/False on every real Stage-1 request."""
    def post(url, payload, key, timeout):
        calls.append(payload); stage = json.loads(payload['messages'][-1]['content'])['stage']; rid = RUBRIC['rubric_id']
        n1 = sum(json.loads(p['messages'][-1]['content'])['stage'] == 'applicability' for p in calls)
        ans = {rid: {'applicable': n1 % 2 == 1, 'evidence_basis': 'e', 'missing_requirement': ''}} if stage == 'applicability' \
            else {rid: {'level': '1', 'anchor_basis': 'x'}}
        return {'choices': [{'message': {'content': json.dumps({'answers': ans})}}], 'usage': {}}
    return post


def score(client, reply):
    return two_stage_structured_scorer(client, observed_context=True)(VIEW, PREFIX, [RUBRIC], reply)[RUBRIC['rubric_id']]['applicable']


def test_without_cache_identical_stage1_requests_can_disagree():
    calls = []; c = LunaDecisionClient(model='m', key='k', system='s', post=flipping_post(calls))
    assert {score(c, 'reply one'), score(c, 'reply two')} == {True, False}


def test_cache_makes_siblings_share_stage1_but_not_stage2():
    calls = []; c = LunaDecisionClient(model='m', key='k', system='s', post=flipping_post(calls), request_cache=REQUEST_CACHE)
    assert score(c, 'reply one') is True and score(c, 'reply two') is True and score(c, 'reply one') is True
    stages = [json.loads(p['messages'][-1]['content'])['stage'] for p in calls]
    assert stages.count('applicability') == 1                       # one real Stage-1 request for the shared state
    assert stages.count('level') == 2                                # different replies are different Stage-2 requests
    assert c.cache_hits == 3


def test_request_key_is_canonical_and_covers_every_field():
    p = {'model': 'm', 'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 5}
    assert request_key(p) == request_key(dict(reversed(list(p.items()))))
    for change in ({'model': 'n'}, {'max_tokens': 6}, {'messages': [{'role': 'user', 'content': 'y'}]}):
        assert request_key({**p, **change}) != request_key(p)


def test_concurrent_identical_requests_are_single_flight():
    calls = []

    def slow(url, payload, key, timeout):
        calls.append(1); time.sleep(0.05)
        return {'choices': [{'message': {'content': 'answer'}}], 'usage': {}}
    c = LunaDecisionClient(model='m', key='k', system='s', post=slow, request_cache=REQUEST_CACHE)
    out = []
    threads = [threading.Thread(target=lambda: out.append(c._chat([{'role': 'user', 'content': 'same'}], {}, 'p'))) for _ in range(6)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert out == ['answer'] * 6 and len(calls) == 1


def test_failed_request_is_not_cached():
    calls = []

    def fail_once(url, payload, key, timeout):
        calls.append(1)
        if len(calls) == 1: raise ValueError('boom')
        return {'choices': [{'message': {'content': 'ok'}}], 'usage': {}}
    c = LunaDecisionClient(model='m', key='k', system='s', post=fail_once, request_cache=REQUEST_CACHE)
    with pytest.raises(ValueError): c._chat([{'role': 'user', 'content': 'x'}], {}, 'p')
    assert c._chat([{'role': 'user', 'content': 'x'}], {}, 'p') == 'ok'


def test_cache_changes_protocol_and_fingerprint_only_when_enabled():
    off = LunaDecisionClient(model='m', key='k', system='s'); on = LunaDecisionClient(model='m', key='k', system='s', request_cache=REQUEST_CACHE)
    assert off.protocol_sha256 != on.protocol_sha256
    with pytest.raises(ValueError): LunaDecisionClient(model='m', key='k', system='s', request_cache='other')
    cfg = X.load_config('configs/experiments/reward_prompt_matrix/production/r05_o05_evolving_300.json')
    assert X.fingerprint(cfg) == '093ad76d54316ad9ef3e9fd44c6bdbff76672f2454e0952a67a5728984459637'
    cfg['judges']['scorer_request_cache'] = REQUEST_CACHE
    assert X.fingerprint(cfg) != '093ad76d54316ad9ef3e9fd44c6bdbff76672f2454e0952a67a5728984459637'


def test_no_frozen_or_dev_config_enables_the_cache():
    for path in sorted((ROOT / 'configs/experiments').rglob('*.json')):
        assert 'scorer_request_cache' not in path.read_text(), path
