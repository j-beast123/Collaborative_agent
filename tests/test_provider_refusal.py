"""A provider content-filter refusal (finish_reason content_filter, null content) is deterministic for a request: it is not
retried, and behavior feedback records it as INVALID feedback, so the trajectory gets no rubric mapping (like leaked feedback)."""
import io
import json

import pytest

from src.collaboration import generator as G
from src.online_rl.tree.backends import api_behavior_feedback

DIALOGUE = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'I think B.'}, {'speaker': 'agent', 'stage': 'A1', 'round': 1, 'text': 'Consider the timeline.'},
            {'speaker': 'doctor', 'stage': 'D2', 'round': 2, 'text': 'Noted.'}, {'speaker': 'agent', 'stage': 'A2', 'round': 2, 'text': 'It still points elsewhere.'}]
VIEW = {'question': 'q', 'options': {'A': 'alpha', 'B': 'beta'}, 'scenario': ''}


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def api(bodies, calls):
    def urlopen(req, timeout=None):
        calls.append(json.loads(req.data.decode())); return Response(json.dumps(bodies[min(len(calls), len(bodies)) - 1]).encode())
    return urlopen


REFUSAL = {'choices': [{'finish_reason': 'content_filter', 'native_finish_reason': 'refusal', 'message': {'content': None}}], 'provider': 'P', 'usage': {}}
ok = lambda text: {'choices': [{'finish_reason': 'stop', 'message': {'content': text}}], 'usage': {}}


def test_refusal_raises_provider_refusal_without_retry(monkeypatch):
    calls = []; monkeypatch.setattr(G.urllib.request, 'urlopen', api([REFUSAL], calls))
    with pytest.raises(G.ProviderRefusal) as e: G.doctor_call('k', 'm', 's', 'u', 16)
    assert len(calls) == 1 and e.value.finish_reason == 'content_filter' and e.value.native_finish_reason == 'refusal'


def test_normal_completion_is_unchanged(monkeypatch):
    calls = []; monkeypatch.setattr(G.urllib.request, 'urlopen', api([ok('  hello  ')], calls))
    assert G.doctor_call('k', 'm', 's', 'u', 16) == 'hello' and len(calls) == 1


def test_refused_feedback_is_invalid_and_not_substituted(monkeypatch):
    calls = []; monkeypatch.setattr(G.urllib.request, 'urlopen', api([REFUSAL], calls))
    fb = api_behavior_feedback('k', 'm', 'system', 64, protocol='behavior_observer_v5')(VIEW, DIALOGUE, 'failure')
    assert fb['valid'] is False and fb['status'] == 'provider_refusal' and fb['text'] == '' and fb['target_reply_id'] is None
    assert fb['provider_refusal']['finish_reason'] == 'content_filter' and len(calls) == 1


def test_refusal_on_the_repair_call_is_also_invalid(monkeypatch):
    calls = []                                                       # first answer has no locus line -> one repair request -> refused
    monkeypatch.setattr(G.urllib.request, 'urlopen', api([ok('The Agent kept its position.'), REFUSAL], calls))
    fb = api_behavior_feedback('k', 'm', 'system', 64, protocol='behavior_observer_v5')(VIEW, DIALOGUE, 'failure')
    assert fb['valid'] is False and fb['status'] == 'provider_refusal' and len(calls) == 2 and fb['raw'] == ['The Agent kept its position.']


def test_refused_feedback_is_excluded_from_rubric_mapping():
    from src.online_rl.tree.posthoc import DoctorFeedback
    f = DoctorFeedback('c/0.1', 0.0, '', 'sha', None, {'status': 'provider_refusal', 'raw': [], 'leakage': {}}, False, 'behavior_observer_v5')
    assert [x for x in [f] if getattr(x, 'valid', True)] == []        # the trainer's rule for trajectories that get a rubric mapping
