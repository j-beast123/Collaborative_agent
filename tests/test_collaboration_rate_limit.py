from email.utils import formatdate

from src.collaboration import generator as G


def test_gemma_overload_backoff_and_retry_after(monkeypatch):
    assert [G._http_retry_delay('google/gemma-3-12b-it', 429, i, None) for i in range(7)] == [10, 20, 40, 60, 60, 60, 60]
    assert G._http_retry_delay('google/gemma-3-12b-it', 429, 0, '30') == 30
    monkeypatch.setattr(G.time, 'time', lambda: 1000)
    assert G._http_retry_delay('google/gemma-3-12b-it', 429, 0, formatdate(1090, usegmt=True)) == 90
    assert G._http_retry_delay('google/gemma-3-12b-it', 429, 0, 'invalid') == 10
    assert G._http_retry_delay('openai/test', 429, 0, None) == 2


def test_gemma_requests_are_paced_without_delaying_other_models(monkeypatch):
    monkeypatch.setenv('OPENROUTER_GEMMA_MIN_INTERVAL', '4')
    clock = [100.0]
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(G.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(G.time, 'sleep', sleep)
    monkeypatch.setattr(G, '_LAST_REQUEST', {})
    G._pace_request('google/gemma-3-12b-it')
    clock[0] += 1
    G._pace_request('google/gemma-3-12b-it')
    G._pace_request('openai/test')
    assert sleeps == [3]
