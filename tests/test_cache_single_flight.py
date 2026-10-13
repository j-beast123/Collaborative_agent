"""CallCache single-flight: concurrent same-key requests run the underlying call once (and record API usage once); different
keys stay parallel; a failed leader stores nothing, releases its waiters, and the retry runs the call exactly once more;
pre-existing records hit without any call; key, stored record and return value are those of the plain cache."""
import json, threading, time
from src.common import usage
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cache import CallCache


def slow(counter, value, delay=0.3, gate=None):
    def fn():
        with counter['lock']: counter['n'] += 1
        if gate is not None: gate.wait(5)
        time.sleep(delay); return value
    return fn


def counter(): return {'n': 0, 'lock': threading.Lock()}


def run_threads(targets):
    out, errs = [None] * len(targets), [None] * len(targets)
    def wrap(i, f):
        try: out[i] = f()
        except BaseException as e: errs[i] = e
    ts = [threading.Thread(target=wrap, args=(i, f)) for i, f in enumerate(targets)]
    for t in ts: t.start()
    for t in ts: t.join(10)
    assert not any(t.is_alive() for t in ts), 'a thread is stuck'
    return out, errs


def test_same_key_three_concurrent_calls_run_fn_once(tmp_path):
    c = CallCache(tmp_path / 'calls.jsonl'); n = counter(); ident = {'node': 'x/0', 'seed': 1}
    out, errs = run_threads([lambda: c.call('rerun', ident, slow(n, {'outcome': 1.0}))] * 3)
    assert errs == [None] * 3 and n['n'] == 1 and out == [{'outcome': 1.0}] * 3
    lines = [json.loads(l) for l in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert lines == [{'key': sha256_json(['rerun', ident]), 'kind': 'rerun', 'identity': ident, 'value': {'outcome': 1.0}}]
    assert c.calls == {'rerun': 1} and c.hits == {'rerun': 2}


def test_different_keys_still_run_in_parallel(tmp_path):
    c = CallCache(None); n = counter(); t0 = time.time()
    out, errs = run_threads([lambda i=i: c.call('rerun', {'seed': i}, slow(n, i, delay=0.5)) for i in range(4)])
    assert errs == [None] * 4 and out == [0, 1, 2, 3] and n['n'] == 4
    assert time.time() - t0 < 1.5, 'different keys were serialised'


def test_failed_leader_stores_nothing_releases_waiters_and_retry_runs_once(tmp_path):
    c = CallCache(tmp_path / 'calls.jsonl'); calls = counter(); started = threading.Event(); release = threading.Event()
    def failing():
        with calls['lock']: calls['n'] += 1
        started.set(); release.wait(5); raise RuntimeError('429 exhausted')
    leader = threading.Thread(target=lambda: errs.append(_try(lambda: c.call('doctor', {'k': 1}, failing))))
    errs = []; leader.start(); started.wait(5)
    ok = counter()                                                 # two waiters queue behind the failing leader
    out = []; ws = [threading.Thread(target=lambda: out.append(c.call('doctor', {'k': 1}, slow(ok, 'fine', 0.2)))) for _ in range(2)]
    for w in ws: w.start()
    time.sleep(0.2); assert calls['n'] == 1 and ok['n'] == 0     # waiters did not call while the leader ran
    release.set(); leader.join(5)
    for w in ws: w.join(5)
    assert isinstance(errs[0], RuntimeError) and not any(w.is_alive() for w in ws)
    assert ok['n'] == 1 and out == ['fine', 'fine']             # one retry call, both waiters get its value
    assert [json.loads(l)['value'] for l in (tmp_path / 'calls.jsonl').read_text().splitlines()] == ['fine']
    assert c._inflight == {}


def _try(f):
    try: f()
    except BaseException as e: return e


def test_failure_without_waiters_then_sequential_retry(tmp_path):
    c = CallCache(tmp_path / 'calls.jsonl'); n = counter()
    def bad():
        with n['lock']: n['n'] += 1
        raise ValueError('boom')
    assert isinstance(_try(lambda: c.call('feedback', {'leaf': 'a'}, bad)), ValueError)
    assert not (tmp_path / 'calls.jsonl').exists() and c._inflight == {}
    assert c.call('feedback', {'leaf': 'a'}, slow(n, 'ok', 0)) == 'ok' and n['n'] == 2


def test_pre_existing_cache_hit_makes_no_call(tmp_path):
    p = tmp_path / 'calls.jsonl'; ident = {'node': 'x/1'}
    CallCache(p).call('agent', ident, lambda: {'completion_text': 'hi'})
    before = p.read_text(); c = CallCache(p); n = counter()
    out, errs = run_threads([lambda: c.call('agent', ident, slow(n, 'NEW'))] * 3)
    assert out == [{'completion_text': 'hi'}] * 3 and n['n'] == 0 and p.read_text() == before


def test_api_usage_recorded_once_for_concurrent_same_key(tmp_path, monkeypatch):
    log = tmp_path / 'api_usage.jsonl'; monkeypatch.setenv(usage.ENV, str(log))
    c = CallCache(None); n = counter()
    def api_call():
        with n['lock']: n['n'] += 1
        time.sleep(0.3); usage.record('tree_response_rubric_score_applicability', 'openai/gpt-5.6-luna', {'prompt_tokens': 10, 'completion_tokens': 2})
        return {'applicable': True}
    out, errs = run_threads([lambda: c.call('rubric_score', {'node': 'x/2', 'rubrics': ['r']}, api_call)] * 3)
    assert errs == [None] * 3 and n['n'] == 1
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['purpose'] == 'tree_response_rubric_score_applicability'


def test_record_and_value_identical_to_plain_cache(tmp_path):
    ident = {'rubric': 'r', 'node': 'x/3', 'level': '2', 'policy': 'p', 'text': 't', 'seed': 7}
    c = CallCache(tmp_path / 'a.jsonl'); v = c.call('rerun', ident, lambda: {'final_option': 'B', 'outcome': 0.0})
    line = (tmp_path / 'a.jsonl').read_text()
    assert v == {'final_option': 'B', 'outcome': 0.0}
    assert line == json.dumps({'key': sha256_json(['rerun', ident]), 'kind': 'rerun', 'identity': ident, 'value': v}, ensure_ascii=False) + '\n'
