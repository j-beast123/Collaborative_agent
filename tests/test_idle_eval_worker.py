import json
import threading
from types import SimpleNamespace

import pytest

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import idle_eval_worker as worker
from src.online_rl.tree.idle_eval_priority import PriorityCoordinator


def manifest(cases):
    return {'job_sha256': 'job', 'policy_sha256': 'policy', 'completed_cases': 20,
            'snapshot_sha256': 'snapshot', 'eval_seed': 0, 'rollout': {'rounds': 2},
            'dataset_cases': len(cases), 'case_ids': [c['case_id'] for c in cases],
            'original_case_sha256': [sha256_json(c) for c in cases]}


def result(case, index):
    return {'job_sha256': 'job', 'policy_sha256': 'policy', 'case_id': case['case_id'],
            'original_case': case, 'index': index, 'outcome': 1, 'num_discussion_rounds': 2,
            'max_discussion_rounds': 2, 'final_option': 'A', 'seed': index, 'snapshot_sha256': 'snapshot'}


def test_claims_exclude_active_case_and_resume_only_missing_results(tmp_path):
    cases = [{'case_id': 'one', 'ground_truth': {'correct_option': 'A'}},
             {'case_id': 'two', 'ground_truth': {'correct_option': 'A'}}]
    info = manifest(cases)
    with worker.claim_case(tmp_path, cases, info) as first:
        assert first[0] == 0
        with worker.claim_case(tmp_path, cases, info) as second:
            assert second[0] == 1
            worker.atomic_write_json(second[2], result(cases[1], 1))
        with worker.claim_case(tmp_path, cases, info) as blocked:
            assert blocked is None
    with worker.claim_case(tmp_path, cases, info) as resumed:
        assert resumed[0] == 0
        worker.atomic_write_json(resumed[2], result(cases[0], 0))
    summary = worker.summarize(tmp_path, info)
    assert summary['status'] == 'complete'
    assert summary['evaluated'] == summary['correct'] == 2
    assert summary['mean_agent_turns'] == 2


def test_committed_original_case_and_adapter_are_validated(tmp_path):
    cases = [{'case_id': 'one', 'question': 'original'}]
    row = result(cases[0], 0)
    row['original_case'] = {**cases[0], 'question': 'changed'}
    worker.atomic_write_json(tmp_path / 'cases/case_000000.json', row)
    with pytest.raises(RuntimeError, match='original case'):
        worker.summarize(tmp_path, manifest(cases))


@pytest.mark.parametrize('field,value,error', [
    ('seed', 8, 'seed'), ('snapshot_sha256', 'another', 'snapshot'),
    ('num_discussion_rounds', 3, 'discussion'), ('outcome', 0, 'outcome')])
def test_committed_result_rejects_inconsistent_evaluation_evidence(field, value, error):
    case = {'case_id': 'one', 'ground_truth': {'correct_option': 'A'}}
    row = result(case, 0)
    row[field] = value
    with pytest.raises(RuntimeError, match=error):
        worker._validate_result(row, manifest([case]))


def test_partial_append_recovery_preserves_completed_native_call(tmp_path):
    path = tmp_path / 'calls.jsonl'
    cache = worker._cache(path)
    value = worker._cached(cache, 'agent', {'seed': 23}, lambda: {'text': 'saved'})
    with path.open('ab') as stream:
        stream.write(b'{"key":"torn')
    resumed = worker._cache(path)
    assert worker._cached(resumed, 'agent', {'seed': 23}, lambda: pytest.fail('repeated call')) == value
    assert path.read_bytes().endswith(b'\n')
    path.write_text('{"key":"corrupt"}\n')
    with pytest.raises(KeyError):
        worker._cache(path)


def test_completed_agent_and_doctor_calls_replay_after_mid_episode_revocation(tmp_path):
    count = {'agent': 0, 'doctor': 0}

    def generate(prompt, max_tokens, seed, segment='agent'):
        count['agent'] += 1
        return {'completion_text': f'{prompt}:{seed}', 'completion_ids': [seed]}

    def http(url, payload, timeout):
        count['doctor'] += 1
        return {'message': {'content': 'Doctor reply'}}

    policy = SimpleNamespace(
        _generate=generate, doctor=SimpleNamespace(http=http, protocol='doctor-v2'),
        agent_turn=lambda *a: {}, ST=SimpleNamespace(generate_with_trace=lambda *a: None),
        precommit=True, args=SimpleNamespace(generation_temperature=.7, top_p=.9, top_k=20),
        cfg={'rollout': {'agent_sampling_protocol': 'per_row_rng_batch_v1'}})
    coordinator = PriorityCoordinator(tmp_path / 'priority', idle_grace_seconds=0, require_training=False)
    residency = coordinator.evaluation_residency(0)
    calls = worker.EvaluationCalls(coordinator, residency, 0)
    cache_path = tmp_path / 'calls.jsonl'
    case = {'case_id': 'one'}
    info = manifest([case])
    _, restore = worker.install_call_cache(policy, calls, worker._cache(cache_path), info, case, 5)
    first = policy._generate('private assessment', 128, 11)
    doctor_reply = policy.doctor.http('http://private/api/chat', {'options': {'seed': 9}}, 60)
    calls.stopping = True
    with pytest.raises(worker.WorkerRevoked):
        policy._generate('visible reply', 400, 12)
    restore()
    calls.stopping = False
    _, restore = worker.install_call_cache(policy, calls, worker._cache(cache_path), info, case, 5)
    try:
        assert policy._generate('private assessment', 128, 11) == first
        assert policy.doctor.http('http://other-private-port/api/chat', {'options': {'seed': 9}}, 60) == doctor_reply
        policy._generate('visible reply', 400, 12)
    finally:
        restore()
        residency.release()
    assert count == {'agent': 2, 'doctor': 1}
    records = [json.loads(line) for line in cache_path.read_text().splitlines()]
    assert [r['identity'].get('seed') for r in records if r['kind'] == 'agent_segment'] == [11, 12]


def test_failed_native_doctor_is_stopped_before_call_lease_release(tmp_path):
    coordinator = PriorityCoordinator(tmp_path, idle_grace_seconds=0, require_training=False)
    residency = coordinator.evaluation_residency(0)
    events = []

    def close():
        assert coordinator.status()['gpus']['0']['calls']
        events.append('doctor-stopped')

    doctor = SimpleNamespace(close=close)
    calls = worker.EvaluationCalls(coordinator, residency, 0, sync=lambda: events.append('cuda-synced'), doctor=doctor)
    with pytest.raises(TimeoutError):
        calls.run(lambda: (_ for _ in ()).throw(TimeoutError('HTTP read failed')))
    assert events == ['cuda-synced', 'doctor-stopped']
    assert not coordinator.status()['gpus']['0']['calls']
    residency.release()


def test_failed_cleanup_keeps_call_and_residency_quarantined(tmp_path):
    coordinator = PriorityCoordinator(tmp_path, idle_grace_seconds=0, require_training=False)
    residency = coordinator.evaluation_residency(4)
    doctor = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError('still resident')))
    calls = worker.EvaluationCalls(coordinator, residency, 4, doctor=doctor)
    with pytest.raises(RuntimeError, match='still resident'):
        calls.run(lambda: (_ for _ in ()).throw(TimeoutError()))
    state = coordinator.status()['gpus']['4']
    assert state['calls'] and state['residents']
    with pytest.raises(RuntimeError, match='active'):
        residency.release()


def test_stop_file_prevents_new_native_work(tmp_path):
    coordinator = PriorityCoordinator(tmp_path / 'priority', idle_grace_seconds=0, require_training=False)
    residency = coordinator.evaluation_residency(0)
    stop = tmp_path / 'STOP.json'
    stop.write_text('{}')
    calls = worker.EvaluationCalls(coordinator, residency, 0, stop_path=stop)
    with pytest.raises(worker.WorkerRevoked):
        calls.run(lambda: pytest.fail('GPU call after stop'))
    assert not coordinator.status()['gpus']['0']['calls']
    residency.release()


def test_releases_policy_memory_before_residency_acknowledgement():
    events = []
    model = SimpleNamespace(to=lambda device: events.append(('model-to', device)))
    policy = SimpleNamespace(model=model, models=[model], _batcher=SimpleNamespace(close=lambda: events.append('batcher-closed')))
    cuda = SimpleNamespace(is_initialized=lambda: True, synchronize=lambda: events.append('synced'),
                           empty_cache=lambda: events.append('empty-cache'))
    worker._free_policy(policy, SimpleNamespace(cuda=cuda))
    assert events == ['batcher-closed', ('model-to', 'cpu'), 'synced', 'empty-cache']
    assert policy.model is None and policy.models == []


def test_ordinary_shutdown_waits_for_training_before_offload(tmp_path, monkeypatch):
    coordinator = PriorityCoordinator(tmp_path, idle_grace_seconds=0, poll_seconds=.001, require_training=False)
    residency = coordinator.evaluation_residency(0)
    attempted, events = threading.Event(), []
    original_call = coordinator.evaluation_call

    def admit(*args):
        attempted.set()
        return original_call(*args)

    monkeypatch.setattr(coordinator, 'evaluation_call', admit)
    monkeypatch.setattr(worker, '_free_policy', lambda *args: events.append('offload'))
    doctor = SimpleNamespace(close=lambda: events.append('doctor-closed'))
    with coordinator.training_activity(0, 'generation'):
        thread = threading.Thread(target=worker.release_worker_resources,
                                  args=(None, None, doctor, coordinator, residency, 0))
        thread.start()
        assert attempted.wait(1)
        assert events == []
    thread.join(1)
    assert not thread.is_alive()
    assert events == ['offload', 'doctor-closed']
    assert residency.released


def test_revoked_residency_cleans_up_without_new_call_admission(tmp_path, monkeypatch):
    coordinator = PriorityCoordinator(tmp_path, idle_grace_seconds=0, require_training=False)
    residency = coordinator.evaluation_residency(4)
    with coordinator._state() as state:
        state['gpus']['4']['residents'][residency.token]['revoked'] = True
    monkeypatch.setattr(coordinator, 'evaluation_call', lambda *args: pytest.fail('admission during update'))
    events = []
    monkeypatch.setattr(worker, '_free_policy', lambda *args: events.append('offload'))
    worker.release_worker_resources(None, None, SimpleNamespace(close=lambda: events.append('doctor-closed')),
                                    coordinator, residency, 4)
    assert events == ['offload', 'doctor-closed'] and residency.released
