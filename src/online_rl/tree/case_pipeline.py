"""Schedule rollouts independently of serialized latest-snapshot bank commits."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
from queue import Queue
from threading import Event
from src.online_rl.integrated.records import atomic_write_json


def ready_cases(jobs, submit, order_path):
    """Keep coordinator rollout queues moving while the caller handles bank work.

    Submit one rollout ahead per coordinator. Bank phases use the same single-
    worker executor, so policy/client context never has concurrent case phases.
    A bank phase may wait for that coordinator's already queued next rollout;
    other coordinators continue independently. Persist bank selection for resume.
    """
    path = Path(order_path)
    order = json.loads(path.read_text()) if path.exists() else []
    indices = {j['index'] for j in jobs}
    if (not isinstance(order, list) or any(type(i) is not int for i in order)
            or len(set(order)) != len(order) or not set(order) <= indices):
        raise RuntimeError('Invalid saved ready-case bank order')
    if not path.exists():
        order = sorted(j['index'] for j in jobs if (Path(j['out']) / 'bank_online.json').exists())
        atomic_write_json(path, order)
    queues = {}
    for job in sorted(jobs, key=lambda j: j['index']):
        queues.setdefault(job['worker'], []).append(job)
    completed = Queue()
    stopped = Event()

    def pump():
        active = {}
        def start(worker):
            if queues[worker] and not stopped.is_set():
                job = queues[worker].pop(0)
                active[submit(job)] = job
        try:
            for worker in queues:
                start(worker)
            while active and not stopped.is_set():
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: active[f]['index']):
                    job = active.pop(future)
                    result = future.result()
                    if len(result) != 1 or result[0]['case_cursor'] != job['index']:
                        raise RuntimeError('Pipeline rollout returned a different case')
                    # Crucially, this does not wait for the caller's bank commit.
                    start(job['worker'])
                    completed.put(('record', result[0]))
        except BaseException as exc:
            completed.put(('error', exc))
        finally:
            wait(active)  # Never change adapters with a submitted phase in flight.
            completed.put(('done', None))

    pending = {}
    position = 0
    with ThreadPoolExecutor(1, thread_name_prefix='rollout-scheduler') as scheduler:
        future = scheduler.submit(pump)
        try:
            while position < len(jobs):
                target = order[position] if position < len(order) else None
                while not pending or (target is not None and target not in pending):
                    kind, value = completed.get()
                    if kind == 'error':
                        raise value
                    if kind == 'done':
                        raise RuntimeError('Rollout scheduler ended before every bank case was selected')
                    cursor = value['case_cursor']
                    if cursor in pending or cursor in order[:position] or cursor not in indices:
                        raise RuntimeError('Duplicate or unexpected pipeline case')
                    pending[cursor] = value
                cursor = target if target is not None else next(iter(pending))
                if position == len(order):
                    order.append(cursor)
                    atomic_write_json(path, order)
                position += 1
                yield pending.pop(cursor)
            future.result()
        finally:
            stopped.set()
            future.result()  # Drain before the caller reaches the update barrier.
