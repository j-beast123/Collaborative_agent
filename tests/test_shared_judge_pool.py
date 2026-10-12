from concurrent.futures import ThreadPoolExecutor
import threading

from src.online_rl.sherpa.backend import ChecklistBackend
from src.online_rl.tree.cache import CallCache


def test_parallel_cases_share_one_bounded_judge_pool():
    backend = ChecklistBackend.__new__(ChecklistBackend)
    backend._judge_pool = ThreadPoolExecutor(2)
    backend.identity = lambda: {'test': 'shared-pool'}
    guard = threading.Lock()
    full = threading.Event()
    release = threading.Event()
    counts = {'active': 0, 'peak': 0}
    def score(case, trajectory, rubrics):
        with guard:
            counts['active'] += 1
            counts['peak'] = max(counts['peak'], counts['active'])
            if counts['active'] == 2: full.set()
        try:
            assert release.wait(5)
            return {'trajectory_id': trajectory['trajectory_id'], 'reward': 1.}
        finally:
            with guard: counts['active'] -= 1
    backend.score = score
    cache = CallCache(None)
    try:
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(backend.score_all, {'case_id': str(i)},
                                   [{'trajectory_id': str(i), 'dialogue': []}], [], cache, {})
                       for i in range(4)]
            try:
                assert full.wait(5)
                assert counts['active'] == 2
            finally:
                release.set()
            assert [future.result()[0]['trajectory_id'] for future in futures] == ['0', '1', '2', '3']
        assert counts == {'active': 0, 'peak': 2}
    finally:
        release.set()
        backend.close()
