import queue
import threading
from types import SimpleNamespace

from src.online_rl.tree.five_gpu import batch_slots
from src.online_rl.tree.gpu_pool import create_pool


def test_asymmetric_batch_uses_two_lanes_on_gpu0_to_3_and_one_on_gpu4():
    slots = batch_slots(0, 120, interval=30, workers=5, lanes=[2, 2, 2, 2, 1])
    assert len(slots) == 9
    assert [sum(slot['worker'] == gpu for slot in slots) for gpu in range(5)] == [2, 2, 2, 2, 1]
    assert len({slot['index'] for slot in slots}) == 9


def test_asymmetric_batch_stops_at_checkpoint_barrier():
    slots = batch_slots(27, 120, interval=30, workers=5, lanes=[2, 2, 2, 2, 1])
    assert [slot['index'] for slot in slots] == [27, 28, 29]


def test_pool_capacity_is_per_gpu():
    manager = SimpleNamespace(RLock=threading.RLock, dict=dict, Queue=queue.Queue)
    spec = create_pool(manager, list(range(5)), agent_capacity=4,
                       doctor_capacity=[2, 2, 2, 2, 1])
    doctors = [spec['doctor'].get_nowait() for _ in range(9)]
    assert sorted(doctors) == [0, 0, 1, 1, 2, 2, 3, 3, 4]
    assert spec['request_workers']['doctor_workers'] == 9
