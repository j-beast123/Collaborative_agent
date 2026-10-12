"""Bounded case overlap: one original rollout at a time per coordinator.

Completed rollout releases the slot while its bank worker drains. All case
futures still join before scoring or changing the frozen policy.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Lock


class RolloutSlot:
    def __init__(self, semaphore):
        self.semaphore = semaphore
        self.guard = Lock()
        self.released = False
        semaphore.acquire()

    def release(self):
        with self.guard:
            if not self.released:
                self.released = True
                self.semaphore.release()


def run(jobs, execute):
    slot = BoundedSemaphore(1)
    def one(job):
        lease = RolloutSlot(slot)
        try: return execute(job, lease)
        finally: lease.release()
    with ThreadPoolExecutor(max_workers=max(1,len(jobs)), thread_name_prefix='case-bank-overlap') as pool:
        return list(pool.map(one,jobs))
