"""Execution-only per-process GPU status and timing; never changes policy/cache identity."""
import collections
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

_status=None


class LiveGPU:
    def __init__(self,root,gpu):
        self.root=Path(root)/'live';self.root.mkdir(parents=True,exist_ok=True)
        self.gpu=gpu;self.pid=os.getpid();self.lock=threading.RLock()
        self.active=collections.Counter();self.calls=collections.Counter();self.seconds=collections.Counter()
        self.metadata={};self.started={};self.last_error=None
    def write(self):
        now=time.time()
        row={'pid':self.pid,'gpu':self.gpu,'updated_at':now,'active':{k:v for k,v in self.active.items() if v},
             'active_since':dict(self.started),'calls':dict(self.calls),'seconds':dict(self.seconds),
             'last_error':self.last_error,**self.metadata}
        path=self.root/f'gpu{self.gpu}.json';tmp=path.with_name(path.name+f'.{self.pid}.tmp')
        # Status is ephemeral: atomic replace, without per-call fsync on the training disk.
        try:tmp.write_text(json.dumps(row));tmp.replace(path)
        except OSError:pass
    def note(self,**metadata):
        with self.lock:self.metadata.update(metadata);self.write()


def configure(root,gpu):
    global _status
    _status=LiveGPU(root,gpu);_status.note(phase='initializing')
    return _status


def note(**metadata):
    if _status is not None:_status.note(**metadata)


@contextmanager
def activity(name):
    status=_status
    if status is None:
        yield;return
    start=time.monotonic()
    with status.lock:
        status.active[name]+=1;status.started.setdefault(name,time.time());status.write()
    outcome='ok'
    try:yield
    except BaseException as e:
        outcome=type(e).__name__
        with status.lock:status.last_error=outcome
        raise
    finally:
        elapsed=time.monotonic()-start
        with status.lock:
            status.active[name]-=1
            if not status.active[name]:status.started.pop(name,None)
            status.calls[name]+=1;status.seconds[name]+=elapsed;status.write()
            try:
                with (status.root/f'timings_gpu{status.gpu}_pid{status.pid}.jsonl').open('a') as f:
                    f.write(json.dumps({'finished_at':time.time(),'pid':status.pid,'gpu':status.gpu,'operation':name,
                                        'seconds':elapsed,'outcome':outcome,**status.metadata})+'\n')
            except OSError:pass
