"""Synchronized append for shared per-run API audit files; no credential logging."""
import fcntl
import json
import threading
from pathlib import Path

_LOCK = threading.Lock()


def append_json(path, row):
    path = Path(path)
    line = json.dumps(row, ensure_ascii=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(line)
        stream.flush()
