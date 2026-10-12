"""Parent-runner stage telemetry; never changes model, cache or learning decisions."""
import json
import os
from pathlib import Path
import time

_root = None
_state = {}


def configure(root):
    global _root, _state
    _root = Path(root)
    _state = {'pid': os.getpid(), 'run_started_at': time.time()}


def emit(event):
    global _state
    now = time.time()
    row = {**event, 'timestamp': now}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if _root is None:
        return
    try:
        batch = event.get('batch', _state.get('batch'))
        previous = _state
        changed_batch = batch != previous.get('batch')
        changed_stage = changed_batch or event['stage'] != previous.get('stage')
        state = {**previous, **row, 'batch': batch,
                 'batch_started_at': now if changed_batch else previous.get('batch_started_at'),
                 'stage_started_at': now if changed_stage else previous.get('stage_started_at', now)}
        if changed_stage and previous.get('stage'):
            row['previous_stage'] = previous['stage']
            row['previous_stage_seconds'] = now-previous['stage_started_at']
        if event['stage'] == 'batch_committed' and state.get('batch_started_at'):
            state['batch_seconds'] = now-state['batch_started_at']
        _state = state
        path = _root/'progress.json'
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False))
        temporary.replace(path)
        with (_root/'progress_events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    except OSError:
        pass
