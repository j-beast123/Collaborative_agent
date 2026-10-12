"""Read-only view of streaming queues and committed chunk receipts."""
from pathlib import Path


def describe(directory, record, progress):
    directory = Path(directory)
    plan = record(directory / 'stream_plan.json').get('chunks', [])
    if not isinstance(plan, list): plan = []
    groups = [group for group in plan if isinstance(group, list)]
    journal = record(directory.parent / 'trajectory_stream' / 'state.json')
    transactions = set(journal.get('transactions', []))
    prefix = directory.name + '_chunk_'
    if journal:
        completed = {i for i in range(len(groups)) if f'{prefix}{i:04d}' in transactions}
    else:
        completed = set(range(min(len(groups), progress.get('chunks_committed', 0))))
    committed = sum(len(groups[i]) for i in completed)
    current = progress.get('current_chunk')
    index = current - 1 if isinstance(current, int) else None
    active_chunks = progress.get('active_chunks')
    if isinstance(active_chunks,dict):
        indices = [int(k)-1 for k in active_chunks if str(k).isdigit() and 0 <= int(k)-1 < len(groups) and int(k)-1 not in completed]
        reserved = sum(len(groups[i]) for i in indices)
        lock_wait = sum(len(groups[i]) for i in indices if 'lock' in str(active_chunks[str(i+1)].get('stage','')).lower() and '대기' in str(active_chunks[str(i+1)].get('stage','')))
        active = bool(indices)
    else:
        active = (index is not None and 0 <= index < len(groups) and index not in completed)
        indices = [index] if active else []
        reserved = len(groups[index]) if active else 0
        lock_wait = reserved if 'lock' in str(progress.get('stage','')).lower() and '대기' in str(progress.get('stage','')) else 0
    processing = reserved - lock_wait
    ready = progress.get('ready_trajectories', 0)
    # A plan can name a saved chunk whose reflections are still replaying on resume.
    waiting = max(0, ready - committed - reserved)
    return {'ready': ready, 'committed_trajectories': committed, 'processing_trajectories': processing,
            'lock_wait_trajectories': lock_wait, 'waiting_trajectories': waiting, 'committed_chunks': len(completed),
            'planned_chunks': len(groups), 'current_chunk': current if active else None,
            'current_chunk_size': reserved, 'rollout_done': bool(progress.get('rollout_done')),
            'active_indices':indices, 'chunk_sizes': [len(g) for g in groups], 'completed_indices': sorted(completed)}
