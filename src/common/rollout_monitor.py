"""Incremental read-only counters from successfully persisted rollout cache calls."""
from collections import Counter
import json
from pathlib import Path


class RolloutCounters:
    def __init__(self):
        self.files = {}

    def read(self, path):
        path=Path(path)
        empty={'agent_nodes':0,'doctor_turns':0,'trajectories':0,'reflections':0,'agent_depths':{},'leaf_depths':{}}
        try:stat=path.stat()
        except OSError:return empty
        state=self.files.get(str(path))
        if state is None or state['inode']!=stat.st_ino or stat.st_size<state['offset']:
            state={'inode':stat.st_ino,'offset':0,'seen':set(),'agent':Counter(),'leaf':Counter(),
                   'doctor':0,'feedback':set()}
            self.files[str(path)]=state
        try:
            with path.open('rb') as stream:
                stream.seek(state['offset'])
                while True:
                    start=stream.tell();line=stream.readline()
                    if not line or not line.endswith(b'\n'):
                        state['offset']=start;break
                    state['offset']=stream.tell()
                    try:row=json.loads(line)
                    except (ValueError,UnicodeDecodeError):continue
                    if not isinstance(row,dict):continue
                    kind=row.get('kind');identity=row.get('identity') or {}
                    if kind not in ('agent','final','doctor_turn','feedback','group_reflection') or not isinstance(identity,dict):continue
                    key=row.get('key') or json.dumps([kind,identity],sort_keys=True)
                    if key in state['seen']:continue
                    state['seen'].add(key)
                    if kind in ('agent','final','doctor_turn') and {'case','policy','protocol','path'}<=identity.keys():
                        path_id=identity['path']
                        if not isinstance(path_id,str):continue
                        depth=len(path_id.split('.')) if path_id else 0
                        if kind=='agent':state['agent'][depth]+=1
                        elif kind=='final':state['leaf'][depth]+=1
                        else:state['doctor']+=1
                    elif kind=='feedback' and isinstance(identity.get('leaf'),str):
                        state['feedback'].add(identity['leaf'])
                    elif kind=='group_reflection' and isinstance(identity.get('group'),str):
                        state['feedback'].add(identity['group'])
        except OSError:pass
        return {'agent_nodes':sum(state['agent'].values()),'doctor_turns':state['doctor'],
                'trajectories':sum(state['leaf'].values()),'reflections':len(state['feedback']),
                'agent_depths':dict(state['agent']),'leaf_depths':dict(state['leaf'])}
