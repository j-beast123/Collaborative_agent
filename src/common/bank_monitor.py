"""Read-only incremental bank work counters; cached calls are not bank commits."""
from collections import Counter
import json
from pathlib import Path

KINDS={'offline_chunk_generation','posthoc_gate','rubric_state_applicability',
       'rubric_match_batch','intervention_instruction_audit','rerun',
       'rubric_behavior_batch','rubric_behavior','rubric_score','bank_delta_dedup'}


class BankCounters:
    def __init__(self):
        self.files={}
        self.records={}

    def record(self,path):
        path=Path(path)
        try:stat=path.stat()
        except OSError:return {}
        stamp=(stat.st_ino,stat.st_mtime_ns,stat.st_size)
        old=self.records.get(str(path))
        if old and old[0]==stamp:return old[1]
        try:
            value=json.loads(path.read_text())
            if not isinstance(value,dict):return {}
        except (OSError,ValueError):return {}
        self.records[str(path)]=(stamp,value)
        return value

    def read(self,directory,discovery_cap=48):
        directory=Path(directory);path=directory/'calls.jsonl'
        try:stat=path.stat()
        except OSError:stat=None
        state=self.files.get(str(path))
        if state is None or (stat and (state['inode']!=stat.st_ino or stat.st_size<state['offset'])):
            state={'inode':stat.st_ino if stat else None,'offset':0,'seen':set(),
                   'calls':Counter(),'chunks':{},'gates':Counter()}
            self.files[str(path)]=state
        if stat:
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
                        if not isinstance(row,dict) or row.get('kind') not in KINDS:continue
                        kind=row['kind'];identity=row.get('identity') or {};value=row.get('value') or {}
                        if not isinstance(identity,dict) or not isinstance(value,dict):continue
                        key=row.get('key') or json.dumps([kind,identity],sort_keys=True)
                        if key in state['seen']:continue
                        state['seen'].add(key);state['calls'][kind]+=1
                        if kind=='offline_chunk_generation':
                            actions=Counter(v.get('action') for v in value.get('decisions',{}).values() if isinstance(v,dict))
                            state['chunks'][identity.get('chunk_index',key)]=actions
                        elif kind=='posthoc_gate':
                            passed=value.get('passed')
                            state['gates']['passed' if passed is True else 'rejected' if passed is False else 'unresolved']+=1
            except OSError:pass
        rolled=self.record(directory/'bank_rollout.json')
        online=self.record(directory/'bank_online.json')
        stream=self.record(directory/'stream_progress.json')
        eligible=sum(1 for row in rolled.get('feedback',[]) if isinstance(row,dict) and row.get('valid',True)) if rolled else None
        if stream.get('rollout_done') and eligible is None:
            eligible=stream.get('ready_trajectories')
        summary=online.get('offline_chunk_bank') or {}
        selected=len(summary['selected_ids']) if isinstance(summary.get('selected_ids'),list) else (eligible if discovery_cap is None else min(discovery_cap,eligible)) if eligible is not None else None
        proposals=Counter()
        for counts in state['chunks'].values():proposals.update(counts)
        selections=online.get('selection',[])
        kept=sum(row.get('retained') is True for row in selections)
        removed=sum(row.get('retained') is False for row in selections)
        before=(online.get('cumulative_bank_input') or {}).get('entries',[])
        after=(online.get('cumulative_bank_output') or {}).get('entries',[])
        added=(sum(c['commit_added_unique'] if 'commit_added_unique' in c else c['bank_after_size']-c['bank_before_size'] for c in summary.get('chunks',[]))
               if summary.get('trajectory_streaming') else len(after)-len(before))
        from src.common.stream_monitor import describe
        flow=describe(directory,self.record,stream) if stream else None
        return {'delta_checks':state['calls']['bank_delta_dedup'],'flow':flow,'stream':stream,'selected':selected,'chunks_total':(selected+7)//8 if selected is not None else None,
                'chunks_generated':len(state['chunks']),'proposals':{key:proposals[key] for key in ('CREATE','REUSE_BANK','PEER')},
                'gate_passed':state['gates']['passed'],'gate_rejected':state['gates']['rejected'],
                'gate_unresolved':state['gates']['unresolved'],'gate_done':state['calls']['posthoc_gate'],
                'source_checks':state['calls']['rubric_state_applicability'],
                'peer_checks':state['calls']['rubric_match_batch'],
                'instruction_audits':state['calls']['intervention_instruction_audit'],
                'reruns':state['calls']['rerun'],
                'score_calls':sum(state['calls'][k] for k in ('rubric_behavior_batch','rubric_behavior','rubric_score')),
                'done':(directory/'bank_online.json').is_file(),'score_done':(directory/'bank_score.json').is_file(),
                'kept':kept if online else None,'removed':removed if online else None,
                'bank_added':added if online else None}


def activity(row,now):
    """Use coordinator cache/phase operations, not remote actor role metadata."""
    labels=[('cache.bank_delta_dedup','추가 rubric 중복 비교 (Gemini)'),('cache.rerun','Verification rerun'),('cache.intervention_instruction_audit','Intervention 검사'),
            ('cache.posthoc_gate','Gate 판정'),('cache.rubric_match_batch','PEER 재확인'),
            ('cache.rubric_state_applicability','Applicability 검사'),
            ('cache.offline_chunk_generation','Chunk 후보 생성'),
            ('cache.rubric_behavior','응답 채점'),('cache.rubric_score','응답 채점'),
            ('bank_online.','Bank 생성·검증'),('bank_score.','전체 bank 채점')]
    for prefix,label in labels:
        names=[name for name,count in row.get('active',{}).items() if count and name.startswith(prefix)]
        if names:
            starts=[row.get('active_since',{}).get(name) for name in names]
            starts=[start for start in starts if isinstance(start,(int,float))]
            return label,now-min(starts) if starts else None
    return None,None
