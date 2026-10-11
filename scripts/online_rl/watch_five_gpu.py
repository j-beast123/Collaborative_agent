"""Read-only live view of GPU0..4, worker caches, BASE progress and API spend.

Works on an already-running job without restart. Cache rows are completed calls;
last completed call is never represented as the operation currently executing.
"""
import argparse
import collections
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
ONE_STEP_PROTOCOLS={'intervention_agent_once_doctor_final_once_v1',
                    'fresh_agent_from_joint_generated_intervention_instruction_then_doctor_final_each_rerun_v4',
                    'fresh_agent_from_joint_generated_intervention_instruction_then_doctor_final_each_rerun_v5'}


def unscorable_suffix(distribution):
    """U is an explicitly unscorable assessment, separate from genuine NA."""
    return ((f' U:{distribution["U"]}' if distribution.get('U',0) else '') +
            (f' NR:{distribution["NR"]}' if distribution.get('NR',0) else ''))


def incomplete_suffix(funnel):
    return f' incomplete {funnel["rubrics_incomplete"]}' if funnel.get('rubrics_incomplete',0) else ''


def case_score_distribution(pipeline):
    saved=pipeline.get('score_distribution',{}).get('union') or {}
    if saved and (pipeline.get('phase_complete',{}).get('union') or saved.get('cases')):
        return saved.get('counts',saved), saved.get('n',pipeline.get('phase_scored_pairs',{}).get('union',0)), 'final'
    return pipeline.get('live_score_distribution',{}), pipeline.get('live_scored_pairs',0), 'live'


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


class JsonlTail:
    def __init__(self):self.offset=0;self.pending=b'';self.inode=None
    def read(self,path):
        path=Path(path)
        try:stat=path.stat()
        except OSError:return []
        if self.inode!=stat.st_ino or stat.st_size<self.offset:
            self.offset=0;self.pending=b'';self.inode=stat.st_ino
        with path.open('rb') as f:
            f.seek(self.offset)
            while True:
                chunk=f.read(65536)
                if not chunk:break
                lines=(self.pending+chunk).split(b'\n');self.pending=lines.pop()
                for line in lines:
                    try:
                        row=json.loads(line)
                        if isinstance(row,dict):yield row
                    except (ValueError,UnicodeDecodeError):pass
            self.offset=f.tell()


class WorkerCache:
    def __init__(self,path):
        self.path=Path(path);self.tail=JsonlTail();self.counts=collections.Counter();self.depths=collections.Counter()
        self.last=None;self.history=collections.deque()
        self.agent_paths=set();self.final_paths=set()
        self.doctor_depths=collections.Counter();self.final_depths=collections.Counter()
        self.feedback_leaves=set();self.scored_nodes=set();self.gate_pass=0;self.gate_reject=0;self.rerun_unparsed=0
        self.one_step_reruns=0;self.one_step_final_calls=0;self.legacy_reruns=0;self.one_step_targets=set()
        self.scored_pairs=set();self.live_score_distribution=collections.Counter()
    def update(self,now):
        for row in self.tail.read(self.path):
            kind=row.get('kind','unknown');self.counts[kind]+=1;self.last=kind
            if kind=='agent':
                path=row.get('identity',{}).get('path')
                if path:
                    self.depths[len(path.split('.'))]+=1
                    self.agent_paths.add(path)
            elif kind=='final':
                path=row.get('identity',{}).get('path')
                if path:
                    self.final_paths.add(path)
                    self.final_depths[len(path.split('.'))]+=1
            elif kind=='doctor_turn':
                path=row.get('identity',{}).get('path')
                if path:self.doctor_depths[len(path.split('.'))]+=1
            elif kind=='feedback':
                leaf=row.get('identity',{}).get('leaf')
                if leaf:self.feedback_leaves.add(leaf)
            elif kind=='rubric_score':
                node=row.get('identity',{}).get('node')
                if node:self.scored_nodes.add(node)
                value=row.get('value') or {}
                for rubric_id,result in value.items():
                    if rubric_id=='_decisions' or not isinstance(result,dict) or not node:continue
                    pair=(node,rubric_id)
                    if pair in self.scored_pairs:continue
                    if result.get('scoring_status') in ('rubric_unscorable','group_unscorable'):label='U'
                    elif result.get('applicable') is False:label='NA'
                    elif result.get('applicable') is True:
                        stage2=result.get('stage2') or {}
                        level=stage2.get('level') if isinstance(stage2,dict) else None
                        if level is None:level=result.get('hard')
                        try:
                            level=int(float(level))
                            if level not in (0,1,2):continue
                            label=str(level)
                        except (TypeError,ValueError):continue
                    else:continue
                    self.scored_pairs.add(pair);self.live_score_distribution[label]+=1
            elif kind=='posthoc_gate':
                passed=(row.get('value') or {}).get('passed')
                self.gate_pass+=passed is True;self.gate_reject+=passed is False
            elif kind=='rerun':
                self.rerun_unparsed+=(row.get('value') or {}).get('outcome') is None
                ident=row.get('identity') or {}
                if ident.get('continuation_protocol') in ONE_STEP_PROTOCOLS:
                    self.one_step_reruns+=1
                else:self.legacy_reruns+=1
            elif kind=='rerun_doctor_turn':
                owner=(row.get('identity') or {}).get('rerun') or {}
                if owner.get('continuation_protocol') in ONE_STEP_PROTOCOLS:
                    turn=(row.get('identity') or {}).get('turn') or {}
                    if turn.get('stage')=='FINAL':self.one_step_final_calls+=1
                    self.one_step_targets.add((owner.get('case_id'),owner.get('rubric'),owner.get('node'),owner.get('level')))
        self.history.append((now,self.counts['agent']))
        while len(self.history)>1 and self.history[1][0]<now-60:self.history.popleft()
    def rate(self):
        if len(self.history)<2:return None
        a,b=self.history[0],self.history[-1]
        return 60*(b[1]-a[1])/(b[0]-a[0]) if b[0]>a[0] else None


def leaf_progress(agent_paths,final_paths,branching,max_turns,min_turns=1):
    """Bounds from the observed prefix tree, without guessing future halt outcomes.

    Observing any child means all G siblings are planned. A saved FINAL closes
    that entire subtree, including unparsed answers. Rerun calls never enter
    these path sets. Min-turn rules force the first levels to expand.
    """
    expanded=set()
    for path in agent_paths:
        parts=path.split('.')
        for length in range(len(parts)):
            expanded.add('.'.join(parts[:length]))
    depth=max((len(p.split('.')) for p in agent_paths),default=0)
    # The builder schedules the entire frontier before generating this level.
    # Even a parent whose first child hasn't finished is already set to expand.
    expanded.update(p for p in agent_paths if len(p.split('.'))<depth and p not in final_paths)
    minimum=min(max_turns,max(1,min_turns))

    def pending(path,depth):
        if path in final_paths:return 0,0
        if depth==max_turns:return 1,1
        if depth<minimum or path in expanded:
            bounds=[pending(f'{path}.{j}' if path else str(j),depth+1) for j in range(branching)]
            return sum(x[0] for x in bounds),sum(x[1] for x in bounds)
        return 1,branching**(max_turns-depth)

    low,high=pending('',0)
    target=branching if depth==1 else branching*sum(
        len(p.split('.'))==depth-1 and p not in final_paths for p in agent_paths) if depth else 0
    return {'completed':len(final_paths),'remaining_min':low,'remaining_max':high,
            'tree_complete':high==0,'depth_target':target}


def first_record_time(path):
    try:
        with Path(path).open() as file:value=json.loads(file.readline())['time']
        try:return float(value)
        except (ValueError,TypeError):return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
    except (OSError,ValueError,KeyError,TypeError):return None


def file_time(path):
    try:return Path(path).stat().st_mtime
    except OSError:return None


def elapsed(start,end):
    return max(0,end-start) if start is not None else None


def summarize_phase(record):
    """Compact confirmed results; None rewards remain distinct from real zero."""
    target_meta=record.get('agent_targets') or {}
    targets=target_meta.get('nodes',{})
    outcome_weight=float((target_meta.get('reward_weights') or {}).get('w_outcome',0.0))
    outcome_values={nid:float(node['outcome_component']) for nid,node in targets.items()
                    if isinstance(node,dict) and node.get('outcome_component') is not None}
    outcome_counts=collections.Counter()
    for value in outcome_values.values():
        label='0' if abs(value)<1e-12 else ('1' if abs(value-1.0)<1e-12 else 'mix')
        outcome_counts[label]+=1
    outcome_groups=[];outcome_signal_targets=set()
    for ids in target_meta.get('groups',{}).values():
        values=[outcome_values[nid] for nid in ids if nid in outcome_values]
        if len(values)>1:
            outcome_groups.append(values)
            if max(values)-min(values)>1e-12:outcome_signal_targets.update(nid for nid in ids if nid in outcome_values)
    outcome_n=len(outcome_values);outcome_sum=sum(outcome_values.values())
    outcome_signal={'targets':outcome_n,'zero':outcome_counts['0'],'mixed':outcome_counts['mix'],
                    'one':outcome_counts['1'],'mean':outcome_sum/outcome_n if outcome_n else None,
                    'mean_contribution':outcome_weight*outcome_sum/outcome_n if outcome_n else None,
                    'contribution_sum':outcome_weight*outcome_sum,
                    'groups':len(outcome_groups),
                    'varying_groups':sum(max(v)-min(v)>1e-12 for v in outcome_groups),
                    'varying_targets':len(outcome_signal_targets)}
    rewarded=[v['reward'] for v in targets.values() if v.get('reward') is not None]
    nodes=record.get('tree',{}).get('nodes',[])
    scores=record.get('response_scores',[])
    witness=record.get('rubric_bank',{}).get('witness',[])
    telemetry=record.get('rubric_bank',{}).get('telemetry',{})
    score_distribution=collections.Counter()
    for item in scores:
        hard=item.get('hard')
        applicable=item.get('applicable',item.get('match',hard is not None or item.get('score') is not None))
        if item.get('scoring_status') == 'not_retrieved':
            score_distribution['NR']+=1
        elif item.get('scoring_status') in ('rubric_unscorable','group_unscorable'):
            score_distribution['U']+=1
        elif not applicable:
            score_distribution['NA']+=1
        elif hard is None and item.get('score') is None:
            score_distribution['other']+=1
        else:
            if hard is None:
                expected=float(item['score'])
                label=str(int(round(expected*2))) if expected in (0.0,0.5,1.0) else 'other'
            else:
                # ResponseRubricScore.hard is normalized by hard_value(): label / 2.
                level=int(round(float(hard)*2))
                label=str(level) if level in (0,1,2) else 'other'
            score_distribution[label]+=1
    return {'total_nodes':len(nodes),'outcome_signal':outcome_signal,
            'scored_nodes':len({s['response_node_id'] for s in scores}),
            'valid_rewards':len(rewarded),'na_rewards':sum(v.get('reward') is None for v in targets.values()),
            'reward_sum':sum(rewarded),'keep':sum(w.get('decision')=='KEEP' for w in witness),
            'not_validated':sum(w.get('decision')=='NOT_VALIDATED' for w in witness),
            'rollout_wall_seconds':(record.get('timings') or {}).get('rollout_wall_seconds'),
            'step_wall_seconds':record.get('seconds'),
            'score_distribution':{**{k:score_distribution.get(k,0) for k in ('0','1','2','NA','U','other')},
                                  **({'NR':score_distribution['NR']} if score_distribution['NR'] else {})},
            'cumulative_reused_trajectories':len(record.get('cumulative_reuse', [])),
            'scored_pairs':len(scores),
            'training_rubrics':len(record.get('rubric_weights',{})),
            'union_rubrics':len(record.get('training_rubrics',[])),
            # `rubrics` is the canonical set that reached witness validation;
            # KEEP is the subset with a causal witness. Keep NA separate.
            'rubrics_generated':int(telemetry.get('created_new',0)),
            'rubrics_validated':int(telemetry.get('rubrics',len(witness))),
            'rubrics_keep':int(telemetry.get('keep',sum(w.get('decision')=='KEEP' for w in witness))),
            'rubrics_incomplete':int(telemetry.get('incomplete',sum(w.get('decision')=='INCOMPLETE' for w in witness))),
            'rubrics_not_validated':int(telemetry.get('not_validated',sum(w.get('decision')=='NOT_VALIDATED' for w in witness)))}


def gpu_sample():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,power.draw',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        if result.returncode:raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        rows={}
        for row in csv.reader(io.StringIO(result.stdout)):
            idx,uuid,used,total,util,power=[v.strip() for v in row]
            rows[int(idx)]={'uuid':uuid,'memory_mb':float(used),'total_mb':float(total),'util':float(util),'power_w':float(power)}
        processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],
                                 capture_output=True,text=True,timeout=5)
        ids={v['uuid']:k for k,v in rows.items()}
        for row in csv.reader(io.StringIO(processes.stdout)):
            if len(row)<3:continue
            uuid,pid,name=[v.strip() for v in row[:3]]
            if uuid in ids:rows[ids[uuid]].setdefault('pids',[]).append(int(pid))
        return rows,None
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as e:return {},str(e)


class Monitor:
    def __init__(self,root,cfg):
        self.root=Path(root);self.cfg=cfg;self.caches={};self.samples=collections.defaultdict(collections.deque)
        self.log=JsonlTail();self.gpu4_stage=None;self.api=JsonlTail();self.spend=0.;self.unpriced=0;self.api_calls=0
        self.api_latency=collections.Counter();self.api_purposes=collections.defaultdict(collections.Counter)
        self.start=time.monotonic()
        self.phase_summaries={}

    def phase_summary(self,path):
        stamp=file_time(path)
        if stamp is None:return None
        key=str(path);saved=self.phase_summaries.get(key)
        if saved is None or saved[0]!=stamp:
            record=read_json(path)
            if record is None:return None
            saved=(stamp,summarize_phase(record));self.phase_summaries[key]=saved
        return saved[1]
    def snapshot(self):
        now=time.monotonic();wall=time.time();gpu,error=gpu_sample()
        for idx,g in gpu.items():
            hist=self.samples[idx];hist.append((now,g['util']))
            while hist and hist[0][0]<now-60:hist.popleft()
            g['mean_util_60s']=sum(v for _,v in hist)/len(hist);g['samples']=len(hist)
        for row in self.log.read(self.root/'train.log'):
            stage=row.get('stage')
            if stage in ('BASE_eval','checkpoint_eval','agent_update'):self.gpu4_stage=row
            elif stage=='batch_committed' and (self.gpu4_stage or {}).get('stage')=='agent_update':self.gpu4_stage=None
        for row in self.api.read(self.root/'api_usage.jsonl'):
            self.api_calls+=1;cost=row.get('cost_usd')
            if cost is None:self.unpriced+=1
            else:self.spend+=cost
            if row.get('latency_seconds') is not None:
                purpose=str(row.get('purpose') or row.get('stage') or 'unknown')
                for metric in ('latency_seconds','request_seconds','retry_wait_seconds'):
                    value=row.get(metric)
                    if value is not None:
                        self.api_latency[metric]+=float(value)
                        self.api_purposes[purpose][metric]+=float(value)
                self.api_latency['timed_calls']+=1
                self.api_purposes[purpose]['calls']+=1
        state=read_json(self.root/'run_state.json',{})
        cumulative=bool(self.cfg.get('rubric',{}).get('cumulative_bank') or state.get('rubric_bank_sha256'))
        eval_jobs=[]
        for job in sorted((self.root/'evaluation').glob('checkpoint_*')):
            manifest=read_json(job/'manifest.json',{})
            summary=read_json(job/'summary.json',{})
            total=int(manifest.get('settings',{}).get('eval_cases',len(manifest.get('case_ids',[]))))
            evaluated=int(summary.get('evaluated',len(list((job/'cases').glob('case_*.json')))))
            eval_jobs.append({'checkpoint_cases':manifest.get('completed_cases'), 'evaluated':evaluated,
                              'total':total, 'status':summary.get('status','pending'),
                              'accuracy':summary.get('accuracy')})
        pending_eval=next((e for e in eval_jobs if e['status']!='complete'),None)
        latest_eval=eval_jobs[-1] if eval_jobs else None
        manifests=sorted(self.root.glob('batches/batch_*/manifest.json'))
        latest=manifests[-1] if manifests else None;manifest=read_json(latest,{}) if latest else {}
        launch=read_json(self.root/'launch.json',{})
        run_start=launch.get('launched_at') or file_time(self.root/'run_manifest.json')
        batch_start=file_time(latest) if latest else None
        batch_number=int(latest.parent.name.split('_')[-1]) if latest else None
        if latest:
            prefix=str(latest.parent)+os.sep
            self.caches={k:v for k,v in self.caches.items() if k.startswith(prefix)}
            self.phase_summaries={k:v for k,v in self.phase_summaries.items() if k.startswith(prefix)}
        batch_end=(file_time(self.root/'run_state.json') or wall) if latest and state.get('batch_cursor',0)>=batch_number else wall
        evals_complete=(not self.cfg['five_gpu'].get('checkpoint_evaluation_enabled',True) or
                       bool(eval_jobs) and all(e['status']=='complete' for e in eval_jobs))
        run_end=(max(file_time(self.root/'run_state.json') or wall,
                     max((file_time(job/'summary.json') or 0 for job in (self.root/'evaluation').glob('checkpoint_*')),default=0))
                 if state.get('case_cursor',0)>=self.cfg['schedule']['cases'] and evals_complete else wall)
        rows=[]
        for idx in range(5):
            g=gpu.get(idx,{});r={'gpu':idx,**g,'phase':'not started','progress':'-','last_completed':None,'last_commit_age_s':None}
            live=read_json(self.root/'live'/f'gpu{idx}.json',{})
            live_ok=live and pid_alive(live.get('pid'))
            if idx in self.cfg['five_gpu']['rollout_gpus'] and latest:
                assignment=next((a for a in manifest.get('assignments',[]) if a['worker']==idx),None)
                if assignment:
                    directory=latest.parent/f'case_{assignment["index"]:06d}'
                    path=directory/'calls.jsonl';key=str(path)
                    cache=self.caches.setdefault(key,WorkerCache(path));cache.update(now)
                    r.update(case_index=assignment['index'],case_id=manifest['case_ids'][manifest['assignments'].index(assignment)])
                    if cumulative:
                        if (directory/'bank_score.json').exists():r['phase']='waiting at batch barrier'
                        elif (directory/'bank_verify.json').exists():r['phase']='bank scoring / wait'
                        elif (directory/'bank_prepare.json').exists():r['phase']='bank rerun verification / wait'
                        else:r['phase']='bank rollout / Doctor reflection'
                    elif (directory/'union.json').exists() or (directory/'result.json').exists():r['phase']='waiting at batch barrier'
                    elif (directory/'P1_verify.json').exists() or (directory/'P1.json').exists():r['phase']='P0+P1 union scoring / wait'
                    elif (directory/'P1_prepare.json').exists():r['phase']='P1 rubric / rerun'
                    elif (directory/'P0_verify.json').exists():r['phase']='P1 generation / wait'
                    elif (directory/'P0_prepare.json').exists():r['phase']='P0 rubric / rerun'
                    elif (directory/'P0.json').exists():r['phase']='P1 verification / reflection wait'
                    elif cache.counts['feedback'] or any('rubric' in k for k in cache.counts):r['phase']='P0 rubric / rerun'
                    else:r['phase']='P0 rollout'
                    depth=max(cache.depths,default=0)
                    ro=self.cfg['rollout']
                    leaves=leaf_progress(cache.agent_paths,cache.final_paths,ro['branching'],ro['rounds'],
                                         ro.get('halting',{}).get('min_agent_turns',2 if ro.get('doctor_backend') else 1))
                    if cumulative and not any((directory/f'bank_{stage}.json').exists() for stage in ('prepare','verify','score')):
                        # Leaf reflection overlaps rollout. Its first completed
                        # API call does not mean a complete tree or rubric bank.
                        active=(live.get('active',{}) if live_ok and live.get('case_index')==assignment['index'] else {})
                        generating=any(cache.counts[k] for k in ('rubric_generation','rubric_repair','rubric_match','rubric_match_batch','posthoc_gate')) or any(
                            active.get(k) for k in ('role.rubric_generator','cache.rubric_generation','cache.rubric_repair','cache.posthoc_gate'))
                        matching=bool(cache.counts['cumulative_bank_match'] or active.get('cache.cumulative_bank_match'))
                        if not leaves['tree_complete']:r['phase']='bank rollout / Doctor reflection'
                        elif generating:r['phase']='bank candidate generation / reuse'
                        elif matching:r['phase']='bank reuse matching'
                        else:r['phase']='bank Doctor reflection / wait'
                    case_start=first_record_time(directory/'execution_runtime.jsonl') or batch_start
                    case_end=(file_time(directory/'bank_score.json') if cumulative else None) or file_time(directory/'union.json') or file_time(directory/'result.json') or wall
                    phases={
                        'P0': next((self.phase_summary(directory/f'{name}.json') for name in
                                    ('P0_verify','P0','P0_prepare') if (directory/f'{name}.json').exists()),None),
                        'P1': next((self.phase_summary(directory/f'{name}.json') for name in
                                    ('P1_verify','P1','P1_prepare') if (directory/f'{name}.json').exists()),None),
                        'union': self.phase_summary(directory/'union.json')}
                    if phases['union'] is None and (directory/'result.json').exists():
                        phases['union']=self.phase_summary(directory/'result.json')
                    bank_prepared=self.phase_summary(directory/'bank_prepare.json') if cumulative else None
                    if cumulative:
                        phases={'P0':self.phase_summary(directory/'bank_verify.json'),
                                'P1':None,'union':self.phase_summary(directory/'bank_score.json')}
                    bank_phases={phase:(directory/f'bank_{phase}.json').exists() for phase in ('prepare','verify','score')}
                    final_scoring=phases['union']
                    pipeline={'feedback_done':len(cache.feedback_leaves),
                              'cumulative':cumulative,
                              'bank_phase_complete':bank_phases,
                              'bank_reused_trajectories':((phases['P0'] or bank_prepared or {}).get('cumulative_reused_trajectories',0) if cumulative else 0),
                              'generation_calls':cache.counts['rubric_generation'],
                              'repair_calls':cache.counts['rubric_repair'],
                              'match_calls':cache.counts['rubric_match'],
                              'gate_pass':cache.gate_pass,'gate_reject':cache.gate_reject,
                              'reruns_done':cache.counts['rerun'],'reruns_unparsed':cache.rerun_unparsed,
                              'one_step_reruns_done':cache.one_step_reruns,
                              'one_step_final_calls':cache.one_step_final_calls,
                              'one_step_targets':len(cache.one_step_targets),
                              'legacy_reruns_preserved':cache.legacy_reruns,
                              'scoring_requests':cache.counts['rubric_score'],'scoring_unique_nodes':len(cache.scored_nodes),
                              'live_scored_pairs':len(cache.scored_pairs),
                              'live_score_distribution':{k:cache.live_score_distribution.get(k,0) for k in ('0','1','2','NA','U')},
                              'phase_complete':({'P0':bank_phases['verify'],'P1':False,'union':bank_phases['score']} if cumulative else
                                                {p:phases[p] is not None for p in phases}),
                              'rollout_seconds':(phases['P0'] or bank_prepared or {}).get('rollout_wall_seconds'),
                              'p0_step_seconds':(phases['P0'] or bank_prepared or {}).get('step_wall_seconds'),
                              'score_distribution':{phase:(phases[phase] or {}).get('score_distribution')
                                                    for phase in ('P0','P1','union')},
                              'outcome_signal':{phase:(phases[phase] or {}).get('outcome_signal')
                                                for phase in ('P0','P1','union')},
                              'phase_scored_pairs':{phase:(phases[phase] or {}).get('scored_pairs')
                                                    for phase in ('P0','P1','union')},
                              'keep':{p:phases[p]['keep'] if phases[p] is not None else None for p in ('P0','P1')},
                              'rubric_funnel':{p:({key:phases[p][key] for key in
                                                  ('rubrics_generated','rubrics_validated','rubrics_keep','rubrics_not_validated','rubrics_incomplete')}
                                                 if phases[p] is not None else None) for p in ('P0','P1')},
                              'final_scored_nodes':final_scoring['scored_nodes'] if final_scoring else 0,
                              'final_nodes_total':final_scoring['total_nodes'] if final_scoring else len(cache.agent_paths),
                              'final_nodes_total_known':final_scoring is not None or leaves['tree_complete'],
                              'final_valid_rewards':final_scoring['valid_rewards'] if final_scoring else 0,
                              'final_na_rewards':final_scoring['na_rewards'] if final_scoring else 0,
                              'final_reward_sum':final_scoring['reward_sum'] if final_scoring else 0,
                              'final_training_rubrics':final_scoring['training_rubrics'] if final_scoring else None}
                    r.update(agent_nodes=cache.counts['agent'],doctor_calls=cache.counts['doctor_turn'],finals=cache.counts['final'],
                             depth=depth,depth_done=cache.depths[depth],
                             depth_doctor_done=cache.doctor_depths[depth],depth_final_done=cache.final_depths[depth],
                             depth_doctor_upper=leaves['depth_target'] if depth<ro['rounds'] else 0,
                             depth_capacity=self.cfg['rollout']['branching']**depth if depth else 0,
                             depth_target=leaves['depth_target'],leaves=leaves,
                             case_elapsed_seconds=elapsed(case_start,case_end),
                             pipeline=pipeline,
                             nodes_per_minute=cache.rate(),last_completed=cache.last,
                             progress=f'A{cache.counts["agent"]} D{cache.counts["doctor_turn"]} F{cache.counts["final"]}')
                    if path.exists():r['last_commit_age_s']=max(0,wall-path.stat().st_mtime)
            if idx==4 and (self.gpu4_stage or {}).get('stage')=='agent_update':
                r.update(phase='Agent update',progress=f'batch {self.gpu4_stage.get("batch")}')
            elif idx==4 and self.cfg['five_gpu'].get('execution_mode')!='shared_ready_tasks_v1':
                summary=read_json(self.root/'evaluation/BASE_raw_no_precommit/summary.json',{})
                stage=(self.gpu4_stage or {}).get('stage')
                if stage=='agent_update':r.update(phase='Agent update',progress=f'batch {self.gpu4_stage.get("batch")}')
                elif stage=='checkpoint_eval':r.update(phase='checkpoint evaluation',progress=f'checkpoint {self.gpu4_stage.get("checkpoint_cases")}')
                elif summary and summary.get('status')!='complete':
                    r.update(phase='BASE dialogue',progress=f'{summary.get("evaluated",0)}/{summary.get("dataset_cases",622)}',
                             mean_episode_seconds=summary.get('mean_episode_seconds'),base_accuracy=summary.get('accuracy'))
                elif summary:r.update(phase='BASE complete / checkpoint eval / wait',progress='622/622')
            if live_ok:
                r['live']=live
                r['active']=live.get('active',{})
                r['activity_seconds']=live.get('seconds',{})
                r['active_elapsed_seconds']={k:max(0,wall-float(t)) for k,t in live.get('active_since',{}).items()}
                if self.cfg['five_gpu'].get('execution_mode')=='shared_ready_tasks_v1':
                    r['pool_calls']=sum(v for k,v in live.get('calls',{}).items() if k.startswith('pool.execute.'))
                    r['progress'] += f' | GPU tasks {r["pool_calls"]}'
            else:r['active']=None
            if not g.get('pids') and g:r['process_state']='no GPU process'
            rows.append(r)
        pipelines=[r['pipeline'] for r in rows if r.get('pipeline')]
        summed=('feedback_done','generation_calls','repair_calls','match_calls','gate_pass','gate_reject',
                'reruns_done','reruns_unparsed','scoring_requests','scoring_unique_nodes','final_scored_nodes',
                'live_scored_pairs',
                'one_step_reruns_done','one_step_final_calls','one_step_targets','legacy_reruns_preserved',
                'final_nodes_total','final_valid_rewards','final_na_rewards','final_reward_sum')
        pipeline={k:sum(p[k] for p in pipelines) for k in summed}
        pipeline['phase_complete']={phase:sum(p['phase_complete'][phase] for p in pipelines) for phase in ('P0','P1','union')}
        for key in ('rollout_seconds','p0_step_seconds'):
            values=[p[key] for p in pipelines if p.get(key) is not None]
            pipeline[key]=sum(values);pipeline[key+'_cases']=len(values)
        pipeline['score_distribution']={}
        union_signals=[p['outcome_signal']['union'] for p in pipelines
                       if p.get('outcome_signal',{}).get('union') is not None]
        outcome_n=sum(x['targets'] for x in union_signals)
        outcome_value_sum=sum((x['mean'] or 0.0)*x['targets'] for x in union_signals)
        pipeline['outcome_signal']={'cases':len(union_signals),'targets':outcome_n,
                                    'zero':sum(x['zero'] for x in union_signals),
                                    'mixed':sum(x['mixed'] for x in union_signals),
                                    'one':sum(x['one'] for x in union_signals),
                                    'mean':outcome_value_sum/outcome_n if outcome_n else None,
                                    'mean_contribution':sum(x['contribution_sum'] for x in union_signals)/outcome_n if outcome_n else None,
                                    'groups':sum(x['groups'] for x in union_signals),
                                    'varying_groups':sum(x['varying_groups'] for x in union_signals),
                                    'varying_targets':sum(x['varying_targets'] for x in union_signals)}
        live_scores=collections.Counter()
        for p in pipelines:live_scores.update(p['live_score_distribution'])
        pipeline['live_score_distribution']={k:live_scores.get(k,0) for k in ('0','1','2','NA','U')}
        for phase in ('P0','P1','union'):
            records=[p['score_distribution'][phase] for p in pipelines if p['score_distribution'].get(phase) is not None]
            totals=collections.Counter()
            scored_pairs=0
            for dist in records:totals.update(dist)
            scored_pairs=sum(p['phase_scored_pairs'][phase] for p in pipelines
                             if p['phase_scored_pairs'].get(phase) is not None)
            pipeline['score_distribution'][phase]={'counts':{k:totals.get(k,0) for k in ('0','1','2','NA','U','NR','other')},
                                                   'cases':len(records),'n':scored_pairs}
        pipeline['cases_total']=len(pipelines)
        if cumulative:
            pipeline['cumulative']=True
            pipeline['bank_phase_complete']={phase:sum(p['bank_phase_complete'][phase] for p in pipelines)
                                             for phase in ('prepare','verify','score')}
            pipeline['bank_reused_trajectories']=sum(p['bank_reused_trajectories'] for p in pipelines)
        pipeline['final_nodes_total_known']=bool(pipelines) and all(p['final_nodes_total_known'] for p in pipelines)
        pipeline['keep']={phase:sum(p['keep'][phase] for p in pipelines if p['keep'][phase] is not None)
                          if any(p['keep'][phase] is not None for p in pipelines) else None for phase in ('P0','P1')}
        pipeline['rubric_funnel']={phase:{key:sum(p['rubric_funnel'][phase][key] for p in pipelines
                                                  if p['rubric_funnel'][phase] is not None)
                                          if any(p['rubric_funnel'][phase] is not None for p in pipelines) else None
                                          for key in ('rubrics_generated','rubrics_validated','rubrics_keep','rubrics_not_validated','rubrics_incomplete')}
                                   for phase in ('P0','P1')}
        restart=read_json(self.root/'pool_request_fanout_restart.json',{})
        return {'time':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),'run_dir':str(self.root),
                'batch':latest.parent.name if latest else None,'completed_cases':state.get('case_cursor',0),
                'total_cases':self.cfg['schedule']['cases'],'gpus':rows,'gpu_query_error':error,
                'run_elapsed_seconds':elapsed(run_start,run_end),'batch_elapsed_seconds':elapsed(batch_start,batch_end),
                'leaves':{'completed':sum(r['leaves']['completed'] for r in rows if r.get('leaves')),
                          'remaining_min':sum(r['leaves']['remaining_min'] for r in rows if r.get('leaves')),
                          'remaining_max':sum(r['leaves']['remaining_max'] for r in rows if r.get('leaves')),
                          'trees_complete':sum(r['leaves']['tree_complete'] for r in rows if r.get('leaves')),
                          'trees_total':sum('leaves' in r for r in rows)},
                'pipeline':pipeline,'committed_batches':state.get('batch_cursor',0),'restart':restart,
                'checkpoint_evaluation':pending_eval or latest_eval,
                'idle_evaluation':idle_evaluation_status(self.root),
                'cumulative_bank':cumulative_bank_status(self.root,state,latest.parent if latest else None,pipeline,self.cfg) if cumulative else None,
                'api':{'recorded_usd':self.spend,'calls':self.api_calls,'unpriced_calls':self.unpriced,
                       'timing':dict(self.api_latency),'by_purpose':{k:dict(v) for k,v in self.api_purposes.items()}},
                'sample_window_seconds':min(60,now-self.start)}


def pid_alive(pid):
    if not pid:return False
    try:os.kill(int(pid),0);return True
    except OSError:return False


def idle_evaluation_status(root):
    reference=read_json(Path(root)/'idle_eval_runtime.json',{})
    # A continued evaluator keeps its immutable job under the original output
    # directory.  The new training run only owns this small read-only link, so
    # monitoring can follow the preserved progress without rewriting the job.
    linked=False
    if not reference.get('output_dir'):
        link=read_json(Path(root)/'idle_eval_link.json',{})
        linked_output=link.get('output') or link.get('output_dir')
        if linked_output:
            reference={'output_dir':linked_output,
                       'checkpoint_cases':link.get('checkpoint_cases',0)}
            linked=True
    if not reference.get('output_dir'):return None
    out=Path(reference['output_dir'])
    if not out.is_absolute():out=Path(root)/out
    controller=read_json(out/'status.json',{})
    priority_dir=Path(reference['priority_dir']) if reference.get('priority_dir') else Path(root)/'idle_eval_priority'
    if reference.get('priority_dir') and not priority_dir.is_absolute():priority_dir=Path(root)/priority_dir
    priority=read_json(priority_dir/'priority.json',{})
    active=[];resident=[]
    for gpu,state in priority.get('gpus',{}).items():
        if any(pid_alive(row.get('owner',{}).get('pid')) for row in state.get('calls',{}).values()):active.append(gpu)
        if any(pid_alive(row.get('owner',{}).get('pid')) for row in state.get('residents',{}).values()):resident.append(gpu)
    if linked:
        # During continuation the controller is external to this run's
        # priority directory.  Its worker map is the authoritative live view.
        active.extend(str(gpu) for gpu,pid in (controller.get('worker_pids') or {}).items() if pid_alive(pid))
    if (controller.get('contract') or reference.get('contract'))=='periodic_idle_checkpoint_eval_v1':
        job=controller.get('current_job') or {}
        evaluation=job.get('evaluation') or controller.get('evaluation')
        # Read only the explicitly selected job. The old fixed C20/622 summary
        # at output/evaluation must never be relabelled as a new balanced job.
        summary={}
        if evaluation:
            directory=Path(evaluation)
            if not directory.is_absolute():directory=out/directory
            summary=read_json(directory/'summary.json',{})
        queued=controller.get('queued_updates',controller.get('queue',[]))
        return {**controller,**summary,'contract':'periodic_idle_checkpoint_eval_v1',
                'checkpoint_cases':job.get('completed_cases',job.get('checkpoint_cases',controller.get('checkpoint_cases'))),
                'optimizer_steps':job.get('optimizer_steps',controller.get('evaluating_update')),
                'dataset_cases':summary.get('dataset_cases',controller.get('dataset_cases',200)),
                'queue_depth':controller.get('queue_depth',len(queued) if isinstance(queued,list) else 0),
                'baseline_accuracy':summary.get('baseline_accuracy',controller.get('baseline_accuracy',reference.get('baseline_accuracy',.5))),
                'controller_stage':controller.get('stage','starting'),
                'controller_alive':pid_alive(controller.get('pid')),
                'active_gpus':sorted(active),'resident_gpus':sorted(resident)}
    summary=read_json(out/'evaluation/summary.json',{})
    return {**summary,'checkpoint_cases':reference.get('checkpoint_cases',20),
            'evaluated':controller.get('evaluated',summary.get('evaluated',0)),
            'controller_stage':controller.get('stage','starting'),
            'controller_alive':pid_alive(controller.get('pid')),
            'active_gpus':sorted(set(active)),'resident_gpus':sorted(set(resident))}


def render_idle_evaluation(idle):
    if not idle:return None
    accuracy=idle.get('accuracy')
    accuracy_text='-' if accuracy is None else f'{accuracy*100:.1f}%'
    stage=('done' if idle.get('status')=='complete' else
           idle.get('controller_stage','starting') if idle.get('controller_alive') else 'stopped')
    active=','.join(str(x) for x in idle.get('active_gpus',[])) or '-'
    if idle.get('contract')!='periodic_idle_checkpoint_eval_v1':
        return (f'Idle eval C{idle["checkpoint_cases"]} · {idle.get("evaluated",0)}/{idle.get("dataset_cases",622)} · '
                f'accuracy {accuracy_text} · {stage} · active GPUs {active}')
    total=idle.get('dataset_cases',200)
    evaluated=idle.get('evaluated',0)
    checkpoint=idle.get('checkpoint_cases');update=idle.get('optimizer_steps')
    identity=f'C{count(checkpoint)}/U{count(update)}' if checkpoint is not None or update is not None else 'waiting checkpoint'
    delta=''
    baseline=idle.get('baseline_accuracy')
    paired_delta=idle.get('paired_accuracy_delta')
    if paired_delta is not None:
        delta=f' · paired ΔC20 {paired_delta*100:+.1f}pp'
    elif accuracy is not None and baseline is not None and evaluated==total:
        delta=f' · ΔC20 {(accuracy-baseline)*100:+.1f}pp'
    elif baseline is not None:
        # The partial evaluated subset need not be balanced even though the
        # full dataset is. A full-set 50% reference is not a paired delta.
        delta=f' · C20 full {baseline*100:.0f}%'
    transitions=idle.get('transition_metrics') or {}
    groups=[]
    for key,label,total_key in [('baseline_correct_retained','kept','baseline_correct_total'),
                                ('baseline_wrong_recovered','recovered','baseline_wrong_total')]:
        value=transitions.get(key,idle.get(key))
        denominator=transitions.get(total_key,idle.get(total_key,100))
        if value is not None:groups.append(f'{label} {value}/{denominator}')
    group_text=' · '+' '.join(groups) if groups else ''
    return (f'Balanced{total} {identity} · {evaluated}/{total} · accuracy {accuracy_text}{delta}{group_text} · '
            f'{stage} · queued {idle.get("queue_depth",0)} latestU{count(idle.get("latest_committed_update"))} · GPUs {active}')


def cumulative_bank_status(root,state,batchdir,pipeline,cfg=None):
    """Saved bank snapshots only; never instantiate the live GPU coordinator."""
    spec=(cfg or {}).get('rubric',{}).get('cumulative_bank') or {}
    digest=state.get('rubric_bank_sha256')
    committed=read_json(Path(root)/'rubric_bank/snapshots'/f'{digest}.json',{}) if digest else {}
    frozen=read_json(Path(batchdir)/'bank_frozen.json',{}) if batchdir else {}
    before=read_json(Path(batchdir)/'bank_input.json',{}) if batchdir else {}
    metrics=read_json(Path(batchdir)/'bank_metrics.json',{}) if batchdir else {}
    guardian=read_json(Path(batchdir)/'bank_guardian.json',{}) if batchdir else {}
    before_digest=before.get('rubric_bank_sha256')
    before_bank=read_json(Path(root)/'rubric_bank/snapshots'/f'{before_digest}.json',{}) if before_digest else committed
    empty_start=bool(spec) and not state and batchdir is None and not spec.get('initial_snapshot_file')
    # An existing batch without a scope field used the legacy shortlist. A new
    # configuration must not relabel that saved batch as full-bank scoring.
    scope=before.get('scoring_scope','top_k') if before else spec.get('scoring_scope','top_k')
    guardian_spec=spec.get('guardian')
    complete=(pipeline.get('bank_phase_complete') or {})
    verify_done=complete.get('verify',0)
    cases_total=pipeline.get('cases_total',0)
    if guardian_spec is None:
        guardian_status=None
    elif guardian:
        guardian_status=('error_fail_open' if guardian.get('status')=='ERROR_FAIL_OPEN' else 'complete')
    elif cases_total and verify_done >= cases_total:
        guardian_status='checking_candidates'
    else:
        guardian_status='waiting_verification'
    return {'committed_version':state.get('rubric_bank_version'),
            'committed_size':len(committed['entries']) if 'entries' in committed else (0 if empty_start else None),
            'empty_start_pending':empty_start,
            'frozen_version':frozen.get('version'),'frozen_size':len(frozen['entries']) if 'entries' in frozen else None,
            'added_unique_rubrics':len(frozen['entries'])-len(before_bank['entries'])
                if 'entries' in frozen and 'entries' in before_bank else metrics.get('added_unique_rubrics'),
            'new_keep':(pipeline.get('keep') or {}).get('P0'),
            'reused_trajectories':metrics.get('reused_trajectories',pipeline.get('bank_reused_trajectories',0)),
            'scoring_scope':scope,'scoring_top_k':None if scope=='all' else spec.get('retrieval_top_k',8),
            'guardian_enabled':guardian_spec is not None,'guardian_status':guardian_status,
            'guardian_verify_done':verify_done,'guardian_cases_total':cases_total,
            'guardian_exact':len(guardian.get('exact',[])) if guardian else None,
            'guardian_semantic':len(guardian.get('semantic',[])) if guardian else None,
            'guardian_aliases':len(guardian.get('aliases',{})) if guardian else None,
            'guardian_accepted':len(guardian.get('accepted',[])) if guardian else None}


def render_bank_progress(s):
    p=s.get('pipeline',{});done=p.get('bank_phase_complete',{});total=p.get('cases_total',0)
    return (f'Phases prepare {done.get("prepare",0)}/{total} · verify {done.get("verify",0)}/{total} · '
            f'score {done.get("score",0)}/{total} · rerun {p.get("one_step_reruns_done",0)} · '
            f'update {s.get("committed_batches",0)}')


def render_bank_status(s):
    b=s.get('cumulative_bank') or {};p=s.get('pipeline',{})
    committed=count(b.get('committed_size'))
    if b.get('empty_start_pending'):committed+=' (not started)'
    elif b.get('committed_version') is not None:committed+=f' (v{b["committed_version"]})'
    scope='all' if b.get('scoring_scope')=='all' else f'top{b.get("scoring_top_k",8)}/state'
    frozen=str(b['frozen_size']) if b.get('frozen_size') is not None else 'pending'
    version=f', v{b["frozen_version"]}' if b.get('frozen_version') is not None else ''
    added=(f'+{b["added_unique_rubrics"]}' if b.get('added_unique_rubrics') is not None else 'pending')
    guardian=''
    if b.get('guardian_enabled'):
        status=b.get('guardian_status')
        if status=='waiting_verification':
            detail=f'waiting verify {b.get("guardian_verify_done",0)}/{b.get("guardian_cases_total",0)}'
        elif status=='checking_candidates':detail='checking candidates'
        elif status=='complete':
            detail=(f'complete exact {count(b.get("guardian_exact"))} semantic {count(b.get("guardian_semantic"))} '
                    f'aliases {count(b.get("guardian_aliases"))} accepted {count(b.get("guardian_accepted"))}')
        else:detail='ERROR fail-open'
        guardian=f' | Guardian {detail}'
    return (f'Bank stored {committed} | scoring {frozen} ({scope}{version}) | new {added} unique | '
            f'reused {b.get("reused_trajectories",0)} trajectories · new KEEP {count(b.get("new_keep"))} · '
            f'gate +{p.get("gate_pass",0)}/-{p.get("gate_reject",0)} · NA stays unscored{guardian}')


def number(v,suffix='',digits=0):return '-' if v is None else f'{v:.{digits}f}{suffix}'


def duration(seconds):
    if seconds is None:return '-'
    minutes,sec=divmod(int(max(0,seconds)),60);hours,minutes=divmod(minutes,60)
    return f'{hours}h {minutes:02d}m {sec:02d}s' if hours else f'{minutes}m {sec:02d}s'


def remaining(leaves):
    low,high=leaves['remaining_min'],leaves['remaining_max']
    return str(low) if low==high else f'{low}~{high}'


def render_outcome_signal(signal):
    signal=signal or {}
    mean=signal.get('mean_contribution')
    contribution='-' if mean is None else f'{mean:.2f}'
    return (f'Outcome/train n={signal.get("targets",0)} '
            f'O0:{signal.get("zero",0)} mix:{signal.get("mixed",0)} O1:{signal.get("one",0)} · '
            f'varying sibling groups {signal.get("varying_groups",0)}/{signal.get("groups",0)} · '
            f'mean reward part {contribution}')


def count(v):return '-' if v is None else str(v)


def final_score_text(p):
    target=str(p['final_nodes_total'])+('' if p.get('final_nodes_total_known') else '+')
    return f'{p["final_scored_nodes"]}/{target}'


def render(s):
    lines=[f'{s["time"]} | {s["batch"] or "no batch"} | completed {s["completed_cases"]}/{s["total_cases"]}',
           'Run: '+s['run_dir'],
           f'Elapsed run {duration(s.get("run_elapsed_seconds"))} | current batch {duration(s.get("batch_elapsed_seconds"))}']
    leaves=s.get('leaves',{})
    if leaves.get('trees_total'):
        lines.append(f'묶음 말단: 완료 {leaves["completed"]} | 남은 {remaining(leaves)} | 트리 완료 {leaves["trees_complete"]}/{leaves["trees_total"]}')
    pipeline=s.get('pipeline',{})
    if pipeline.get('cases_total'):
        p=pipeline;total=p['cases_total'];done=p['phase_complete']
        lines.append(f'묶음 후처리: 피드백 {p["feedback_done"]} | 루브릭 생성 {p["generation_calls"]}·보정 {p["repair_calls"]}·재사용판정 {p["match_calls"]} '
                     f'| 게이트 통과/탈락 {p["gate_pass"]}/{p["gate_reject"]} | 리런 {p["reruns_done"]} (미파싱 {p["reruns_unparsed"]})')
        lines.append(f'새 리런(Agent 1회→Doctor FINAL): 완료 {p["one_step_reruns_done"]} | 개입 대상 {p["one_step_targets"]} '
                     f'| FINAL 호출 {p["one_step_final_calls"]} | 구 프로토콜 리런 캐시 보관·미사용 {p["legacy_reruns_preserved"]}')
        lines.append(f'채점 완료 요청 {p["scoring_requests"]} (대상 응답 {p["scoring_unique_nodes"]}개) '
                     f'| 최종 학습채점 확정 {final_score_text(p)} | 유효보상 {p["final_valid_rewards"]}·NA {p["final_na_rewards"]}')
        if s.get('cumulative_bank'):
            lines += [render_bank_progress(s),render_bank_status(s)]
        else:
            lines.append(f'단계 완료: P0 {done["P0"]}/{total} | P1 {done["P1"]}/{total} | 최종채점 {done["union"]}/{total} '
                         f'| KEEP P0 {count(p["keep"]["P0"])} / P1 {count(p["keep"]["P1"])} | 업데이트 완료 {s.get("committed_batches",0)}회')
    elif s.get('cumulative_bank'):
        lines.append(render_bank_status(s))
    evaluation=s.get('checkpoint_evaluation')
    if evaluation:
        accuracy='' if evaluation.get('accuracy') is None else f' | accuracy {100*evaluation["accuracy"]:.1f}%'
        label='running' if evaluation['status']!='complete' else 'complete'
        lines.append(f'622-set evaluation · checkpoint {evaluation["checkpoint_cases"]} · '
                     f'{evaluation["evaluated"]}/{evaluation["total"]} · {label}{accuracy}')
    idle_line=render_idle_evaluation(s.get('idle_evaluation'))
    if idle_line:lines.append(idle_line)
    lines.append('GPU  util  avg60  VRAM GiB   watt   stage / progress')
    for r in s['gpus']:
        active=r.get('active')
        activity=','.join(f'{k}:{v}' for k,v in active.items()) if active else None
        phase=activity or r['phase']
        lines.append(f'{r["gpu"]:>3}  {number(r.get("util"),"%"):>4}  {number(r.get("mean_util_60s"),"%"):>5} '
                     f'{number(r.get("memory_mb",0)/1024,"",1) if "memory_mb" in r else "-":>9} '
                     f'{number(r.get("power_w"),"W"):>6}   {phase} | {r["progress"]}')
        if r.get('depth'):
            rate=number(r.get('nodes_per_minute'),'/min',1)
            lines.append(f'     case {r["case_index"]+1} | elapsed {duration(r.get("case_elapsed_seconds"))} '
                         f'| depth {r["depth"]}: Agent {r["depth_done"]}/{r.get("depth_target",r["depth_capacity"])} '
                         f'| Agent {rate} | last completed: {r["last_completed"]} ({number(r["last_commit_age_s"],"s")} ago)')
        elif r.get('mean_episode_seconds') is not None:
            lines.append(f'     BASE mean {r["mean_episode_seconds"]:.1f}s/problem | accuracy {100*r["base_accuracy"]:.1f}%')
        live=r.get('live',{})
        if r.get('leaves'):
            leaf=r['leaves'];label='완료' if leaf['tree_complete'] else '진행 중'
            lines.append(f'     말단: 완료 {leaf["completed"]} | 남은 {remaining(leaf)} | 롤아웃 트리 {label}')
            if r.get('depth') and not leaf['tree_complete']:
                doctor=(f'완료 {r["depth_doctor_done"]} (최대 {r["depth_doctor_upper"]})'
                        if r['depth_doctor_upper'] else '턴 제한으로 생략')
                lines.append(f'     현재 깊이 {r["depth"]} 호출: Doctor {doctor} | FINAL 완료 {r["depth_final_done"]}')
        if r.get('pipeline'):
            p=r['pipeline'];done=p['phase_complete'];ready='완료' if done['union'] else '대기/진행'
            lines.append(f'     후처리: 피드백 {p["feedback_done"]} | 생성 {p["generation_calls"]}·보정 {p["repair_calls"]} '
                         f'| 게이트 통과/탈락 {p["gate_pass"]}/{p["gate_reject"]} | 리런 {p["reruns_done"]} '
                         f'| 채점요청 {p["scoring_requests"]} | 학습채점 확정 {final_score_text(p)} ({ready})')
            lines.append(f'     새 리런 완료 {p["one_step_reruns_done"]} | 개입 대상 {p["one_step_targets"]} '
                         f'| Doctor FINAL {p["one_step_final_calls"]} | 구 프로토콜 캐시 {p["legacy_reruns_preserved"]}건(미사용)')
            if p.get('cumulative'):
                lines.append(f'     Bank new KEEP {count(p["keep"]["P0"])} · reused {p.get("bank_reused_trajectories",0)} trajectories')
            else:
                lines.append(f'     P0 {"완료" if done["P0"] else "대기/진행"} · P1 {"완료" if done["P1"] else "대기/진행"} '
                             f'| KEEP P0 {count(p["keep"]["P0"])} / P1 {count(p["keep"]["P1"])}')
        if live.get('pool_last_case_id'):
            lines.append(f'     last GPU task: {live["pool_last_operation"]} for {live["pool_last_case_id"]} (coordinator GPU{live["pool_last_source_gpu"]})')
    api=s['api'];lines.append(f'API recorded ${api["recorded_usd"]:.4f} | calls {api["calls"]} | unpriced {api["unpriced_calls"]}')
    lines.append(f'Average uses sampled data from last {s["sample_window_seconds"]:.0f}s (warmup until 60s). A/D/F = completed Agent/Doctor/FINAL calls.')
    lines.append('Stage is inferred from saved progress unless live instrumentation is present; last completed != currently active.')
    if leaves.get('trees_total'):
        lines.append('남은 말단은 초기 롤아웃의 최소~최대 범위(조기 종료에 따라 변동). 루브릭·리런 작업은 별도. Elapsed includes waits.')
        lines.append('Doctor 최대 수는 현재 깊이의 분기 수(Agent 조기 종료 시 감소). Doctor 단계가 끝난 뒤 FINAL이 처리되어 말단 수가 증가.')
    if pipeline.get('cases_total'):
        lines.append('채점요청에는 개입 검증 채점도 포함. 최종 학습채점은 union 저장으로 확정. KEEP - = 미확정, 노드수 + = 아직 증가 가능. NA는 실제 0점과 구분.')
    restart=s.get('restart') or {}
    if restart.get('rerun_transition'):
        lines.append(f'재시작: 저장 트리 호출캐시 재사용 | 이전 P0 결과 {len(restart["rerun_transition"].get("old_p0_results_archived",[]))}개 별도 보관 | 새 프로토콜로 P0 재계산')
    if s['gpu_query_error']:lines.append('GPU query unavailable: '+s['gpu_query_error'][:160])
    if s.get('handoff'):
        h=s['handoff']
        if h.get('agent_batch_size') is not None and h.get('doctor_concurrency') is not None:
            capacity=f'Agent batch {h["agent_batch_size"]} / Doctor requests {h["doctor_concurrency"]}'
        else:capacity=f'Agent/Doctor parallel {h.get("parallel","-")}'
        lines.append(f'Controller: {h.get("stage")} | {capacity}')
        if h.get('stage')=='waiting_for_batch_halt4':
            lines.append(f'4턴 종료조건 예약: batch {h["target_batch"]} 업데이트·저장 완료 뒤 자동 재개 '
                         '(4턴부터 선택 안정·반론 없음이면 종료, 공유 큐 Agent20/Doctor5 제출)')
            if h.get('pending_rubric_schema'):
                lines.append('루브릭 구조 예약: User 상황 / Agent의 이전 발화·행동 / 0·1·2점 행동 (숨겨진 belief 제외)')
            if h.get('pending_candidate_seed'):
                lines.append('P1 파이프라인 예약: 첫 P0 완료 case로 공통 P1 고정 → case별 평가·학습채점 즉시 → 5문제 전체 점수로 승자 선택')
        if h.get('rubric_schema'):
            lines.append('루브릭: User 상황 / 이전 Agent 행동 / 점수별 행동')
        if h.get('candidate_seed'):
            lines.append('공통 P1: 첫 P0 완료 case 기반, case별 평가·학습채점 즉시, 최종 선택은 5문제 전체 점수')
        if h.get('pending_repeat_agent_reply_after_turns') is not None:
            lines.append(f'반복응답 종료 예약: Agent {h["pending_repeat_agent_reply_after_turns"]}턴부터 이전 발화와 문장부호·대소문자 제외 완전 일치하면 즉시 종료')
        if h.get('repeat_agent_reply_after_turns') is not None:
            lines.append(f'반복응답 종료 적용 중: Agent {h["repeat_agent_reply_after_turns"]}턴부터 이전 발화와 정규화 후 완전 일치하면 즉시 종료')
        if h.get('stable_choice_no_objection_after_turns') is not None:
            lines.append(f'추가 종료조건 적용 중: Agent {h["stable_choice_no_objection_after_turns"]}턴부터 '
                         '선택 안정·반론 없음이면 종료 (기존 최소 2턴 조건 유지)')
        if h.get('case_agent_workers') is not None and h.get('case_doctor_workers') is not None:
            lines.append(f'공유 큐 제출: case당 Agent 최대 {h["case_agent_workers"]} / Doctor 최대 {h["case_doctor_workers"]}개 '
                         '(GPU별 처리 용량과 별도)')
        if h.get('stage')=='resume_launched' and not (ROOT/'outputs/diagnostics/doctor_parallel_20261008/joint_decision.json').exists():
            lines.append('1/4/8 benchmark incomplete; resumed fallback configuration. See joint_benchmark.log.')
    if s.get('benchmark'):
        b=s['benchmark'];lines.append(f'Benchmark: {b.get("stage")} | parallel {b.get("parallel","-")} | repetition {b.get("repeat",-1)+1}/2 | training paused')
        for r in s.get('benchmark_results',[]):
            lines.append(f'  p{r["parallel"]} run{r["repeat"]+1}: Doctor {r["wall_seconds"]:.1f}s | Agent {r.get("agent_wall_seconds",0):.1f}s')
    return '\n'.join(lines)


def render_dashboard(s):
    """Short live view: spend/progress first, one compact status row per GPU."""
    api=s.get('api',{})
    batch=s.get('batch') or 'no batch'
    lines=[f'{s["time"]}  {batch} · run {duration(s.get("run_elapsed_seconds"))} · '
           f'batch {duration(s.get("batch_elapsed_seconds"))} · '
           f'API recorded ${api.get("recorded_usd",0):.2f} · {api.get("calls",0)} calls'
           f'{" · unpriced " + str(api["unpriced_calls"]) if api.get("unpriced_calls") else ""}']
    leaves=s.get('leaves',{})
    if leaves.get('trees_total'):
        lines.append(f'Cases {s.get("completed_cases",0)}/{s.get("total_cases",0)} · '
                     f'leaves {leaves.get("completed",0)} done / {remaining(leaves)} left · '
                     f'trees {leaves.get("trees_complete",0)}/{leaves.get("trees_total",0)}')
    evaluation=s.get('checkpoint_evaluation')
    if evaluation:
        accuracy='' if evaluation.get('accuracy') is None else f' · accuracy {100*evaluation["accuracy"]:.1f}%'
        status='running' if evaluation['status']!='complete' else 'done'
        lines.append(f'622 eval · checkpoint {evaluation["checkpoint_cases"]} · '
                     f'{evaluation["evaluated"]}/{evaluation["total"]} · {status}{accuracy}')
    idle_line=render_idle_evaluation(s.get('idle_evaluation'))
    if idle_line:lines.append(idle_line)
    p=s.get('pipeline',{})
    if p.get('cases_total'):
        done=p.get('phase_complete',{})
        if s.get('cumulative_bank'):
            lines.append(render_bank_progress(s))
        else:
            lines.append(f'Phases · P0 verified {done.get("P0",0)}/{p["cases_total"]} · '
                         f'P1 verified {done.get("P1",0)}/{p["cases_total"]} · '
                         f'final scoring {done.get("union",0)}/{p["cases_total"]} · '
                         f'rerun {p.get("one_step_reruns_done",0)} · updates {s.get("committed_batches",0)}')
        funnel=p.get('rubric_funnel',{})
        def rubric_status(phase):
            f=funnel.get(phase)
            if f is None or f.get('rubrics_validated') is None:return f'{phase} verification pending'
            return (f'{phase} KEEP {f["rubrics_keep"]}/{f["rubrics_validated"]}'
                    f' (new {f["rubrics_generated"]}){incomplete_suffix(f)}')
        if s.get('cumulative_bank'):lines.append(render_bank_status(s))
        else:lines.append('Rubrics confirmed · '+' | '.join(rubric_status(phase) for phase in ('P0','P1'))+
                          f' · gate +{p.get("gate_pass",0)}/-{p.get("gate_reject",0)}')
        lines.append(render_outcome_signal(p.get('outcome_signal')))
        if p.get('rollout_seconds_cases'):
            lines.append(f'Rollout {duration(p["rollout_seconds"])} total · '
                         f'{duration(p["rollout_seconds"]/p["rollout_seconds_cases"])} avg')
        train=p['score_distribution']['union']
        if train['cases']:
            counts=train['counts']
            lines.append(f'Train scores ({train["cases"]}/{p["cases_total"]} cases) · n={train["n"]} '
                         f'0:{counts["0"]} 1:{counts["1"]} 2:{counts["2"]} NA:{counts["NA"]}{unscorable_suffix(counts)}')
        else:
            lines.append(f'Train scores · waiting for final rubric set (union {done.get("union",0)}/{p["cases_total"]})')
    elif s.get('cumulative_bank'):
        lines.append(render_bank_status(s))
    def active_task(row):
        elapsed=row.get('active_elapsed_seconds',{})
        preferred=('agent_generate','agent_batch_generate','doctor_generate','doctor_queue_wait',
                   'role.rubric_generator','role.rubric_scorer','role.rubric_gate',
                   'role.feedback_generator','role.prompt_reflection','role.prompt_refiner')
        key=next((name for name in preferred if name in elapsed),None)
        if key is None and elapsed:
            key=min(elapsed,key=elapsed.get)
        if key is None:return None
        label=key.removeprefix('role.').replace('_',' ')
        aliases={'agent generate':'Agent','agent batch generate':'Agent batch',
                 'doctor generate':'Doctor','doctor queue wait':'Doctor queue'}
        return f'{aliases.get(label,label)} {duration(elapsed[key])}'
    for r in s.get('gpus',[]):
        util=number(r.get('util'),'%').rjust(4)
        mem=(f'{r["memory_mb"]/1024:.0f}G'
             if r.get('memory_mb') is not None and r.get('total_mb') else '-')
        if r.get('case_index') is not None:
            leaf=r.get('leaves',{})
            work=(f'case {r["case_index"]+1} {r.get("phase")} · '
                  f'A{r.get("agent_nodes",0)} D{r.get("doctor_calls",0)} F{r.get("finals",0)} · '
                  f'L{leaf.get("completed",0)}/{remaining(leaf)}')
        else:
            work=f'{r.get("phase","idle")} · {r.get("progress","-")}'
            if r.get('mean_episode_seconds') is not None:
                work+=f' · {100*r["base_accuracy"]:.1f}% · {r["mean_episode_seconds"]:.1f}s/case'
        tasks=r.get('pool_calls')
        if tasks is not None:work+=f' · T{tasks}'
        active=active_task(r)
        if active:work+=' · '+active
        lines.append(f'GPU{r.get("gpu","-")} {util} {mem} · {work}')
        case_pipeline=r.get('pipeline')
        if case_pipeline:
            funnel=case_pipeline.get('rubric_funnel',{})
            def keep_text(phase):
                f=funnel.get(phase)
                if f is None or f.get('rubrics_validated') is None:return f'{phase} pending'
                return f'{phase} {f["rubrics_keep"]}/{f["rubrics_validated"]}{incomplete_suffix(f)}'
            dist,scored,_=case_score_distribution(case_pipeline)
            keep_label=(f'new KEEP {count(case_pipeline.get("keep",{}).get("P0"))} · reused {case_pipeline.get("bank_reused_trajectories",0)}'
                        if case_pipeline.get('cumulative') else f'KEEP {keep_text("P0")} / {keep_text("P1")}')
            lines.append(f'     case stats · {keep_label} · '
                         f'score n={scored} 0:{dist.get("0",0)} 1:{dist.get("1",0)} '
                         f'2:{dist.get("2",0)} NA:{dist.get("NA",0)}{unscorable_suffix(dist)}')
    if s.get('gpu_query_error'):
        lines.append('GPU telemetry unavailable: '+s['gpu_query_error'][:100])
    timing=api.get('timing',{})
    if timing.get('timed_calls'):
        avg=timing.get('latency_seconds',0)/timing['timed_calls']
        lines.append(f'API time {duration(timing.get("latency_seconds",0))} total · {avg:.1f}s avg · '
                     f'retry {duration(timing.get("retry_wait_seconds",0))}')
    if p.get('cases_total'):
        lines.append('Per-GPU KEEP/scores follow the assigned case; shared-queue work may execute on another GPU.')
    return '\n'.join(lines)


def render_compact_dashboard(s):
    """One terminal line per GPU so all five workers fit on short screens."""
    api=s.get('api',{}); p=s.get('pipeline',{}); leaves=s.get('leaves',{})
    batch=s.get('batch') or 'no batch'
    lines=[f'{s["time"]} {batch} · run {duration(s.get("run_elapsed_seconds"))} · '
           f'batch {duration(s.get("batch_elapsed_seconds"))} · '
           f'recorded ${api.get("recorded_usd",0):.2f} · {api.get("calls",0)} API calls'
           f'{" · unpriced " + str(api["unpriced_calls"]) if api.get("unpriced_calls") else ""}']
    if leaves.get('trees_total'):
        done=p.get('phase_complete',{})
        lines.append(f'Cases {s.get("completed_cases",0)}/{s.get("total_cases",0)} · '
                     f'leaves {leaves.get("completed",0)}/{remaining(leaves)} · '
                     f'trees {leaves.get("trees_complete",0)}/{leaves.get("trees_total",0)}')
        if s.get('cumulative_bank'):
            lines.append(render_bank_progress(s))
        else:
            lines.append(f'Phases P0 verified {done.get("P0",0)}/{p["cases_total"]} · '
                         f'P1 verified {done.get("P1",0)}/{p["cases_total"]} · '
                         f'final scored {done.get("union",0)}/{p["cases_total"]} · '
                         f'rerun {p.get("one_step_reruns_done",0)} · update {s.get("committed_batches",0)}')
        funnel=p.get('rubric_funnel',{})
        def compact_rubrics(phase):
            f=funnel.get(phase)
            if f is None or f.get('rubrics_validated') is None:return f'{phase} verifying'
            return f'{phase} KEEP {f["rubrics_keep"]}/{f["rubrics_validated"]} (new {f["rubrics_generated"]}){incomplete_suffix(f)}'
        if s.get('cumulative_bank'):lines.append(render_bank_status(s))
        else:lines.append('Rubrics · '+compact_rubrics('P0')+' · '+compact_rubrics('P1')+
                          f' · gate +{p.get("gate_pass",0)}/-{p.get("gate_reject",0)}')
        lines.append(render_outcome_signal(p.get('outcome_signal')))
        train=p.get('score_distribution',{}).get('union',{})
        if train.get('cases'):
            counts=train['counts']
            lines.append(f'Train scores {train["cases"]}/{p["cases_total"]} cases · n={train["n"]} '
                         f'0:{counts["0"]} 1:{counts["1"]} 2:{counts["2"]} NA:{counts["NA"]}{unscorable_suffix(counts)}')
        else:
            lines.append(f'Train scores waiting · final set scored {done.get("union",0)}/{p["cases_total"]}')
    elif s.get('cumulative_bank'):
        lines.append(render_bank_status(s))
    idle_line=render_idle_evaluation(s.get('idle_evaluation'))
    if idle_line:lines.append(idle_line)
    for r in s.get('gpus',[]):
        util=number(r.get('util'),'%').rjust(3)
        mem=(f'{r["memory_mb"]/1024:.0f}G' if r.get('memory_mb') is not None and r.get('total_mb') else '-')
        idx=r.get('gpu','-')
        if r.get('case_index') is not None:
            leaf=r.get('leaves',{})
            keep=r.get('pipeline',{}).get('rubric_funnel',{})
            def case_keep(phase):
                f=keep.get(phase)
                if f is None or f.get('rubrics_validated') is None:return '-'
                return f'{f["rubrics_keep"]}/{f["rubrics_validated"]}{incomplete_suffix(f)}'
            case_pipeline=r.get('pipeline',{})
            dist,n,score_label=case_score_distribution(case_pipeline)
            keep_label=(f'newK{case_keep("P0")} R{case_pipeline.get("bank_reused_trajectories",0)}'
                        if case_pipeline.get('cumulative') else f'K{case_keep("P0")}/{case_keep("P1")}')
            work=(f'C{r["case_index"]+1} {r.get("phase")} · '
                  f'A{r.get("agent_nodes",0)} D{r.get("doctor_calls",0)} F{r.get("finals",0)} · '
                  f'L{leaf.get("completed",0)}/{remaining(leaf)} · '
                  f'{keep_label} · '
                  f'{score_label} score n={n} [0:{dist.get("0",0)} 1:{dist.get("1",0)} '
                  f'2:{dist.get("2",0)} NA:{dist.get("NA",0)}{unscorable_suffix(dist)}]')
        else:
            work=f'{r.get("phase","idle")} · {r.get("progress","-")}'
        active=next((k for k in ('agent_generate','agent_batch_generate','doctor_generate','role.rubric_generator',
                                 'role.rubric_scorer','role.rubric_gate','role.feedback_generator',
                                 'role.prompt_reflection','role.prompt_refiner') if k in r.get('active_elapsed_seconds',{})),None)
        if active:
            label=active.removeprefix('role.').replace('_',' ')
            work+=f' · {label} {duration(r["active_elapsed_seconds"][active])}'
        lines.append(f'GPU{idx} {util} {mem} | {work}')
    timing=api.get('timing',{})
    if timing.get('timed_calls'):
        avg=timing.get('latency_seconds',0)/timing['timed_calls']
        lines.append(f'API time {duration(timing.get("latency_seconds",0))} · {avg:.1f}s avg · '
                     f'retry {duration(timing.get("retry_wait_seconds",0))}')
    if s.get('gpu_query_error'):lines.append('GPU telemetry unavailable: '+s['gpu_query_error'][:100])
    return '\n'.join(lines)


def main():
    from src.online_rl.tree.experiment import load_config
    p=argparse.ArgumentParser();p.add_argument('--config',help='Pin a configuration; default follows the current controller')
    p.add_argument('--run-dir');p.add_argument('--interval',type=float,default=2);p.add_argument('--once',action='store_true');p.add_argument('--json',action='store_true');p.add_argument('--compact',action='store_true',help='One line per GPU; fits five GPUs on short terminals')
    args=p.parse_args()
    if args.interval<.5:p.error('--interval must be >=0.5 seconds')
    monitor=None;current=None
    try:
        while True:
            handoff=read_json(ROOT/'outputs/diagnostics/doctor_parallel_20261008/handoff_status.json',{})
            if args.config and handoff.get('config')!=args.config:handoff={}
            if args.config:
                cfg=load_config(ROOT/args.config)
                root=ROOT/(args.run_dir or cfg['output_dir'])
            elif args.run_dir:
                root=Path(args.run_dir)
                if not root.is_absolute():root=ROOT/root
                saved=read_json(root/'run_manifest.json',{})
                cfg=saved.get('config') or load_config(ROOT/'configs/online_rl/tree_v21_five_gpu_shared5.json')
            else:
                config=handoff.get('config') or 'configs/online_rl/tree_v21_five_gpu_shared5.json'
                cfg=load_config(ROOT/config)
                root=ROOT/handoff.get('run_dir',cfg['output_dir'])
            if current!=str(root):monitor=Monitor(root,cfg);current=str(root)
            s=monitor.snapshot();s['handoff']=handoff;output=json.dumps(s,ensure_ascii=False) if args.json else (render_compact_dashboard(s) if args.compact else render_dashboard(s))
            if handoff.get('benchmark_dir'):
                benchdir=ROOT/handoff['benchmark_dir'];s['benchmark']=read_json(benchdir/'progress.json',{})
                s['benchmark_results']=read_json(benchdir/'results.json',[])
                b=s['benchmark']
                for r in s['gpus']:
                    r['phase']='training paused';r['progress']='-';r['active']=None
                    if r['gpu']==b.get('gpu'):
                        r['phase']='benchmark '+b.get('stage','');r['progress']=f'parallel {b.get("parallel","-")}'
                output=json.dumps(s,ensure_ascii=False) if args.json else render_dashboard(s)
            if sys.stdout.isatty() and not args.once and not args.json:print('\033[2J\033[H',end='')
            print(output,flush=True)
            if args.once:break
            time.sleep(args.interval)
    except KeyboardInterrupt:pass

if __name__=='__main__':main()
