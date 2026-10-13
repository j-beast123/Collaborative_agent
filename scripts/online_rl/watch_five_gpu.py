"""Read-only live view of GPU0..4, worker caches, BASE progress and API spend.

Works on an already-running job without restart. Cache rows are completed calls;
last completed call is never represented as the operation currently executing.
"""
import argparse
import collections
import csv
import fcntl
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
        # Final witness rows are persisted only when P0/P1 finishes. Rebuild
        # confirmed KEEP evidence from completed cache rows for live display.
        self.witness_applicable=set();self.witness_outcomes=collections.defaultdict(dict)
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
                    if result.get('applicable') is False:label='NA'
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
            elif kind=='rubric_applicability':
                ident=row.get('identity') or {};rubric=ident.get('rubric');node=ident.get('node')
                result=(row.get('value') or {}).get(rubric) if rubric else None
                if rubric and node and isinstance(result,dict) and result.get('applicable') is True:
                    self.witness_applicable.add((rubric,node))
            elif kind=='posthoc_gate':
                passed=(row.get('value') or {}).get('passed')
                self.gate_pass+=passed is True;self.gate_reject+=passed is False
            elif kind=='rerun':
                self.rerun_unparsed+=(row.get('value') or {}).get('outcome') is None
                ident=row.get('identity') or {}
                if ident.get('continuation_protocol')=='intervention_agent_once_doctor_final_once_v1':
                    self.one_step_reruns+=1
                    rubric,node,level=ident.get('rubric'),ident.get('node'),ident.get('level')
                    outcome=(row.get('value') or {}).get('outcome')
                    if rubric and node and level in ('0','2'):
                        # None is an executed miss, never evidence of a target hit.
                        self.witness_outcomes[(rubric,node,level)][ident.get('seed') or row.get('key')]=outcome
                else:self.legacy_reruns+=1
            elif kind=='rerun_doctor_turn':
                owner=(row.get('identity') or {}).get('rerun') or {}
                if owner.get('continuation_protocol')=='intervention_agent_once_doctor_final_once_v1':
                    turn=(row.get('identity') or {}).get('turn') or {}
                    if turn.get('stage')=='FINAL':self.one_step_final_calls+=1
                    self.one_step_targets.add((owner.get('case_id'),owner.get('rubric'),owner.get('node'),owner.get('level')))
        self.history.append((now,self.counts['agent']))
        while len(self.history)>1 and self.history[1][0]<now-60:self.history.popleft()
    def rate(self):
        if len(self.history)<2:return None
        a,b=self.history[0],self.history[-1]
        return 60*(b[1]-a[1])/(b[0]-a[0]) if b[0]>a[0] else None
    def live_witness_keep_ids(self):
        """Rubrics with >=2 parsed target outcomes at an applicable support."""
        kept=set()
        for (rubric,node,level),runs in self.witness_outcomes.items():
            if (rubric,node) not in self.witness_applicable:continue
            target=1.0 if level=='2' else 0.0
            if sum(value is not None and value==target for value in runs.values())>=2:
                kept.add(rubric)
        return kept


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


def runner_lock_active(root):
    """The run-directory flock is authoritative across relaunches; handoff PIDs can be stale."""
    path=Path(root)/'runner.lock'
    if not path.exists():return False
    try:
        with path.open('a') as file:
            try:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return True
            fcntl.flock(file,fcntl.LOCK_UN)
            return False
    except OSError:return False


def elapsed(start,end):
    return max(0,end-start) if start is not None else None


def wilson_interval(correct,total,z=1.959963984540054):
    """Approximate 95% binomial interval for a partial checkpoint-eval sample."""
    if total <= 0:return None
    p=correct/total;z2=z*z;den=1+z2/total
    center=(p+z2/(2*total))/den
    margin=z*((p*(1-p)/total+z2/(4*total*total))**0.5)/den
    return max(0.0,center-margin),min(1.0,center+margin)


def summarize_phase(record):
    """Compact confirmed results; None rewards remain distinct from real zero."""
    targets=(record.get('agent_targets') or {}).get('nodes',{})
    rewarded=[v['reward'] for v in targets.values() if v.get('reward') is not None]
    nodes=record.get('tree',{}).get('nodes',[])
    scores=record.get('response_scores',[])
    witness=record.get('rubric_bank',{}).get('witness',[])
    telemetry=record.get('rubric_bank',{}).get('telemetry',{})
    score_distribution=collections.Counter()
    for item in scores:
        hard=item.get('hard')
        if not item.get('applicable', hard is not None) or (hard is None and item.get('score') is None):
            score_distribution['NA']+=1
        else:
            if hard is None:
                expected=float(item['score'])
                label=str(int(round(expected*2))) if expected in (0.0,0.5,1.0) else 'other'
            else:
                # ResponseRubricScore.hard is normalized by hard_value(): label / 2.
                level=int(round(float(hard)*2))
                label=str(level) if level in (0,1,2) else 'other'
            score_distribution[label]+=1
    return {'total_nodes':len(nodes),
            'scored_nodes':len({s['response_node_id'] for s in scores}),
            'valid_rewards':len(rewarded),'na_rewards':sum(v.get('reward') is None for v in targets.values()),
            'reward_sum':sum(rewarded),'keep':sum(w.get('decision')=='KEEP' for w in witness),
            'not_validated':sum(w.get('decision')=='NOT_VALIDATED' for w in witness),
            'rollout_wall_seconds':(record.get('timings') or {}).get('rollout_wall_seconds'),
            'step_wall_seconds':record.get('seconds'),
            'score_distribution':{k:score_distribution.get(k,0) for k in ('0','1','2','NA','other')},
            'scored_pairs':len(scores),
            'training_rubrics':len(record.get('rubric_weights',{})),
            'union_rubrics':len(record.get('training_rubrics',[])),
            # `rubrics` is the canonical set that reached witness validation;
            # KEEP is the subset with a causal witness. Keep NA separate.
            'rubrics_generated':int(telemetry.get('created_new',0)),
            'rubrics_validated':int(telemetry.get('rubrics',len(witness))),
            'rubrics_keep':int(telemetry.get('keep',sum(w.get('decision')=='KEEP' for w in witness))),
            'keep_rubric_ids':[w.get('rubric_id') for w in witness if w.get('decision')=='KEEP' and w.get('rubric_id')],
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
        runner_active=runner_lock_active(self.root)
        eval_jobs=[]
        for job in sorted((self.root/'evaluation').glob('checkpoint_*')):
            manifest=read_json(job/'manifest.json',{})
            summary=read_json(job/'summary.json',{})
            total=int(manifest.get('settings',{}).get('eval_cases',len(manifest.get('case_ids',[]))))
            # The evaluator atomically commits each case before refreshing its
            # aggregate summary (which may only happen every 25 cases). Read
            # those commits directly so the live estimate advances per case.
            case_files=sorted((job/'cases').glob('case_*.json'))
            outcomes=[read_json(path,{}) for path in case_files]
            outcomes=[row.get('outcome') for row in outcomes if isinstance(row,dict)]
            outcomes=[value for value in outcomes if value in (0,1,0.0,1.0) or value is None]
            evaluated=len(outcomes)
            correct=sum(value in (1,1.0) for value in outcomes)
            unparsed=sum(value is None for value in outcomes)
            accuracy=correct/evaluated if evaluated else None
            ci=wilson_interval(int(correct),evaluated) if correct is not None else None
            eval_jobs.append({'checkpoint_cases':manifest.get('completed_cases'), 'evaluated':evaluated,
                              'total':total, 'status':summary.get('status','pending'),
                              'accuracy':accuracy,'correct':correct,'unparsed':unparsed,
                              'accuracy_ci95':ci})
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
                    if (directory/'union.json').exists() or (directory/'result.json').exists():r['phase']='waiting at batch barrier'
                    elif (directory/'P1.json').exists():r['phase']='P0+P1 union scoring / wait'
                    elif (directory/'P0.json').exists():r['phase']='P1 verification / reflection wait'
                    elif cache.counts['feedback'] or any('rubric' in k for k in cache.counts):r['phase']='P0 rubric / rerun'
                    else:r['phase']='P0 rollout'
                    depth=max(cache.depths,default=0)
                    ro=self.cfg['rollout']
                    leaves=leaf_progress(cache.agent_paths,cache.final_paths,ro['branching'],ro['rounds'],
                                         ro.get('halting',{}).get('min_agent_turns',2 if ro.get('doctor_backend') else 1))
                    case_start=first_record_time(directory/'execution_runtime.jsonl') or batch_start
                    case_end=file_time(directory/'union.json') or file_time(directory/'result.json') or wall
                    phases={p:self.phase_summary(directory/(p+'.json')) for p in ('P0','P1','union')}
                    if phases['union'] is None and (directory/'result.json').exists():
                        phases['union']=self.phase_summary(directory/'result.json')
                    final_scoring=phases['union']
                    saved_keep_ids={rubric_id for phase in ('P0','P1') if phases[phase]
                                    for rubric_id in phases[phase].get('keep_rubric_ids',[])}
                    live_witness_keep_ids=cache.live_witness_keep_ids()-saved_keep_ids
                    pipeline={'feedback_done':len(cache.feedback_leaves),
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
                              'live_witness_keeps':len(live_witness_keep_ids),
                              'live_score_distribution':{k:cache.live_score_distribution.get(k,0) for k in ('0','1','2','NA')},
                              'phase_complete':{p:phases[p] is not None for p in phases},
                              'rollout_seconds':(phases['P0'] or {}).get('rollout_wall_seconds'),
                              'p0_step_seconds':(phases['P0'] or {}).get('step_wall_seconds'),
                              'score_distribution':{phase:(phases[phase] or {}).get('score_distribution')
                                                    for phase in ('P0','P1','union')},
                              'phase_scored_pairs':{phase:(phases[phase] or {}).get('scored_pairs')
                                                    for phase in ('P0','P1','union')},
                              'keep':{p:phases[p]['keep'] if phases[p] is not None else None for p in ('P0','P1')},
                              'rubric_funnel':{p:({key:phases[p][key] for key in
                                                  ('rubrics_generated','rubrics_validated','rubrics_keep','rubrics_not_validated')}
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
                'live_scored_pairs','live_witness_keeps',
                'one_step_reruns_done','one_step_final_calls','one_step_targets','legacy_reruns_preserved',
                'final_nodes_total','final_valid_rewards','final_na_rewards','final_reward_sum')
        pipeline={k:sum(p[k] for p in pipelines) for k in summed}
        pipeline['phase_complete']={phase:sum(p['phase_complete'][phase] for p in pipelines) for phase in ('P0','P1','union')}
        for key in ('rollout_seconds','p0_step_seconds'):
            values=[p[key] for p in pipelines if p.get(key) is not None]
            pipeline[key]=sum(values);pipeline[key+'_cases']=len(values)
        pipeline['score_distribution']={}
        live_scores=collections.Counter()
        for p in pipelines:live_scores.update(p['live_score_distribution'])
        pipeline['live_score_distribution']={k:live_scores.get(k,0) for k in ('0','1','2','NA')}
        for phase in ('P0','P1','union'):
            records=[p['score_distribution'][phase] for p in pipelines if p['score_distribution'].get(phase) is not None]
            totals=collections.Counter()
            scored_pairs=0
            for dist in records:totals.update(dist)
            scored_pairs=sum(p['phase_scored_pairs'][phase] for p in pipelines
                             if p['phase_scored_pairs'].get(phase) is not None)
            pipeline['score_distribution'][phase]={'counts':{k:totals.get(k,0) for k in ('0','1','2','NA','other')},
                                                   'cases':len(records),'n':scored_pairs}
        pipeline['cases_total']=len(pipelines)
        pipeline['final_nodes_total_known']=bool(pipelines) and all(p['final_nodes_total_known'] for p in pipelines)
        pipeline['keep']={phase:sum(p['keep'][phase] for p in pipelines if p['keep'][phase] is not None)
                          if any(p['keep'][phase] is not None for p in pipelines) else None for phase in ('P0','P1')}
        pipeline['rubric_funnel']={phase:{key:sum(p['rubric_funnel'][phase][key] for p in pipelines
                                                  if p['rubric_funnel'][phase] is not None)
                                          if any(p['rubric_funnel'][phase] is not None for p in pipelines) else None
                                          for key in ('rubrics_generated','rubrics_validated','rubrics_keep','rubrics_not_validated')}
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
                'runner_active':runner_active,
                'checkpoint_evaluation':pending_eval or latest_eval,
                'api':{'recorded_usd':self.spend,'calls':self.api_calls,'unpriced_calls':self.unpriced,
                       'timing':dict(self.api_latency),'by_purpose':{k:dict(v) for k,v in self.api_purposes.items()}},
                'sample_window_seconds':min(60,now-self.start)}


def pid_alive(pid):
    if not pid:return False
    try:os.kill(int(pid),0);return True
    except OSError:return False


def number(v,suffix='',digits=0):return '-' if v is None else f'{v:.{digits}f}{suffix}'


def duration(seconds):
    if seconds is None:return '-'
    minutes,sec=divmod(int(max(0,seconds)),60);hours,minutes=divmod(minutes,60)
    return f'{hours}h {minutes:02d}m {sec:02d}s' if hours else f'{minutes}m {sec:02d}s'


def remaining(leaves):
    low,high=leaves['remaining_min'],leaves['remaining_max']
    return str(low) if low==high else f'{low}~{high}'


def count(v):return '-' if v is None else str(v)


def final_score_text(p):
    target=str(p['final_nodes_total'])+('' if p.get('final_nodes_total_known') else '+')
    return f'{p["final_scored_nodes"]}/{target}'


def render(s):
    lines=[f'{s["time"]} | {s["batch"] or "no batch"} | completed {s["completed_cases"]}/{s["total_cases"]}',
           'Run: '+s['run_dir'],
           f'Elapsed run {duration(s.get("run_elapsed_seconds"))} | current batch {duration(s.get("batch_elapsed_seconds"))}']
    handoff=s.get('handoff',{})
    if s.get('runner_active'):
        lines.append('RUNNER ACTIVE · run lock held')
    elif handoff.get('stage')=='training_running':
        lines.append('RUNNER STOPPED · run lock free; elapsed time includes downtime')
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
        lines.append(f'단계 완료: P0 {done["P0"]}/{total} | P1 {done["P1"]}/{total} | 최종채점 {done["union"]}/{total} '
                     f'| KEEP P0 {count(p["keep"]["P0"])} / P1 {count(p["keep"]["P1"])} | 업데이트 완료 {s.get("committed_batches",0)}회')
    evaluation=s.get('checkpoint_evaluation')
    if evaluation:
        if evaluation.get('accuracy') is None:accuracy=''
        elif evaluation['status']!='complete' and evaluation.get('accuracy_ci95'):
            lo,hi=evaluation['accuracy_ci95']
            correct=evaluation.get('correct')
            numerator=f'{correct}/' if correct is not None else ''
            accuracy=f' | est {100*evaluation["accuracy"]:.1f}% ({numerator}{evaluation["evaluated"]}; Wilson 95% {100*lo:.1f}–{100*hi:.1f}%)'
        else:accuracy=f' | accuracy {100*evaluation["accuracy"]:.1f}%'
        label='running' if evaluation['status']!='complete' else 'complete'
        lines.append(f'622-set evaluation · checkpoint {evaluation["checkpoint_cases"]} · '
                     f'{evaluation["evaluated"]}/{evaluation["total"]} · {label}{accuracy}')
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
    lines=[f'{s["time"]}  {batch} · {duration(s.get("run_elapsed_seconds"))} · '
           f'API ${api.get("recorded_usd",0):.2f} · {api.get("calls",0)} calls']
    leaves=s.get('leaves',{})
    if leaves.get('trees_total'):
        lines.append(f'Cases {s.get("completed_cases",0)}/{s.get("total_cases",0)} · '
                     f'leaves {leaves.get("completed",0)} done / {remaining(leaves)} left · '
                     f'trees {leaves.get("trees_complete",0)}/{leaves.get("trees_total",0)}')
    evaluation=s.get('checkpoint_evaluation')
    if evaluation:
        if evaluation.get('accuracy') is None:accuracy=''
        elif evaluation['status']!='complete' and evaluation.get('accuracy_ci95'):
            lo,hi=evaluation['accuracy_ci95']
            correct=evaluation.get('correct')
            numerator=f'{correct}/' if correct is not None else ''
            accuracy=f' · est {100*evaluation["accuracy"]:.1f}% ({numerator}{evaluation["evaluated"]}; 95% {100*lo:.1f}–{100*hi:.1f}%)'
        else:accuracy=f' · accuracy {100*evaluation["accuracy"]:.1f}%'
        status='running' if evaluation['status']!='complete' else 'done'
        lines.append(f'622 eval · checkpoint {evaluation["checkpoint_cases"]} · '
                     f'{evaluation["evaluated"]}/{evaluation["total"]} · {status}{accuracy}')
    p=s.get('pipeline',{})
    if p.get('cases_total'):
        done=p.get('phase_complete',{})
        lines.append(f'P0 {done.get("P0",0)}/{p["cases_total"]} · '
                     f'P1 {done.get("P1",0)}/{p["cases_total"]} · '
                     f'score {done.get("union",0)}/{p["cases_total"]} · '
                     f'rerun {p.get("one_step_reruns_done",0)} · updates {s.get("committed_batches",0)}')
        keep=p.get('keep',{})
        funnel=p.get('rubric_funnel',{})
        def rubric_status(phase):
            f=funnel.get(phase)
            if f is None or f.get('rubrics_validated') is None:return f'{phase} pending'
            return (f'{phase} {f["rubrics_keep"]}/{f["rubrics_validated"]} KEEP'
                    f' · {f["rubrics_generated"]} new')
        lines.append('Rubrics · '+' | '.join(rubric_status(phase) for phase in ('P0','P1'))+
                     f' · live KEEP +{p.get("live_witness_keeps",0)} · gate +{p.get("gate_pass",0)}/-{p.get("gate_reject",0)}')
        if p.get('rollout_seconds_cases'):
            lines.append(f'Rollout {duration(p["rollout_seconds"])} total · '
                         f'{duration(p["rollout_seconds"]/p["rollout_seconds_cases"])} avg')
        available=[(label,p['score_distribution'][phase]) for label,phase in
                   (('P0','P0'),('P1','P1'),('train','union')) if p['score_distribution'][phase]['cases']]
        if available:
            labels=[]
            if p.get('live_scored_pairs'):
                live=p['live_score_distribution']
                labels.append(f'live n={p["live_scored_pairs"]} 0:{live["0"]} 1:{live["1"]} 2:{live["2"]} NA:{live["NA"]}')
            labels.extend(
                f'{label} n={dist["n"]} 0:{dist["counts"]["0"]} 1:{dist["counts"]["1"]} '
                f'2:{dist["counts"]["2"]} NA:{dist["counts"]["NA"]}'
                for label,dist in available)
            lines.append('Scores/rubric-response  '+' · '.join(labels))
        elif p.get('live_scored_pairs'):
            live=p['live_score_distribution']
            lines.append(f'Live rubric scores n={p["live_scored_pairs"]} · '
                         f'0:{live["0"]} 1:{live["1"]} 2:{live["2"]} NA:{live["NA"]}')
        else:
            lines.append('Score distribution pending · no rubric-response scores recorded yet')
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
                return f'{phase} {f["rubrics_keep"]}/{f["rubrics_validated"]}'
            dist=case_pipeline.get('live_score_distribution',{})
            scored=case_pipeline.get('live_scored_pairs',0)
            lines.append(f'     case stats · KEEP {keep_text("P0")} / {keep_text("P1")} · '
                         f'score n={scored} 0:{dist.get("0",0)} 1:{dist.get("1",0)} '
                         f'2:{dist.get("2",0)} NA:{dist.get("NA",0)}')
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
    lines=[f'{s["time"]} {batch} · {duration(s.get("run_elapsed_seconds"))} · '
           f'${api.get("recorded_usd",0):.2f} · {api.get("calls",0)} API calls']
    handoff=s.get('handoff',{})
    if s.get('runner_active'):
        lines.append('RUNNER ACTIVE · run lock held')
    elif handoff.get('stage')=='training_running':
        lines.append('RUNNER STOPPED · run lock free · elapsed includes downtime')
    if leaves.get('trees_total'):
        done=p.get('phase_complete',{})
        lines.append(f'Cases {s.get("completed_cases",0)}/{s.get("total_cases",0)} · '
                     f'leaves {leaves.get("completed",0)}/{remaining(leaves)} · '
                     f'trees {leaves.get("trees_complete",0)}/{leaves.get("trees_total",0)} · '
                     f'P0/P1/score {done.get("P0",0)}/{done.get("P1",0)}/{done.get("union",0)} · '
                     f'rerun {p.get("one_step_reruns_done",0)} · update {s.get("committed_batches",0)}')
        funnel=p.get('rubric_funnel',{})
        def compact_keep(phase):
            f=funnel.get(phase)
            if f is None or f.get('rubrics_validated') is None:return f'{phase} pending'
            return f'{phase} {f["rubrics_keep"]}/{f["rubrics_validated"]}K/{f["rubrics_generated"]}new'
        live=p.get('live_score_distribution',{})
        lines.append(f'Rubrics {compact_keep("P0")} · {compact_keep("P1")} · '
                     f'live KEEP +{p.get("live_witness_keeps",0)} · '
                     f'gate +{p.get("gate_pass",0)}/-{p.get("gate_reject",0)} · '
                     f'live scores n={p.get("live_scored_pairs",0)} '
                     f'0:{live.get("0",0)} 1:{live.get("1",0)} '
                     f'2:{live.get("2",0)} NA:{live.get("NA",0)}')
    evaluation=s.get('checkpoint_evaluation')
    if evaluation:
        if evaluation.get('status')!='complete' and evaluation.get('accuracy_ci95'):
            lo,hi=evaluation['accuracy_ci95'];correct=evaluation.get('correct')
            detail=(f'est {100*evaluation["accuracy"]:.1f}% · Wilson95 {100*lo:.1f}–{100*hi:.1f}%'
                    if evaluation.get('accuracy') is not None else 'estimating')
            if correct is not None:detail+=f' · {correct} correct'
        elif evaluation.get('accuracy') is not None:
            detail=f'accuracy {100*evaluation["accuracy"]:.1f}%'
        else:detail='estimating'
        status='running' if evaluation.get('status')!='complete' else 'done'
        lines.append(f'622 eval C{evaluation.get("checkpoint_cases")} · '
                     f'{evaluation.get("evaluated",0)}/{evaluation.get("total",622)} · {status} · {detail}')
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
                return f'{f["rubrics_keep"]}/{f["rubrics_validated"]}'
            dist=r.get('pipeline',{}).get('live_score_distribution',{})
            n=r.get('pipeline',{}).get('live_scored_pairs',0)
            live_keep=r.get('pipeline',{}).get('live_witness_keeps',0)
            work=(f'C{r["case_index"]+1} {r.get("phase")} · '
                  f'A{r.get("agent_nodes",0)} D{r.get("doctor_calls",0)} F{r.get("finals",0)} · '
                  f'L{leaf.get("completed",0)}/{remaining(leaf)} · liveK+{live_keep} · '
                  f'K{case_keep("P0")}/{case_keep("P1")} · '
                  f'S{n}[0:{dist.get("0",0)} 1:{dist.get("1",0)} 2:{dist.get("2",0)} NA:{dist.get("NA",0)}]')
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
    p=argparse.ArgumentParser();p.add_argument('--config',help='Pin a configuration; default follows the current controller')
    p.add_argument('--run-dir');p.add_argument('--interval',type=float,default=2);p.add_argument('--once',action='store_true');p.add_argument('--json',action='store_true');p.add_argument('--compact',action='store_true',help='One line per GPU; fits five GPUs on short terminals')
    args=p.parse_args()
    if args.interval<.5:p.error('--interval must be >=0.5 seconds')
    monitor=None;current=None
    try:
        while True:
            handoff=read_json(ROOT/'outputs/diagnostics/doctor_parallel_20261008/handoff_status.json',{})
            if args.config and handoff.get('config')!=args.config:handoff={}
            config=args.config or handoff.get('config') or 'configs/online_rl/tree_v21_five_gpu_shared5.json'
            cfg=json.loads((ROOT/config).read_text());root=ROOT/(args.run_dir or handoff.get('run_dir') or cfg['output_dir'])
            if args.config and not args.run_dir:root=ROOT/cfg['output_dir']
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
