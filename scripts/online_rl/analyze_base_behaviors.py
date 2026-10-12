"""Tag the saved base dialogues with Luna; resumable, outcome blind and GPU free."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.online_rl.integrated.records import atomic_write_json, read_json, sha256_json, sha256_text
from src.online_rl.sherpa.behavior_analysis import (
    TAGS, associations, decode_annotations, dialogue_view, direct_tag_contrast, evidence_passages,
    numbered_dialogue, tag_schema, validate_annotations)


class TrainingGate:
    def __init__(self, home, allow_overlap=False):
        self.home, self.allow_overlap = Path(home), allow_overlap
        self.lock = threading.Lock()
        self.checked, self.allowed = 0.0, False

    def ready(self):
        if self.allow_overlap:
            return True
        with self.lock:
            now = time.monotonic()
            if now - self.checked < .5:
                return self.allowed
            self.checked = now
            try:
                progress = read_json(self.home / 'progress.json')
                pid = progress.get('pid')
                if pid:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        self.allowed = True
                        return True
                self.allowed = progress.get('stage') in ('agent_update', 'run_finished')
            except FileNotFoundError:
                self.allowed = True
            return self.allowed

    def wait(self):
        while not self.ready():
            time.sleep(.5)


def load_inputs(cfg, root, batch_size):
    from src.online_rl.cases import load_cases, file_sha256
    source = Path(root) / cfg['prompt_optimization']['output_dir']
    manifest = read_json(source / 'base_manifest.json')
    cases = {r['case_id']: r for r in load_cases(Path(root) / cfg['data']['train'])}
    if (not manifest['complete'] or manifest['identity'].get('mode') != 'original_base_no_adapter'
            or len(manifest['case_ids']) != 120
            or set(manifest['case_ids']) != set(cases)
            or manifest['identity']['training_sha256'] != file_sha256(Path(root) / cfg['data']['train'])):
        raise ValueError('Need all 120 pinned original-base cases')
    by_case, provenance = {}, {}
    for cid in manifest['case_ids']:
        path = source / 'base_rollouts' / cid / 'rollouts.json'
        row = read_json(path)
        if row['identity'] != manifest['identity'] or row['case_id'] != cid or len(row['trajectories']) != 32:
            raise ValueError('Invalid base rollout input')
        if {t['outcome'] for t in row['trajectories']} - {0, 1}:
            raise ValueError('Expected binary recorded outcomes')
        by_case[cid] = row['trajectories']
        provenance[cid] = file_sha256(path)
    case_ids = list(manifest['case_ids'])
    random.Random(20261013).shuffle(case_ids)
    jobs = []
    # Interleave cases, rather than annotating one case before the next. All
    # trajectories are included regardless of outcome or annotation result.
    for offset in range(0, 32, batch_size):
        for cid in case_ids:
            group = by_case[cid][offset:offset+batch_size]
            group = sorted(group, key=lambda r: sha256_text(r['trajectory_id']))
            mapping = {f'dialogue_{i}': t for i, t in enumerate(group)}
            case = {k: cases[cid].get(k) for k in ('scenario', 'question', 'options')}
            jobs.append({'case_id': cid, 'trajectories': mapping, 'request': {
                'case': case, 'dialogues': [{'dialogue_id': key, 'dialogue': numbered_dialogue(dialogue_view(t))}
                                           for key, t in mapping.items()]}})
    return jobs, {'base_identity': manifest['identity'], 'source_rollout_sha256': provenance}


def write_report(home, rows, complete):
    stats = associations(rows)
    report = {'complete': complete, 'trajectories': len(rows), 'cases': len({r['case_id'] for r in rows}),
              'overall_success_rate': sum(r['outcome'] for r in rows)/len(rows) if rows else None,
              'contradiction_vs_agreement_only': direct_tag_contrast(rows),
              'tags': stats, 'limitations': [
                  'Observational association, not a causal intervention or effect of training.',
                  'Tags are multi-label; rows cannot be added as mutually exclusive categories.',
                  'Within-case differences compare tagged versus untagged conversations in the same case.',
                  'Mixed-only differences exclude uniform-outcome cases that have zero outcome variation.',
                  'Bootstrap resamples cases, not trajectories; intervals are exploratory and uncorrected for multiple comparisons.',
                  'Each tag has its own eligible case denominator. Sparse tags need manual evidence review.',
                  'Temporal direction is unresolved: assistant behavior can respond to earlier physician reasoning.']}
    atomic_write_json(home/'report.json', report)
    atomic_write_json(home/'annotations.json', rows)
    with (home/'tag_summary.csv').open('w') as handle:
        values = [{k:v for k,v in r.items() if k!='case_strata'} for r in stats]
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = [r['tag'].replace('_',' ') for r in stats]
        fig, ax = plt.subplots(1, 2, figsize=(14,5), constrained_layout=True)
        xs = list(range(len(stats)))
        ax[0].bar([i-.18 for i in xs], [100*r['tagged_success_rate'] if r['tagged_success_rate'] is not None else float('nan') for r in stats], width=.36, label='Tag present')
        ax[0].bar([i+.18 for i in xs], [100*r['untagged_success_rate'] if r['untagged_success_rate'] is not None else float('nan') for r in stats], width=.36, label='Tag absent')
        ax[0].set_ylabel('Success rate (%)');ax[0].set_title('Unadjusted association');ax[0].legend()
        for i,r in enumerate(stats):
            mean=r['mixed_within_case_mean_difference'];ci=r['mixed_within_case_bootstrap_95_interval']
            if mean is None:continue
            if r['mixed_within_case_count']<2:
                ax[1].text(i,0,'n=1\ninsufficient cases',ha='center',va='bottom',fontsize=8,color='gray')
                continue
            ax[1].plot(i,100*mean,'o',color='#2680c2')
            if ci:ax[1].vlines(i,100*ci[0],100*ci[1],color='#2680c2')
        ax[1].axhline(0,color='black',linewidth=1);ax[1].set_ylabel('Mean within-case difference (percentage points)');ax[1].set_title('Mixed cases: case-adjusted association / 95% case bootstrap')
        ax[0].set_xticks(xs,[f'{label}\nN={r["tagged"]}' for label,r in zip(labels,stats)],rotation=40,ha='right')
        ax[1].set_xticks(xs,[f'{label}\nn={r["mixed_within_case_count"]} cases' for label,r in zip(labels,stats)],rotation=40,ha='right')
        fig.suptitle(f'Base-model behavior audit: {len(rows)} dialogues; descriptive, not causal')
        fig.savefig(home/'behavior_associations.png',dpi=180);plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/ours.json')
    parser.add_argument('--output-dir',default='outputs/base_behavior_luna_v2')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--concurrency',type=int,default=64)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--max-batches',type=int)
    parser.add_argument('--allow-overlap',action='store_true',help='Allow requests during training API stages')
    args=parser.parse_args()
    if not 1<=args.concurrency<=100 or not 1<=args.batch_size<=32 or args.max_batches is not None and args.max_batches<1:
        parser.error('Invalid concurrency, batch size or batch limit')
    os.chdir(ROOT)
    from src.online_rl.tree.experiment import load_config
    cfg=load_config(args.config)
    jobs,source=load_inputs(cfg,ROOT,args.batch_size)
    prompt=(ROOT/'prompts/sherpa/behavior_tags_v1.txt').read_text()
    settings={'api_provider':'openai_direct','max_tokens':8192,'reasoning_effort':'low',
              'response_format':'json_schema_strict','temperature':0.0}
    identity={**source,'tag_prompt_sha256':sha256_text(prompt),'model':'gpt-6-luna','settings':settings,'batch_size':args.batch_size}
    home=ROOT/args.output_dir
    print(json.dumps({'trajectories':3840,'batches':len(jobs),'model':identity['model'],'concurrency':args.concurrency,
                      'training_api_overlap':args.allow_overlap,'output_dir':str(home),'launch':args.launch}),flush=True)
    if not args.launch:return
    home.mkdir(parents=True,exist_ok=True)
    with (home/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=home/'manifest.json'
        if manifest.exists() and read_json(manifest)['identity']!=identity:raise RuntimeError('Analysis identity changed; choose a fresh output directory')
        atomic_write_json(manifest,{'identity':identity,'total_trajectories':3840,'total_batches':len(jobs)})
        os.environ['API_USAGE_LOG']=str(home/'api_usage.jsonl');os.environ['API_EVENT_LOG']=str(home/'api_events.jsonl')
        os.environ['API_USAGE_STAGE']='behavior_analysis'
        os.environ['SHERPA_RATE_LIMIT_RETRIES']='0'
        from scripts.online_rl.run_tree import load_key
        from src.online_rl.tree.openai_direct import responses_json
        key=load_key({'api_provider':'openai_direct'})
        gate=TrainingGate(ROOT/cfg['output_dir'],args.allow_overlap)
        stop=threading.Event()
        def annotate(job):
            cache_identity={'analysis':sha256_json(identity),'case_id':job['case_id'],'request':sha256_json(job['request'])}
            path=home/'batches'/f'{sha256_json(cache_identity)}.json'
            views={key:dialogue_view(t) for key,t in job['trajectories'].items()}
            evidence_ids=sorted({key for dialogue in views.values() for key in evidence_passages(dialogue)})
            if path.exists():
                record=read_json(path)
                if record['identity']!=cache_identity:raise RuntimeError('Cached annotation identity changed')
                validate_annotations(record['result'],views)
                return record
            if stop.is_set():return None
            while not gate.ready():
                if stop.wait(.5):return None
            request=dict(job['request'])
            for repair in range(2):
                if stop.is_set():return None
                raw,meta=responses_json(key,'gpt-6-luna',prompt,json.dumps(request,ensure_ascii=False),settings,
                                        tag_schema(views,evidence_ids),'assistant_behavior_tags','behavior_analysis')
                try:
                    value=decode_annotations(json.loads(raw),views)
                except (ValueError,TypeError,KeyError) as exc:
                    atomic_write_json(home/'validation_errors'/f'{path.stem}_{repair}.json',{'error':str(exc),'raw':raw,'response_id':meta.get('response_id')})
                    if repair:raise
                    request['repair']={'error':str(exc),'instruction':'Return all annotations with valid assistant evidence IDs and consistent tags.'}
                    while not gate.ready():
                        if stop.wait(.5):return None
                    continue
                record={'identity':cache_identity,'case_id':job['case_id'],'mapping':{
                    name:{'trajectory_id':t['trajectory_id'],'outcome':t['outcome']} for name,t in job['trajectories'].items()},
                    'result':value,'meta':{k:meta.get(k) for k in ('response_id','model','usage','status')}}
                atomic_write_json(path,record)
                return record
        selected=jobs[:args.max_batches] if args.max_batches else jobs
        completed=0
        started=time.perf_counter()
        records=[]
        with ThreadPoolExecutor(args.concurrency) as pool:
            futures=[pool.submit(annotate,job) for job in selected]
            try:
                for future in as_completed(futures):
                    record=future.result()
                    if record is None:continue
                    records.append(record);completed+=1
                    progress={'completed_batches':completed,'total_batches':len(selected),'seconds':time.perf_counter()-started}
                    atomic_write_json(home/'progress.json',progress)
                    if completed%20==0 or completed==len(selected):print(json.dumps(progress),flush=True)
            except BaseException:
                stop.set()
                for future in futures:future.cancel()
                raise
        rows=[]
        for record in records:
            for annotation in record['result']['annotations']:
                rows.append({**record['mapping'][annotation['dialogue_id']],'case_id':record['case_id'],
                             'tags':[b['tag'] for b in annotation['behaviors']],'evidence':annotation['behaviors']})
        if len({r['trajectory_id'] for r in rows})!=len(rows):raise RuntimeError('Duplicate trajectory annotation')
        report=write_report(home,rows,completed==len(jobs))
        print(json.dumps({'complete':report['complete'],'annotated':len(rows),'report':str(home/'report.json')},indent=2),flush=True)


if __name__=='__main__':main()
