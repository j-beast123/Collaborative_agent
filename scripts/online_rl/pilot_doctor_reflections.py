"""Re-reflect on a few saved dialogues through the running frozen Doctor server."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from scripts.online_rl.analyze_base_behaviors import TrainingGate
from src.online_rl.integrated.records import atomic_write_json,read_json,sha256_json,sha256_text
from src.online_rl.sherpa.behavior_analysis import dialogue_view,evidence_passages,numbered_dialogue


def reflection_evidence(trajectory,full_turns=False):
    view=dialogue_view(trajectory)
    return ({f'a{row["assistant_turn"]}':{'assistant_turn':row['assistant_turn'],'quote':row['text']}
             for row in view if row['speaker']=='agent'} if full_turns else evidence_passages(view))


def grounded_schema(trajectory,full_turns=False):
    passages=reflection_evidence(trajectory,full_turns)
    fields={'evidence_id':{'type':'string','enum':['none',*passages]},'reflection':{'type':'string','maxLength':900}}
    item={'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
    properties={key:item for key in ('helpful','hindering','wished_for')}
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def grounded_reflection(raw,trajectory,full_turns=False):
    value=json.loads(raw)
    if not isinstance(value,dict) or set(value)!={'helpful','hindering','wished_for'}:
        raise ValueError('Expected exactly three grounded reflection fields')
    passages=reflection_evidence(trajectory,full_turns)
    labels=('What helped:','What hindered:','What I would have wanted:')
    evidence={};texts=[]
    for key,label in zip(('helpful','hindering','wished_for'),labels):
        item=value[key]
        if not isinstance(item,dict) or set(item)!={'evidence_id','reflection'} or not isinstance(item['reflection'],str) or not item['reflection'].strip():
            raise ValueError('Invalid grounded reflection item')
        ident=item['evidence_id']
        if ident=='none':
            if not item['reflection'].strip().startswith('None identified'):
                raise ValueError('No evidence must be represented as None identified')
            quote='';evidence[key]=None
        else:
            if ident not in passages:raise ValueError('Reflection evidence must refer to an assistant passage')
            evidence[key]={'evidence_id':ident,**passages[ident]}
            quote=' [Assistant evidence: '+json.dumps(passages[ident]['quote'],ensure_ascii=False)+']'
        texts.append(label+' '+item['reflection']+quote)
    text='\n\n'.join(texts)
    reflection_sections(text)
    return text,value,evidence


def reflection_sections(text):
    labels=('What helped:','What hindered:','What I would have wanted:')
    positions=[text.find(label) for label in labels]
    if any(i<0 for i in positions) or positions!=sorted(positions):
        raise ValueError('Private reflection needs all three ordered sections')
    if re.search(r'(?im)^\s*(?:Discussion|STATUS|PRIVATE_STATE|Final answer)\s*:',text):
        raise ValueError('Doctor returned a discussion/decision rather than private reflection')
    return {label[:-1]:text[pos+len(label):(positions[i+1] if i+1<len(labels) else len(text))].strip()
            for i,(label,pos) in enumerate(zip(labels,positions))}


def choose_samples(source,count,seed):
    pools=[]
    for path in sorted((source/'reflections').glob('*.json')):
        rows=read_json(path)
        groups={y:[r for r in rows if r['trajectory']['outcome']==y] for y in (0,1)}
        if all(groups.values()):pools.append((path.stem,groups))
    if len(pools)<count:raise ValueError('Insufficient paired success/failure reflection cases')
    rng=random.Random(seed);selected=[]
    for cid,groups in rng.sample(pools,count):
        for outcome in (0,1):
            row=rng.choice(groups[outcome]);selected.append({'case_id':cid,**row})
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/ours.json')
    p.add_argument('--cases',type=int,default=4)
    p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261013)
    p.add_argument('--output-dir',default='outputs/doctor_reflection_pilot_v1')
    p.add_argument('--launch',action='store_true')
    p.add_argument('--grounded',action='store_true',help='Select source passage IDs instead of generating quotations')
    p.add_argument('--full-turn-evidence',action='store_true',help='Use whole assistant turns as evidence, with no text fragmentation')
    a=p.parse_args()
    if not 1<=a.cases<=20 or not 1<=a.concurrency<=8:p.error('Invalid sample size or concurrency')
    if a.full_turn_evidence and not a.grounded:p.error('--full-turn-evidence requires --grounded')
    os.chdir(ROOT)
    from src.online_rl.tree.experiment import load_config
    from src.online_rl.cases import load_cases
    from src.online_rl.doctor import make_doctor
    from src.online_rl.sherpa.backend import case_view
    cfg=load_config(a.config);source=ROOT/cfg['prompt_optimization']['output_dir'];training=ROOT/cfg['output_dir']
    samples=choose_samples(source,a.cases,a.seed)
    prompt=(ROOT/('prompts/sherpa/trajectory_doctor_reflection_pilot_v5.txt' if a.full_turn_evidence else
                 'prompts/sherpa/trajectory_doctor_reflection_pilot_v4.txt' if a.grounded else
                  'prompts/sherpa/trajectory_doctor_reflection_pilot_v3.txt')).read_text()
    doctor=make_doctor(cfg['rollout'],(ROOT/cfg['rollout']['doctor_system_prompt_file']).read_text())
    base_manifest=read_json(source/'base_manifest.json')
    if base_manifest['identity']['doctor']!=doctor.protocol:raise ValueError('Doctor protocol differs from original base')
    by_id={r['case_id']:r for r in load_cases(ROOT/cfg['data']['train'])}
    identity={'source_base':base_manifest['identity'],'prompt_sha256':sha256_text(prompt),'doctor':doctor.protocol,
              'selected_trajectories':[s['trajectory']['trajectory_id'] for s in samples],
              'max_tokens':1536,'system_mode':'standalone_private_reflection_grounded' if a.grounded else 'standalone_private_reflection','new_rollouts':0,
              'seed_policy':'reuse original reflection seed'}
    if a.grounded:identity['context_representation']=('full_assistant_turn_evidence_v1' if a.full_turn_evidence else
                                                    'full_original_assistant_text_with_citation_fragments_v1')
    home=ROOT/a.output_dir
    print(json.dumps({'sample_cases':a.cases,'reflections':len(samples),'doctor':doctor.model,
                      'concurrency':a.concurrency,'new_rollouts':0,'launch':a.launch,'output_dir':str(home)}),flush=True)
    if not a.launch:return
    done=home/'results.json'
    if done.exists():
        saved=read_json(done)
        if saved['identity']!=identity or not saved['complete']:
            raise RuntimeError('Saved pilot protocol changed')
        print(json.dumps({'complete':True,'cached_reflections':len(saved['reflections']),
                          'results':str(done),'inference_started':False}),flush=True)
        return
    runtime=read_json(training/'doctor_server/inference_runtime.json')
    if runtime['engine']!='vllm' or len(runtime['servers'])!=1:raise RuntimeError('Need the existing frozen Doctor server')
    endpoint=runtime['servers'][0]
    if urllib.parse.urlparse(endpoint).hostname!='127.0.0.1':raise ValueError('Only the existing loopback Doctor server is allowed')
    os.kill(runtime['pids'][0],0)
    with urllib.request.urlopen(endpoint+'/health',timeout=5) as response:
        if response.status!=200:raise RuntimeError('Doctor server unavailable')
    doctor.local_endpoint=endpoint
    home.mkdir(parents=True,exist_ok=True)
    with (home/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=home/'manifest.json'
        if manifest.exists() and read_json(manifest)['identity']!=identity:raise RuntimeError('Pilot protocol changed; choose a fresh output directory')
        atomic_write_json(manifest,{'identity':identity,'endpoint':endpoint,'samples':samples})
        gate=TrainingGate(training)
        def wait_for_idle_doctor():
            # Use the beginning of GPU-0 optimization, leaving a buffer before
            # the next training rollout needs the frozen Doctor on GPU 1.
            while True:
                progress=read_json(training/'progress.json')
                if not gate.ready():time.sleep(.5);continue
                if progress.get('stage')=='agent_update' and time.time()-progress['stage_started_at']<=40:return
                os.kill(runtime['pids'][0],0)
                time.sleep(.5)
        def reflect(sample):
            trajectory=sample['trajectory'];old=sample['doctor_reflection']
            cid=sample['case_id'];path=home/'reflections'/f'{sha256_text(trajectory["trajectory_id"])}.json'
            cache_id={'pilot':sha256_json(identity),'trajectory':sha256_json(trajectory),'seed':old['seed']}
            if path.exists():
                result=read_json(path)
                if result['identity']!=cache_id:raise RuntimeError('Pilot cache identity changed')
                return result
            request={'task':'private retrospective collaboration reflection',
                     'case':case_view(by_id[cid]),'dialogue':trajectory['dialogue'],
                     'recorded_outcome':trajectory['outcome']}
            if a.grounded:
                view=dialogue_view(trajectory)
                if a.full_turn_evidence:
                    request['dialogue']=[{**row,'evidence_id':f'a{row["assistant_turn"]}'} if row['speaker']=='agent' else row for row in view]
                else:
                    numbered=numbered_dialogue(view)
                    request['dialogue']=[{**indexed,'text':original['text']} for indexed,original in zip(numbered,view)]
                request['recorded_final_choice']=trajectory['final_option']
                request['outcome_label']='SUCCESS' if trajectory['outcome'] else 'FAILURE'
            schema=grounded_schema(trajectory,a.full_turn_evidence) if a.grounded else None
            started=time.perf_counter()
            for attempt in range(2):
                wait_for_idle_doctor()
                raw=doctor._chat(json.dumps(request,ensure_ascii=False),old['seed']+attempt,1536,prompt,format_schema=schema)
                try:
                    structured=evidence=None
                    text=raw
                    if a.grounded:text,structured,evidence=grounded_reflection(raw,trajectory,a.full_turn_evidence)
                    sections=reflection_sections(text)
                except ValueError as exc:
                    atomic_write_json(home/'validation_errors'/f'{path.stem}_{attempt}.json',{'error':str(exc),'text':raw})
                    if attempt:raise
                    request['repair']='Write all three reflection sections; no discussion or new final answer.'
                    continue
                result={'identity':cache_id,'case_id':cid,'trajectory_id':trajectory['trajectory_id'],
                        'outcome':trajectory['outcome'],'old_reflection':old['text'],'new_reflection':text,
                        'sections':sections,'seed':old['seed']+attempt,'seconds':time.perf_counter()-started}
                if a.grounded:result.update(structured=structured,evidence=evidence,raw=raw)
                atomic_write_json(path,result);return result
        records=[]
        with ThreadPoolExecutor(a.concurrency) as pool:
            for future in as_completed([pool.submit(reflect,s) for s in samples]):
                result=future.result();records.append(result)
                print(json.dumps({'completed_reflections':len(records),'total':len(samples),'trajectory_id':result['trajectory_id']}),flush=True)
        records.sort(key=lambda r:(r['case_id'],r['outcome']))
        atomic_write_json(home/'results.json',{'identity':identity,'complete':True,'reflections':records})
        print(json.dumps({'complete':True,'reflections':len(records),'results':str(home/'results.json')}),flush=True)


if __name__=='__main__':main()
