"""Prepare exact cached Doctor inputs; benchmark on an otherwise empty GPU only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def request(url,payload=None):
    req=urllib.request.Request(url,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as r:return json.load(r)


def rows(path):
    with path.open() as f:
        for line in f:
            try:yield json.loads(line)
            except ValueError:continue


def prepare(cfg,out,count):
    from src.online_rl.doctor import turn_prompt
    from src.collaboration.generator import final_decision_prompt
    run=ROOT/cfg['output_dir'];system=(ROOT/cfg['rollout']['doctor_system_prompt_file']).read_text()
    inputs=[]
    for manifest_path in sorted(run.glob('batches/batch_*/manifest.json')):
        manifest=json.loads(manifest_path.read_text())
        for assignment,cid in zip(manifest['assignments'],manifest['case_ids']):
            opening=next((r for r in rows(run/'openings'/f'train_gpu{assignment["worker"]}.jsonl') if r['identity']['case']['case_id']==cid),None)
            if opening is None:continue
            case=opening['identity']['case'];rounds=opening['identity']['rounds']
            cache=manifest_path.parent/f'case_{assignment["index"]:06d}'/'calls.jsonl'
            if not cache.exists():continue
            records=list(rows(cache));agents={r['identity']['path']:r['value'] for r in records if r['kind']=='agent'}
            doctors={r['identity']['path']:r['value'] for r in records if r['kind']=='doctor_turn'}
            for r in records:
                if r['kind'] not in ('doctor_turn','final'):continue
                path=r['identity'].get('path','')
                if not path:continue
                dialogue=[{'speaker':'doctor','stage':'D1','round':1,'text':opening['value']['visible']}]
                parts=path.split('.');valid=True
                for i in range(1,len(parts)+1):
                    ancestor='.'.join(parts[:i])
                    if ancestor not in agents:valid=False;break
                    dialogue.append({'speaker':'agent','stage':f'A{i}','round':i,'text':agents[ancestor]['completion_text']})
                    if i<len(parts) or (r['kind']=='final' and ancestor in doctors):
                        if ancestor not in doctors:valid=False;break
                        dialogue.append({'speaker':'doctor','stage':f'D{i+1}','round':i+1,'text':doctors[ancestor]['visible']})
                if not valid:continue
                user=(turn_prompt(case,dialogue,len(parts)+1,rounds,True) if r['kind']=='doctor_turn' else final_decision_prompt(case,dialogue))
                payload={'model':cfg['rollout']['doctor_backend']['model'],'stream':False,'keep_alive':-1,
                         'options':{**cfg['rollout']['doctor_backend']['options'],'num_predict':cfg['rollout']['max_tokens_doctor'],'seed':r['identity']['seed']},
                         'messages':[{'role':'user','content':system+'\n\n'+user}]}
                inputs.append({'case_id':cid,'path':path,'kind':r['kind'],'payload':payload})
    # Deterministic, distributed selection avoids using only the first case's short inputs.
    inputs.sort(key=lambda r:hashlib.sha256((r['case_id']+'/'+r['path']+'/'+r['kind']).encode()).hexdigest())
    if len(inputs)<count:raise RuntimeError(f'Only {len(inputs)} completed requests; need {count}')
    selected=inputs[:count];out.mkdir(parents=True,exist_ok=True)
    target=out/'requests.json'
    if target.exists():raise RuntimeError('Use a fresh output directory')
    target.write_text(json.dumps({'config':cfg,'requests':selected,'source_run':str(run),'diagnostic_only':True},ensure_ascii=False))
    print(json.dumps({'prepared':count,'kinds':{k:sum(r['kind']==k for r in selected) for k in ('doctor_turn','final')},'file':str(target)}))


def idle(gpu,allowed_pid=None):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
    uuid=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
    if any(line.split(',')[0].strip()==uuid and int(line.split(',')[1])!=allowed_pid for line in result.stdout.splitlines()):
        raise RuntimeError(f'GPU {gpu} has active compute processes. Training was not stopped or changed.')


def benchmark(out,gpu,port,repeats,with_agent=False,both=False,arms=(1,4,8),limit=None,num_batch=None,ollama_bin='/usr/local/bin/ollama'):
    idle(gpu)
    data=json.loads((out/'requests.json').read_text());cfg=data['config'];url=f'http://127.0.0.1:{port}'
    if limit:
        data['requests']=data['requests'][:limit]
        if both:data['agent_requests']=data['agent_requests'][:limit]
    if num_batch:
        for item in data['requests']:item['payload']['options']['num_batch']=num_batch
    (out/'benchmark_settings.json').write_text(json.dumps({'arms':list(arms),'requests':len(data['requests']),
        'agent_actions':len(data.get('agent_requests',[])) if both else 0,'repeats':repeats,'num_batch':num_batch,'auto_resume':False}))
    try:request(url+'/api/version')
    except OSError:pass
    else:raise RuntimeError('Benchmark port already owned; refusing to reuse it')
    backend=cfg['rollout']['doctor_backend'];results=[]
    def progress(stage,**fields):
        row={'stage':stage,'time':time.time(),'pid':os.getpid(),'gpu':gpu,**fields}
        path=out/'progress.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(path)
    progress('loading_agent' if with_agent else 'starting')
    if with_agent:
        if both:cfg['rollout']['agent_sampling_protocol']='per_row_rng_batch_v1'
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['SHERPA_AGENT_REPLICAS']='1'
        os.environ.pop('SHERPA_DOCTOR_ENDPOINTS',None)
        from src.online_rl.runtime import SharedPolicy
        policy=SharedPolicy(cfg,key='',device='cuda:0')
        print(json.dumps({'stage':'agent_resident','gpu':gpu}),flush=True)
    # Rotate order to reduce systematic warmup/order bias across repetitions.
    for repeat in range(repeats):
        order=list(arms);shift=repeat%len(order)
        for parallel in (order[shift:]+order[:shift]):
            progress('warming_doctor',repeat=repeat,parallel=parallel)
            idle(gpu,os.getpid() if with_agent else None)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OLLAMA_HOST':url.removeprefix('http://'),
                 'OLLAMA_MODELS':'/home/kai/.ollama/models','OLLAMA_NUM_PARALLEL':str(parallel),'OLLAMA_MAX_LOADED_MODELS':'1'}
            proc=None
            try:
                with (out/f'server_r{repeat}_p{parallel}.log').open('w') as log:
                    proc=subprocess.Popen([ollama_bin,'serve'],env=env,stdout=log,stderr=log,start_new_session=True)
                for _ in range(60):
                    if proc.poll() is not None:raise RuntimeError('Benchmark server exited')
                    try:version=request(url+'/api/version')['version'];break
                    except OSError:time.sleep(.5)
                else:raise RuntimeError('Benchmark server startup timeout')
                if version!=backend['ollama_version']:raise RuntimeError('Ollama version mismatch')
                tags=request(url+'/api/tags')['models']
                if not any(t['name']==backend['model'] and t['digest']==backend['model_digest'] for t in tags):raise RuntimeError('Doctor digest mismatch')
                warm={**data['requests'][0]['payload'],'options':{**data['requests'][0]['payload']['options'],'num_predict':1}}
                with ThreadPoolExecutor(parallel) as pool:list(pool.map(lambda _:request(url+'/api/chat',warm),range(parallel)))
                loaded=request(url+'/api/ps')['models']
                if not loaded or any(m.get('size_vram',0)<m.get('size',0) for m in loaded):raise RuntimeError('Doctor is not fully on GPU')
                def call(item):
                    start=time.perf_counter();body=request(url+'/api/chat',item['payload'])
                    if body.get('error'):raise RuntimeError(body['error'])
                    return {'seconds':time.perf_counter()-start,'tokens':body.get('eval_count',0),'prompt_tokens':body.get('prompt_eval_count',0),
                            'response':body.get('message',{}).get('content',''),'kind':item['kind'],'case_id':item['case_id'],'path':item['path']}
                samples=[];stop=threading.Event()
                def sample_gpu():
                    while not stop.is_set():
                        try:
                            row=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                            if row.returncode==0:samples.append(tuple(float(x.strip()) for x in row.stdout.strip().split(',')))
                        except (OSError,ValueError,subprocess.TimeoutExpired):pass
                        stop.wait(1)
                sampler=threading.Thread(target=sample_gpu,daemon=True);sampler.start()
                progress('doctor_requests',repeat=repeat,parallel=parallel,total=len(data['requests']))
                start=time.perf_counter()
                try:
                    with ThreadPoolExecutor(parallel) as pool:outputs=list(pool.map(call,data['requests']))
                finally:stop.set()
                elapsed=time.perf_counter()-start;latencies=sorted(r['seconds'] for r in outputs)
                sampler.join(timeout=6)
                result={'repeat':repeat,'parallel':parallel,'wall_seconds':elapsed,'requests_per_second':len(outputs)/elapsed,
                        'tokens_per_second':sum(r['tokens'] for r in outputs)/elapsed,'p50_seconds':statistics.median(latencies),
                        'p95_seconds':latencies[min(len(latencies)-1,int(.95*len(latencies)))],
                        'empty_responses':sum(not r['response'].strip() for r in outputs),'doctor_only':True,
                        'agent_resident':with_agent,'peak_vram_mb':max((s[0] for s in samples),default=None),
                        'mean_sampled_gpu_util':statistics.mean(s[1] for s in samples) if samples else None}
                if both:
                    from src.online_rl.agent_batch import Batcher
                    if hasattr(policy,'_batcher'):policy._batcher.close()
                    policy._batcher=Batcher(policy,parallel)
                    first=data['agent_requests'][0]
                    with ThreadPoolExecutor(parallel) as pool:
                        list(pool.map(lambda _:policy.agent_turn(first['case'],first['prefix'],first['turn'],first['seed']),range(parallel)))
                    print(json.dumps({'stage':'Agent_benchmark','repeat':repeat,'parallel':parallel,'actions':len(data['agent_requests'])}),flush=True)
                    progress('agent_requests',repeat=repeat,parallel=parallel,total=len(data['agent_requests']))
                    def agent_call(item):
                        began=time.perf_counter();trace=policy.agent_turn(item['case'],item['prefix'],item['turn'],item['seed'])
                        return {'seconds':time.perf_counter()-began,'case_id':item['case']['case_id'],'path':item['path'],
                                'tokens':sum(len(s['completion_ids']) for s in trace['segments']),
                                'private_assessment':trace['private_assessment'],'completion_text':trace['completion_text']}
                    agent_start=time.perf_counter()
                    with ThreadPoolExecutor(parallel) as pool:agent_outputs=list(pool.map(agent_call,data['agent_requests']))
                    agent_seconds=time.perf_counter()-agent_start
                    result.update(agent_wall_seconds=agent_seconds,agent_tokens_per_second=sum(r['tokens'] for r in agent_outputs)/agent_seconds,
                                  agent_empty_responses=sum(not r['completion_text'].strip() for r in agent_outputs),
                                  combined_workload_seconds=elapsed+agent_seconds)
                    (out/f'agent_responses_r{repeat}_p{parallel}.json').write_text(json.dumps(agent_outputs,ensure_ascii=False))
                (out/f'responses_r{repeat}_p{parallel}.json').write_text(json.dumps(outputs,ensure_ascii=False))
                results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)
                progress('arm_complete',repeat=repeat,parallel=parallel,arms_completed=len(results))
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        try:request(url+'/api/generate',{'model':backend['model'],'keep_alive':0})
                        finally:os.killpg(proc.pid,signal.SIGTERM)
                    proc.wait(timeout=30)
                # Owned runner termination can lag server exit; wait before the next idle check.
                time.sleep(2)
    progress('complete',arms_completed=len(results))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--config',default='configs/online_rl/tree_v21_five_gpu.json')
    p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,default=1);p.add_argument('--port',type=int,default=11571)
    p.add_argument('--requests',type=int,default=64);p.add_argument('--repeats',type=int,default=3);p.add_argument('--with-agent',action='store_true');p.add_argument('--both',action='store_true')
    p.add_argument('--arms',type=int,nargs='+',default=[1,4,8]);p.add_argument('--limit',type=int);p.add_argument('--num-batch',type=int)
    p.add_argument('--ollama-bin',default='/usr/local/bin/ollama')
    a=p.parse_args();out=ROOT/a.out
    if a.requests<1 or a.repeats<1:p.error('requests and repeats must be positive')
    if a.prepare:prepare(json.loads((ROOT/a.config).read_text()),out,a.requests)
    else:benchmark(out,a.gpu,a.port,a.repeats,a.with_agent or a.both,a.both,tuple(a.arms),a.limit,a.num_batch,a.ollama_bin)

if __name__=='__main__':main()
