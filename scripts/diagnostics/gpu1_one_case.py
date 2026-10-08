"""Local, no-API-cost one-case latency probe: one HF Agent and one Ollama Doctor on GPU1."""
import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def request(url,payload=None,timeout=30):
    data=None if payload is None else json.dumps(payload).encode()
    with urllib.request.urlopen(urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'}),timeout=timeout) as r:
        return json.load(r)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--config',default='configs/online_rl/tree_v21_five_gpu.json')
    p.add_argument('--out',default='outputs/diagnostics/gpu1_one_case')
    p.add_argument('--doctor-model',default='gemma3:12b');p.add_argument('--port',type=int,default=11561)
    p.add_argument('--case-index',type=int,default=0);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the one physical GPU before launching Python')
    os.environ['SHERPA_AGENT_REPLICAS']='1';os.environ.pop('SHERPA_DOCTOR_ENDPOINTS',None)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('One visible CUDA GPU is required')
    from src.online_rl.cases import load_cases
    from src.online_rl.integrated.records import atomic_write_json
    from src.online_rl.environment import final_outcome
    from src.online_rl.integrated.continuation import adapter_sha256
    from src.online_rl.runtime import SharedPolicy
    cfg=json.loads((ROOT/a.config).read_text());out=ROOT/a.out
    if (out/'summary.json').exists():raise RuntimeError('Use a fresh --out to measure a new probe')
    out.mkdir(parents=True,exist_ok=True)
    cases=load_cases(ROOT/cfg['data']['train']);order=cfg['five_gpu'].get('train_order_file')
    if order:
        ids=json.loads((ROOT/order).read_text());case={c['case_id']:c for c in cases}[ids[a.case_index]]
    else:case=cases[a.case_index]
    url=f'http://127.0.0.1:{a.port}'
    try:request(url+'/api/version',timeout=2)
    except OSError:pass
    else:raise RuntimeError('Port already in use; never adopt an unrelated Doctor server')
    env={**os.environ,'OLLAMA_HOST':url.removeprefix('http://'),'OLLAMA_MODELS':'/home/kai/.ollama/models',
         'OLLAMA_MAX_LOADED_MODELS':'1','OLLAMA_NUM_PARALLEL':'1','OLLAMA_KEEP_ALIVE':'30m'}
    proc=None
    try:
        with (out/'doctor_server.log').open('w') as log:
            proc=subprocess.Popen(['/usr/local/bin/ollama','serve'],env=env,stdout=log,stderr=log,start_new_session=True)
        atomic_write_json(out/'owned_server.json',{'pid':proc.pid,'physical_gpu':a.gpu,'url':url})
        for _ in range(60):
            if proc.poll() is not None:raise RuntimeError('Owned Doctor server exited; inspect doctor_server.log')
            try:version=request(url+'/api/version')['version'];break
            except OSError:time.sleep(.5)
        else:raise RuntimeError('Doctor startup timeout')
        tags=request(url+'/api/tags')['models'];tag=next((m for m in tags if m['name']==a.doctor_model),None)
        if tag is None:raise RuntimeError('Requested local Doctor model is not installed')
        cfg=copy.deepcopy(cfg);backend=cfg['rollout']['doctor_backend']
        backend.update(endpoint=url,model=a.doctor_model,model_digest=tag['digest'],ollama_version=version,
                       quantization=tag['details']['quantization_level'],gpu=f'physical GPU {a.gpu}')
        cfg['rollout']['doctor_model']=f"ollama/{a.doctor_model}@{tag['digest'][:12]}"
        cfg['rollout'].pop('doctor_opening',None)
        atomic_write_json(out/'manifest.json',{'physical_gpu':a.gpu,'device_name':torch.cuda.get_device_name(0),
            'case_id':case['case_id'],'agent':cfg['policy'],'doctor':backend,'rollout':cfg['rollout'],
            'diagnostic_only':True,'training_update':False,'doctor_differs_from_training_pin':True,
            'torch':torch.__version__,'model_load_and_warmup_excluded_from_response_timings':True})
        print(json.dumps({'stage':'load_agent','gpu':a.gpu,'case_id':case['case_id']}),flush=True)
        t=time.perf_counter();policy=SharedPolicy(cfg,key='');load_seconds=time.perf_counter()-t
        verification=policy.doctor.verify()
        if verification['problems']:raise RuntimeError(str(verification['problems']))
        print(json.dumps({'stage':'warmup','agent_load_seconds':load_seconds}),flush=True)
        prompt=policy.ST.chat_prompt(policy.tok,policy.agent_system,'Say hello briefly.')
        policy._generate(prompt,8,12345)
        request(url+'/api/generate',{'model':a.doctor_model,'prompt':'Hello','stream':False,'keep_alive':-1,
                                    'options':{'num_predict':1,'num_ctx':8192}},timeout=180)
        timings=[];actions=[];doctor_timings=[];doctor_inference=[];model_generations=[]
        def save_timings():
            atomic_write_json(out/'timings.json',{'agent_segments':timings,'agent_replies':actions,
                'doctor_calls':doctor_timings,'doctor_inference':doctor_inference})
        original_model_generate=policy.model.generate
        def model_generate(*args,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter()
            result=original_model_generate(*args,**kwargs);torch.cuda.synchronize()
            model_generations.append(time.perf_counter()-start);return result
        policy.model.generate=model_generate
        original_doctor_http=policy.doctor.http
        def doctor_http(endpoint,payload,timeout):
            start=time.perf_counter();body=original_doctor_http(endpoint,payload,timeout)
            if endpoint.endswith('/api/chat'):
                doctor_inference.append({'http_seconds':time.perf_counter()-start,
                    'load_seconds':(body.get('load_duration') or 0)/1e9,
                    'prompt_eval_seconds':(body.get('prompt_eval_duration') or 0)/1e9,
                    'decode_seconds':(body.get('eval_duration') or 0)/1e9,
                    'prompt_tokens':body.get('prompt_eval_count'),'generated_tokens':body.get('eval_count')})
                save_timings()
            return body
        policy.doctor.http=doctor_http
        orig_generate=policy._generate
        def generate(*args,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter();r=orig_generate(*args,**kwargs)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            row={'kind':'agent_segment','role':'private_assessment' if len(timings)%2==0 else 'visible_reply',
                 'tokens':len(r['completion_ids']),'seconds':elapsed,
                 'model_generate_seconds':model_generations[-1],
                 'trace_and_tokenization_seconds':elapsed-model_generations[-1],
                 'tokens_per_second':len(r['completion_ids'])/model_generations[-1]}
            timings.append(row);save_timings();print(json.dumps(row),flush=True);return r
        policy._generate=generate
        orig_turn=policy.agent_turn
        def agent_turn(case,prefix,turn,seed):
            torch.cuda.synchronize();start=time.perf_counter();r=orig_turn(case,prefix,turn,seed)
            torch.cuda.synchronize();row={'kind':'agent_reply','turn':turn,'seconds':time.perf_counter()-start}
            actions.append(row);save_timings();print(json.dumps(row),flush=True);return r
        policy.agent_turn=agent_turn
        for name in ('turn','final'):
            fn=getattr(policy.doctor,name)
            def timed(*args,_fn=fn,_name=name,**kwargs):
                start=time.perf_counter();r=_fn(*args,**kwargs)
                row={'kind':'doctor_'+_name,'seconds':time.perf_counter()-start}
                doctor_timings.append(row);save_timings();print(json.dumps(row),flush=True);return r
            setattr(policy.doctor,name,timed)
        start=time.perf_counter();episode=policy.generate(case,seed=cfg['schedule']['seed']);total=time.perf_counter()-start
        atomic_write_json(out/'trajectory.json',{'case_id':case['case_id'],**episode})
        save_timings()
        memory=request(url+'/api/ps')
        snapshot=subprocess.check_output(['nvidia-smi','--id',str(a.gpu),'--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()
        summary={'case_id':case['case_id'],'physical_gpu':a.gpu,'agent_replies':len(actions),
            'mean_agent_reply_seconds':sum(r['seconds'] for r in actions)/len(actions),
            'mean_private_assessment_seconds':sum(r['seconds'] for r in timings if r['role']=='private_assessment')/len(actions),
            'mean_visible_reply_seconds':sum(r['seconds'] for r in timings if r['role']=='visible_reply')/len(actions),
            'mean_doctor_call_seconds':sum(r['seconds'] for r in doctor_timings)/len(doctor_timings),
            'agent_load_seconds':load_seconds,'episode_seconds':total,'final_option':episode['final_option'],
            'outcome':final_outcome(case,episode['final_option']),'gpu_snapshot':snapshot,'doctor_loaded':memory,
            'agent_policy_sha256':adapter_sha256(policy.model),'doctor_quantization':backend['quantization'],
            'limitations':['One case and one dialogue path, not the G=4 full tree.',
                           'Initial seeded LoRA, not a trained checkpoint; Doctor Q4_K_M differs from training Q8_0.']}
        atomic_write_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    finally:
        if proc is not None and proc.poll() is None:
            try:request(url+'/api/generate',{'model':a.doctor_model,'keep_alive':0},timeout=30)
            except Exception:pass
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

if __name__=='__main__':main()
