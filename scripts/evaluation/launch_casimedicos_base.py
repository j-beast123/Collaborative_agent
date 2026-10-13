"""Launch four detached baseline workers and dedicated Ollama servers on GPUs 1..4."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
run=ROOT/'outputs/test/base/casimedicos_exp_5turn_seed20261006'
run.mkdir(parents=True,exist_ok=True)
info=[]
for gpu in range(1,5):
    config=f'configs/online_rl/casimedicos_base_gpu{gpu}.json'
    cfg=json.loads((ROOT/config).read_text());out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OLLAMA_HOST':f'127.0.0.1:{11435+gpu}',
         'OLLAMA_MODELS':'/opt/anonymous/.ollama/models','OLLAMA_MAX_LOADED_MODELS':'2',
         'OLLAMA_NUM_PARALLEL':'1','OLLAMA_KEEP_ALIVE':'30m','OMP_NUM_THREADS':'2',
         'MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
    pidfile=out/'worker.pid'
    if pidfile.exists() and Path('/proc',pidfile.read_text().strip(),'cmdline').exists():
        info.append({'gpu':gpu,'worker_pid':int(pidfile.read_text()),'server_pid':int((out/'ollama.pid').read_text()),'assigned_cases':156 if gpu<3 else 155,'config':config})
        print('Already running GPU',gpu,flush=True)
        continue
    endpoint=cfg['pilot']['ollama_url']
    try:
        urllib.request.urlopen(endpoint+'/api/tags',timeout=10)
    except (urllib.error.URLError, TimeoutError):
        with (out/'ollama.log').open('a') as log:
            server=subprocess.Popen(['/usr/local/bin/ollama','serve'],cwd=ROOT,env=env,
                                      stdout=log,stderr=log,start_new_session=True)
        (out/'ollama.pid').write_text(str(server.pid))
        for attempt in range(100):
            try:
                with urllib.request.urlopen(endpoint+'/api/tags',timeout=10) as response:json.load(response)
                break
            except (urllib.error.URLError, TimeoutError):time.sleep(.1)
        else:raise RuntimeError(f'Ollama on GPU {gpu} did not start')
    else:
        owned=int((out/'ollama.pid').read_text())
        actual=Path(f'/proc/{owned}/environ').read_bytes()
        assert f'CUDA_VISIBLE_DEVICES={gpu}'.encode() in actual
        assert f'OLLAMA_HOST=127.0.0.1:{11435+gpu}'.encode() in actual
        server=type('ExistingServer',(),{'pid':owned})()
    (out/'runtime.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('OLLAMA_') or k=='CUDA_VISIBLE_DEVICES'},indent=2)+'\n')
    with (out/'run.log').open('a') as log:
        worker=subprocess.Popen(['python3','scripts/evaluation/run_casimedicos_base.py','--config',config],
                                 cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
    pidfile.write_text(str(worker.pid))
    entry={'gpu':gpu,'worker_pid':worker.pid,'server_pid':server.pid,'assigned_cases':156 if gpu<3 else 155,'config':config}
    info.append(entry);print(json.dumps(entry),flush=True)
(run/'run_manifest.json').write_text(json.dumps({'dataset':'HiTZ/casimedicos-exp','language':'en','total_cases':622,
    'protocol':'five Doctor/Agent rounds then Doctor FINAL; no belief, feedback, rubric, intervention or training',
    'shards':info},indent=2)+'\n')
