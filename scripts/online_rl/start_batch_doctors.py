"""Start four dedicated GPU-pinned Ollama Doctor servers; no experiment runs."""
import argparse,json,os,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def start(cfg):
    out=ROOT/cfg['output_dir']/'doctor_servers';out.mkdir(parents=True,exist_ok=True)
    gpu_ids=cfg['batch'].get('doctor_gpu_ids',list(range(len(cfg['batch']['doctor_urls']))))
    if len(gpu_ids)!=len(cfg['batch']['doctor_urls']):raise ValueError('Doctor GPU IDs and URLs must have identical lengths')
    for gpu,url in zip(gpu_ids,cfg['batch']['doctor_urls']):
        # A live foreign server is never adopted merely because its port matches.
        pidfile=out/f'gpu{gpu}.pid'
        if pidfile.exists():
            try:
                pid=int(pidfile.read_text());env=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
                if f'CUDA_VISIBLE_DEVICES={gpu}'.encode() in env and f'OLLAMA_HOST={url.removeprefix("http://")}'.encode() in env:
                    if f'OLLAMA_NUM_PARALLEL={cfg["batch"].get("doctor_parallelism",1)}'.encode() in env:
                        with urllib.request.urlopen(url+'/api/tags',timeout=3):pass
                        continue
                    os.kill(pid,15)
                    for _ in range(40):
                        if not (Path('/proc')/str(pid)).exists():break
                        time.sleep(.25)
            except (OSError,ValueError):pass
        try:
            with urllib.request.urlopen(url+'/api/tags',timeout=2):pass
        except OSError:pass
        else:raise RuntimeError(f'Unowned server at {url}; select a free endpoint')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OLLAMA_HOST':url.removeprefix('http://'),
             'OLLAMA_MODELS':'/opt/anonymous/.ollama/models','OLLAMA_MAX_LOADED_MODELS':'1',
             'OLLAMA_NUM_PARALLEL':str(cfg['batch'].get('doctor_parallelism',1)),'OLLAMA_KEEP_ALIVE':'30m'}
        with (out/f'gpu{gpu}.log').open('a') as log:
            proc=subprocess.Popen(['/usr/local/bin/ollama','serve'],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
        pidfile.write_text(str(proc.pid))
        for attempt in range(40):
            if proc.poll() is not None:raise RuntimeError(f'Doctor server GPU{gpu} exited; see its log')
            try:
                with urllib.request.urlopen(url+'/api/tags',timeout=1) as response:
                    models=json.load(response)['models']
                if not any(m['name']==cfg['doctor_backend']['model'] and m['digest']==cfg['doctor_backend']['digest'] for m in models):
                    raise RuntimeError('Pinned Doctor model missing or changed')
                break
            except OSError:time.sleep(.25)
        else:raise RuntimeError(f'Doctor server GPU{gpu} not ready')
        print(f'GPU{gpu} Doctor server ready: {url}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/online_rl/tree_batch12_haiku_gemini_luna.json');p.add_argument('--launch',action='store_true');a=p.parse_args();cfg=json.loads((ROOT/a.config).read_text())
    if a.launch:start(cfg)
    else:print(json.dumps({'doctor_gpus':[0,1,2,3],'endpoints':cfg['batch']['doctor_urls'],'launch':False},indent=2))
