"""Start five dedicated GPU-pinned Ollama Doctor servers; no experiment runs."""
import argparse,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.online_rl.tree.devices import physical_gpus


def start(cfg, *, require_model=True, selected_gpus=None):
    out=ROOT/cfg['output_dir']/'doctor_servers';out.mkdir(parents=True,exist_ok=True)
    configured_binary=os.environ.get('SHERPA_OLLAMA_BINARY',cfg['five_gpu'].get('ollama_binary','ollama'))
    found=shutil.which(configured_binary) if '/' not in configured_binary else None
    binary=str(Path(found or configured_binary).resolve())
    if not Path(binary).is_file():raise FileNotFoundError(f'Ollama binary not found: {configured_binary}; set SHERPA_OLLAMA_BINARY')
    gpu_ids=cfg['five_gpu'].get('doctor_gpu_ids',list(range(len(cfg['five_gpu']['doctor_urls']))))
    if len(gpu_ids)!=len(cfg['five_gpu']['doctor_urls']):raise ValueError('Doctor GPU IDs and URLs must have identical lengths')
    requested=set(gpu_ids if selected_gpus is None else selected_gpus)
    unknown=requested-set(gpu_ids)
    if unknown:raise ValueError(f'Doctor GPU IDs not configured: {sorted(unknown)}')
    physical=physical_gpus()
    for gpu,url in zip(gpu_ids,cfg['five_gpu']['doctor_urls']):
        if gpu not in requested:continue
        device=physical[gpu]  # logical Doctor GPU -> physical
        per_gpu=cfg['five_gpu'].get('doctor_concurrency_by_gpu')
        configured=(per_gpu[gpu] if per_gpu is not None else
                    cfg['five_gpu'].get('doctor_concurrency',cfg['five_gpu']['lanes_per_gpu']))
        parallelism=int(os.environ.get('SHERPA_DOCTOR_CONCURRENCY',configured))
        if parallelism not in (1,2,4,8):raise ValueError('Doctor concurrency must be 1, 2, 4 or 8')
        # A live foreign server is never adopted merely because its port matches.
        pidfile=out/f'gpu{gpu}.pid'
        if pidfile.exists():
            try:
                pid=int(pidfile.read_text());env=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
                if f'CUDA_VISIBLE_DEVICES={device}'.encode() in env and f'OLLAMA_HOST={url.removeprefix("http://")}'.encode() in env:
                    if f'OLLAMA_NUM_PARALLEL={parallelism}'.encode() in env and str((Path('/proc')/str(pid)/'exe').resolve())==binary:
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
        models_dir=os.environ.get('SHERPA_OLLAMA_MODELS',cfg['five_gpu'].get('ollama_models_dir',str(Path.home()/'.ollama/models')))
        env={**os.environ,'CUDA_VISIBLE_DEVICES':device,'OLLAMA_HOST':url.removeprefix('http://'),
             'OLLAMA_MODELS':models_dir,'OLLAMA_MAX_LOADED_MODELS':'1',
             'OLLAMA_NUM_PARALLEL':str(parallelism),'OLLAMA_KEEP_ALIVE':'30m'}
        with (out/f'gpu{gpu}.log').open('a') as log:
            proc=subprocess.Popen([binary,'serve'],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
        pidfile.write_text(str(proc.pid))
        for attempt in range(40):
            if proc.poll() is not None:raise RuntimeError(f'Doctor server GPU{gpu} exited; see its log')
            try:
                with urllib.request.urlopen(url+'/api/tags',timeout=1) as response:
                    models=json.load(response)['models']
                if require_model and not any(m['name']==cfg['rollout']['doctor_backend']['model'] and m['digest']==cfg['rollout']['doctor_backend']['model_digest'] for m in models):
                    raise RuntimeError('Pinned Doctor model missing or changed')
                break
            except OSError:time.sleep(.25)
        else:raise RuntimeError(f'Doctor server GPU{gpu} not ready')
        print(f'GPU{gpu} Doctor server ready: {url}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/online_rl/tree_v21_five_gpu_shared5.json');p.add_argument('--launch',action='store_true');p.add_argument('--gpu',type=int,action='append',help='Start only these logical Doctor GPUs (physical ids from CUDA_VISIBLE_DEVICES); repeat to select several')
    a=p.parse_args()
    from src.online_rl.tree.experiment import load_config
    cfg=load_config(a.config)
    if a.launch:start(cfg,selected_gpus=a.gpu)
    else:print(json.dumps({'doctor_gpus':a.gpu or cfg['five_gpu'].get('doctor_gpu_ids',list(range(len(cfg['five_gpu']['doctor_urls'])))),'physical_gpus':physical_gpus(),'endpoints':cfg['five_gpu']['doctor_urls'],'launch':False},indent=2))
