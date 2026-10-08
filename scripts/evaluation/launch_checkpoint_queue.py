"""Detach an evaluator that registers and drains published checkpoint snapshots."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='outputs/training/tree_batch12_haiku_gemini_luna_seed20261006');p.add_argument('--config',default='configs/evaluation/tree_checkpoint_queue.json');p.add_argument('--launch',action='store_true');a=p.parse_args()
    out=(ROOT/a.run_dir).resolve();settings=json.loads((ROOT/a.config).read_text())
    print(json.dumps({'gpu':4,'checkpoint_interval':30,'run_dir':str(out),'evaluation_data':settings['data_file'],'launch':a.launch},indent=2),flush=True)
    if not a.launch:return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<5:raise RuntimeError('CUDA GPU4 unavailable')
    dest=out/'evaluation';dest.mkdir(parents=True,exist_ok=True);pidpath=dest/'worker.pid'
    if pidpath.exists():
        try:
            pid=int(pidpath.read_text());cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
            if b'run_checkpoint_queue.py' in cmd:print('Evaluator already running',pid);return
        except (OSError,ValueError):pass
    # Reuse dedicated-server ownership validation for the GPU4 Doctor.
    from scripts.online_rl.start_batch_doctors import start
    train=json.loads((ROOT/'configs/online_rl/tree_batch12_haiku_gemini_luna.json').read_text())
    train['batch']['doctor_urls']=train['batch']['doctor_urls']+[settings['doctor_backend']['url']]
    train['output_dir']=str(out.relative_to(ROOT));start(train)
    cmd=[sys.executable,'scripts/evaluation/run_checkpoint_queue.py','--run-dir',str(out),'--config',a.config]
    with (dest/'worker.log').open('a') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=log,start_new_session=True)
    pidpath.write_text(str(proc.pid));print(json.dumps({'pid':proc.pid,'log':str(dest/'worker.log')},indent=2))

if __name__=='__main__':main()
