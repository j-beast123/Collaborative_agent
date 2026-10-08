"""Start local Doctor servers and a detached batch runner only with --launch."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.online_rl.run_tree_batch import preflight
from scripts.online_rl.start_batch_doctors import start


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/online_rl/tree_batch12_haiku_gemini_luna.json');parser.add_argument('--launch',action='store_true');parser.add_argument('--max-batches',type=int)
    a=parser.parse_args();cfg=json.loads((ROOT/a.config).read_text());errors=preflight(cfg)
    print(json.dumps({'blockers':errors,'launch':a.launch,'config':a.config},indent=2),flush=True)
    if errors:raise SystemExit(1)
    if not a.launch:return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<5:raise RuntimeError('Need CUDA-enabled PyTorch and five visible GPUs')
    from scripts.online_rl.run_tree import load_key
    load_key() # Check credentials without printing or passing them on command line.
    out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True);pidpath=out/'runner.pid'
    if pidpath.exists():
        try:
            pid=int(pidpath.read_text());cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
            if b'run_tree_batch.py' in cmd:raise RuntimeError(f'Batch runner already active: {pid}')
        except (OSError,ValueError):pass
    start(cfg)
    cmd=[sys.executable,'scripts/online_rl/run_tree_batch.py','--config',a.config,'--launch']
    if a.max_batches is not None:cmd+=['--max-batches',str(a.max_batches)]
    with (out/'runner.log').open('a') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},stdout=log,stderr=log,start_new_session=True)
    pidpath.write_text(str(process.pid))
    subprocess.run([sys.executable,'scripts/evaluation/launch_checkpoint_queue.py','--run-dir',str(out),'--launch'],cwd=ROOT,check=True)
    print(json.dumps({'pid':process.pid,'log':str(out/'runner.log')},indent=2))

if __name__=='__main__':main()
