"""Finish the active first-batch pilot, then resume the configured full schedule."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/online_rl/tree_batch12_haiku_gemini_luna.json');a=p.parse_args();cfg=json.loads((ROOT/a.config).read_text());out=ROOT/cfg['output_dir']
    import fcntl
    with (out/'continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid=int((out/'runner.pid').read_text())
        while True:
            try:
                cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
                if b'run_tree_batch.py' not in cmd:break
            except OSError:break
            time.sleep(5)
        print('First pilot runner ended; resuming full schedule',flush=True)
        subprocess.run([sys.executable,'scripts/online_rl/launch_tree_batch.py','--config',a.config,'--launch'],cwd=ROOT,check=True)

if __name__=='__main__':main()
