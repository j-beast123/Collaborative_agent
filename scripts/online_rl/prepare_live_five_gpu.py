"""Prepare the owned GPU4 Doctor server, Q8 model and local runtime pins."""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/online_rl/tree_v21_five_gpu.json');a=p.parse_args()
    from scripts.online_rl.start_five_gpu_doctors import start
    cfg=json.loads((ROOT/a.config).read_text());bootstrap=copy.deepcopy(cfg)
    bootstrap['five_gpu']['doctor_urls']=[cfg['five_gpu']['doctor_urls'][4]]
    bootstrap['five_gpu']['doctor_gpu_ids']=[4]
    start(bootstrap,require_model=False)
    env={**os.environ,'OLLAMA_HOST':cfg['five_gpu']['doctor_urls'][4].removeprefix('http://')}
    subprocess.run(['/usr/local/bin/ollama','pull',cfg['rollout']['doctor_backend']['model']],env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/online_rl/pin_local_doctor.py'),'--config',a.config],check=True)
