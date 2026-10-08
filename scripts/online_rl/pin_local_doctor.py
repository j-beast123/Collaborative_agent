"""Record the installed Q8 Doctor digest/version before a fresh five-GPU run."""
import argparse
import json
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/online_rl/tree_v21_five_gpu.json');a=p.parse_args()
    path=ROOT/a.config;c=json.loads(path.read_text());b=c['rollout']['doctor_backend'];url=c['five_gpu']['doctor_urls'][4]
    with urllib.request.urlopen(url+'/api/tags',timeout=10) as r:tags=json.load(r)['models']
    tag=next((t for t in tags if t['name']==b['model']),None)
    if tag is None or tag['details'].get('quantization_level')!='Q8_0':raise RuntimeError('Requested Q8 Doctor is not installed')
    with urllib.request.urlopen(url+'/api/version',timeout=10) as r:version=json.load(r)['version']
    b.update(model_digest=tag['digest'],quantization='Q8_0',ollama_version=version)
    c['rollout']['doctor_model']=f'ollama/{b["model"]}@{tag["digest"][:12]}'
    c['five_gpu']['opening_identity'].update(model_digest=tag['digest'],quantization='Q8_0',ollama_version=version)
    if (ROOT/c['output_dir']/'run_state.json').exists():raise RuntimeError('Refusing to repin an existing training run')
    path.write_text(json.dumps(c,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps({'doctor_model':c['rollout']['doctor_model'],'quantization':'Q8_0','ollama_version':version}))

if __name__=='__main__':main()
