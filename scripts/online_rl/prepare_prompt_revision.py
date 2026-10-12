"""Prepare isolated v5 search inputs after iteration one; no inference or API calls."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def prepare(cfg, training_cfg, root):
    from src.online_rl.doctor import make_doctor
    from src.online_rl.integrated.records import atomic_write_json, read_json
    from src.online_rl.sherpa.config import config_problems
    from src.online_rl.sherpa.prompt_search import base_identity
    root = Path(root)
    problems = config_problems(cfg, root)
    if problems:
        raise ValueError('; '.join(problems))
    state = read_json(root / training_cfg['output_dir'] / 'run_state.json')
    if state['completed_steps'] < 30:
        raise RuntimeError('Wait for the first 30 training steps to commit')
    source = root / training_cfg['prompt_optimization']['output_dir']
    target = root / cfg['prompt_optimization']['output_dir']
    if source.resolve() == target.resolve():
        raise ValueError('The revised search needs a separate output directory')
    manifest = read_json(source / 'base_manifest.json')
    doctor = make_doctor(cfg['rollout'], (root / cfg['rollout']['doctor_system_prompt_file']).read_text())
    if not manifest['complete'] or manifest['identity'] != base_identity(cfg, SimpleNamespace(doctor=doctor)):
        raise RuntimeError('Completed base inputs do not match the revised rollout protocol')
    sources = [(source / 'base_rollouts' / key / 'rollouts.json') for key in manifest['case_ids']]
    for path in sources:
        row = read_json(path)
        if row['identity'] != manifest['identity'] or row['case_id'] != path.parent.name or len(row['trajectories']) != 32:
            raise RuntimeError('Invalid original-base evidence')
    # Only copy immutable rollout inputs. Fresh reflection/generation/judging
    # evidence must use the new backend identity in this separate directory.
    target.mkdir(parents=True, exist_ok=True)
    for path in sources:
        destination = target / 'base_rollouts' / path.parent.name / 'rollouts.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != path.read_bytes():
                raise RuntimeError('Revised search already contains different base inputs')
        else:
            shutil.copy2(path, destination)
    saved = target / 'base_manifest.json'
    if saved.exists() and read_json(saved) != manifest:
        raise RuntimeError('Revised search base manifest changed')
    if not saved.exists():
        atomic_write_json(saved, manifest)
    return {'completed_training_steps': state['completed_steps'], 'copied_base_cases': len(sources),
            'output_dir': str(target), 'inference_started': False}


if __name__ == '__main__':
    import json
    from src.online_rl.tree.experiment import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/sherpa_qwen_doctor_v5.json')
    parser.add_argument('--training-config', default='configs/ours.json')
    args = parser.parse_args()
    os.chdir(ROOT)
    print(json.dumps(prepare(load_config(args.config), load_config(args.training_config), ROOT), indent=2))
