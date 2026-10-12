"""Isolate uncached prompt runs without removing existing evidence."""
import copy
import shutil
import tempfile
from pathlib import Path

from src.online_rl.integrated.records import atomic_write_json


def fresh_prompt_run(cfg, root, phase):
    cfg = copy.deepcopy(cfg)
    spec = cfg['prompt_optimization']
    source = Path(root) / spec['output_dir']
    if phase == 'optimize-prompt' and not (source / 'base_manifest.json').is_file():
        raise ValueError('optimize-prompt needs completed base rollouts; run --phase all first')
    source.parent.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix=source.name + '_nocache_', dir=source.parent))
    if phase == 'optimize-prompt':
        # The fixed rollout dataset is input to search, not an inference cache.
        shutil.copy2(source / 'base_manifest.json', home / 'base_manifest.json')
        for saved in (source / 'base_rollouts').glob('*/rollouts.json'):
            target = home / 'base_rollouts' / saved.parent.name / saved.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved, target)
    spec.update(output_dir=str(home), final_artifact=str(home / 'final_prompt.json'),
                cache_enabled=False)
    config_path = home / 'run_config.json'
    atomic_write_json(config_path, cfg)
    return cfg, config_path
