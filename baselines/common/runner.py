"""Isolated original runner; no imports from excluded methods."""
import hashlib
import importlib.util
from pathlib import Path
from baselines.common.protocol import ROOT
from src.online_rl.integrated.records import sha256_json


def source_hash():
    paths=[]
    for directory in ('baselines/common','baselines/rar','baselines/online_rubrics','baselines/outcome_only'):
        paths += [p for p in (ROOT/directory).glob('*.py') if not p.name.startswith('test_')]
    paths += [ROOT/'scripts/online_rl/run_five_gpu.py',ROOT/'baselines/common/settings.json']
    return sha256_json({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def isolated_runner(hook):
    path=(ROOT/'baselines/common/four_gpu.py' if 'four_gpu' in hook.cfg else ROOT/'scripts/online_rl/run_five_gpu.py')
    spec=importlib.util.spec_from_file_location('comparison_original_runner',path)
    module=importlib.util.module_from_spec(spec)
    if 'four_gpu' in hook.cfg:
        spec.loader.exec_module(module)
    else:
        # Preserve the original five-GPU algorithm; only redirect model loading
        # and spawned initialization through the same BF16 keyword adapter.
        source=path.read_text()
        replacements={
            'from src.online_rl.runtime import SharedPolicy':'from baselines.common.policy_loader import load_policy as SharedPolicy',
            'from src.online_rl.tree import five_gpu_worker as worker':'from baselines.common import five_gpu_worker_compat as worker',
        }
        for old,new in replacements.items():
            if source.count(old)!=1:raise RuntimeError('Original outcome runner import changed; review compatibility adapter')
            source=source.replace(old,new)
        exec(compile(source,str(path),'exec'),module.__dict__)
    module.experiment=hook
    return module
