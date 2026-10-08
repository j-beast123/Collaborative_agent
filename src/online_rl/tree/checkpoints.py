"""Periodic review/evaluation snapshots; recovery saves stay separate."""
import json
import shutil
import tempfile
from pathlib import Path
from src.online_rl.integrated.records import atomic_write_json, sha256_json


def save_periodic_snapshot(out: Path, state: dict, interval: int, prompt_text: str):
    if type(interval) is not int or interval <= 0:
        raise ValueError('checkpoint.every_cases must be a positive integer')
    count = state['case_cursor']
    if not count or count % interval:
        return None
    parent = Path(out) / 'checkpoints' / 'periodic'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / f'case_{count:06d}'
    if target.exists():
        if json.loads((target / 'run_state.json').read_text()) != state:
            raise RuntimeError('Periodic checkpoint differs from committed state')
        return target
    tmp = Path(tempfile.mkdtemp(prefix=target.name + '.pending-', dir=parent))
    try:
        source = state.get('checkpoint')
        if source:
            shutil.copytree(source, tmp / 'model')
        atomic_write_json(tmp / 'run_state.json', state)
        (tmp / 'rubric_generator_prompt.txt').write_text(prompt_text)
        manifest = Path(out) / 'run_manifest.json'
        if manifest.exists():
            shutil.copy2(manifest, tmp / 'run_manifest.json')
        atomic_write_json(tmp / 'COMPLETE.json', {
            'completed_cases': count, 'every_cases': interval, 'model_saved': bool(source),
            'source_recovery_checkpoint': source, 'run_state_sha256': sha256_json(state),
            'policy_sha256': state['policy_sha256'], 'prompt_sha256': state['prompt_sha256'],
            'case_records': {'directory': '../../../cases', 'first': 'case_000000.json',
                             'last': f'case_{count - 1:06d}.json'},
        })
        tmp.rename(target)
        atomic_write_json(parent / 'latest.json', {'completed_cases': count, 'path': target.name})
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target
