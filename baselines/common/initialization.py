"""Record actual local seeded initial tensors; compare across methods and resumes."""
import fcntl,json
from pathlib import Path
from src.online_rl.integrated.records import atomic_write_json,sha256_json


def initial_identity(cfg):
    cm=cfg['comparison_method']
    return {'policy':cfg['policy'],'local_model':cm.get('local_model_identity'),
        'packages':cm.get('package_versions'), 'initialization':'src.online_rl.runtime.seeded_lora'}


def expected_initial(cfg):
    cm=cfg['comparison_method']
    if cm.get('initial_policy_sha256'):return cm['initial_policy_sha256']
    if not cm.get('initialization_registry'):return None
    identity=initial_identity(cfg)
    path=Path(cm['initialization_registry'])/(sha256_json(identity)+'.json')
    if not path.exists():return None
    row=json.loads(path.read_text())
    if row['identity']!=identity or row['identity_sha256']!=sha256_json(identity):raise RuntimeError('Initial identity registry differs')
    return row['initial_policy_sha256']


def record_initial(cfg,observed):
    expected=cfg['comparison_method'].get('initial_policy_sha256')
    if expected and observed!=expected:raise RuntimeError('Initial adapter differs from the supplied expected hash')
    cm=cfg['comparison_method']
    identity=initial_identity(cfg)
    row={'identity':identity,'identity_sha256':sha256_json(identity),'initial_policy_sha256':observed}
    registry=cm.get('initialization_registry')
    if registry:
        root=Path(registry);root.mkdir(parents=True,exist_ok=True);path=root/(row['identity_sha256']+'.json')
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists():
                if json.loads(path.read_text())!=row:raise RuntimeError('Seeded initial model differs across methods; no rollout started')
            else:atomic_write_json(path,row)
    own=Path(cfg['output_dir'])/'initial_policy.json'
    if own.exists():
        if json.loads(own.read_text())!=row:raise RuntimeError('Initial model differs from this run on resume')
    else:atomic_write_json(own,row)
    return row
