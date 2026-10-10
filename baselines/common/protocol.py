"""Shared train120 overlay. Original configs remain byte-for-byte untouched."""
import copy
import hashlib
import json
from pathlib import Path
from src.online_rl.integrated.records import sha256_json

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = Path(__file__).with_name('settings.json')
COMMON = ('policy', 'data', 'rollout', 'schedule', 'optimizer', 'advantage_normalization', 'five_gpu')


def settings():
    return json.loads(SETTINGS.read_text())


def apply_common(config):
    cfg, spec = copy.deepcopy(config), settings()
    # Explicit user-approved OpenRouter routing ID (not an inferred date alias).
    cfg['rubric']['generator_model'] = spec['rubric_generator_model']
    cfg['data'].update(train=spec['train_file'], train_sha256=spec['train_sha256'],
                       train_manifest='baselines/common/settings.json')
    cfg['schedule']['cases'] = spec['train_cases']
    cfg['five_gpu'].update(train_order_file=spec['train_order_file'], train_order_sha256=spec['train_order_sha256'],
        eval_file=spec['test_file'], eval_sha256=spec['test_sha256'], checkpoint_evaluation_enabled=False)
    cfg['five_gpu']['base_evaluation']['enabled'] = False
    # User-approved Doctor replacement. One pin governs train, final evaluation,
    # opening identity and the offline asset verifier; original configs stay intact.
    doctor=spec['doctor'];backend=cfg['rollout']['doctor_backend']
    backend.update(model=doctor['model'],model_digest=doctor['model_digest'])
    cfg['rollout']['doctor_model']=f"ollama/{doctor['model']}@{doctor['model_digest'][:12]}"
    cfg['five_gpu']['opening_identity'].update(model=doctor['model'],model_digest=doctor['model_digest'])
    if cfg['prompt_update']['enabled']:
        raise ValueError('Use the no-prompt-optimization reference')
    if 'experiment' in cfg:
        cfg['experiment']['train_order_sha256'] = spec['train_order_sha256']
    return cfg


def identity(config):
    """Comparable across servers: paths/ports are deployment details, not data identity."""
    result = {k: copy.deepcopy(config[k]) for k in COMMON}
    for section, keys in {'data': ('train','train_manifest'),
                          'five_gpu': ('train_order_file','eval_file','ollama_binary','ollama_models_dir','doctor_urls','doctor_gpu_ids','rollout_gpus','update_gpu')}.items():
        for key in keys:
            result[section].pop(key, None)
    result['rollout']['doctor_backend'].pop('endpoint', None)
    # Tensor identity is checked separately; an identical adapter may live at
    # another local path, or be loaded from the verified initial artifact.
    result['policy'].pop('init_adapter', None)
    # Model path/Hub name is deployment-specific; audit base revision separately.
    result['policy'].pop('base_model', None)
    # Final-only scheduling is this comparison's user-requested evaluation rule.
    result['five_gpu']['checkpoint_evaluation_enabled'] = False
    return result


def verify_sherpa_manifest(path, cfg, *, initial_policy_sha256, base_model_revision):
    """Read an exported manifest; never executes or contacts the SHERPA server."""
    if not path:
        return ['Missing SHERPA identity manifest (resolved config, initial adapter hash, base revision)']
    value = json.loads(Path(path).read_text())
    errors = []
    remote = value.get('config')
    if not remote or identity(remote) != identity(cfg):
        errors.append('SHERPA common settings differ from train120 comparison settings')
    if not remote or remote.get('prompt_update', {}).get('enabled') is not False:
        errors.append('SHERPA manifest must explicitly disable prompt optimization')
    if value.get('initial_policy_sha256') != initial_policy_sha256:
        errors.append('SHERPA initial adapter hash differs or is missing')
    if value.get('base_model_revision') != base_model_revision:
        errors.append('SHERPA base model revision differs or is missing')
    return errors


def manifest_template(cfg):
    return {'config': cfg, 'initial_policy_sha256': None, 'base_model_revision': None,
            'note': 'Populate from actual remote SHERPA run, not from a fresh seed-only assumption.',
            'common_identity_sha256': sha256_json(identity(cfg))}


def mixture(remote_config):
    from src.online_rl.tree.experiment import validate_weights
    value={k:remote_config['agent_reward'][k] for k in ('w_outcome','w_rubric')}
    validate_weights(value['w_rubric'],value['w_outcome'])
    return value
