"""Static destination preflight for a production reward-ablation run: no model inference, no Doctor, no training.
Checks git, GPUs, software, local model artifacts, data/prompt hashes, protocol and the accepted runtime profile, compares the
host with the reference environment, and optionally records launch provenance (never secrets).
  preflight_destination.py --config <cfg> [--runtime-config <overlay>] [--expected-commit SHA] [--write-provenance]
  preflight_destination.py --capture-reference <out.json>"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import experiment as X
from src.online_rl.tree.devices import physical_gpus

PROD = 'configs/experiments/reward_prompt_matrix/production'
PROFILE = f'{PROD}/runtime_profile.json'
REFERENCE_ENV = f'{PROD}/reference_env.json'
EXPECTED = {'batch_cases': 5, 'branching': 4, 'rounds': 4, 'cases': 300, 'prompt_mode': ('evolving', 'p0_fixed'),
            'p0_sha256': 'f8d6e39470d7f8a93ce8224cea9e5aa2ed2fa07ced9d982cc5e16d0e495d9bad',
            'feedback_model': 'anthropic/claude-4.6-sonnet-20260217', 'ollama_version': '0.12.5',
            'theta0_policy_sha256': '9018ce3bbacd319049576e8ace88afb4777621a8769919bf46676fca8a85b17c'}
# Protocol families other than the 300-case reward matrix, by experiment.variant prefix: expected values and output root.
PROTOCOLS = {
    'dev_': {'cases': 60, 'rounds': 4, 'prompt_mode': ('evolving',), 'root': 'outputs/dev_opt/', 'scorer_request_cache': None},
    't2dev_': {'cases': 30, 'rounds': 2, 'prompt_mode': ('p0_fixed',), 'root': 'outputs/fixed_t2_dev/', 'scorer_request_cache': 'canonical_request_sha256_v1'},
    't2rw': {'cases': 30, 'rounds': 2, 'prompt_mode': ('p0_fixed',), 'root': 'outputs/fixed_t2_rw/', 'scorer_request_cache': 'canonical_request_sha256_v1',
             'outcome_only_allowed': True},
    't2dis': {'cases': 60, 'rounds': 2, 'prompt_mode': ('p0_fixed',), 'root': 'outputs/fixed_t2_dis/', 'scorer_request_cache': 'canonical_request_sha256_v1',
              'outcome_only_allowed': True},
    't2f120': {'cases': 120, 'rounds': 2, 'prompt_mode': ('p0_fixed',), 'root': 'outputs/fixed_t2_final120/', 'scorer_request_cache': 'canonical_request_sha256_v1'},
    't2bal': {'cases': 60, 'rounds': 2, 'prompt_mode': ('p0_fixed',), 'root': 'outputs/fixed_t2_bal/', 'scorer_request_cache': 'canonical_request_sha256_v1'},
    't2m': {'cases': 120, 'rounds': 2, 'prompt_mode': ('p0_fixed',), 'root': 'outputs/fixed_t2_main120/', 'scorer_request_cache': 'canonical_request_sha256_v1'},
    't2psmoke': {'cases': 5, 'rounds': 2, 'prompt_mode': ('p0_fixed',), 'root': 'outputs/fixed_t2_pilot_smoke/', 'scorer_request_cache': 'canonical_request_sha256_v1',
                 'outcome_only_allowed': True},
    't2p': {'cases': 60, 'rounds': 2, 'prompt_mode': ('p0_fixed',), 'root': 'outputs/fixed_t2_pilot/', 'scorer_request_cache': 'canonical_request_sha256_v1',
            'outcome_only_allowed': True},
    't2smoke': {'cases': 5, 'rounds': 2, 'prompt_mode': ('p0_fixed',), 'root': 'outputs/fixed_t2_smoke/', 'scorer_request_cache': 'canonical_request_sha256_v1'},
    't2_': {'cases': 120, 'rounds': 2, 'prompt_mode': ('p0_fixed',), 'root': 'outputs/fixed_t2_120/', 'scorer_request_cache': 'canonical_request_sha256_v1'},
}


def protocol_family(variant):
    return next((PROTOCOLS[k] for k in PROTOCOLS if (variant or '').startswith(k)), None)
# Fields whose difference changes generated rollouts (Agent bf16 kernels, llama.cpp kernels).
NUMERIC_FIELDS = ('gpu_name', 'compute_capability', 'torch', 'cuda', 'transformers', 'ollama_version', 'agent_dtype', 'attn_implementation')


def sh(*cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired): return ''


def environment():
    """Hardware/software stack of the GPUs this run would use (CUDA_VISIBLE_DEVICES mapping)."""
    import torch, transformers
    ids = physical_gpus()
    rows = [r.split(', ') for r in sh('nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.used,driver_version,compute_cap',
                                       '--format=csv,noheader,nounits').splitlines() if r]
    gpus = [dict(zip(('index', 'name', 'uuid', 'memory_total_mib', 'memory_used_mib', 'driver', 'compute_capability'), r)) for r in rows]
    used = [g for g in gpus if g['index'] in ids]
    binary = os.environ.get('SHERPA_OLLAMA_BINARY', 'ollama')
    ollama = sh(binary, '--version').split()[-1] if sh(binary, '--version') else None
    return {'host': platform.node(), 'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'transformers': transformers.__version__, 'driver': used[0]['driver'] if used else None,
            'gpu_name': used[0]['name'] if used else None, 'compute_capability': used[0]['compute_capability'] if used else None,
            'gpus': used, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'ollama_binary': binary,
            'ollama_version': ollama, 'agent_dtype': 'bfloat16', 'attn_implementation': 'transformers default (sdpa)'}


def doctor_digest(cfg):
    models = os.environ.get('SHERPA_OLLAMA_MODELS', str(Path.home() / '.ollama/models'))
    name, tag = cfg['rollout']['doctor_backend']['model'].split(':')
    manifest = Path(models) / 'manifests/registry.ollama.ai/library' / name / tag
    return hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None


def agent_snapshot(cfg):
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(cfg['policy']['base_model'], 'config.json')
        return str(Path(hit).parent.name) if isinstance(hit, str) else None
    except ImportError: return None


def check(cfg_path, runtime_cfg, expected_commit):
    from scripts.online_rl.run_five_gpu import preflight
    cfg = X.load_config(runtime_cfg or cfg_path); prod = X.load_config(cfg_path)
    profile = json.loads((ROOT / PROFILE).read_text())
    env = environment(); hard, warn = [], []
    head = sh('git', '-C', str(ROOT), 'rev-parse', 'HEAD'); dirty = [l for l in sh('git', '-C', str(ROOT), 'status', '--porcelain').splitlines() if l]
    if not head: warn.append('not a git checkout: commit cannot be verified')
    elif expected_commit and not head.startswith(expected_commit): hard.append(f'HEAD {head[:12]} != expected {expected_commit}')
    if dirty: hard.append(f'working tree not clean ({len(dirty)} entries)')
    try: ids = physical_gpus()
    except ValueError as e: hard.append(str(e)); ids = []
    if len(env['gpus']) != len(ids): hard.append(f'mapped GPUs found {len(env["gpus"])} of {len(ids)}')
    busy = [g['index'] for g in env['gpus'] if int(g['memory_used_mib']) > 1000]
    if busy: hard.append(f'GPUs already in use: {busy}')
    if env['ollama_version'] != EXPECTED['ollama_version']: hard.append(f'Ollama {env["ollama_version"]} != {EXPECTED["ollama_version"]}')
    if doctor_digest(cfg) != cfg['rollout']['doctor_backend']['model_digest']: hard.append('Doctor model digest missing or different')
    if not agent_snapshot(cfg): hard.append(f"Agent model {cfg['policy']['base_model']} not in local HF cache")
    hard += [f'runner preflight: {e}' for e in preflight(cfg)]
    e, ar, ro = cfg['experiment'], cfg['agent_reward'], cfg['rollout']
    proto = {'batch_cases': cfg['five_gpu']['batch_cases'], 'branching': ro['branching'], 'rounds': ro['rounds'],
             'cases': cfg['schedule']['cases'], 'prompt_mode': e['prompt_mode'], 'feedback_model': cfg['feedback']['model']}
    fam = protocol_family(e.get('variant'))
    expected = {**EXPECTED, **{k: fam[k] for k in ('cases', 'rounds', 'prompt_mode')}} if fam else EXPECTED
    hard += [f'{k}={v!r} expected {expected[k]!r}' for k, v in proto.items() if (v not in expected[k] if k == 'prompt_mode' else v != expected[k])]
    if fam and not cfg['output_dir'].startswith(fam['root']): hard.append(f"variant {e.get('variant')} must write under {fam['root']}")
    if cfg['judges'].get('scorer_request_cache') != (fam or {}).get('scorer_request_cache'):
        hard.append(f"judges.scorer_request_cache {cfg['judges'].get('scorer_request_cache')!r} does not match the protocol")
    outcome_only = ar['w_rubric'] == 0 and (fam or {}).get('outcome_only_allowed')
    if not cfg['rubric']['enabled'] and not outcome_only: hard.append('rubric reward pipeline not enabled')
    if cfg['prompt_update']['enabled'] != (e['prompt_mode'] == 'evolving'): hard.append('prompt evolution does not match prompt_mode')
    if cfg['rubric']['prompt_sha256'] != EXPECTED['p0_sha256']: hard.append('initial P0 differs from the production P0')
    if X.fingerprint(cfg) != X.fingerprint(prod): hard.append('runtime overlay changed the research fingerprint')
    if profile.get('status') != 'decided': hard.append('runtime profile not decided (smoke pending)')
    elif cfg['five_gpu']['api_workers'] != profile['api_workers']: hard.append('config api_workers differs from the runtime profile')
    if os.environ.get('SHERPA_PARALLEL_STRATA', '0') != str(int(bool(profile.get('parallel_strata')))): hard.append('SHERPA_PARALLEL_STRATA differs from the runtime profile')
    if profile.get('checkpoint_evaluation') == 'async' and cfg['five_gpu'].get('checkpoint_evaluation_enabled', True):
        hard.append('runtime profile uses async eval but the runner would block on checkpoint eval')
    ref = json.loads((ROOT / REFERENCE_ENV).read_text()) if (ROOT / REFERENCE_ENV).exists() else None
    diffs = {k: (ref.get(k), env.get(k)) for k in NUMERIC_FIELDS if ref and ref.get(k) != env.get(k)}
    cross = {'reference_host': ref and ref.get('host'), 'differences': diffs,
             'status': 'OK' if not diffs else 'WARNING',
             'verdict': None if not diffs else ('B: host/software confounded with reward condition; not suitable as a paired main comparison '
                                                 'unless all compared arms run on this stack' if set(diffs) & {'gpu_name', 'compute_capability', 'torch', 'transformers'}
                                                 else 'A: runtime differs but rollout kernels match; acceptable with the difference reported')}
    report = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'commit': head, 'config': cfg_path, 'runtime_config': runtime_cfg,
              'run_id': e['run_id'], 'output_dir': cfg['output_dir'], 'resolved_config_sha256': sha256_json(cfg),
              'experiment_fingerprint': X.fingerprint(cfg), 'variant': e.get('variant'),
              'training_hyperparameters': X.training_hyperparameters(cfg), 'reward': {'w_outcome': ar['w_outcome'], 'w_rubric': ar['w_rubric']},
              'protocol': proto, 'feedback_prompt_sha256': cfg['feedback']['prompt_sha256'], 'p0_sha256': cfg['rubric']['prompt_sha256'],
              'expected_theta0_policy_sha256': EXPECTED['theta0_policy_sha256'], 'data': {'train_sha256': cfg['data']['train_sha256'],
              'train_order_sha256': cfg['five_gpu']['train_order_sha256'], 'eval_sha256': cfg['five_gpu']['eval_sha256'], 'seed': cfg['schedule']['seed']},
              'doctor': {'model': ro['doctor_backend']['model'], 'digest': doctor_digest(cfg), 'urls': cfg['five_gpu']['doctor_urls']},
              'agent_snapshot': agent_snapshot(cfg), 'runtime_profile': profile,
              'runtime': {'api_workers': cfg['five_gpu']['api_workers'], 'SHERPA_PARALLEL_STRATA': os.environ.get('SHERPA_PARALLEL_STRATA', '0')},
              'environment': env, 'cross_host': cross, 'hard_failures': hard, 'warnings': warn}
    return report


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--config'); a.add_argument('--runtime-config'); a.add_argument('--expected-commit')
    a.add_argument('--write-provenance', action='store_true'); a.add_argument('--capture-reference')
    x = a.parse_args()
    if x.capture_reference:
        Path(x.capture_reference).write_text(json.dumps(environment(), indent=2) + '\n'); print(x.capture_reference); return
    r = check(x.config, x.runtime_config, x.expected_commit)
    print(json.dumps({k: r[k] for k in ('run_id', 'output_dir', 'reward', 'protocol', 'runtime', 'cross_host', 'hard_failures', 'warnings')}, indent=2))
    if r['cross_host']['status'] == 'WARNING': print('CROSS-HOST COMPARABILITY: WARNING -', r['cross_host']['verdict'], file=sys.stderr)
    if x.write_provenance and not r['hard_failures']:
        out = ROOT / r['output_dir']; out.mkdir(parents=True, exist_ok=True); p = out / 'LAUNCH_PROVENANCE.json'
        rec = json.loads(p.read_text()) if p.exists() else {'run_id': r['run_id'], 'launches': []}
        rec['launches'].append(r); p.write_text(json.dumps(rec, indent=2) + '\n')
    sys.exit(1 if r['hard_failures'] else 0)


if __name__ == '__main__': main()
