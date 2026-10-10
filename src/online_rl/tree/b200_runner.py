"""Two dedicated GPUs, five concurrent cases, one frozen-policy update barrier.

CPU-importable. Reuses the original tree builder, GRPO updater, checkpoint
publication/restore and evaluator. No model/service is started by preflight.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

from src.online_rl.integrated.records import atomic_write_json, sha256_json
from src.online_rl.tree.five_gpu import (batch_slots, training_order, save_checkpoint,
    load_checkpoint, update_batch, EvaluationQueue, cache_openings)
from src.online_rl.tree.b200_diagnostics import batch_diagnostics, usage_report, attempt_report, error_report

ROOT = Path(__file__).resolve().parents[3]
CONFIG = 'configs/experiments/train120_max4_rubric_only_b200.json'
ENV_PATHS = {'SHERPA_BASE_MODEL': ('policy', 'base_model'),
             'SHERPA_TRAIN_FILE': ('data', 'train'),
             'SHERPA_TRAIN_ORDER': ('five_gpu', 'train_order_file'),
             'SHERPA_TEST_FILE': ('five_gpu', 'eval_file'),
             'SHERPA_OLLAMA_BINARY': ('five_gpu', 'ollama_binary'),
             'SHERPA_OLLAMA_MODELS': ('five_gpu', 'ollama_models_dir')}


def positive_learning_rate(value):
    try:
        rate = float(value)
    except (TypeError, ValueError):
        raise ValueError('Learning rate must be a finite positive number') from None
    if isinstance(value, bool) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Learning rate must be a finite positive number')
    return rate


def load_config(path=CONFIG, *, learning_rate=None):
    cfg = json.loads((ROOT / path).read_text())
    # Explicit CLI > environment > config. Equivalent numeric forms have the
    # same configuration identity; a changed LR starts a separate experiment.
    requested = learning_rate if learning_rate is not None else os.environ.get('SHERPA_LEARNING_RATE')
    baseline = positive_learning_rate(cfg['optimizer']['learning_rate'])
    rate = baseline if requested is None else positive_learning_rate(requested)
    if rate != baseline:
        cfg['optimizer']['learning_rate'] = rate
        cfg['output_dir'] += '/lr_' + repr(rate)
        cfg['experiment']['output_root'] = cfg['output_dir']
        cfg['notes']['changes'].append(f'Run-start learning rate override: {baseline} -> {rate}; separate output/cache/checkpoints')
    from src.online_rl.tree.b200_doctor_setup import receipt_path, apply_receipt
    receipt = receipt_path()
    if receipt.is_file():
        cfg = apply_receipt(cfg, receipt)
    elif os.environ.get('SHERPA_DOCTOR_RECEIPT'):
        raise ValueError(f'Doctor receipt is missing: {receipt}; run the Doctor setup pull command first')
    for env, (section, key) in ENV_PATHS.items():
        if os.environ.get(env):
            cfg[section][key] = str(Path(os.environ[env]).expanduser().resolve())
    return cfg


def validate_resume_identity(state, cfg, source_sha):
    if state and (state['config_sha256'] != sha256_json(cfg) or state['source_sha256'] != source_sha):
        raise RuntimeError('Config/code changed (including learning rate); resume requires the original run settings')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def code_identity():
    files = sorted([*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/online_rl/*.py')])
    return sha256_json({str(p.relative_to(ROOT)): file_sha(p) for p in files})


def physical_devices():
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
    if len(devices) != 2 or len(set(devices)) != 2 or any(not x.strip() for x in devices):
        raise ValueError('CUDA_VISIBLE_DEVICES must contain exactly two distinct physical GPUs; Doctor first, Assistant second')
    return [d.strip() for d in devices]


def preflight(cfg, *, resources=True):
    from scripts.online_rl.run_tree import blockers
    from src.online_rl.cases import load_cases
    from src.online_rl.guards import assert_not_protected
    errors = blockers(cfg)
    rb = cfg['rubric']
    if rb.get('bank_mode') != 'parallel_candidates_consolidate_v1':
        errors.append('This B200 version requires the parallel candidate/consolidate bank')
    if type(rb.get('candidate_batch_size')) is not int or rb['candidate_batch_size'] < 1:
        errors.append('rubric.candidate_batch_size must be a positive integer')
    cap = cfg.get('execution', {}).get('global_api_max_in_flight')
    if type(cap) is not int or cap < 1: errors.append('global_api_max_in_flight must be a positive integer')
    for name, role, model in (('generator', rb.get('generation', {}), rb.get('generator_model')),
                              ('consolidate', rb.get('consolidate', {}), rb.get('consolidate', {}).get('model'))):
        if model != 'openai/gpt-6-luna' or role.get('reasoning_effort') != 'none' or role.get('strict_provider_parameters') is not True:
            errors.append(f'{name} requires gpt-6-luna, explicit reasoning none and strict provider parameters')
    cc = rb.get('consolidate', {})
    cp = ROOT / cc.get('prompt_file', '__missing__')
    if not cp.is_file() or file_sha(cp) != cc.get('prompt_sha256'):
        errors.append('Consolidate prompt missing or hash mismatch')
    expected = {'doctor_gpu': 0, 'assistant_gpu': 1, 'assistant_replicas': 1, 'doctor_replicas': 1,
                'case_workers': 5, 'batch_cases': 5, 'agent_batch_size': 4, 'doctor_concurrency': 1, 'evaluation_workers': 4}
    if cfg.get('two_gpu') != expected: errors.append('Two-GPU topology/concurrency differs from the validated configuration')
    if cfg['schedule']['cases'] != 120 or cfg['rollout']['rounds'] != 4 or cfg['rollout']['branching'] != 4:
        errors.append('Requires train120, max4, branching4')
    if cfg['agent_reward'] != {'w_outcome': 0.0, 'w_rubric': 1.0, 'r_a_weighting': 'equal'}:
        errors.append('Requires rubric-only, equal KEEP weights, without outcome fallback')
    if cfg['rollout']['agent_protocol'] != 'precommit_v1' or cfg['prompt_update']['enabled']:
        errors.append('Requires precommit_v1 and fixed P0')
    if cfg['verification']['mode'] != 'fixed_source_linear_majority_v1' or cfg['intervention']['reruns_per_intervention'] != 3:
        errors.append('Requires fixed creator source and three planned linear trials')
    if not cfg.get('doctor_need', {}).get('enabled') or not cfg.get('fixed_accuracy', {}).get('enabled'):
        errors.append('This version requires prospective Doctor needs and one fixed accuracy criterion')
    fg = cfg['five_gpu']
    if fg['checkpoint_evaluation_enabled'] or fg['base_evaluation']['enabled'] or cfg.get('evaluation'):
        errors.append('Training evaluation must be disabled; use the evaluate action')
    if fg['doctor_urls'] != [cfg['rollout']['doctor_backend']['endpoint']] or fg['doctor_gpu_ids'] != [0]:
        errors.append('Dedicated Doctor endpoint/GPU mapping differs')
    if fg['opening_identity'].get('private_need') != cfg.get('doctor_need'):
        errors.append('D1 cache identity must include the prospective need protocol')
    if fg['checkpoint_every_cases'] != 30 or fg['batch_cases'] != 5:
        errors.append('Requires five-case batches and 30-case milestone checkpoints')
    try: physical_devices()
    except ValueError as exc: errors.append(str(exc))
    data_ok = True
    for section, key, digest in ((cfg['data'], 'train', 'train_sha256'), (fg, 'train_order_file', 'train_order_sha256'), (fg, 'eval_file', 'eval_sha256')):
        p = ROOT / section[key]
        if not p.is_file(): errors.append(f'Missing file: {p}'); data_ok = False
        elif file_sha(p) != section[digest]: errors.append(f'Hash differs: {p}'); data_ok = False
    if data_ok:
        try:
            train, test = load_cases(ROOT / cfg['data']['train']), load_cases(ROOT / fg['eval_file'])
            training_order(cfg, train, ROOT)
            assert_not_protected(train, 'B200 train120')
            if len(train) != 120 or len(test) != 622 or len({c['case_id'] for c in test}) != 622:
                errors.append('Dataset counts/IDs differ')
            norm = lambda c: ' '.join(c['question'].split()).casefold()
            if {norm(c) for c in train} & {norm(c) for c in test}: errors.append('Train/test question overlap')
        except (ValueError, RuntimeError, OSError) as exc: errors.append(str(exc))
    if resources:
        if os.environ.get('SHERPA_HF_RUN_ID'):
            from src.online_rl.tree.b200_hub import run_id
            try: run_id(os.environ['SHERPA_HF_RUN_ID'])
            except ValueError as exc: errors.append(str(exc))
            if not os.environ.get('HF_TOKEN', '').strip(): errors.append('HF_TOKEN is required for enabled HF backups (write access to the model repo)')
        model = Path(cfg['policy']['base_model'])
        if not model.is_dir() or not (model / 'config.json').is_file():
            errors.append('Set SHERPA_BASE_MODEL to the existing local Qwen3-8B snapshot')
        else:
            index = model / 'model.safetensors.index.json'
            if index.is_file():
                shards = set(json.loads(index.read_text())['weight_map'].values())
                if any(not (model / s).is_file() for s in shards): errors.append('Local Qwen weight shards are incomplete')
            elif not (model / 'model.safetensors').is_file(): errors.append('Local Qwen weights missing')
        binary = fg.get('ollama_binary')
        if not binary or not Path(binary).is_file(): errors.append('Set SHERPA_OLLAMA_BINARY to Ollama 0.12.5')
        else:
            import subprocess
            result = subprocess.run([binary, '--version'], capture_output=True, text=True, timeout=15)
            if '0.12.5' not in result.stdout + result.stderr: errors.append('Ollama executable version must be 0.12.5')
        models = fg.get('ollama_models_dir')
        manifest = Path(models or '/__unset__') / 'manifests/registry.ollama.ai/library/gemma3/12b-it-q8_0'
        if not manifest.is_file(): errors.append('Set SHERPA_OLLAMA_MODELS to a store containing gemma3:12b-it-q8_0')
        elif file_sha(manifest) != cfg['rollout']['doctor_backend']['model_digest']:
            errors.append('Doctor manifest digest differs from the effective pinned configuration; refusing model substitution')
        else:
            m = json.loads(manifest.read_text())
            for blob in [m['config'], *m['layers']]:
                p = Path(models) / 'blobs' / blob['digest'].replace(':', '-')
                if not p.is_file() or p.stat().st_size != blob['size']: errors.append(f'Missing/incomplete Doctor blob: {p.name}')
        receipt = cfg.get('doctor_download_receipt')
        if receipt:
            if binary != receipt['binary'] or models != receipt['models']:
                errors.append('Environment paths differ from the downloaded Doctor receipt')
            if binary and Path(binary).is_file() and file_sha(binary) != receipt['binary_sha256']:
                errors.append('Pinned Ollama executable changed')
        if not os.environ.get('OPENROUTER_API_KEY', '').strip(): errors.append('OPENROUTER_API_KEY is missing in this terminal')
    return errors


def configure_execution(cfg, out):
    from src.common.api_limiter import configure
    configure(cfg.get('execution', {}).get('global_api_max_in_flight', 100))
    os.environ['SHERPA_AGENT_REPLICAS'] = '1'
    os.environ['SHERPA_AGENT_BATCH_SIZE'] = '4'
    os.environ['SHERPA_DOCTOR_CONCURRENCY'] = '1'
    os.environ['SHERPA_RUBRIC_SCORE_BATCH_SIZE'] = '4'
    os.environ['SHERPA_API_WORKERS'] = str(cfg['five_gpu']['api_workers'])
    os.environ['SHERPA_MEMBER_WORKERS'] = '1'
    os.environ['SHERPA_PARALLEL_STRATA'] = '1'
    os.environ['SHERPA_AUDIT_ALL_ERRORS'] = '1'
    os.environ['API_USAGE_LOG'] = str(out / 'api_usage.jsonl')
    os.environ['SHERPA_RETRY_AUDIT'] = str(out / 'retry_audit.jsonl')
    os.environ['SHERPA_HTTP_AUDIT'] = str(out / 'http_attempts.jsonl')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.pop('SHERPA_DOCTOR_ENDPOINTS', None)


def run_case(cfg, policy, case, index, batchdir, expected):
    from scripts.online_rl.run_tree import build
    from src.common import usage
    local = copy.deepcopy(cfg)
    local['output_dir'] = str(batchdir / f'case_{index:06d}')
    local['schedule']['cases'] = 1
    with usage.case(case['case_id']):
        trainer = build(local, policy=policy, cases=[case], defer_update=True)
        if trainer.state.policy_sha256 != expected: raise RuntimeError('Case policy differs at batch barrier')
        if trainer.state.case_cursor:
            record = json.loads((Path(local['output_dir']) / 'cases/case_000000.json').read_text())
        else:
            record = trainer.step()
        trainer.validate_member_interventions(record)
        if record['policy_sha256'] != expected: raise RuntimeError('Stale saved case policy')
        record['case_cursor'] = index
        # This is not an optimizer step; the parent owns the five-case boundary.
        record['update'] = {'optimizer_step': False, 'deferred_to_batch': True, 'update_gpu': 1}
        return record


def commit_batch(cfg, out, state, records, updater, actual, *, save=save_checkpoint, load=load_checkpoint):
    """Publish checkpoint first, state last. Recover without applying a second update."""
    if len(records) != 5 or [r['case_cursor'] for r in records] != list(range(state['case_cursor'], state['case_cursor'] + 5)):
        raise RuntimeError('Exactly five ordered cases are required before update')
    batch = state['batch_cursor'] + 1
    cursor = state['case_cursor'] + 5
    milestone = cursor % 30 == 0 or cursor == 120
    checkpoint = out / 'checkpoints' / (f'checkpoint_{cursor:06d}' if milestone else f'recovery_{batch:06d}')
    identity = {'records_sha256': sha256_json(records), 'source_policy_sha256': state['policy_sha256'],
                'config_sha256': state['config_sha256'], 'source_sha256': state['source_sha256'], 'completed_cases': cursor}
    if checkpoint.exists():
        proof = json.loads((checkpoint / 'COMPLETE.json').read_text())
        if any(proof.get(k) != v for k, v in identity.items()): raise RuntimeError('Recovery checkpoint identity differs')
        load(updater, checkpoint)
        if actual() != proof['policy_sha256']: raise RuntimeError('Recovered adapter differs')
        update, seconds = proof['update'], proof['update_seconds']
    else:
        started = time.monotonic()
        update = update_batch(updater, records, state['policy_sha256'], actual)
        seconds = time.monotonic() - started
        proof = {**identity, 'policy_sha256': actual(), 'update': update, 'update_seconds': seconds}
        save(updater, checkpoint, proof)
    diagnostics = batch_diagnostics(records, update, seconds)
    diagnostics['api_usage_by_role'] = usage_report(out / 'api_usage.jsonl', {r['case_id'] for r in records})
    diagnostics['actual_http_attempts_by_role'] = attempt_report(out, {r['case_id'] for r in records})
    diagnostics['failed_cache_operations_by_kind'] = error_report(out / 'batches' / f'batch_{batch:06d}')
    atomic_write_json(out / 'diagnostics' / f'batch_{batch:06d}.json', diagnostics)
    for record, diagnostic in zip(records, diagnostics['cases']):
        diagnostic['api_usage_by_role'] = usage_report(out / 'api_usage.jsonl', {record['case_id']})
        diagnostic['actual_http_attempts_by_role'] = attempt_report(out, {record['case_id']})
        atomic_write_json(out / 'cases' / f"case_{record['case_cursor']:06d}.json", {**record, 'batch_update': update})
        atomic_write_json(out / 'diagnostics' / f"case_{record['case_cursor']:06d}.json", diagnostic)
    next_state = {**state, 'case_cursor': cursor, 'batch_cursor': batch, 'checkpoint': str(checkpoint), 'policy_sha256': actual()}
    atomic_write_json(out / 'run_state.json', next_state)
    return next_state


def model_identity(cfg):
    base = Path(cfg['policy']['base_model'])
    return {p.name: file_sha(p) for p in sorted(base.iterdir())
            if p.is_file() and (p.suffix in ('.json', '.safetensors') or p.name.endswith('.model'))}


def run(cfg, *, max_batches=None, evaluate=False):
    import torch
    from src.common import usage
    from src.online_rl.runtime import SharedPolicy
    from src.online_rl.integrated.grpo import make_updater
    from src.online_rl.integrated.continuation import adapter_sha256, protocol_sha256
    from src.online_rl.cases import load_cases
    from src.online_rl.environment import final_outcome
    from src.online_rl.tree.b200_hub import storage_settings
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Exactly two visible CUDA GPUs are required')
    out = (ROOT / cfg['output_dir']).resolve(); out.mkdir(parents=True, exist_ok=True)
    configure_execution(cfg, out)
    with ExitStack() as stack:
        for p in [out / 'runner.lock', *[Path(f'/tmp/sherpa_tree_physical_gpu{d}.lock') for d in physical_devices()]]:
            lock = stack.enter_context(p.open('a'))
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError(f'Another run owns {p.name}')
        config_sha, source_sha = sha256_json(cfg), code_identity()
        path = out / 'run_state.json'
        state = json.loads(path.read_text()) if path.exists() else None
        validate_resume_identity(state, cfg, source_sha)
        if evaluate and (not state or state['case_cursor'] != 120):
            raise RuntimeError('Final evaluation requires all 120 training cases committed')
        if state is None and any((out / 'checkpoints').glob('*')):
            raise RuntimeError('Orphan initial checkpoint; inspect before restarting; do not overwrite')
        base_identity = model_identity(cfg)
        if state and state['base_model_identity'] != base_identity: raise RuntimeError('Local base model changed')
        torch.manual_seed(cfg['schedule']['seed'])
        policy = SharedPolicy(cfg, os.environ['OPENROUTER_API_KEY'], device='cuda:1')
        if policy.doctor.verify()['problems']: raise RuntimeError('Running Doctor does not match pinned identity')
        policy.seed_all = lambda seed: None
        initial = adapter_sha256(policy.model)
        expected_initial = cfg.get('expected_initial_adapter_sha256')
        if expected_initial and expected_initial != initial: raise RuntimeError('Initial adapter hash differs from SHERPA')
        if state and state['initial_adapter_sha256'] != initial: raise RuntimeError('Reinitialized adapter differs on resume')
        updater = make_updater(policy, cfg)
        cache_openings(policy.doctor, out / 'openings/common.jsonl', cfg['five_gpu']['opening_identity'])
        original_generate = policy._generate
        def measured_generate(prompt, max_new_tokens, seed, segment='agent'):
            started = time.monotonic()
            value = original_generate(prompt, max_new_tokens, seed, segment)
            usage.record('local_agent_private' if max_new_tokens == 120 else 'local_agent_visible', cfg['policy']['base_model'],
                {'prompt_tokens': len(value['prompt_input_ids']), 'completion_tokens': len(value['completion_ids'])},
                latency_seconds=time.monotonic() - started)
            return value
        policy._generate = measured_generate
        if state:
            load_checkpoint(updater, state['checkpoint'])
            if adapter_sha256(policy.model) != state['policy_sha256']: raise RuntimeError('Restored policy hash mismatch')
        else:
            checkpoint = out / 'checkpoints/checkpoint_000000'
            state = {'case_cursor': 0, 'batch_cursor': 0, 'config_sha256': config_sha, 'source_sha256': source_sha,
                     'base_model_identity': base_identity, 'initial_adapter_sha256': initial,
                     'policy_sha256': initial, 'checkpoint': str(checkpoint)}
            save_checkpoint(updater, checkpoint, state)
            atomic_write_json(out / 'run_manifest.json', {'config': cfg, 'initial_state': state,
                'logical_to_physical_gpu': physical_devices(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'hf_storage': storage_settings()})
            atomic_write_json(path, state)
        actual = lambda: adapter_sha256(policy.model)
        try:
            from src.online_rl.tree.b200_hub import sync_barrier
            # Retry a failed remote backup before advancing the locally committed run.
            if state['case_cursor'] > 0: sync_barrier(out)
            if evaluate:
                final_evaluation(cfg, out, state, policy, protocol_sha256(cfg, ROOT), final_outcome)
                sync_barrier(out)
                return
            cases = load_cases(ROOT / cfg['data']['train']); byid = {c['case_id']: c for c in cases}
            order = training_order(cfg, cases, ROOT)
            count = 0
            while state['case_cursor'] < 120 and (max_batches is None or count < max_batches):
                batch = state['batch_cursor'] + 1
                batchdir = out / 'batches' / f'batch_{batch:06d}'
                slots = batch_slots(state['case_cursor'], 120, interval=30, workers=1, lanes=5)
                manifest = {'indices': [s['index'] for s in slots], 'case_ids': [order[s['index']] for s in slots],
                            'policy_sha256': state['policy_sha256']}
                manifest_path = batchdir / 'manifest.json'
                if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                    raise RuntimeError('Pending batch order/policy differs')
                atomic_write_json(manifest_path, manifest)
                print(json.dumps({'stage': 'case_pipeline', 'batch': batch, 'cases': 5, 'assistant_gpu': 1, 'doctor_gpu': 0}), flush=True)
                try:
                    with ThreadPoolExecutor(max_workers=5) as pool:
                        futures = [pool.submit(run_case, cfg, policy, byid[order[s['index']]], s['index'], batchdir, state['policy_sha256']) for s in slots]
                        records = [future.result() for future in futures]
                except Exception as exc:
                    atomic_write_json(out / 'diagnostics' / f'pending_batch_{batch:06d}.json',
                        {'batch': batch, 'optimizer_step': False, 'error_type': type(exc).__name__,
                         'failed_cache_operations_by_kind': error_report(batchdir),
                         'actual_http_attempts_by_role': attempt_report(out, set(manifest['case_ids']))})
                    raise
                # Every generation/scoring future has finished; no update before this barrier.
                state = commit_batch(cfg, out, state, records, updater, actual)
                count += 1
                print(json.dumps({'stage': 'batch_committed', 'batch': state['batch_cursor'], 'completed_cases': state['case_cursor'],
                                  'optimizer_steps': updater.steps}), flush=True)
                sync_barrier(out)
        finally:
            if hasattr(policy, '_batcher'): policy._batcher.close()
            atomic_write_json(out / 'api_usage_summary.json', usage_report(out / 'api_usage.jsonl'))


def final_evaluation(cfg, out, state, policy, protocol, outcome_fn):
    from src.online_rl.cases import load_cases
    if state['case_cursor'] != 120 or Path(state['checkpoint']).name != 'checkpoint_000120':
        raise RuntimeError('Only the completed final checkpoint can be evaluated')
    selection = {k: state[k] for k in ('checkpoint', 'policy_sha256', 'config_sha256', 'source_sha256')}
    selection.update(test_sha256=cfg['five_gpu']['eval_sha256'], test_seed=cfg['five_gpu']['eval_seed'])
    selected = out / 'final_evaluation_selection.json'
    if selected.exists() and json.loads(selected.read_text()) != selection: raise RuntimeError('Final evaluation identity changed')
    atomic_write_json(selected, selection)
    queue = EvaluationQueue(out / 'evaluation', load_cases(ROOT / cfg['five_gpu']['eval_file']),
                            {'seed': cfg['five_gpu']['eval_seed'], 'data_sha256': cfg['five_gpu']['eval_sha256'], 'protocol_sha256': protocol})
    if any(p.name != 'checkpoint_000120' for p in queue.root.glob('checkpoint_*')):
        raise RuntimeError('Non-final evaluation job found')
    queue.register(state['checkpoint'], 120, state['policy_sha256'])
    def play(manifest, case, seed):
        from src.common import usage
        from src.online_rl.tree.backends import with_role
        with usage.case(case['case_id']):
            episode = with_role('evaluation', lambda: policy.generate(case, seed))
        return {'dialogue': episode['dialogue'], 'final_option': episode['final_option'],
                'num_discussion_rounds': episode['num_discussion_rounds'], 'outcome': outcome_fn(case, episode['final_option']),
                'private_needs': episode.get('private_needs', [])}
    queue.run_all_parallel(play, workers=cfg['two_gpu']['evaluation_workers'],
        progress=lambda manifest, done, total: print(json.dumps({'stage': 'final_evaluation', 'evaluated': done, 'total': total}), flush=True))
    summary = json.loads((out / 'evaluation/checkpoint_000120/summary.json').read_text())
    if summary['status'] != 'complete' or summary['evaluated'] != 622: raise RuntimeError('Final evaluation incomplete')
    atomic_write_json(out / 'final_accuracy.json', {**summary, 'selection': selection,
        'metric': 'Doctor FINAL accuracy; denominator includes unparsed outcomes'})
