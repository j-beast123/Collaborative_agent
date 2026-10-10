"""One-time, offline migration of the two failed, zero-update first batches.

No GPU/model/API calls. Exact before/after source pins and a structural config
comparison prevent this repair from authorizing unrelated experiment changes.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from baselines.common.runner import source_hash
from src.online_rl.integrated.records import atomic_write_json, sha256_json

OLD_MODEL = 'google/gemini-3.8-flash-20260902'
NEW_MODEL = 'google/gemini-3.8-flash'
OLD_SOURCE = '67a2c4d5d41f79046bc8acb6a88e19c2233bb44b136979f648e1c946269010be'
NEW_SOURCE = 'f9e3c74a0cc10dc7f41d35f2df7e325cbdf9feb63ff2c5bec8ea697abdfaec80'
REPAIR = 'generator_id_20261009'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def convert(cfg):
    value = copy.deepcopy(cfg)
    cm = value['comparison_method']
    require(cm['source_sha256'] == OLD_SOURCE, 'Unexpected previous source')
    require(value['rubric']['generator_model'] == OLD_MODEL, 'Unexpected previous generator')
    generator = cm['engine']['generator']
    require(all(generator[k] == OLD_MODEL for k in ('model', 'revision', 'request_model')),
            'Unexpected previous generator identity')
    require(generator['accepted_response_models'] == [OLD_MODEL], 'Unexpected accepted IDs')
    value['rubric']['generator_model'] = NEW_MODEL
    cm['source_sha256'] = NEW_SOURCE
    generator.update(model=NEW_MODEL, revision=NEW_MODEL, request_model=NEW_MODEL,
                     accepted_response_models=[NEW_MODEL])
    return value


def normalized_hash(cfg):
    value = copy.deepcopy(cfg)
    value['five_gpu']['checkpoint_evaluation_enabled'] = False
    return sha256_json(value)


def plan(cfg):
    out = Path(cfg['output_dir']).resolve()
    old = read(out / 'resolved_launch_config.json')
    require(convert(old) == cfg, 'Changes beyond the approved generator/source fix')
    require(read(out / 'method_binding.json') == old['comparison_method'], 'Old method binding differs')
    state = read(out / 'run_state.json')
    require(state['case_cursor'] == state['batch_cursor'] == 0, 'Training already advanced')
    require(state['policy_sha256'] == state['theta0_policy_sha256'], 'Policy already changed')
    require(state['config_sha256'] == normalized_hash(old), 'Old config hash differs')
    old_fp = sha256_json(old['comparison_method'])
    new_fp = sha256_json(cfg['comparison_method'])
    require(state['experiment_fingerprint'] == old_fp, 'Old state identity differs')
    checkpoint = out / 'checkpoints/checkpoint_000000'
    require(Path(state['checkpoint']).resolve() == checkpoint, 'Unexpected checkpoint')
    require(list((out / 'checkpoints').iterdir()) == [checkpoint], 'Pending or trained checkpoint exists')
    proof = read(checkpoint / 'COMPLETE.json')
    require(proof['completed_cases'] == 0 and proof['experiment_fingerprint'] == old_fp
            and proof['policy_sha256'] == state['policy_sha256']
            and proof['protocol_sha256'] == state['protocol_sha256'], 'Initial checkpoint proof differs')
    require(not list((out / 'cases').glob('*.json')) and not (out / 'evaluation').exists(),
            'Committed training or evaluation artifacts exist')
    attempts = [json.loads(line) for line in (out / 'reward_api_attempts.jsonl').read_text().splitlines()]
    require(bool(attempts) and all(r['status'] == 'failed' and r['requested_model'] == OLD_MODEL
            and r.get('response_model') == NEW_MODEL
            and r.get('error') == 'Returned model does not match the configured accepted model IDs'
            for r in attempts), 'This repair requires only the observed model-ID failure')
    for pattern in ('rewards/cache/*.jsonl', 'rewards/reference_generation.jsonl', 'method_reports/*.json'):
        require(not any(p.stat().st_size for p in out.glob(pattern)), 'Reward/reference work already exists')
    manifest = read(out / 'run_manifest.json')
    require(manifest['config'] == old and manifest['protocol_sha256'] == state['protocol_sha256'],
            'Parent manifest differs')
    changes = {
        'resolved_launch_config.json': cfg,
        'method_binding.json': cfg['comparison_method'],
        'run_manifest.json': {**manifest, 'config': cfg},
        'run_state.json': {**state, 'config_sha256': normalized_hash(cfg), 'experiment_fingerprint': new_fp},
        'checkpoints/checkpoint_000000/COMPLETE.json': {**proof, 'experiment_fingerprint': new_fp},
    }
    batches = list((out / 'batches').iterdir())
    require(batches == [out / 'batches/batch_000001'], 'Unexpected batches')
    batch = batches[0]
    bm = read(batch / 'manifest.json')
    require(bm['source_policy_sha256'] == state['policy_sha256'] and len(bm['case_ids']) == 5,
            'Batch policy/case count differs')
    preserved = [batch / 'manifest.json', out / 'reward_api_attempts.jsonl', out / 'initial_policy.json']
    preserved += [p for p in checkpoint.rglob('*') if p.is_file() and p.name != 'COMPLETE.json']
    for index in range(5):
        case = batch / f'case_{index:06d}'
        man = read(case / 'run_manifest.json')
        s = read(case / 'run_state.json')
        require(s['case_cursor'] == 1 and s['checkpoint'] is None
                and s['config_sha256'] == man['config_sha256'] == sha256_json(man['config'])
                and s['policy_sha256'] == state['policy_sha256'], 'Incomplete/different saved case')
        require(man['config']['comparison_method'] == old['comparison_method'], 'Case binding differs')
        for filename in ('result.json', 'cases/case_000000.json'):
            record = read(case / filename)
            require(record['case_id'] == bm['case_ids'][index]
                    and record['policy_sha256'] == state['policy_sha256']
                    and not record.get('update', {}).get('optimizer_step'), 'Saved tree identity/update differs')
        updated = convert(man['config'])
        relative = str(case.relative_to(out))
        changes[relative + '/run_manifest.json'] = {**man, 'config': updated, 'config_sha256': sha256_json(updated)}
        changes[relative + '/run_state.json'] = {**s, 'config_sha256': sha256_json(updated)}
    # Every other existing artifact remains byte-identical, including call caches,
    # tree records, behavior logprobs, optimizer/adapter tensors and failed costs.
    preserved += [p for p in batch.rglob('*') if p.is_file() and str(p.relative_to(out)) not in changes]
    return changes, {str(p.relative_to(out)): sha(p) for p in preserved}


def apply(cfg):
    require(source_hash() == NEW_SOURCE, 'Source differs from this reviewed repair')
    out = Path(cfg['output_dir']).resolve()
    with ExitStack() as stack:
        devices = cfg['four_gpu']['doctor_physical_gpus'] + cfg['four_gpu']['assistant_physical_gpus']
        for name in [*(f'/tmp/comparison_launcher_gpu{d}.lock' for d in sorted(devices)), str(out / 'runner.lock')]:
            lock = stack.enter_context(open(name, 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_dir = out / 'repairs' / REPAIR
        journal_path = journal_dir / 'journal.json'
        if journal_path.exists():
            journal = read(journal_path)
            require(journal['new_config_sha256'] == sha256_json(cfg), 'Repair target changed')
        else:
            changes, preserved = plan(cfg)
            journal = {'reason': 'User explicitly selected google/gemini-3.8-flash after the dated ID failed response validation',
                       'utc': datetime.now(timezone.utc).isoformat(), 'old_source': OLD_SOURCE, 'new_source': NEW_SOURCE,
                       'old_request_model': OLD_MODEL, 'new_request_model': NEW_MODEL,
                       'new_config_sha256': sha256_json(cfg), 'preserved_sha256': preserved, 'files': {}}
            # Snapshot exact original text and intended replacement before writing
            # any run metadata. A partial application can safely be retried.
            for name, value in changes.items():
                journal['files'][name] = {'before': (out / name).read_text(), 'after': json.dumps(value, ensure_ascii=False, default=str)}
            journal_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(journal_path, journal)
        for name, expected in journal['preserved_sha256'].items():
            require(sha(out / name) == expected, 'Preserved artifact changed: ' + name)
        for name, versions in journal['files'].items():
            require((out / name).read_text() in versions.values(), 'Unexpected concurrent metadata change: ' + name)
        for name, versions in journal['files'].items():
            if (out / name).read_text() != versions['after']:
                # Atomic replacement; journal retains the exact original bytes.
                atomic_write_json(out / name, json.loads(versions['after']))
        # atomic_write_json formatting is normalized in the journal for idempotence.
        for name, versions in journal['files'].items():
            require(read(out / name) == json.loads(versions['after']), 'Repair write failed: ' + name)
        for name, expected in journal['preserved_sha256'].items():
            require(sha(out / name) == expected, 'Preservation verification failed: ' + name)
        result = {'method': cfg['comparison_method']['method'], 'generator': NEW_MODEL,
                  'saved_trees': 5, 'optimizer_steps': 0, 'gpu_or_api_work_started': False,
                  'backup_and_change_record': str(journal_path), 'preserved_files': len(journal['preserved_sha256'])}
        atomic_write_json(journal_dir / 'COMPLETE.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('rar', 'online_rubrics'), required=True)
    parser.add_argument('--runtime', default='baselines/runtime.local.json')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    from baselines.common.rubric_launch import setup
    spec = read(Path(f'baselines/{args.method}/method.json'))
    cfg, _, _, _, errors = setup(args.method, spec, read(Path(args.runtime)))
    require(not errors, '; '.join(errors))
    if args.apply:
        print(json.dumps(apply(cfg), indent=2))
    else:
        changes, preserved = plan(cfg)
        print(json.dumps({'metadata_files': len(changes), 'preserved_files': len(preserved), 'apply': False}))


if __name__ == '__main__':
    main()
