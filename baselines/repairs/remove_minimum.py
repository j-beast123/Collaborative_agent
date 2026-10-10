"""Offline, exact transition of the two zero-update runs to no minimum rubric size.

Keeps old cache entries as evidence and adds explicitly labelled reusable entries
for the relaxed prompt. No GPU, provider requests, or weight changes.
"""
import copy
import fcntl
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from baselines.common.protocol import ROOT
from baselines.common.runner import source_hash
from baselines.common.rubrics import INITIAL, EXTRACT, DEDUP, GRADE, criteria
from baselines.repairs.generator_id import require, read, sha, normalized_hash
from src.online_rl.integrated.records import sha256_json, atomic_write_json

OLD_SOURCE = 'f9e3c74a0cc10dc7f41d35f2df7e325cbdf9feb63ff2c5bec8ea697abdfaec80'
NEW_SOURCE = '4787fc56ebb99afeffcf62cb6b890c5857af3cefcc67db3948d93e126643a745'
OLD_PROMPTS = '477a594e7e677c467b341d3921ff559ffeee92bfb4c60e42b6db18c3734c22aa'
NEW_PROMPTS = 'f843e24ca9c2c4bffcd9788f76818e5f0246fb665e48ac2cb77bc10a016b9fee'
OLD_INITIAL = INITIAL.replace('Create up to 20 atomic', 'Create 7 to 20 atomic', 1)
REPAIR = 'remove_rubric_minimum_20261009'


def convert(cfg):
    value = copy.deepcopy(cfg); cm = value['comparison_method']
    require(cm['source_sha256'] == OLD_SOURCE, 'Unexpected previous source')
    require(cm['engine']['prompts_sha256'] == OLD_PROMPTS, 'Unexpected previous prompts')
    cm['source_sha256'] = NEW_SOURCE
    cm['engine']['prompts_sha256'] = NEW_PROMPTS
    return value


def alias(row):
    require(row['key'] == sha256_json(['initial_rubric', row['identity']]), 'Corrupt initial cache key')
    require(row['identity']['system'] == OLD_INITIAL, 'Unexpected cached prompt')
    criteria(row['value'])
    require(len(row['value']['criteria']) >= 7, 'Old successful cache violates its old contract')
    result = copy.deepcopy(row)
    result['identity']['system'] = INITIAL
    result['key'] = sha256_json(['initial_rubric', result['identity']])
    result['reuse_provenance'] = {'repair': REPAIR, 'original_key': row['key'],
        'original_prompt_sha256': sha256_json(OLD_INITIAL),
        'reason': 'Existing criteria satisfy relaxed count contract; reused without a new generation'}
    return result


def plan_run(cfg):
    out = Path(cfg['output_dir']).resolve(); old = read(out/'resolved_launch_config.json')
    require(convert(old) == cfg, 'Unrelated configuration change')
    require(read(out/'method_binding.json') == old['comparison_method'], 'Old binding differs')
    state = read(out/'run_state.json'); oldfp = sha256_json(old['comparison_method']); newfp = sha256_json(cfg['comparison_method'])
    require(state['case_cursor'] == state['batch_cursor'] == 0, 'Training already advanced')
    require(state['policy_sha256'] == state['theta0_policy_sha256'], 'Policy changed')
    require(state['config_sha256'] == normalized_hash(old) and state['experiment_fingerprint'] == oldfp, 'Old state hash differs')
    ckpt = out/'checkpoints/checkpoint_000000'
    require(Path(state['checkpoint']).resolve() == ckpt and list((out/'checkpoints').iterdir()) == [ckpt], 'Trained/pending checkpoint exists')
    proof = read(ckpt/'COMPLETE.json')
    require(proof['completed_cases'] == 0 and proof['policy_sha256'] == state['policy_sha256']
        and proof['experiment_fingerprint'] == oldfp and proof['protocol_sha256'] == state['protocol_sha256'], 'Initial proof differs')
    require(not list((out/'cases').glob('*.json')) and not (out/'evaluation').exists(), 'Committed train/test exists')
    manifest = read(out/'run_manifest.json'); require(manifest['config'] == old, 'Manifest differs')
    values = {out/'resolved_launch_config.json': cfg, out/'method_binding.json': cfg['comparison_method'],
        out/'run_manifest.json': {**manifest, 'config': cfg},
        out/'run_state.json': {**state, 'config_sha256': normalized_hash(cfg), 'experiment_fingerprint': newfp},
        ckpt/'COMPLETE.json': {**proof, 'experiment_fingerprint': newfp}}
    batch = out/'batches/batch_000001'
    require(list((out/'batches').iterdir()) == [batch], 'Unexpected batches')
    bm = read(batch/'manifest.json')
    require(bm['source_policy_sha256'] == state['policy_sha256'] and len(bm['case_ids']) == 5, 'Batch differs')
    for index in range(5):
        case = batch/f'case_{index:06d}'; man = read(case/'run_manifest.json'); cs = read(case/'run_state.json')
        require(cs['case_cursor'] == 1 and cs['checkpoint'] is None and cs['policy_sha256'] == state['policy_sha256']
            and cs['config_sha256'] == man['config_sha256'] == sha256_json(man['config']), 'Saved case differs')
        require(man['config']['comparison_method'] == old['comparison_method'], 'Case binding differs')
        for f in ('result.json', 'cases/case_000000.json'):
            row = read(case/f)
            require(row['policy_sha256'] == state['policy_sha256'] and row['case_id'] == bm['case_ids'][index]
                and not row.get('update', {}).get('optimizer_step'), 'Tree updated/different')
        updated = convert(man['config'])
        values[case/'run_manifest.json'] = {**man, 'config': updated, 'config_sha256': sha256_json(updated)}
        values[case/'run_state.json'] = {**cs, 'config_sha256': sha256_json(updated)}
    edits = {p: json.dumps(v, ensure_ascii=False, default=str) for p, v in values.items()}
    cache = out/'rewards/cache/initial_rubrics.jsonl'
    if cache.exists():
        original = cache.read_text(); rows = [json.loads(s) for s in original.splitlines() if s.strip()]
        added = [alias(r) for r in rows]
        require(len({r['key'] for r in rows + added}) == len(rows + added), 'Duplicate/new cache entry already exists')
        edits[cache] = original + ('' if not original or original.endswith('\n') else '\n') + ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in added)
    # Old method reports are historical evidence. A resumed score overwrites them
    # with the current engine after loading the same cached rubric/grades.
    preserved = {str(p): sha(p) for sub in ('batches', 'checkpoints', 'rewards', 'method_reports', 'openings')
        for p in (out/sub).rglob('*') if p.is_file() and p not in edits}
    for name in ('reward_api_attempts.jsonl', 'initial_policy.json', 'costs.json'):
        if (out/name).exists(): preserved[str(out/name)] = sha(out/name)
    return edits, preserved


def apply(configs):
    require(source_hash() == NEW_SOURCE, 'Source differs from reviewed repair')
    require(sha256_json([INITIAL, EXTRACT, DEDUP, GRADE]) == NEW_PROMPTS, 'New prompt differs')
    directory = ROOT/'outputs/repairs'/REPAIR; journalpath = directory/'journal.json'
    with ExitStack() as stack:
        devices = sorted({d for cfg in configs for role in ('doctor_physical_gpus', 'assistant_physical_gpus') for d in cfg['four_gpu'][role]})
        locks = [*(Path(f'/tmp/comparison_launcher_gpu{d}.lock') for d in devices), *(Path(c['output_dir'])/'runner.lock' for c in configs)]
        for path in locks:
            lock = stack.enter_context(path.open('a')); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = sha256_json(configs)
        if journalpath.exists():
            journal = read(journalpath); require(journal['target_sha256'] == target, 'Repair target changed')
        else:
            edits = {}; preserved = {}
            for cfg in configs:
                changes, keep = plan_run(cfg); edits.update(changes); preserved.update(keep)
            registries = {Path(c['comparison_method']['method_spec']['initial_rubric_registry']).resolve() for c in configs}
            for root in registries:
                for path in root.glob('*.json'):
                    row = read(path)
                    require(row['value_sha256'] == sha256_json(row['value']), 'Registry value differs')
                    changed = alias(row); newpath = root/(changed['key']+'.json')
                    require(not newpath.exists(), 'New registry key already exists')
                    edits[newpath] = json.dumps(changed, ensure_ascii=False)
                    preserved[str(path)] = sha(path)
            journal = {'reason': 'User removed the minimum rubric count; empty initial rubric still has no reward denominator',
                'utc': datetime.now(timezone.utc).isoformat(), 'old_source': OLD_SOURCE, 'new_source': NEW_SOURCE,
                'target_sha256': target, 'preserved_sha256': preserved,
                'files': {str(p): {'before': p.read_text() if p.exists() else None, 'after': text} for p, text in edits.items()}}
            atomic_write_json(journalpath, journal)
        for name, digest in journal['preserved_sha256'].items(): require(sha(Path(name)) == digest, 'Preserved artifact changed: '+name)
        for name, versions in journal['files'].items():
            p = Path(name); actual = p.read_text() if p.exists() else None
            require(actual in versions.values(), 'Unexpected metadata/cache change: '+name)
        for name, versions in journal['files'].items():
            p = Path(name)
            if not p.exists() or p.read_text() != versions['after']:
                p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name+'.repair-tmp')
                tmp.write_text(versions['after']); tmp.replace(p)
        for name, digest in journal['preserved_sha256'].items(): require(sha(Path(name)) == digest, 'Preservation failed: '+name)
        result = {'repair': REPAIR, 'methods': [c['comparison_method']['method'] for c in configs],
            'preserved_files': len(journal['preserved_sha256']), 'changed_or_added_files': len(journal['files']),
            'gpu_or_api_work_started': False, 'journal': str(journalpath)}
        atomic_write_json(directory/'COMPLETE.json', result); return result


if __name__ == '__main__':
    from baselines.common.rubric_launch import setup
    configs = []
    for method in ('rar', 'online_rubrics'):
        cfg, _, _, _, errors = setup(method, read(ROOT/f'baselines/{method}/method.json'), read(ROOT/'baselines/runtime.local.json'))
        require(not errors, '; '.join(errors)); configs.append(cfg)
    print(json.dumps(apply(configs), indent=2))
