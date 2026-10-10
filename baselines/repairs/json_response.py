"""Offline upgrade of the exact pre-fix OpenAI profile to JSON wrapper handling.

Preserves checkpoint/optimizer/cache bytes. Run only after the launcher exits.
"""
import argparse
import copy
import fcntl
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from baselines.common.protocol import ROOT
from baselines.common.runner import source_hash
from baselines.parallel_api.launch import BASE_SOURCE
from baselines.parallel_api.openai_profile import apply_profile_config
from baselines.repairs.generator_id import require, read, sha, normalized_hash
from src.online_rl.integrated.records import atomic_write_json, sha256_json

NAME = 'json_response_v1'
OLD_ENGINE = '4b7503f60bea71e0eef2e01fac740a1e56cc4d43db662911545036687a7f243b'
NEW_ENGINE = '21ade316a84a103a3dd26fbd6cab152043a29f797afaf371c0df7bc86098b361'
PARSER = '70dcaf2c8dbd10ab91a2521cc8e738dfaa435576d0c0ae089d9007db88174fef'


def protocol_digest(cfg):
    from src.online_rl.integrated.continuation import protocol_sha256
    fg = cfg['five_gpu']
    return sha256_json({'base': protocol_sha256(cfg, ROOT), 'D1': fg['opening_identity'],
        'mode': fg['opening_mode'], 'na': 'exclude_unassessed_v1',
        'execution_mode': 'shared_ready_tasks_v1', 'batch_cases': 5})


def plan(cfg):
    out = Path(cfg['output_dir']).resolve()
    old = read(out/'resolved_launch_config.json')
    require(old['comparison_method']['source_sha256'] == BASE_SOURCE, 'Previous source differs')
    previous = old['comparison_method']['api_profile']['execution_source_sha256']
    expected = copy.deepcopy(previous)
    require(previous.get('engine.py') == OLD_ENGINE and 'json_response.py' not in previous, 'Unexpected previous response parser')
    expected.update({'engine.py': NEW_ENGINE, 'json_response.py': PARSER})
    require(cfg['comparison_method']['api_profile']['execution_source_sha256'] == expected, 'Changes beyond the reviewed JSON response fix')
    require(old['comparison_method']['engine']['grader']['request_model'] == 'gpt-5.6-luna', 'Previous grader differs')
    require(apply_profile_config(old) == cfg, 'Unrelated config change: only the reviewed response parser source may change')
    require(read(out/'method_binding.json') == old['comparison_method'], 'Old binding differs')
    state = read(out/'run_state.json'); cursor = state['case_cursor']; batch = state['batch_cursor']
    require(cursor == 5*batch and 0 <= cursor < cfg['schedule']['cases'], 'Expected unfinished, five-case-aligned training')
    oldfp = sha256_json(old['comparison_method']); newfp = sha256_json(cfg['comparison_method'])
    require(state['config_sha256'] == normalized_hash(old) and state['experiment_fingerprint'] == oldfp, 'Old state identity differs')
    require(protocol_digest(old) == protocol_digest(cfg) == state['protocol_sha256'], 'Dialogue/training protocol changed')
    checkpoint = Path(state['checkpoint']).resolve()
    require(checkpoint.parent == out/'checkpoints', 'Checkpoint outside this run')
    for path in (out/'checkpoints').iterdir():
        require(path.is_dir() and not path.is_symlink() and not path.name.endswith('.pending'),
                'Incomplete checkpoint: recover with the original launcher before transition')
        proof = read(path/'COMPLETE.json')
        require(proof['completed_cases'] <= cursor, 'Uncommitted checkpoint: recover with the original launcher before transition')
    proof = read(checkpoint/'COMPLETE.json')
    require(proof['completed_cases'] == cursor and proof['policy_sha256'] == state['policy_sha256']
        and proof['experiment_fingerprint'] == oldfp and proof['protocol_sha256'] == state['protocol_sha256'], 'Checkpoint proof differs')
    require(sorted(p.name for p in (out/'cases').glob('case_*.json')) == [f'case_{i:06d}.json' for i in range(cursor)],
            'Committed case records differ; recover with the original launcher first')
    manifest = read(out/'run_manifest.json')
    require(manifest['config'] == old and manifest['protocol_sha256'] == state['protocol_sha256'], 'Run manifest differs')
    require(not (out/'evaluation').exists(), 'Test evaluation already started')
    values = {
        out/'resolved_launch_config.json': cfg, out/'method_binding.json': cfg['comparison_method'],
        out/'run_manifest.json': {**manifest, 'config': cfg},
        out/'run_state.json': {**state, 'config_sha256': normalized_hash(cfg), 'experiment_fingerprint': newfp},
        checkpoint/'COMPLETE.json': {**proof, 'experiment_fingerprint': newfp}}
    order = read(Path(cfg['five_gpu']['train_order_file']))
    for directory in sorted((out/'batches').glob('batch_*')):
        number = int(directory.name.removeprefix('batch_'))
        if number <= batch:
            continue  # Completed training evidence is never relabelled/re-scored.
        require(number == batch+1, 'Unexpected future batch')
        bm = read(directory/'manifest.json')
        require(bm['source_policy_sha256'] == state['policy_sha256'] and bm['case_ids'] == order[cursor:cursor+5], 'Pending batch policy/order differs')
        for case in sorted(directory.glob('case_*')):
            index = int(case.name.removeprefix('case_'))
            require(cursor <= index < cursor+5, 'Unexpected pending case')
            mp = case/'run_manifest.json'; sp = case/'run_state.json'
            if not mp.exists() and not sp.exists():
                require(not (case/'result.json').exists(), 'Tree without manifest/state')
                continue  # Only a partial generation call cache exists.
            require(mp.exists() and sp.exists(), 'Incomplete child metadata; resume original launcher first')
            man = read(mp); cs = read(sp)
            require(man['config']['comparison_method'] == old['comparison_method'], 'Child binding differs')
            require(cs['config_sha256'] == man['config_sha256'] == sha256_json(man['config'])
                and cs['case_cursor'] in (0, 1) and cs['checkpoint'] is None
                and cs['policy_sha256'] == state['policy_sha256'], 'Child policy/config differs')
            for recordpath in (case/'result.json', case/'cases/case_000000.json'):
                if recordpath.exists():
                    record = read(recordpath)
                    require(record['case_id'] == order[index] and record['policy_sha256'] == state['policy_sha256']
                        and not record.get('update', {}).get('optimizer_step'), 'Pending tree policy/order/update differs')
            updated = apply_profile_config(man['config'])
            values[mp] = {**man, 'config': updated, 'config_sha256': sha256_json(updated)}
            values[sp] = {**cs, 'config_sha256': sha256_json(updated)}
    edits = {str(p.relative_to(out)): json.dumps(v, ensure_ascii=False, default=str) for p, v in values.items()}
    # No cache aliases: the new provider/parameters make new grade keys.
    preserved = {}
    for p in sorted(out.rglob('*')):
        require(not p.is_symlink(), 'Unexpected symlink in run artifacts: '+str(p))
        if p.is_file() and p not in values and p.name != 'runner.lock' and 'repairs' not in p.relative_to(out).parts:
            preserved[str(p.relative_to(out))] = sha(p)
    return edits, preserved, state


def write_text_atomic(path, value):
    temporary = path.with_name(path.name+'.provider-transition-tmp')
    temporary.write_text(value)
    temporary.replace(path)


def apply(cfg):
    require(source_hash() == BASE_SOURCE, 'Base source changed')
    out = Path(cfg['output_dir']).resolve()
    directory = out/'repairs'/NAME; journalpath = directory/'journal.json'
    with ExitStack() as stack:
        devices = cfg['four_gpu']['doctor_physical_gpus'] + cfg['four_gpu']['assistant_physical_gpus']
        paths = [*(Path(f'/tmp/comparison_launcher_gpu{d}.lock') for d in sorted(devices)), out/'runner.lock']
        for path in paths:
            lock = stack.enter_context(path.open('a'))
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Training is still running/holds a lock; stop it normally first: '+str(path)) from None
        target = sha256_json(cfg)
        if journalpath.exists():
            journal = read(journalpath)
            require(journal['target_sha256'] == target, 'Transition target/source changed')
            if (directory/'COMPLETE.json').exists():
                require(read(out/'method_binding.json') == cfg['comparison_method'], 'Completed transition binding differs')
                return {**read(directory/'COMPLETE.json'), 'already_applied': True}
        else:
            edits, preserved, state = plan(cfg)
            journal = {'name': NAME, 'utc': datetime.now(timezone.utc).isoformat(), 'target_sha256': target,
                'committed_cases': state['case_cursor'], 'committed_batches': state['batch_cursor'],
                'policy_sha256': state['policy_sha256'], 'new_profile': cfg['comparison_method']['api_profile'],
                'scope': 'Exact outer JSON/heredoc wrapper removal and at most 3 same-request format attempts; unchanged model, prompts, params, schema and valid cache entries',
                'preserved_sha256': preserved,
                'files': {name: {'before': (out/name).read_text(), 'after': text} for name, text in edits.items()}}
            atomic_write_json(journalpath, journal)
        for name, digest in journal['preserved_sha256'].items():
            require(sha(out/name) == digest, 'Preserved artifact changed: '+name)
        for name, versions in journal['files'].items():
            require((out/name).read_text() in versions.values(), 'Metadata changed outside the transition: '+name)
        for name, versions in journal['files'].items():
            if (out/name).read_text() != versions['after']:
                write_text_atomic(out/name, versions['after'])
        for name, digest in journal['preserved_sha256'].items():
            require(sha(out/name) == digest, 'Artifact preservation failed: '+name)
        result = {'method': cfg['comparison_method']['method'], 'committed_cases': journal['committed_cases'],
            'repair': NAME, 'provider': 'openai', 'model': 'gpt-5.6-luna',
            'preserved_files': len(journal['preserved_sha256']), 'backup_and_change_record': str(journalpath),
            'gpu_or_api_work_started': False}
        atomic_write_json(directory/'COMPLETE.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=('rar', 'online_rubrics'))
    parser.add_argument('--runtime', default='baselines/runtime.local.json')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    from baselines.parallel_api.launch import install
    from baselines.parallel_api.openai_profile import install_profile
    launcher = install(20, 100); install_profile(launcher)
    cfg, _, _, _, errors = launcher.setup(args.method,
        read(ROOT/f'baselines/parallel_api/{args.method}_openai.json'), read(Path(args.runtime)))
    require(not errors, '; '.join(errors))
    out = Path(cfg['output_dir'])
    if read(out/'method_binding.json') == cfg['comparison_method'] and read(out/'resolved_launch_config.json') == cfg:
        result = {'repair': NAME, 'already_current': True, 'files_modified': 0}
    elif args.apply:
        result = apply(cfg)
    else:
        edits, preserved, state = plan(cfg)
        result = {'apply': False, 'metadata_files': len(edits), 'preserved_files': len(preserved),
                  'committed_cases': state['case_cursor'], 'note': 'Read-only snapshot; --apply replans under exclusive locks'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
