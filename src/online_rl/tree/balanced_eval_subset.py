"""Freeze an outcome-stratified diagnostic evaluation view; never training data.

The saved C20 labels define this view's 50% baseline by construction. They do
not estimate population accuracy. Original case objects and sampling seeds are
preserved; no label, outcome, or selection metadata is added to model inputs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from src.online_rl.cases import normalize
from src.online_rl.integrated.records import sha256_json

CONTRACT = 'balanced_checkpoint_eval_v1'
PER_STRATUM = 100


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _manifest_path(path):
    path = Path(path)
    return path / 'manifest.json' if path.is_dir() else path


def _relative(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
        raise RuntimeError('Selection payload path must be a local regular file')
    if not path.is_file() or path.resolve().parent != (root / Path(name).parent).resolve():
        raise RuntimeError('Selection payload is missing or invalid')
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Selection payload escaped its directory')
    return path


def _validate_row(row, manifest):
    # Import lazily: the worker may itself use load_selection(). This helper is
    # CPU-only and never constructs a policy, coordinator, or model client.
    from src.online_rl.tree.idle_eval_worker import _validate_result
    _validate_result(row, manifest)
    outcome = row.get('outcome')
    if outcome is not None and (isinstance(outcome, bool) or outcome not in (0, 1)):
        raise RuntimeError('Baseline outcome must be 0, 1, or None')


def _sample(pool, seed):
    rng = random.Random(seed)
    correct = sorted(r['original_index'] for r in pool if r['baseline_outcome'] == 1)
    wrong = sorted(r['original_index'] for r in pool if r['baseline_outcome'] == 0)
    if len(correct) < PER_STRATUM or len(wrong) < PER_STRATUM:
        raise RuntimeError('Need at least 100 parsed correct and 100 parsed incorrect baseline cases')
    return sorted(rng.sample(correct, PER_STRATUM) + rng.sample(wrong, PER_STRATUM))


def validate_selection(path):
    """Validate only frozen local payloads; return the manifest without writes."""
    path = _manifest_path(path)
    if path.is_symlink():
        raise RuntimeError('Selection manifest cannot be a symlink')
    root = path.parent
    m = _read(path)
    if m.get('contract') != CONTRACT or m.get('per_stratum') != PER_STRATUM:
        raise RuntimeError('Unknown balanced evaluation contract or stratum size')
    if type(m.get('selection_seed')) is not int:
        raise RuntimeError('Selection seed must be an integer')
    if m.get('sha256') != sha256_json({k: v for k, v in m.items() if k != 'sha256'}):
        raise RuntimeError('Selection manifest hash differs')
    for name, digest in m['files'].items():
        if _sha(_relative(root, name).read_bytes()) != digest:
            raise RuntimeError(f'Selection payload hash differs: {name}')
    baseline = _read(_relative(root, 'baseline_manifest.json'))
    if (baseline.get('completed_cases') != 20
            or sha256_json(baseline) != m['baseline']['manifest_sha256']
            or any(m['baseline'][key] != baseline[key] for key in
                   ('job_sha256', 'policy_sha256', 'snapshot_sha256', 'protocol_sha256', 'eval_seed'))
            or baseline['eval_sha256'] != m['original_dataset_sha256']):
        raise RuntimeError('Baseline C20 identity differs')
    if baseline['job_sha256'] != sha256_json({k: v for k, v in baseline.items() if k != 'job_sha256'}):
        raise RuntimeError('Baseline job hash differs')
    pool = m['baseline_pool']
    rows = pool['rows']
    indices = [r['original_index'] for r in rows]
    if (indices != sorted(set(indices)) or len({r['case_id'] for r in rows}) != len(rows)
            or pool['evaluated'] != len(rows) or pool['dataset_cases'] != baseline['dataset_cases']
            or pool['complete'] != (len(rows) == baseline['dataset_cases'])):
        raise RuntimeError('Baseline pool count or identities differ')
    for row in rows:
        i = row['original_index']
        if (type(i) is not int or not 0 <= i < baseline['dataset_cases']
                or row['case_id'] != baseline['case_ids'][i]
                or isinstance(row['baseline_outcome'], bool)
                or row['baseline_outcome'] not in (0, 1, None)):
            raise RuntimeError('Invalid baseline pool row')
    if any(pool[key] != sum(r['baseline_outcome'] == value for r in rows)
           for key, value in [('correct', 1), ('incorrect', 0), ('unparsed', None)]):
        raise RuntimeError('Baseline pool outcome counts differ')
    selected = _sample(rows, m['selection_seed'])
    if m['selected_original_indices'] != selected or m['dataset_cases'] != 2 * PER_STRATUM:
        raise RuntimeError('Selection is not the declared deterministic balanced sample')
    originals = _read(_relative(root, 'data.json'))
    jsonl = [json.loads(line) for line in _relative(root, 'data.jsonl').read_text().splitlines() if line.strip()]
    if originals != jsonl or len(originals) != 2 * PER_STRATUM or len(m['cases']) != len(originals):
        raise RuntimeError('Selected JSON/JSONL data differ')
    by_index = {r['original_index']: r for r in rows}
    outcomes = []
    for item, original, i in zip(m['cases'], originals, selected):
        row_path = _relative(root, item['result_path'])
        data = row_path.read_bytes()
        result = json.loads(data)
        _validate_row(result, baseline)
        pool_row = by_index[i]
        if (item['original_index'] != i or item['case_id'] != baseline['case_ids'][i]
                or result['index'] != i or original != result['original_case']
                or sha256_json(original) != item['original_case_sha256']
                or item['original_case_sha256'] != baseline['original_case_sha256'][i]
                or item['seed'] != result['seed'] or item['seed'] != baseline['eval_seed'] + i
                or item['baseline_outcome'] != result['outcome']
                or item['baseline_outcome'] != pool_row['baseline_outcome']
                or item['result_file_sha256'] != _sha(data)
                or item['result_file_sha256'] != pool_row['result_file_sha256']):
            raise RuntimeError('Selected case, seed, label, or result hash differs')
        outcomes.append(result['outcome'])
    if outcomes.count(1) != PER_STRATUM or outcomes.count(0) != PER_STRATUM:
        raise RuntimeError('Selected outcomes are not exactly 100 correct and 100 incorrect')
    return m


def build_selection(baseline_eval_dir, output_dir, full_dataset_path, seed=0):
    """Copy a stopped evaluation's completed rows into an immutable 200-case view.

    Caller must stop the baseline writer first. The builder additionally checks
    that the complete source file inventory and bytes remain stable while read.
    Existing selections are validated and reused; they are never resampled.
    """
    if type(seed) is not int:
        raise ValueError('Selection seed must be an integer')
    src, dest, dataset = Path(baseline_eval_dir).resolve(), Path(output_dir).resolve(), Path(full_dataset_path).resolve()
    if dest == src or dest.is_relative_to(src) or src.is_relative_to(dest):
        raise ValueError('Selection output must be separate from the baseline evaluation')
    dataset_bytes = dataset.read_bytes()
    dataset_sha = _sha(dataset_bytes)
    baseline_bytes = (src / 'manifest.json').read_bytes()
    baseline = json.loads(baseline_bytes)
    if baseline.get('completed_cases') != 20 or baseline.get('eval_sha256') != dataset_sha:
        raise RuntimeError('Baseline is not C20 or original dataset hash differs')
    if dest.exists():
        existing = validate_selection(dest)
        if (existing['selection_seed'] != seed or existing['original_dataset_sha256'] != dataset_sha
                or existing['baseline']['manifest_sha256'] != sha256_json(baseline)):
            raise RuntimeError('Existing selection belongs to another baseline or selection seed')
        return existing
    originals = [json.loads(line) for line in dataset_bytes.decode().splitlines() if line.strip()]
    if (len(originals) != baseline['dataset_cases']
            or len({c['case_id'] for c in originals}) != len(originals)
            or [c['case_id'] for c in originals] != baseline['case_ids']
            or [sha256_json(c) for c in originals] != baseline['original_case_sha256']):
        raise RuntimeError('Original dataset does not match the baseline manifest')
    paths = sorted((src / 'cases').glob('case_*.json'))
    frozen = {}
    for path in paths:
        data = path.read_bytes()
        row = json.loads(data)
        _validate_row(row, baseline)
        i = row['index']
        if i in frozen:
            raise RuntimeError('Duplicate baseline result index')
        frozen[i] = (path, data, row)
    if (paths != sorted((src / 'cases').glob('case_*.json'))
            or baseline_bytes != (src / 'manifest.json').read_bytes()
            or any(data != path.read_bytes() for path, data, _ in frozen.values())):
        raise RuntimeError('Baseline changed while freezing; stop its writers first')
    pool_rows = [{'original_index': i, 'case_id': row['case_id'], 'baseline_outcome': row['outcome'],
                  'result_file_sha256': _sha(data)} for i, (_, data, row) in sorted(frozen.items())]
    selected = _sample(pool_rows, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{dest.name}.pending-', dir=dest.parent))
    try:
        (temporary / 'baseline_results').mkdir()
        (temporary / 'baseline_manifest.json').write_bytes(baseline_bytes)
        selected_original = [originals[i] for i in selected]
        (temporary / 'data.json').write_text(json.dumps(selected_original, ensure_ascii=False, indent=2) + '\n')
        (temporary / 'data.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in selected_original))
        entries = []
        for i in selected:
            path, data, row = frozen[i]
            relative = f'baseline_results/case_{i:06d}.json'
            (temporary / relative).write_bytes(data)
            entries.append({'case_id': row['case_id'], 'original_index': i,
                            'original_case_sha256': sha256_json(originals[i]), 'seed': row['seed'],
                            'baseline_outcome': row['outcome'], 'result_file_sha256': _sha(data),
                            'result_path': relative, 'source_result_path': str(path)})
        m = {'contract': CONTRACT, 'purpose': 'fixed_balanced_diagnostic_evaluation_only',
             'selection_seed': seed, 'per_stratum': PER_STRATUM, 'dataset_cases': len(selected),
             'selection_method': 'random_sample_correct_then_incorrect_sorted_original_indices',
             'original_dataset_path': str(dataset), 'original_dataset_sha256': dataset_sha,
             'baseline_eval_dir': str(src), 'baseline': {k: baseline[k] for k in
                ('job_sha256', 'policy_sha256', 'snapshot_sha256', 'protocol_sha256', 'eval_seed')},
             'baseline_pool': {'evaluated': len(pool_rows), 'dataset_cases': len(originals),
                'complete': len(pool_rows) == len(originals),
                'correct': sum(r['baseline_outcome'] == 1 for r in pool_rows),
                'incorrect': sum(r['baseline_outcome'] == 0 for r in pool_rows),
                'unparsed': sum(r['baseline_outcome'] is None for r in pool_rows), 'rows': pool_rows},
             'selected_original_indices': selected, 'cases': entries,
             'interpretation': 'Recorded C20 accuracy is 50% by selection, not population accuracy; rerunning C20 need not reproduce every stochastic answer.',
             'files': {f: _sha((temporary / f).read_bytes()) for f in
                       ('baseline_manifest.json', 'data.json', 'data.jsonl')}}
        m['baseline']['manifest_sha256'] = sha256_json(baseline)
        m['sha256'] = sha256_json(m)
        (temporary / 'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=2) + '\n')
        validate_selection(temporary)
        os.rename(temporary, dest)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return m


def load_selection(path, full_cases, full_original, fg):
    """Return (selected normalized cases, originals, metadata) without mutation.

    ``fg`` describes the original complete evaluation dataset. The worker must
    use metadata['seeds'][subset_index], not eval_seed + subset_index.
    """
    m = validate_selection(path)
    baseline = _read(_manifest_path(path).parent / 'baseline_manifest.json')
    if (fg.get('eval_sha256') != m['original_dataset_sha256']
            or fg.get('eval_cases') != baseline['dataset_cases']
            or fg.get('eval_seed') != baseline['eval_seed']
            or len(full_cases) != baseline['dataset_cases'] or len(full_original) != len(full_cases)
            or [c['case_id'] for c in full_original] != baseline['case_ids']
            or [sha256_json(c) for c in full_original] != baseline['original_case_sha256']
            or any(normalize(original) != case for original, case in zip(full_original, full_cases))):
        raise RuntimeError('Full evaluation dataset/config differs from the frozen selection')
    indices = m['selected_original_indices']
    metadata = {'contract': CONTRACT, 'sha256': m['sha256'], 'manifest_path': str(_manifest_path(path).resolve()),
                'original_dataset_sha256': m['original_dataset_sha256'], 'original_indices': indices,
                'seeds': [c['seed'] for c in m['cases']],
                'baseline_outcomes': [c['baseline_outcome'] for c in m['cases']],
                'baseline': m['baseline'], 'baseline_pool': {k: v for k, v in m['baseline_pool'].items() if k != 'rows'}}
    return (copy.deepcopy([full_cases[i] for i in indices]),
            copy.deepcopy([full_original[i] for i in indices]), copy.deepcopy(metadata))
