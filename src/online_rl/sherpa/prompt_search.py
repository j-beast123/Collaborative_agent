"""Base-only, train-only offline generator-prompt optimization by mean case AUC."""
from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.records import atomic_write_json, read_json, sha256_json, sha256_text
from src.online_rl.sherpa.pipeline import assess, reflections
from src.online_rl.sherpa.rollouts import collect, schedule
from src.online_rl.tree.base_evaluation import original_base_policy
from src.online_rl.tree.cache import CallCache


def base_identity(cfg, policy):
    from src.online_rl.cases import file_sha256
    identity = {'base_model': cfg['policy']['base_model'], 'doctor': policy.doctor.protocol,
            'rollout': cfg['rollout'], 'seed': cfg['schedule']['seed'],
            'training_sha256': file_sha256(cfg['data']['train']), 'mode': 'original_base_no_adapter'}
    if cfg['prompt_optimization'].get('base_collection_identity') is not None:
        identity['base_collection'] = cfg['prompt_optimization']['base_collection_identity']
    return identity


def optimization_order(cfg, cases, root):
    from src.online_rl.cases import file_sha256
    order = [case for batch in schedule(cfg, cases, root)[:30] for case in batch]
    spec = cfg['prompt_optimization']
    if not spec.get('optimization_case_ids_file'):
        return order
    path = Path(root) / spec['optimization_case_ids_file']
    if file_sha256(path) != spec['optimization_case_ids_sha256']:
        raise RuntimeError('Fixed prompt optimization case selection changed')
    ids = read_json(path)
    by_id = {case['case_id']: case for case in order}
    if not isinstance(ids, list) or len(ids) != spec['mixed_cases'] or len(set(ids)) != len(ids) or not set(ids) <= set(by_id):
        raise ValueError('Fixed optimization case IDs must be a unique subset of the training cases')
    return [by_id[key] for key in ids]


def collect_base(cfg, policy, cases, root, *, max_cases=None):
    home = Path(root) / cfg['prompt_optimization']['output_dir']
    home.mkdir(parents=True, exist_ok=True)
    identity = base_identity(cfg, policy)
    manifest = home / 'base_manifest.json'
    if manifest.exists() and read_json(manifest)['identity'] != identity:
        raise RuntimeError('Offline base data identity changed; choose a fresh output directory')
    order = optimization_order(cfg, cases, root)
    total = len(order)
    openings = {}
    spec = cfg['prompt_optimization']
    if spec.get('fixed_openings_file'):
        from src.online_rl.cases import file_sha256
        path = Path(root) / spec['fixed_openings_file']
        if file_sha256(path) != spec['fixed_openings_sha256']:
            raise RuntimeError('Pinned screening Doctor openings changed')
        openings = read_json(path)
        if set(openings) != {case['case_id'] for case in order}:
            raise ValueError('Each selected optimization case needs its screening Doctor opening')
    if manifest.exists():
        previous = read_json(manifest)
        if previous['case_ids'] != [case['case_id'] for case in order]:
            raise RuntimeError('Base collection case selection changed')
    else:
        previous = {'identity': identity, 'case_ids': [case['case_id'] for case in order], 'complete': False}
        atomic_write_json(manifest, previous)
    completed = 0
    def one(case):
        folder = home / 'base_rollouts' / case['case_id']
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder / 'rollouts.json'
        if cfg['prompt_optimization'].get('cache_enabled', True) and saved.exists():
            row = read_json(saved)
            if row['identity'] != identity or row['case_id'] != case['case_id']:
                raise RuntimeError('Base rollout identity mismatch')
        else:
            kwargs = ({'fixed_opening': openings[case['case_id']],
                       'seed_override': derive_seed(cfg['schedule']['seed'], case['case_id'], 'plan_b_auc32', 0)}
                      if openings else {})
            _, trajectories = collect(case, cfg, policy, CallCache(folder / 'calls.jsonl'), 0, behavior=False, **kwargs)
            outcomes = [row['outcome'] for row in trajectories]
            row = {'identity': identity, 'case_id': case['case_id'], 'trajectories': trajectories,
                   'outcomes': outcomes, 'success_rate': sum(outcomes) / 32,
                   'mixed': set(outcomes) == {0, 1}}
            atomic_write_json(saved, row)
        return case['case_id'], row
    # All replicas have adapters disabled before any worker starts. Up to four
    # cases overlap Doctor I/O with batched generation on the two GPU queues.
    with original_base_policy(policy):
        with ThreadPoolExecutor(cfg['prompt_optimization'].get('rollout_case_workers', 1)) as pool:
            futures = [pool.submit(one, case) for case in (order[:max_cases] if max_cases else order)]
            try:
                for future in as_completed(futures):
                    case_id, row = future.result()
                    completed += 1
                    previous.update(complete=completed == total, completed_cases=completed, total_cases=total)
                    atomic_write_json(manifest, previous)
                    print(f'Base rollout {completed}/{total}: {case_id} success={row["success_rate"]:.3f}', flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    previous.update(complete=completed == total, completed_cases=completed, total_cases=total)
    atomic_write_json(manifest, previous)
    return previous


def optimize(cfg, policy, backend, cases, root):
    home = Path(root) / cfg['prompt_optimization']['output_dir']
    manifest = read_json(home / 'base_manifest.json')
    if not manifest['complete'] or manifest['identity'] != base_identity(cfg, policy):
        raise RuntimeError('Complete the configured original-base rollout collection before optimizing the prompt')
    by_id = {case['case_id']: case for case in cases}
    train_ids = [case['case_id'] for case in optimization_order(cfg, cases, root)]
    if manifest.get('case_ids') is not None and manifest['case_ids'] != train_ids:
        raise RuntimeError('Base manifest differs from the fixed optimization case selection')
    observations = {key: read_json(home / 'base_rollouts' / key / 'rollouts.json') for key in train_ids}
    mixed = [key for key in train_ids if observations[key]['mixed']]
    mixed.sort(key=lambda key: derive_seed(cfg['schedule']['seed'], key, 'prompt_case_selection', 0))
    required = cfg['prompt_optimization'].get('mixed_cases', 24)
    if type(required) is not int or not 1 <= required <= 120:
        raise ValueError('mixed_cases must be an integer from 1 to 120')
    if len(mixed) < required:
        atomic_write_json(home / 'optimization_split.json', {'mixed_cases': mixed, 'required': required, 'complete': False})
        raise RuntimeError(f'Need {required} mixed-outcome training cases; found {len(mixed)}')
    selected = mixed[:required]
    split = {'case_ids': selected, 'all_mixed_case_ids': mixed, 'complete': True,
             'base_identity': manifest['identity'], 'test_used': False}
    split_path = home / 'optimization_split.json'
    if split_path.exists() and read_json(split_path).get('complete') and read_json(split_path) != split:
        raise RuntimeError('Prompt optimization split changed')
    atomic_write_json(split_path, split)
    cache = CallCache(home / 'optimization_calls.jsonl')
    # Exactly one fixed reflection sample per selected base case, shared by every candidate.
    examples = {}
    def reflect_case(key):
        rows = reflections(by_id[key], observations[key]['trajectories'], backend, cache,
                           derive_seed(cfg['schedule']['seed'], key, 'base_reflection_selection', 0))
        atomic_write_json(home / 'reflections' / f'{key}.json', rows)
        return key, rows
    with ThreadPoolExecutor(cfg['prompt_optimization'].get('candidate_workers', 1)) as pool:
        examples.update(pool.map(reflect_case, selected))

    evaluation_locks = {}
    evaluation_guard = threading.Lock()

    def evaluate(text, candidate_id, *, confirmation_repeat=None, selection_repeat=None):
        # Duplicate proposals share one immutable evidence directory. Serialize
        # only identical prompts; distinct candidates are evaluated concurrently.
        key = (sha256_text(text), confirmation_repeat, selection_repeat)
        with evaluation_guard:
            lock = evaluation_locks.setdefault(key, threading.Lock())
        with lock:
            return evaluate_locked(text, candidate_id, confirmation_repeat=confirmation_repeat, selection_repeat=selection_repeat)

    def evaluate_locked(text, candidate_id, *, confirmation_repeat=None, selection_repeat=None):
        digest = sha256_text(text)
        scope = ('offline_prompt' if confirmation_repeat is None
                 else f'offline_prompt_confirmation_{confirmation_repeat}')
        folder = (home / 'candidates' / digest if confirmation_repeat is None else
                  home / 'confirmation' / f'repeat_{confirmation_repeat}' / digest)
        if selection_repeat is not None:
            scope = f'offline_prompt_selection_{selection_repeat}'
            folder = home / 'selection' / f'repeat_{selection_repeat}' / digest
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'prompt.txt').write_text(text)
        identity = {'base': manifest['identity'], 'case_ids': selected, 'prompt': digest,
                    'backend': backend.identity(), 'scope': scope}
        def evaluate_case(key):
            path = folder / 'cases' / f'{key}.json'
            if cfg['prompt_optimization'].get('cache_enabled', True) and path.exists():
                result = read_json(path)
                if result['identity'] != identity:
                    raise RuntimeError('Prompt candidate evidence identity mismatch')
            else:
                result = assess(by_id[key], observations[key]['trajectories'], examples[key], text,
                                backend, cache, {'scope': scope, 'candidate': digest})
                # Invalid sets earn zero fitness; the fixed case denominator never shrinks.
                result.update(case_id=key, identity=identity, fitness_auc=result['auc'] if result['valid'] else 0.0)
                atomic_write_json(path, result)
            if result['fitness_auc'] is None:
                raise RuntimeError('Selected mixed case has no defined AUC')
            return {'case_id': key, 'auc': result['auc'], 'fitness_auc': result['fitness_auc'],
                    'valid': result['valid'], 'rubrics': result['generation']['rubrics']}
        # Each candidate overlaps independent cases; API semaphores remain global.
        # Consume in fixed case order so AUC aggregation/tie breaking is unchanged.
        with ThreadPoolExecutor(cfg['prompt_optimization'].get('case_workers', 4)) as pool:
            futures = [pool.submit(evaluate_case, key) for key in selected]
            try:
                scores = [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        evaluated = {'candidate_id': candidate_id, 'prompt_sha256': digest, 'text': text,
                     'case_scores': scores, 'mean_auc': sum(row['fitness_auc'] for row in scores) / len(selected),
                     'identity': identity}
        atomic_write_json(folder / 'score.json', evaluated)
        print(f'Prompt {candidate_id}: mean AUC={evaluated["mean_auc"]:.4f} scope={scope}', flush=True)
        return evaluated

    spec = cfg['prompt_optimization']
    selection_results = {}
    def stabilize(candidate):
        digest = candidate['prompt_sha256']
        if digest not in selection_results:
            runs = [candidate] + [evaluate(candidate['text'], candidate['candidate_id'], selection_repeat=repeat)
                                  for repeat in range(1, spec.get('selection_repeats', 0) + 1)]
            result = dict(candidate)
            result['mean_auc'] = sum(run['mean_auc'] for run in runs) / len(runs)
            result['selection_mean_aucs'] = [run['mean_auc'] for run in runs]
            result['case_scores'] = [{**row, 'fitness_auc': sum(run['case_scores'][index]['fitness_auc'] for run in runs) / len(runs)}
                                     for index, row in enumerate(candidate['case_scores'])]
            selection_results[digest] = result
        return selection_results[digest]
    initial = (Path(root) / spec['initial_prompt_file']).read_text()
    winner = stabilize(evaluate(initial, 'initial'))
    initial_result = winner
    history = [winner]
    minimum_gain = spec.get('min_auc_improvement', 0.0)
    def select_parent_or_challenger(parent, challengers):
        challenger = max([parent, *challengers], key=lambda row: row['mean_auc'])
        return challenger if challenger['mean_auc'] > parent['mean_auc'] + minimum_gain else parent
    for index, path in enumerate(spec.get('additional_initial_prompt_files', [])):
        candidate = stabilize(evaluate((Path(root) / path).read_text(), f'seed_{index}'))
        history.append(candidate)
        winner = select_parent_or_challenger(winner, [candidate])
    started = time.perf_counter()
    iterations = spec['iterations']
    if iterations not in (2, 3) or spec['candidates_per_iteration'] != 4:
        raise ValueError('Prompt search requires two or three iterations, four candidates each')
    for iteration in range(1, iterations + 1):
        low = sorted(winner['case_scores'], key=lambda row: (row['fitness_auc'], row['case_id']))[:2]
        from src.online_rl.sherpa.prompt_feedback import discrimination_feedback
        evidence = []
        for row in low:
            key = row['case_id']
            assessment = read_json(home / 'candidates' / winner['prompt_sha256'] / 'cases' / f'{key}.json')
            evidence.append({'case': {field: by_id[key].get(field) for field in ('question', 'scenario', 'options')},
                             'auc': row['fitness_auc'], 'rubrics': row['rubrics'], 'examples': examples[key],
                             'scoring_feedback': discrimination_feedback(observations[key]['trajectories'], assessment)})
        def propose_and_evaluate(index):
            proposal = cache.call('prompt_proposal', {'parent': winner['prompt_sha256'], 'iteration': iteration,
                                 'index': index, 'evidence': sha256_json(evidence), 'backend': backend.identity()},
                                  lambda: backend.propose(winner['text'], evidence, iteration, index))
            atomic_write_json(home / 'proposals' / f'iteration_{iteration}_candidate_{index}.json', proposal)
            return evaluate(proposal['text'], f'iteration_{iteration}_candidate_{index}')
        with ThreadPoolExecutor(spec.get('candidate_workers', 1)) as pool:
            candidates = list(pool.map(propose_and_evaluate, range(4)))
        # Deterministic ties keep the parent; no test or checkpoint selection enters fitness.
        finalists = sorted(candidates, key=lambda candidate: candidate['mean_auc'], reverse=True)[:2]
        contenders = [winner, *[stabilize(candidate) for candidate in finalists]] if spec.get('selection_repeats', 0) else [winner, *candidates]
        winner = select_parent_or_challenger(winner, contenders)
        history.extend(candidates)
        atomic_write_json(home / 'history.json', {'candidates': history, 'winner': winner['prompt_sha256'],
                                                 'selection': list(selection_results.values()),
                                                 'completed_iterations': iteration})
        print(f'Prompt search iteration {iteration}: mean AUC={winner["mean_auc"]:.4f}', flush=True)
        if iteration >= 2 and spec.get('max_seconds') is not None and time.perf_counter() - started >= spec['max_seconds']:
            break
    # Fresh generation/judging calls on the same fixed cases and reflections.
    # These observations report repeatability; they never reselect the winner.
    confirmation = {'same_optimization_cases': True, 'held_out': False,
                    'used_for_selection': False, 'repeats': [],
                    'search_initial_mean_auc': initial_result['mean_auc'],
                    'search_final_mean_auc': winner['mean_auc']}
    for repeat in range(1, spec.get('confirmation_repeats', 0) + 1):
        with ThreadPoolExecutor(min(2, spec.get('candidate_workers', 1))) as pool:
            first = pool.submit(evaluate, initial, 'confirmation_initial', confirmation_repeat=repeat)
            last = pool.submit(evaluate, winner['text'], 'confirmation_final', confirmation_repeat=repeat)
            repeated_initial, repeated_final = first.result(), last.result()
        initial_by_case = {row['case_id']: row['fitness_auc'] for row in repeated_initial['case_scores']}
        confirmation['repeats'].append({
            'repeat': repeat, 'initial_prompt_sha256': repeated_initial['prompt_sha256'],
            'final_prompt_sha256': repeated_final['prompt_sha256'],
            'same_prompt': repeated_initial['prompt_sha256'] == repeated_final['prompt_sha256'],
            'initial_mean_auc': repeated_initial['mean_auc'],
            'final_mean_auc': repeated_final['mean_auc'],
            'delta_mean_auc': repeated_final['mean_auc'] - repeated_initial['mean_auc'],
            'case_deltas': [{'case_id': row['case_id'],
                             'delta_auc': row['fitness_auc'] - initial_by_case[row['case_id']]}
                            for row in repeated_final['case_scores']]})
        atomic_write_json(home / 'confirmation.json', confirmation)
    prompt_path = home / 'final_prompt.txt'
    prompt_path.write_text(winner['text'])
    final = {'schema': 'sherpa_auc_prompt_v1', 'prompt_file': str(prompt_path.resolve()),
             'prompt_sha256': winner['prompt_sha256'], 'mean_auc': winner['mean_auc'],
             'case_scores': winner['case_scores'], 'optimization_case_ids': selected,
             'selection_mean_aucs': winner.get('selection_mean_aucs', [winner['mean_auc']]),
             'base_identity': manifest['identity'], 'backend': backend.identity(), 'test_used': False,
             'history_file': str((home / 'history.json').resolve())}
    atomic_write_json(Path(root) / spec['final_artifact'], final)
    return final


def frozen_prompt(cfg, root, *, policy=None, backend=None):
    artifact = read_json(Path(root) / cfg['prompt_optimization']['final_artifact'])
    if artifact['schema'] != 'sherpa_auc_prompt_v1' or artifact['test_used']:
        raise RuntimeError('Training requires a train-only AUC-optimized generator prompt')
    text = Path(artifact['prompt_file']).read_text()
    if sha256_text(text) != artifact['prompt_sha256']:
        raise RuntimeError('Frozen optimized prompt changed')
    if policy is not None and artifact['base_identity'] != base_identity(cfg, policy):
        raise RuntimeError('Optimized prompt belongs to a different base/model/training-data setup')
    revision_file = cfg.get('rubric_pipeline', {}).get('generator_prompt_file')
    if revision_file:
        # Explicit user revision: retain the immutable optimized source and never
        # claim its AUC for the edited text/new judge. No implicit fallback.
        path = (Path(root) / revision_file).resolve()
        revised = path.read_text()
        if not revised.strip():
            raise RuntimeError('User-revised generator prompt is empty')
        return revised, {'schema': 'sherpa_user_prompt_revision_v1',
            'prompt_file': str(path), 'prompt_sha256': sha256_text(revised),
            'prompt_source': 'explicit_user_revision', 'auc_validated': False,
            'source_optimized_artifact': cfg['prompt_optimization']['final_artifact'],
            'source_optimized_prompt_sha256': artifact['prompt_sha256'],
            'base_identity': artifact['base_identity'],
            **({'backend': backend.identity()} if backend is not None else {})}
    if backend is not None and artifact['backend'] != backend.identity():
        raise RuntimeError('Optimized prompt generator/judge contract changed')
    return text, artifact
