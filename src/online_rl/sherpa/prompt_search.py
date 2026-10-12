"""Base-only, train-only offline generator-prompt optimization by mean case AUC."""
from __future__ import annotations

import time
from pathlib import Path

from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.records import atomic_write_json, read_json, sha256_json, sha256_text
from src.online_rl.sherpa.pipeline import assess, reflections
from src.online_rl.sherpa.rollouts import collect, schedule
from src.online_rl.tree.base_evaluation import original_base_policy
from src.online_rl.tree.cache import CallCache


def base_identity(cfg, policy):
    from src.online_rl.cases import file_sha256
    return {'base_model': cfg['policy']['base_model'], 'doctor': policy.doctor.protocol,
            'rollout': cfg['rollout'], 'seed': cfg['schedule']['seed'],
            'training_sha256': file_sha256(cfg['data']['train']), 'mode': 'original_base_no_adapter'}


def collect_base(cfg, policy, cases, root, *, max_cases=None):
    home = Path(root) / cfg['prompt_optimization']['output_dir']
    home.mkdir(parents=True, exist_ok=True)
    identity = base_identity(cfg, policy)
    manifest = home / 'base_manifest.json'
    if manifest.exists() and read_json(manifest)['identity'] != identity:
        raise RuntimeError('Offline base data identity changed; choose a fresh output directory')
    order = [case for batch in schedule(cfg, cases, root)[:30] for case in batch]
    if manifest.exists():
        previous = read_json(manifest)
    else:
        previous = {'identity': identity, 'case_ids': [case['case_id'] for case in order], 'complete': False}
        atomic_write_json(manifest, previous)
    completed = 0
    # Offline observations come from the actual frozen base, not a trained adapter.
    with original_base_policy(policy):
        for case in order:
            folder = home / 'base_rollouts' / case['case_id']
            folder.mkdir(parents=True, exist_ok=True)
            saved = folder / 'rollouts.json'
            if saved.exists():
                row = read_json(saved)
                if row['identity'] != identity or row['case_id'] != case['case_id']:
                    raise RuntimeError('Base rollout identity mismatch')
            else:
                _, trajectories = collect(case, cfg, policy, CallCache(folder / 'calls.jsonl'), 0, behavior=False)
                outcomes = [row['outcome'] for row in trajectories]
                row = {'identity': identity, 'case_id': case['case_id'], 'trajectories': trajectories,
                       'outcomes': outcomes, 'success_rate': sum(outcomes) / 32,
                       'mixed': set(outcomes) == {0, 1}}
                atomic_write_json(saved, row)
            completed += 1
            print(f'Base rollout {completed}/120: {case["case_id"]} success={row["success_rate"]:.3f}', flush=True)
            if max_cases is not None and completed >= max_cases:
                break
    previous.update(complete=completed == 120, completed_cases=completed)
    atomic_write_json(manifest, previous)
    return previous


def optimize(cfg, policy, backend, cases, root):
    home = Path(root) / cfg['prompt_optimization']['output_dir']
    manifest = read_json(home / 'base_manifest.json')
    if not manifest['complete'] or manifest['identity'] != base_identity(cfg, policy):
        raise RuntimeError('Collect all 120 original-base rollouts before optimizing the prompt')
    by_id = {case['case_id']: case for case in cases}
    train_ids = [case['case_id'] for batch in schedule(cfg, cases, root)[:30] for case in batch]
    observations = {key: read_json(home / 'base_rollouts' / key / 'rollouts.json') for key in train_ids}
    mixed = [key for key in train_ids if observations[key]['mixed']]
    mixed.sort(key=lambda key: derive_seed(cfg['schedule']['seed'], key, 'prompt_case_selection', 0))
    if len(mixed) < 24:
        atomic_write_json(home / 'optimization_split.json', {'mixed_cases': mixed, 'required': 24, 'complete': False})
        raise RuntimeError(f'Need 24 mixed-outcome training cases; found {len(mixed)}')
    selected = mixed[:24]
    split = {'case_ids': selected, 'all_mixed_case_ids': mixed, 'complete': True,
             'base_identity': manifest['identity'], 'test_used': False}
    split_path = home / 'optimization_split.json'
    if split_path.exists() and read_json(split_path).get('complete') and read_json(split_path) != split:
        raise RuntimeError('Prompt optimization split changed')
    atomic_write_json(split_path, split)
    cache = CallCache(home / 'optimization_calls.jsonl')
    # Exactly one fixed reflection sample per selected base case, shared by every candidate.
    examples = {}
    for key in selected:
        examples[key] = reflections(by_id[key], observations[key]['trajectories'], backend, cache,
                                   derive_seed(cfg['schedule']['seed'], key, 'base_reflection_selection', 0))
        atomic_write_json(home / 'reflections' / f'{key}.json', examples[key])

    def evaluate(text, candidate_id):
        digest = sha256_text(text)
        folder = home / 'candidates' / digest
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'prompt.txt').write_text(text)
        identity = {'base': manifest['identity'], 'case_ids': selected, 'prompt': digest,
                    'backend': backend.identity()}
        scores = []
        for key in selected:
            path = folder / 'cases' / f'{key}.json'
            if path.exists():
                result = read_json(path)
                if result['identity'] != identity:
                    raise RuntimeError('Prompt candidate evidence identity mismatch')
            else:
                result = assess(by_id[key], observations[key]['trajectories'], examples[key], text,
                                backend, cache, {'scope': 'offline_prompt', 'candidate': digest})
                # Invalid sets earn zero fitness; the fixed 24-case denominator never shrinks.
                result.update(case_id=key, identity=identity, fitness_auc=result['auc'] if result['valid'] else 0.0)
                atomic_write_json(path, result)
            if result['fitness_auc'] is None:
                raise RuntimeError('Selected mixed case has no defined AUC')
            scores.append({'case_id': key, 'auc': result['auc'], 'fitness_auc': result['fitness_auc'],
                           'valid': result['valid'], 'rubrics': result['generation']['rubrics']})
        evaluated = {'candidate_id': candidate_id, 'prompt_sha256': digest, 'text': text,
                     'case_scores': scores, 'mean_auc': sum(row['fitness_auc'] for row in scores) / 24,
                     'identity': identity}
        atomic_write_json(folder / 'score.json', evaluated)
        return evaluated

    spec = cfg['prompt_optimization']
    initial = (Path(root) / spec['initial_prompt_file']).read_text()
    winner = evaluate(initial, 'initial')
    history = [winner]
    started = time.perf_counter()
    iterations = spec['iterations']
    if iterations not in (2, 3) or spec['candidates_per_iteration'] != 4:
        raise ValueError('Prompt search requires two or three iterations, four candidates each')
    for iteration in range(1, iterations + 1):
        low = sorted(winner['case_scores'], key=lambda row: (row['fitness_auc'], row['case_id']))[:2]
        evidence = [{'case': {key: by_id[row['case_id']].get(key) for key in ('question', 'scenario', 'options')},
                     'auc': row['fitness_auc'], 'rubrics': row['rubrics'],
                     'examples': examples[row['case_id']]} for row in low]
        candidates = []
        for index in range(4):
            proposal = cache.call('prompt_proposal', {'parent': winner['prompt_sha256'], 'iteration': iteration,
                                 'index': index, 'evidence': sha256_json(evidence), 'backend': backend.identity()},
                                  lambda: backend.propose(winner['text'], evidence, iteration, index))
            atomic_write_json(home / 'proposals' / f'iteration_{iteration}_candidate_{index}.json', proposal)
            candidates.append(evaluate(proposal['text'], f'iteration_{iteration}_candidate_{index}'))
        # Deterministic ties keep the parent; no test or checkpoint selection enters fitness.
        winner = max([winner, *candidates], key=lambda candidate: candidate['mean_auc'])
        history.extend(candidates)
        atomic_write_json(home / 'history.json', {'candidates': history, 'winner': winner['prompt_sha256'],
                                                 'completed_iterations': iteration})
        print(f'Prompt search iteration {iteration}: mean AUC={winner["mean_auc"]:.4f}', flush=True)
        if iteration >= 2 and spec.get('max_seconds') is not None and time.perf_counter() - started >= spec['max_seconds']:
            break
    prompt_path = home / 'final_prompt.txt'
    prompt_path.write_text(winner['text'])
    final = {'schema': 'sherpa_auc_prompt_v1', 'prompt_file': str(prompt_path.resolve()),
             'prompt_sha256': winner['prompt_sha256'], 'mean_auc': winner['mean_auc'],
             'case_scores': winner['case_scores'], 'optimization_case_ids': selected,
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
    if backend is not None and artifact['backend'] != backend.identity():
        raise RuntimeError('Optimized prompt generator/judge contract changed')
    return text, artifact
