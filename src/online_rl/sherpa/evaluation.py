"""Full-source base and iteration-end checkpoint evaluation."""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

from src.common import usage
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.integrated.records import atomic_write_json, read_json, sha256_json
from src.online_rl.tree.base_evaluation import original_base_policy
from src.online_rl.tree.five_gpu import evaluation_adapter
from src.online_rl.tree.cache import CallCache
from src.online_rl.sherpa.rollouts import collect


def evaluation_cases(cfg, root):
    from src.online_rl.cases import load_cases, file_sha256
    spec = cfg['evaluation']
    cases = load_cases(Path(root) / spec['file'])
    manifest = read_json(Path(root) / spec['split_manifest'])
    source_sha256 = file_sha256(Path(root) / spec['file'])
    if (source_sha256 != manifest['source_sha256']
            or source_sha256 != spec['source_sha256']):
        raise RuntimeError('Evaluation source file differs from the fixed split manifest')
    by_id = {case['case_id']: case for case in cases}
    test_ids = manifest['test_case_ids']
    if (len(by_id) != len(cases) or len(test_ids) != len(cases)
            or len(test_ids) != spec['test_cases'] or set(test_ids) != set(by_id)):
        raise ValueError('Evaluation test split must contain every source case exactly once')
    if spec['checkpoint_steps'] != [30, 60, 90]:
        raise ValueError('Evaluation checkpoints must be the iteration ends: steps 30, 60 and 90')
    return [by_id[key] for key in test_ids], manifest


def solo(policy, case, seed):
    from src.collaboration.generator import case_text, final_answer_option
    user = case_text(case) + '\n\nSolve this question independently. End with "Final answer: <option label>".'
    text = policy.respond_with_system('You solve medical multiple-choice questions independently.', user, seed)
    return {'text': text, 'option': final_answer_option(text)}


def evaluate_job(cfg, policy, cases, folder, identity):
    from src.online_rl.environment import final_outcome
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'results.json'
    if done.exists():
        result = read_json(done)
        if result['identity'] != identity:
            raise RuntimeError('Evaluation result identity changed')
        return result
    rows = []
    started = time.perf_counter()
    for case in cases:
        path = folder / 'cases' / f'{case["case_id"]}.json'
        if path.exists():
            row = read_json(path)
            if row['identity'] != identity:
                raise RuntimeError('Evaluation case identity changed')
        else:
            seed = derive_seed(cfg['evaluation']['seed'], case['case_id'], 'collaboration_eval', 0)
            with usage.case(case['case_id']):
                cache = CallCache(folder / 'calls' / f'{case["case_id"]}.jsonl')
                _, episodes = collect(case, cfg, policy, cache, 0, behavior=False,
                                      branching=1, seed_override=seed)
                trajectory = episodes[0]
                independent = solo(policy, case, derive_seed(cfg['evaluation']['seed'], case['case_id'], 'solo_eval', 0))
            row = {'case_id': case['case_id'], 'identity': identity,
                   'collaboration_correct': int(final_outcome(case, trajectory['final_option']) == 1),
                   'solo_correct': int(final_outcome(case, independent['option']) == 1),
                   'final_option': trajectory['final_option'], 'solo': independent,
                   'termination_reason': trajectory['stop_reason'],
                   'agent_turns': trajectory['agent_turns'], 'dialogue': trajectory['dialogue']}
            atomic_write_json(path, row)
        rows.append(row)
    result = {'identity': identity, 'case_count': len(rows),
              'collaboration_accuracy': statistics.mean(row['collaboration_correct'] for row in rows),
              'solo_accuracy': statistics.mean(row['solo_correct'] for row in rows),
              'termination_reasons': dict(Counter(row['termination_reason'] for row in rows)),
              'mean_agent_turns': statistics.mean(row['agent_turns'] for row in rows),
              'case_ids': [row['case_id'] for row in rows], 'seconds': time.perf_counter() - started}
    atomic_write_json(done, result)
    return result


def evaluate(cfg, policy, root, *, base_only=False):
    home = Path(root) / cfg['output_dir']
    test, manifest = evaluation_cases(cfg, root)
    common = {'doctor_protocol': policy.doctor.protocol, 'rollout': cfg['rollout'],
              'evaluation_seed': cfg['evaluation']['seed'], 'split': sha256_json(manifest),
              'model': cfg['policy']['base_model']}
    with original_base_policy(policy):
        base = evaluate_job(cfg, policy, test, home / 'evaluation' / 'base',
                            {**common, 'policy': 'original_base', 'case_ids': [case['case_id'] for case in test]})
    if base_only:
        return {'base': base}
    state = read_json(home / 'run_state.json')
    iterations = []
    final = None
    for step in cfg['evaluation']['checkpoint_steps']:
        if step > state['completed_steps'] or (step == 90 and not state.get('complete')):
            continue
        checkpoint = home / 'checkpoints' / f'checkpoint_{step:06d}'
        if step == 90:
            checkpoint = Path(state['final_checkpoint'])
        receipt = read_json(checkpoint / 'COMPLETE.json')
        with evaluation_adapter(policy, checkpoint):
            folder = home / 'evaluation' / ('final' if step == 90 else f'iteration_{step // 30}')
            result = evaluate_job(cfg, policy, test, folder,
                                  {**common, 'step': step, 'policy': receipt['policy_sha256'],
                                   'method': cfg['method'], 'case_ids': [case['case_id'] for case in test]})
        iterations.append({'iteration': step // 30, 'step': step, **result})
        if step == 90:
            final = result
    results = {'base': base, 'iteration_results': iterations, 'main_result': final,
               'main_checkpoint_selection': 'last_step_90', 'test_selects_checkpoint': False}
    atomic_write_json(home / 'evaluation' / 'summary.json', results)
    return results
