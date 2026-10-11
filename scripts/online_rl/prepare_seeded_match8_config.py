"""Build a fresh-run config that reuses a saved first-batch rollout and gated rubric candidates.

This only writes a config. It never stops or launches a training process. The source
must have five durable first-batch bank_prepare records from the same initial policy.
The source run may have advanced after those records were saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.tree.experiment import deep_merge, load_raw


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_count(record: dict) -> int:
    count = 0
    for candidate in record.get('trajectory_rubrics') or []:
        rubric = candidate.get('rubric') or {}
        gate = candidate.get('gate') or {}
        source = candidate.get('source_check') or {}
        stage = source.get('second') if source.get('status') == 'REPAIRED_PASS' else source.get('first')
        if (candidate.get('valid') is True and gate.get('protocol') == 'public_mc_gate_v3' and
                gate.get('passed') is True and gate.get('public_criterion_version') == 'public_behavior_app_alignment_v5' and
                rubric.get('criterion_version') == 'public_behavior_app_alignment_v5' and
                source.get('status') in ('PASS', 'REPAIRED_PASS') and (stage or {}).get('applicable') is True and
                isinstance(candidate.get('intervention_instruction'), str) and
                candidate['intervention_instruction'].strip()):
            count += 1
    return count


def prepare(source_runs: list[Path], config_out: Path, output_root: str, run_label: str) -> dict:
    if not source_runs:
        raise RuntimeError('At least one source run is required')
    sources = []
    for source_arg in source_runs:
        source_run = source_arg.resolve()
        state_path = source_run / 'run_state.json'
        manifest_path = source_run / 'run_manifest.json'
        batch = source_run / 'batches' / 'batch_000001'
        batch_manifest_path = batch / 'manifest.json'
        if not state_path.is_file() or not manifest_path.is_file() or not batch_manifest_path.is_file():
            raise RuntimeError(f'Source run is missing durable first-batch metadata: {source_run}')
        manifest = json.loads(manifest_path.read_text())
        batch_manifest = json.loads(batch_manifest_path.read_text())
        cases = batch_manifest.get('case_ids') or []
        if len(batch_manifest.get('assignments') or []) != 5 or len(cases) != 5:
            raise RuntimeError(f'Expected five cases in source batch: {source_run}')
        source_policy = batch_manifest.get('source_policy_sha256')
        initial_receipt_path = source_run / 'checkpoints' / 'checkpoint_000000' / 'COMPLETE.json'
        if not source_policy or not initial_receipt_path.is_file():
            raise RuntimeError(f'Source has no verifiable initial checkpoint receipt: {source_run}')
        receipt = json.loads(initial_receipt_path.read_text())
        if receipt.get('policy_sha256') != source_policy or receipt.get('completed_cases') != 0:
            raise RuntimeError(f'Source first-batch rollout does not match its initial base policy: {source_run}')
        resolved_config = manifest.get('config')
        if not isinstance(resolved_config, dict):
            raise RuntimeError(f'Source manifest lacks resolved config: {source_run}')
        rubric_cfg = resolved_config.get('rubric') or {}
        sources.append({
            'run': source_run, 'batch': batch, 'manifest': manifest, 'manifest_path': manifest_path,
            'cases': cases, 'policy': source_policy, 'prompt': rubric_cfg.get('prompt_sha256'),
            'generator_model': rubric_cfg.get('generator_model'),
        })
    first = sources[0]
    for source in sources[1:]:
        if source['cases'] != first['cases'] or source['policy'] != first['policy'] or source['prompt'] != first['prompt']:
            raise RuntimeError('Source runs must share first-batch case order, initial policy, and rubric prompt hash')

    frozen = {}
    selected_sources = []
    for index, case_id in enumerate(first['cases']):
        options = []
        for source in sources:
            path = source['batch'] / f'case_{index:06d}' / 'bank_prepare.json'
            if not path.is_file():
                continue
            record = json.loads(path.read_text())
            bank = record.get('rubric_bank') or {}
            if (record.get('case_cursor') != index or record.get('case_id') != case_id or
                    record.get('policy_sha256') != source['policy'] or record.get('prompt_sha256') != source['prompt'] or
                    not record.get('tree') or not record.get('feedback') or not record.get('trajectory_inputs') or
                    not isinstance(record.get('trajectory_rubrics'), list) or bank.get('verification_pending') is not True):
                continue
            options.append(( _candidate_count(record), source, path, record))
        if not options:
            raise RuntimeError(f'No complete compatible frozen rollout exists for case {index} ({case_id})')
        # Prefer the saved rollout with the largest number of independently Gate-passed
        # candidates. Equal counts preserve command-line source order.
        count, source, path, _record = max(options, key=lambda item: item[0])
        try:
            source_file = str(path.relative_to(ROOT))
        except ValueError:
            source_file = str(path.resolve())
        frozen[str(index)] = {
            'file': source_file, 'sha256': sha256(path), 'source_run': str(source['run']),
            'source_run_manifest_sha256': sha256(source['manifest_path']),
            'source_generator_model': source['generator_model'], 'reusable_gated_candidates': count,
        }
        selected_sources.append({'case_cursor': index, 'case_id': case_id,
                                 'source_run': str(source['run']), 'generator_model': source['generator_model'],
                                 'reusable_gated_candidates': count})

    base = 'configs/experiments/train120_max4_batched_bank_reuse3_v1.json'
    # Flatten the resolved source config here. Adding another extends layer would
    # exceed the repository's bounded config-chain depth.
    config = deep_merge(load_raw(base), {
        'experiment': {'output_root': output_root, 'run_label': run_label},
        'rubric': {'match_batch_size': 8},
        'five_gpu': {
            'reuse_frozen_rubric_candidates': True,
            'first_batch_frozen_records': frozen,
            'frozen_source_runs': [
                {'run': str(s['run']), 'manifest_sha256': sha256(s['manifest_path']),
                 'generator_model': s['generator_model']} for s in sources
            ],
        },
        'notes': {
            'first_batch_reuse': (
                'Reuse only the hash-pinned five-case rollout, Doctor feedback, and individually Gate-passed '
                'source rubric candidates. Recompute same-case semantic assignments in batches of up to eight; '
                'generate/Gate only trajectories unmatched to valid seeded candidates. Do not reuse prior '
                'assignments, witness/KEEP, score, reward, advantage, or optimizer records. Re-run witness '
                'verification and scoring before the single first-batch update.'
            )
        },
    })
    out = config_out if config_out.is_absolute() else ROOT / config_out
    resolved_output = ROOT / output_root
    if resolved_output.exists():
        raise RuntimeError(f'Fresh output root already exists: {resolved_output}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    return {'config': str(out), 'source_runs': [str(s['run']) for s in sources],
            'selected_sources_by_case': selected_sources, 'source_policy_sha256': first['policy'],
            'source_prompt_sha256': first['prompt'], 'frozen_cases': len(frozen),
            'candidate_reuse': 'valid source candidates only; assignments and witness decisions recomputed',
            'match_batch_size': 8, 'launched': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', action='append', required=True,
                        help='Saved run with durable first-batch bank_prepare records; repeat to offer alternatives')
    parser.add_argument('--config-out', default='configs/experiments/train120_max4_batched_bank_reuse8_seeded_v1.json')
    parser.add_argument('--output-root', default='outputs/train120_max4_batched_bank_reuse8_seeded_20261011')
    parser.add_argument('--run-label', default='train120_max4_batched_bank_reuse8_seeded_v1')
    args = parser.parse_args()
    print(json.dumps(prepare([Path(p) for p in args.source_run], Path(args.config_out),
                             args.output_root, args.run_label), indent=2))


if __name__ == '__main__':
    main()
