"""Five-case transactions for a cumulative, immutable KEEP rubric bank."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.online_rl.integrated.records import atomic_write_json, sha256_json

EXACT_ASSIGNMENT_PROTOCOL = 'exact_generated_source_case_v1'
ONLINE_WORKING_BANK_PROTOCOL = 'trajectory_online_working_bank_v1'


def config_identity(cfg, root):
    spec = cfg.get('rubric', {}).get('cumulative_bank')
    if spec is None:
        return None
    from src.online_rl.tree.cumulative_bank import VERSION, validate_snapshot
    required = {'protocol', 'retrieval_top_k', 'matcher_prompt_file', 'matcher_prompt_sha256'}
    optional = {'initial_snapshot_file', 'initial_snapshot_sha256', 'scoring_scope', 'working_bank_protocol'}
    if not isinstance(spec, dict) or not required <= set(spec) or set(spec) - required - optional:
        raise ValueError('Invalid cumulative rubric bank configuration')
    if spec['protocol'] != VERSION or type(spec['retrieval_top_k']) is not int or not 1 <= spec['retrieval_top_k'] <= 32:
        raise ValueError('Cumulative bank requires its versioned protocol and retrieval_top_k 1..32')
    if spec.get('scoring_scope', 'top_k') not in ('top_k', 'all'):
        raise ValueError('Cumulative bank scoring_scope must be top_k or all')
    if spec.get('working_bank_protocol') not in (None, ONLINE_WORKING_BANK_PROTOCOL):
        raise ValueError('Unknown cumulative working-bank protocol')
    exact = cfg['rubric'].get('assignment_protocol') == EXACT_ASSIGNMENT_PROTOCOL
    if exact and (spec.get('scoring_scope') != 'all' or spec.get('initial_snapshot_file') or spec.get('initial_snapshot_sha256')):
        raise ValueError('Exact generated assignment requires an empty initial bank and full frozen-bank scoring')
    if (cfg.get('prompt_update', {}).get('enabled') or not cfg['rubric'].get('enabled', True)
            or cfg.get('five_gpu', {}).get('execution_mode') != 'shared_ready_tasks_v1'
            or cfg.get('agent_reward', {}).get('r_a_weighting') != 'equal'):
        raise ValueError('Cumulative bank requires fixed P0, rubric rewards, shared five-GPU mode and equal rubric weights')
    prompt = Path(root) / spec['matcher_prompt_file']
    if hashlib.sha256(prompt.read_bytes()).hexdigest() != spec['matcher_prompt_sha256']:
        raise ValueError('Cumulative bank matcher prompt hash differs')
    if bool(spec.get('initial_snapshot_file')) != bool(spec.get('initial_snapshot_sha256')):
        raise ValueError('Initial cumulative bank needs both file and snapshot hash')
    if spec.get('initial_snapshot_file'):
        initial = json.loads((Path(root) / spec['initial_snapshot_file']).read_text())
        validate_snapshot(initial)
        if initial['sha256'] != spec['initial_snapshot_sha256']:
            raise ValueError('Initial cumulative bank snapshot hash differs')
    identity = {'protocol': VERSION, 'retrieval_top_k': spec['retrieval_top_k'],
            'matcher_prompt_sha256': spec['matcher_prompt_sha256'],
            'initial_snapshot_sha256': spec.get('initial_snapshot_sha256'),
            'matcher_model': cfg['rubric']['generator_model'],
            'matcher_generation': cfg['rubric'].get('generation', {}),
            'reuse_effect_verification': 'historical_witness_no_reverification',
            'batch_scoring': 'freeze_full_bank_after_all_five_verifications',
            'weighting': 'equal_unique_criterion',
            'retrieval': 'deterministic_prefix_only_state_shortlist_v1'}
    # Omitted scope preserves the complete legacy identity and its hash.
    if spec.get('scoring_scope', 'top_k') == 'all':
        identity.update(scoring_scope='all', scoring='full_bank_shared_state_applicability_v1',
                        retrieval='generation_reuse_trajectory_shortlist_v1')
    if exact:
        identity.update(assignment_protocol=EXACT_ASSIGNMENT_PROTOCOL,
                        generation_assignment='all_eligible_trajectories_then_exact_public_content_dedup_v1',
                        semantic_matcher='disabled', reuse_effect_verification='new_source_witness_only',
                        retrieval='disabled_for_generation_full_bank_for_scoring_v1',
                        matcher_model=None, matcher_generation=None)
    if spec.get('working_bank_protocol') == ONLINE_WORKING_BANK_PROTOCOL:
        if exact or spec.get('scoring_scope') != 'all':
            raise ValueError('Online working bank requires semantic assignment and full-bank scoring')
        identity.update(working_bank_protocol=ONLINE_WORKING_BANK_PROTOCOL,
                        generation_assignment='trajectory_match_then_generate_gate_rerun_keep_v1',
                        batch_scoring='freeze_full_bank_after_ordered_trajectory_transactions',
                        reuse_effect_verification='historical_keep_no_reverification')
    return identity


def _read(path):
    return json.loads(Path(path).read_text())


def snapshot_path(out, digest):
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise RuntimeError('Invalid committed bank digest')
    return Path(out) / 'rubric_bank' / 'snapshots' / f'{digest}.json'


def publish_snapshot(out, snapshot):
    from src.online_rl.tree.cumulative_bank import validate_snapshot
    validate_snapshot(snapshot)
    path = snapshot_path(out, snapshot['sha256'])
    if path.exists() and _read(path) != snapshot:
        raise RuntimeError('Immutable cumulative bank snapshot changed')
    if not path.exists():
        atomic_write_json(path, snapshot)
    return path


def initial_snapshot(cfg, root):
    from src.online_rl.tree.cumulative_bank import empty_snapshot, validate_snapshot
    spec = cfg['rubric']['cumulative_bank']
    value = (_read(Path(root) / spec['initial_snapshot_file']) if spec.get('initial_snapshot_file')
             else empty_snapshot())
    validate_snapshot(value)
    if spec.get('initial_snapshot_sha256') and value['sha256'] != spec['initial_snapshot_sha256']:
        raise RuntimeError('Bootstrap bank hash differs')
    return value


def bank_fields(snapshot):
    return {'rubric_bank_sha256': snapshot['sha256'], 'rubric_bank_version': snapshot['version']}


def committed_snapshot(cfg, root, out, state):
    from src.online_rl.tree.cumulative_bank import validate_snapshot
    if 'rubric_bank_sha256' not in state:
        raise RuntimeError('Committed run state has no cumulative bank identity')
    value = _read(snapshot_path(out, state['rubric_bank_sha256']))
    validate_snapshot(value)
    if any(state.get(k) != v for k, v in bank_fields(value).items()):
        raise RuntimeError('Committed bank version differs from run state')
    receipt = _read(Path(state['checkpoint']) / 'COMPLETE.json')
    if receipt.get('rubric_bank_sha256') is None:
        # The first handoff/checkpoint is saved before the bank is attached to
        # run state. Its exact initial bank remains pinned by the config hash.
        initial = initial_snapshot(cfg, root)
        allowed_name = f"handoff_{state['case_cursor']:06d}" if cfg.get('supervision_handoff') else 'checkpoint_000000'
        if Path(state['checkpoint']).name != allowed_name or value != initial:
            raise RuntimeError('Checkpoint receipt is missing its bank identity')
    elif any(receipt.get(k) != v for k, v in bank_fields(value).items()):
        raise RuntimeError('Checkpoint and run state cumulative banks differ')
    return value


def initialize_bank_state(cfg, root, out, state):
    """Attach the pinned seed bank once; later resumes must match the receipt."""
    if 'rubric_bank_sha256' not in state:
        value = initial_snapshot(cfg, root)
        allowed_name = f"handoff_{state['case_cursor']:06d}" if cfg.get('supervision_handoff') else 'checkpoint_000000'
        if Path(state['checkpoint']).name != allowed_name:
            raise RuntimeError('Cannot add a bank to an already advanced run')
        publish_snapshot(out, value)
        state = {**state, **bank_fields(value)}
        atomic_write_json(Path(out) / 'run_state.json', state)
    committed_snapshot(cfg, root, out, state)
    return state


def round_records(cfg, root, out, state, batch, batchdir, collect):
    """Publish a tentative scoring bank; only runner checkpoint/state commits it."""
    from src.online_rl.tree.cumulative_bank import extend_snapshot
    before = committed_snapshot(cfg, root, out, state)
    batchdir = Path(batchdir)
    identity = {'batch': batch, 'source_policy_sha256': state['policy_sha256'],
                'source_case_cursor': state['case_cursor'], **bank_fields(before)}
    exact = cfg['rubric'].get('assignment_protocol') == EXACT_ASSIGNMENT_PROTOCOL
    if exact:
        identity['assignment_protocol'] = EXACT_ASSIGNMENT_PROTOCOL
    scoring_scope = cfg['rubric']['cumulative_bank'].get('scoring_scope', 'top_k')
    if scoring_scope not in ('top_k', 'all'):
        raise ValueError('Cumulative bank scoring_scope must be top_k or all')
    if scoring_scope == 'all':
        identity['scoring_scope'] = 'all'
    path = batchdir / 'bank_input.json'
    if path.exists() and _read(path) != identity:
        raise RuntimeError('Pending batch cumulative bank/policy changed')
    if not path.exists():
        atomic_write_json(path, identity)
    prompt = {'text': (Path(root) / cfg['rubric']['generator_prompt_file']).read_text(),
              'id': cfg['rubric']['prompt_id'], 'sha256': cfg['rubric']['prompt_sha256'],
              'snapshot': before}
    timings = {}
    online = cfg['rubric']['cumulative_bank'].get('working_bank_protocol') == ONLINE_WORKING_BANK_PROTOCOL
    if online:
        from src.online_rl.tree.online_working_bank import VERSION as JOURNAL_VERSION
        working = before; records = []; decisions = []
        started = time.monotonic()
        for case_cursor in range(state['case_cursor'], state['case_cursor'] + cfg['five_gpu']['batch_cases']):
            rows = collect('bank_online', {**prompt, 'snapshot': working}, [case_cursor])
            if len(rows) != 1 or rows[0].get('case_cursor') != case_cursor:
                raise RuntimeError('Online working bank requires one deterministic case at a time')
            record = rows[0]
            if record['cumulative_bank_input']['sha256'] != working['sha256']:
                raise RuntimeError('Online case did not consume the latest working bank')
            next_snapshot = record.get('cumulative_bank_output')
            from src.online_rl.tree.cumulative_bank import validate_snapshot
            validate_snapshot(next_snapshot)
            tail = working['sha256']
            for row in record.get('online_working_bank_decisions', []):
                if row['input_bank_sha256'] != tail:
                    raise RuntimeError('Online trajectory did not consume the immediately prior working bank')
                if row['action'] == 'GENERATED_KEEP':
                    if row['output_bank_sha256'] == tail:
                        raise RuntimeError('Online KEEP failed to advance the working bank')
                elif row['output_bank_sha256'] != tail:
                    raise RuntimeError('Only online KEEP may advance the working bank')
                tail = row['output_bank_sha256']
                decisions.append(row)
            if tail != next_snapshot['sha256']:
                raise RuntimeError('Online decision journal does not reach the case output bank')
            working = next_snapshot; records.append(record)
        timings['bank_online'] = time.monotonic() - started
        after = working
        atomic_write_json(batchdir / 'online_working_bank_journal.json', {
            'protocol': JOURNAL_VERSION, 'batch': batch, 'source_policy_sha256': state['policy_sha256'],
            'initial_bank_sha256': before['sha256'], 'working_bank_sha256': after['sha256'],
            'decisions': decisions})
    else:
        records = None
    for phase in ('bank_prepare', 'bank_verify'):
        if online: break
        print(json.dumps({'stage': phase, 'batch': batch, 'bank_version': before['version'],
                          'bank_rubrics': len(before['entries'])}), flush=True)
        started = time.monotonic()
        records = collect(phase, prompt)
        timings[phase] = time.monotonic() - started
        if len(records) != cfg['five_gpu']['batch_cases'] or len({r['case_id'] for r in records}) != len(records):
            raise RuntimeError('Cumulative bank requires all five distinct case records')
        if any(r['policy_sha256'] != state['policy_sha256'] or
               r['cumulative_bank_input']['sha256'] != before['sha256'] for r in records):
            raise RuntimeError('Case preparation used a different policy or bank snapshot')
        if exact and any(r.get('cumulative_reuse') for r in records):
            raise RuntimeError('Exact-generated assignment cannot inherit a semantic bank match')
    if not online:
        after = extend_snapshot(before, records)
    publish_snapshot(out, after)
    frozen_path = batchdir / 'bank_frozen.json'
    if frozen_path.exists() and _read(frozen_path) != after:
        raise RuntimeError('Frozen batch scoring bank changed')
    if not frozen_path.exists():
        atomic_write_json(frozen_path, after)
    print(json.dumps({'stage': 'bank_score', 'batch': batch, 'bank_version': after['version'],
                      'bank_rubrics': len(after['entries'])}), flush=True)
    started = time.monotonic()
    scored = collect('bank_score', {**prompt, 'snapshot': after})
    timings['bank_score'] = time.monotonic() - started
    if sorted(r['case_id'] for r in scored) != sorted(r['case_id'] for r in records):
        raise RuntimeError('Final bank scoring case coverage changed')
    if any(r['policy_sha256'] != state['policy_sha256'] or
           r.get('cumulative_bank_scoring', {}).get('snapshot_sha256') != after['sha256'] for r in scored):
        raise RuntimeError('All five cases must use the same frozen bank and policy')
    if scoring_scope == 'all' and any(
            r.get('cumulative_bank_scoring', {}).get('scoring_scope') != 'all' for r in scored):
        raise RuntimeError('All five cases must score the full cumulative bank')
    atomic_write_json(batchdir / 'bank_metrics.json', {
        **identity, 'scoring_bank_sha256': after['sha256'],
        'before_rubrics': len(before['entries']), 'after_rubrics': len(after['entries']),
        'added_unique_rubrics': len(after['entries']) - len(before['entries']),
        'reused_trajectories': sum(len(r.get('cumulative_reuse', [])) for r in scored),
        **({'semantic_pre_generation_reuse_disabled': True} if exact else {}),
        'timings_wall_seconds': timings})
    return scored, after
