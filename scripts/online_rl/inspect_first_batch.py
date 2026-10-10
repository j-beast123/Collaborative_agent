"""Read-only first-batch evidence index for the isolated source-case run.

This does not judge rubric quality. It collects the immutable source records,
score evidence, witness decisions, and reward diagnostics for human review.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _selected_reply(record: dict, rubric: dict) -> str | None:
    origin = (rubric.get('rubric') or {}).get('source_provenance') or {}
    turn = (origin.get('analysis') or {}).get('selected_agent_turn_id')
    leaf_id = rubric.get('leaf_id')
    leaves = {row['leaf_id']: row for row in record['tree']['leaves']}
    nodes = {row['node_id']: row for row in record['tree']['nodes']}
    if leaf_id not in leaves or type(turn) is not int or not 1 <= turn <= len(leaves[leaf_id]['path']):
        return None
    return nodes[leaves[leaf_id]['path'][turn - 1]]['response']


def _score_summary(rows: list[dict]) -> dict:
    counts = Counter(str(r['hard']) if r['scoring_status'] == 'scored' else r['scoring_status'] for r in rows)
    examples = {}
    for row in rows:
        label = str(row['hard']) if row['scoring_status'] == 'scored' else row['scoring_status']
        if label not in examples:
            evidence = row.get('scoring_evidence') or {}
            # The full wire requests and raw responses remain in the saved case.
            # This index keeps just the judge's short adjudication.
            stage2 = evidence.get('stage2') or {}
            examples[label] = {'response_node_id': row['response_node_id'],
                               'rubric_id': row['rubric_id'], 'stage2': stage2}
    return {'counts': dict(sorted(counts.items())), 'one_example_per_result': examples}


def _review_only_pairs(rubrics: dict[str, dict]) -> list[dict]:
    """Lexical shortlist only; similarity never establishes rubric equivalence."""
    def tokens(row):
        public = row['rubric']
        fields = [public.get('required_behavior', ''), public.get('minimum_performance', ''),
                  public.get('completion_condition', '')]
        return set(re.findall(r'[^\W_]+', ' '.join(fields).casefold(), re.UNICODE))
    ids = sorted(rubrics)
    terms = {rid: tokens(rubrics[rid]) for rid in ids}
    pairs = []
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            a, b = terms[left], terms[right]
            overlap = len(a & b) / len(a | b) if a | b else 0.0
            if overlap >= 0.6:
                pairs.append({'rubric_ids': [left, right], 'lexical_jaccard': round(overlap, 3),
                              'status': 'manual_review_candidate_only'})
    return pairs


def inspect(run_dir: Path) -> dict:
    state = _read(run_dir / 'run_state.json')
    if state.get('case_cursor', 0) < 5 or state.get('batch_cursor', 0) < 1:
        raise RuntimeError('First five cases have not all committed')
    manifest = _read(run_dir / 'run_manifest.json')
    cfg = manifest['config']
    cumulative = cfg['rubric'].get('cumulative_bank') or {}
    if (cfg['policy'].get('init_adapter') is not None or
            cumulative.get('scoring_scope') != 'all' or cumulative.get('initial_snapshot_file')):
        raise RuntimeError('This report requires a fresh base run and an initially empty full-scoring bank')
    if cfg['rubric'].get('assignment_protocol') != 'exact_generated_source_case_v1':
        raise RuntimeError('Exact source-case assignment protocol is not pinned')
    records = [_read(run_dir / 'cases' / f'case_{i:06d}.json') for i in range(5)]
    if [r['case_cursor'] for r in records] != list(range(5)):
        raise RuntimeError('First-batch case cursors are not complete')
    if len({r['policy_sha256'] for r in records}) != 1 or len({r['prompt_sha256'] for r in records}) != 1:
        raise RuntimeError('First-batch cases used different policies or rubric prompts')
    bank_sha = state.get('rubric_bank_sha256')
    if not bank_sha:
        raise RuntimeError('Committed cumulative bank identity is missing')
    snapshot = _read(run_dir / 'rubric_bank' / 'snapshots' / f'{bank_sha}.json')
    if snapshot.get('sha256') != bank_sha or snapshot.get('version') != 1:
        raise RuntimeError('First-batch frozen bank identity or version differs')
    from src.online_rl.tree.cumulative_bank import empty_snapshot, validate_snapshot
    validate_snapshot(snapshot)
    if snapshot.get('parent_sha256') != empty_snapshot()['sha256']:
        raise RuntimeError('First-batch bank was not built from an empty bank')
    entries = {entry['bank_id']: entry for entry in snapshot['entries']}
    if any(proof['case_id'] not in {r['case_id'] for r in records}
           for entry in entries.values() for proof in entry['sources']):
        raise RuntimeError('Frozen bank imported a source outside the first five cases')
    cases = []
    for record in records:
        if record.get('cumulative_reuse'):
            raise RuntimeError('Pre-generation semantic bank reuse occurred')
        if (record.get('cumulative_bank_input') or {}).get('entries'):
            raise RuntimeError('First-batch generation did not start from an empty bank')
        scoring = record.get('cumulative_bank_scoring') or {}
        if scoring.get('snapshot_sha256') != bank_sha or scoring.get('scoring_scope') != 'all':
            raise RuntimeError('Case did not score the same frozen full bank')
        cid = record['case_id']
        bank = record.get('rubric_bank') or {}
        selections = {row['rubric_id']: row for row in record['selection']}
        rubrics = {row['rubric_id']: row for row in record['trajectory_rubrics']}
        witness = {row['rubric_id']: row for row in bank.get('witness', [])}
        kept = {rid for rid, row in selections.items() if row['retained']}
        if not kept <= set(rubrics) or not kept <= set(witness):
            raise RuntimeError(f'{cid}: KEEP rubric lacks a source or witness')
        scored_ids = {row['rubric_id'] for row in record['response_scores']}
        if scored_ids != set(entries):
            raise RuntimeError(f'{cid}: case did not evaluate every frozen bank criterion')
        observed_pairs = [(row['response_node_id'], row['rubric_id']) for row in record['response_scores']]
        expected_pairs = {(node['node_id'], rid) for node in record['tree']['nodes'] for rid in entries}
        if len(observed_pairs) != len(set(observed_pairs)) or set(observed_pairs) != expected_pairs:
            raise RuntimeError(f'{cid}: incomplete or duplicate response-by-bank scoring matrix')
        if any(row['scoring_status'] == 'not_retrieved' for row in record['response_scores']):
            raise RuntimeError(f'{cid}: full-bank scoring emitted a not_retrieved result')
        reviews = []
        for rid, raw in rubrics.items():
            public = raw['rubric']
            origin = public.get('source_provenance') or {}
            if origin.get('source_case_id') != cid:
                raise RuntimeError(f'{cid}: source provenance crosses case boundary')
            reviews.append({'rubric_id': rid, 'source_trajectory_id': raw['leaf_id'],
                            'source_outcome': origin.get('source_outcome'),
                            'source_reply': _selected_reply(record, raw),
                            'applicability': public.get('applicability'),
                            'required_behavior': public.get('required_behavior'),
                            'minimum_performance': public.get('minimum_performance'),
                            'completion_condition': public.get('completion_condition'),
                            'scores': public.get('scores'), 'criterion_version': public.get('criterion_version'),
                            'gate': raw.get('gate'), 'source_check': raw.get('source_check'),
                            'valid': raw['valid'], 'problems': raw.get('problems'),
                            'witness': witness.get(rid), 'selection': selections.get(rid)})
        by_rubric = defaultdict(list)
        for row in record['response_scores']:
            by_rubric[row['rubric_id']].append(row)
        cases.append({'case_cursor': record['case_cursor'], 'case_id': cid,
                      'case_record': str(run_dir / 'cases' / f"case_{record['case_cursor']:06d}.json"),
                      'counts': record['counts'], 'assignment_telemetry': bank.get('telemetry'),
                      'assignments': bank.get('assignments'),
                      'generated_candidates': bank.get('generated_candidates'),
                      'review_only_similar_pairs': _review_only_pairs(rubrics),
                      'rubrics_generated_here': reviews,
                      'frozen_bank_rubric_ids': sorted(entries),
                      'score_distributions': {rid: _score_summary(rows)
                                                                for rid, rows in by_rubric.items()},
                      'group_diagnostics': record.get('group_diagnostics'),
                      'agent_targets': record.get('agent_targets'),
                      'batch_update': record.get('batch_update')})
    return {'schema': 'first_five_source_case_evidence_v1', 'run_dir': str(run_dir),
            'config_sha256': state.get('config_sha256'), 'protocol_sha256': manifest.get('protocol_sha256'),
            'prompt_sha256': records[0]['prompt_sha256'], 'initial_policy_sha256': records[0]['policy_sha256'],
            'committed_cases_at_inspection': state['case_cursor'],
            'checkpoint': state.get('checkpoint'), 'frozen_bank_sha256': bank_sha,
            'frozen_bank_unique_rubrics': len(entries),
            'frozen_bank_sources': {rid: [p['case_id'] for p in entry['sources']]
                                    for rid, entry in entries.items()},
            'review_only_bank_similar_pairs': _review_only_pairs(
                {rid: entry['rubric'] for rid, entry in entries.items()}),
            'requires_manual_semantic_review': True, 'cases': cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = inspect(args.run_dir.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(args.out.resolve()), 'cases': len(report['cases']),
                      'checkpoint': report['checkpoint']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
