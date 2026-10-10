"""Counts with explicit denominators; no imputed third verification trial."""
from collections import Counter, defaultdict
import json
from pathlib import Path


def fraction(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'ratio': numerator / denominator if denominator else None}


def case_diagnostics(record):
    from src.online_rl.tree.fixed_accuracy import ID as ACCURACY
    bank = record.get('rubric_bank') or {}
    assignments = bank.get('assignments', [])
    witnesses = bank.get('witness', [])
    targets = record['agent_targets']
    scores = record.get('response_scores', [])
    by_node = defaultdict(list)
    for score in scores:
        by_node[score['response_node_id']].append(score)
    groups = list(targets['groups'].values())
    applicable = differing = same_outcome_differing = na = excluded = unscorable = 0
    partner_applicable = partner_differing = accuracy_only_eligible = accuracy_only_used = used = 0
    for ids in groups:
        rows = [targets['nodes'][nid] for nid in ids]
        values = [r['R_A'] for r in rows]
        usable = [r['reward'] for r in rows if r['reward'] is not None]
        applicable += any(s['match'] for nid in ids for s in by_node[nid])
        has_difference = all(v is not None for v in values) and len(set(values)) > 1
        differing += has_difference
        outcomes = [r['outcome_component'] for r in rows]
        same_outcome_differing += has_difference and all(o is not None for o in outcomes) and len(set(outcomes)) == 1
        na += not any(s['match'] for nid in ids for s in by_node[nid])
        unscorable += any(s.get('scoring_status') == 'group_unscorable' for nid in ids for s in by_node[nid])
        excluded += len(usable) < 2 or len(set(usable)) <= 1
        is_used = len(usable) >= 2 and len(set(usable)) > 1
        used += is_used
        partner = [[s for s in by_node[nid] if s['rubric_id'] != ACCURACY] for nid in ids]
        partner_applicable += any(s['match'] for rs in partner for s in rs)
        values = [[s['score'] for s in rs if s.get('scoring_status') == 'scored'] for rs in partner]
        means = [sum(v) / len(v) if v else None for v in values]
        partner_differing += all(v is not None for v in means) and len(set(means)) > 1
        only_accuracy = not any(values) and all(any(s['rubric_id'] == ACCURACY and s.get('scoring_status') == 'scored'
                                                  for s in by_node[nid]) for nid in ids)
        accuracy_only_eligible += only_accuracy
        accuracy_only_used += only_accuracy and is_used
    n = len(groups)
    statuses = Counter(s.get('scoring_status') for s in scores)
    needs = {n['need_id']: n for leaf in record['tree']['leaves'] for n in leaf.get('private_needs', [])}
    def distribution(values):
        import statistics
        return {'count': len(values), 'mean': statistics.mean(values) if values else None,
                'std': statistics.pstdev(values) if values else None, 'min': min(values) if values else None,
                'max': max(values) if values else None, 'histogram': dict(Counter(values))}
    need_values = list(needs.values())
    attempts = [a for n in need_values for a in n.get('audit', {}).get('attempts', [])]
    return {'case_id': record['case_id'], 'case_cursor': record['case_cursor'],
            'candidate_bank': bank.get('candidate_bank'),
            'rubrics': {'created': sum(a['assignment'] in ('CREATED_NEW', 'CREATE_REJECTED') for a in assignments),
                        'reused': sum(a['assignment'] == 'REUSED_EXISTING' for a in assignments),
                        'gate_passed': sum(r['valid'] for r in record.get('trajectory_rubrics', [])),
                        'verification_decisions': dict(Counter(w['decision'] for w in witnesses)),
                        'rejected_insufficient_public_need': sum(any('need_public_evidence_insufficient' in p for p in r.get('problems', [])) for r in record.get('trajectory_rubrics', []))},
            'needs': {'enabled': record.get('doctor_need_enabled', True), 'unique_records': len(needs), 'status_counts': dict(Counter(n['status'] for n in need_values)),
                      'missing_node_records': (sum(n.get('prefix_doctor_need') is None for n in record['tree']['nodes'])
                                               if record.get('doctor_need_enabled', True) else 0),
                      'format_errors': sum(len(n.get('audit', {}).get('format_errors', [])) for n in need_values),
                      'format_retries': sum(n.get('audit', {}).get('format_retries', 0) for n in need_values),
                      'transport_retries': sum(a.get('transport_retries', 0) for a in attempts),
                      'combined_generation_output_tokens': sum(a.get('output_tokens', 0) for a in attempts),
                      'combined_generation_seconds': sum(n.get('audit', {}).get('combined_generation_seconds', 0) for n in need_values),
                      'private_need_incremental_tokens': None, 'private_need_incremental_seconds': None,
                      'note': ('Totals describe joint public+need generation; isolated private overhead is not observable.'
                               if record.get('doctor_need_enabled', True) else
                               'Prospective needs disabled: no need generation; absence is intentional, not missing evidence.')},
            'verification': {'actual_trials': sum(w['num_reruns_executed'] for w in witnesses),
                             'early_decisions': sum(t.get('early_decision', False) for w in witnesses for t in w['tested']),
                             'unparsed_trials': sum(t.get('unparsed', 0) for w in witnesses for t in w['tested']),
                             'details': witnesses},
            'groups': n, 'applicable_groups': fraction(applicable, n),
            'different_rubric_score_groups': fraction(differing, n),
            'same_outcome_different_rubric_groups': fraction(same_outcome_differing, n),
            'na_groups': fraction(na, n), 'unscorable_groups': fraction(unscorable, n),
            'training_excluded_groups': fraction(excluded, n),
            'partner_applicable_groups': fraction(partner_applicable, n),
            'partner_different_score_groups': fraction(partner_differing, n),
            'accuracy_only_eligible_groups': fraction(accuracy_only_eligible, n),
            'accuracy_only_trained_groups': fraction(accuracy_only_used, used),
            'accuracy_score_distribution': distribution([s['score'] for s in scores if s['rubric_id'] == ACCURACY and s.get('scoring_status') == 'scored']),
            'partner_score_distribution': distribution([s['score'] for s in scores if s['rubric_id'] != ACCURACY and s.get('scoring_status') == 'scored']),
            'score_status_counts': dict(statuses),
            'score_na_rate': fraction(statuses['not_applicable'], len(scores)),
            'score_unscorable_rate': fraction(statuses['group_unscorable'], len(scores)),
            'operational_scoring_failures': 'abort batch; see attempts/errors, never converted to a score',
            'timings': record.get('timings', {})}


def batch_diagnostics(records, update, seconds):
    cases = [case_diagnostics(record) for record in records]
    counts = Counter()
    decisions = Counter()
    fractions = defaultdict(lambda: [0, 0])
    timing = Counter()
    needs = Counter()
    candidate_counts = Counter()
    need_status = Counter()
    distributions = {key: Counter() for key in ('accuracy_score_distribution', 'partner_score_distribution')}
    for case in cases:
        candidate_counts.update({k: v for k, v in (case.get('candidate_bank') or {}).items()
                                 if type(v) is int and k != 'candidate_batch_size'})
        counts.update({k: v for k, v in case['rubrics'].items() if isinstance(v, int)})
        decisions.update(case['rubrics']['verification_decisions'])
        counts['verification_actual_trials'] += case['verification']['actual_trials']
        counts['verification_early_decisions'] += case['verification']['early_decisions']
        for key, value in case.items():
            if isinstance(value, dict) and 'denominator' in value:
                fractions[key][0] += value['numerator']; fractions[key][1] += value['denominator']
        timing.update(case['timings'])
        needs.update({k: v for k, v in case['needs'].items() if type(v) in (int, float)})
        need_status.update(case['needs']['status_counts'])
        for key, histogram in distributions.items():
            histogram.update(case[key]['histogram'])
    import statistics
    summaries = {}
    for key, histogram in distributions.items():
        values = list(histogram.elements())
        summaries[key] = {'count': len(values), 'mean': statistics.mean(values) if values else None,
                          'std': statistics.pstdev(values) if values else None,
                          'min': min(values) if values else None, 'max': max(values) if values else None,
                          'histogram': dict(histogram)}
    return {'cases': cases, 'counts': dict(counts), 'verification_decisions': dict(decisions),
            'candidate_bank_counts': dict(candidate_counts),
            'needs': {**dict(needs), 'status_counts': dict(need_status),
                      'private_need_incremental_tokens': None, 'private_need_incremental_seconds': None},
            **summaries,
            'rates': {k: fraction(*v) for k, v in fractions.items()},
            'sum_case_phase_seconds': dict(timing), 'update_seconds': seconds,
            'optimizer_update': update, 'actual_optimizer_steps_this_batch': int(bool(update.get('optimizer_step')))}


def usage_report(path, case_ids=None):
    groups = defaultdict(lambda: {'calls': 0, 'retries': 0, 'input_tokens': 0, 'output_tokens': 0,
                                  'unknown_token_calls': 0, 'reported_cost_usd': 0.0, 'unpriced_calls': 0})
    path = Path(path)
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if case_ids is not None and row.get('case_id') not in case_ids:
            continue
        g = groups[str(row.get('stage') or row.get('purpose'))]
        g['calls'] += 1; g['retries'] += row.get('retry_count', 0)
        for field in ('input_tokens', 'output_tokens'):
            g[field] += row.get(field) or 0
        g['unknown_token_calls'] += row.get('input_tokens') is None or row.get('output_tokens') is None
        g['reported_cost_usd'] += row.get('cost_usd') or 0
        g['unpriced_calls'] += row.get('cost_usd') is None
    return dict(groups)


def attempt_report(out, case_ids):
    """Actual HTTP attempts (including failures), separately from logical calls."""
    out = Path(out)
    by_role = defaultdict(Counter)
    path = out / 'http_attempts.jsonl'
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if row.get('case_id') not in case_ids: continue
        counter = by_role[str(row.get('role'))]
        counter['actual_http_attempts'] += 1
        counter['errors'] += bool(row.get('exception_type') or row.get('provider_error'))
        counter['input_tokens_reported'] += row.get('input_tokens') or 0
        counter['output_tokens_reported'] += row.get('output_tokens') or 0
        counter['unknown_usage_attempts'] += row.get('input_tokens') is None or row.get('output_tokens') is None
        counter['seconds'] += row.get('seconds', 0)
        counter['limiter_wait_seconds'] += row.get('limiter_wait_seconds', 0)
        counter['rate_limit_responses'] += row.get('http_status') == 429
        counter['reported_cost_usd'] += row.get('cost_usd') or 0
        counter['unpriced_attempts'] += row.get('cost_usd') is None
        counter['max_actual_global_in_flight'] = max(counter['max_actual_global_in_flight'], row.get('in_flight_at_entry', 0))
    retry_path = out / 'retry_audit.jsonl'
    for line in retry_path.read_text().splitlines() if retry_path.exists() else []:
        row = json.loads(line)
        if row.get('case_id') in case_ids and row.get('event') == 'retry':
            by_role[str(row.get('role'))]['retry_events'] += 1
    return {key: {**dict(value), 'error_rate': fraction(value['errors'], value['actual_http_attempts'])}
            for key, value in by_role.items()}


def error_report(batchdir):
    """Cache failures can be nested: label them operations, not API requests."""
    result = Counter()
    for path in Path(batchdir).glob('case_*/calls.errors.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            result[row['kind']] += 1
    return dict(result)
