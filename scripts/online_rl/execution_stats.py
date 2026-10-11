"""Execution-robustness statistics of one run (read-only): judge calls, local retries, EXECUTION_UNRESOLVED terminals.

    python scripts/online_rl/execution_stats.py <run output dir> <run log> [<run log> ...]

Scientific UNCERTAIN / abstain is reported on its own line and never merged with execution failures."""
import json, sys
from collections import Counter
from pathlib import Path

JUDGE_KINDS = ('rubric_state_applicability', 'rubric_behavior_batch', 'rubric_behavior', 'posthoc_gate',
               'intervention_instruction_audit', 'rubric_match_batch')


def collect(run_dir, logs):
    run_dir = Path(run_dir); stored = Counter(); terminal = {}; uncertain = Counter()
    for path in run_dir.rglob('*.jsonl'):
        name = path.name
        if name.endswith('.execution_unresolved.jsonl'):
            for line in path.read_text().splitlines():
                r = json.loads(line); terminal[(r['stage'], r.get('case_id'), r.get('node_id'), r.get('rubric_id'))] = r
        elif name.endswith('.applicability_uncertain.jsonl'):
            seen = {}
            for line in path.read_text().splitlines():
                r = json.loads(line); seen[json.dumps([r.get('case_id'), r.get('node_id'), r.get('rubric_id')])] = (r.get('audit') or r).get('final')
            uncertain.update(seen.values())
        elif not name.endswith('.errors.jsonl'):
            for line in path.read_text().splitlines():
                try: kind = json.loads(line).get('kind')
                except (ValueError, AttributeError): continue
                if kind in JUDGE_KINDS: stored[kind] += 1
    at = Counter()                                               # (stage, attempt k) -> requests that needed at least k retries
    for log in logs:
        for line in Path(log).read_text(errors='replace').splitlines():
            if '"judge_execution_retry"' not in line: continue
            try: e = json.loads(line[line.index('{'):])
            except ValueError: continue
            at[(e.get('stage'), e.get('attempt'))] += 1
    stages = sorted({s for s, _ in at})
    histogram = {s: {k: at[(s, k)] - at[(s, k + 1)] for k in sorted(a for st, a in at if st == s) if at[(s, k)] - at[(s, k + 1)]} for s in stages}
    retried = sum(at[(s, 1)] for s in stages); total = sum(stored.values()) + len(terminal)
    by_stage = Counter(k[0] for k in terminal)
    return {'judge_results_stored_by_kind': dict(stored), 'total_judge_results': total,
            'requests_needing_a_retry': retried, 'retry_rate': round(retried / total, 4) if total else None,
            'retry_count_histogram_by_stage': histogram, 'local_retry_requests_sent': sum(at.values()),
            'execution_unresolved_terminal': len(terminal), 'execution_unresolved_rate': round(len(terminal) / total, 5) if total else None,
            'execution_unresolved_by_stage': dict(by_stage),
            'scientific_uncertain_final_states (separate)': dict(uncertain)}


if __name__ == '__main__':
    print(json.dumps(collect(sys.argv[1], sys.argv[2:]), indent=1, ensure_ascii=False))
