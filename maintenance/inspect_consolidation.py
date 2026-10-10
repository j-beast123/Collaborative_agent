"""Read-only summary of unresolved consolidation failures; no API/model work."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.online_rl.tree.parallel_bank import membership_report
from src.online_rl.integrated.records import sha256_json


def inspect(run):
    run = Path(run)
    path = run / 'diagnostics/consolidate_validation.jsonl'
    if not path.is_file(): raise FileNotFoundError(f'No consolidation audit: {path}')
    latest = {}
    with path.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                latest[row['input_sha256']] = row
    pending = {key: row for key, row in latest.items() if not row['accepted']}
    originals = {}
    if pending:
        for cache in sorted(run.glob('batches/**/calls.jsonl')):
            with cache.open() as stream:
                for line in stream:
                    if not line.strip(): continue
                    record = json.loads(line)
                    if record.get('kind') == 'consolidate_input':
                        values = record['value']; key = sha256_json(values)
                        if key in pending: originals[key] = (values, str(cache))
    summary = []
    for key, row in sorted(pending.items()):
        original, cache = originals.get(key, (None, None))
        coverage = row.get('membership_report')
        if coverage is None and original is not None:
            try: value = json.loads(row['raw'])
            except (ValueError, TypeError): value = None
            if value is not None: coverage = membership_report(value, original)
        summary.append({'input_sha256': key, 'cache': cache, 'attempt': row['attempt'],
            'time': row.get('time'), 'validation_error': row['validation_error'],
            'membership_report': coverage,
            'note': None if coverage is not None else 'Coverage unavailable: source cache missing or response not valid JSON'})
    return {'run': str(run), 'unresolved_inputs': len(summary), 'failures': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.run), ensure_ascii=False, indent=2))
