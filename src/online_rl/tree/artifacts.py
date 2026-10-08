"""Reviewable exports derived from immutable case/event records; no model or API calls."""
from pathlib import Path
import difflib
import csv
import json
from src.online_rl.integrated.records import atomic_write_json, sha256_json


def read(path):
    return json.loads(path.read_text())


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    tmp.replace(path)


def export_run(run_dir):
    root = Path(run_dir).resolve()
    if not (root / 'run_manifest.json').exists():
        raise ValueError('Not a tree run: run_manifest.json missing')
    dest = root / 'review'; dest.mkdir(exist_ok=True)
    manifest = read(root / 'run_manifest.json')
    atomic_write_json(dest / 'config.json', manifest)
    cases = [read(p) for p in sorted((root / 'cases').glob('case_*.json'))]
    events = [read(p) for p in sorted((root / 'prompt_events').glob('event_*.json'))]
    score_rows, feedback_rows, versions, acceptances, diffs = [], [], [], [], []
    prompt_names = {manifest['initial_prompt_sha256']: manifest['config']['rubric']['prompt_id']}
    for event in events:
        event_id = event['event']
        common = {'event': event_id, 'policy_sha256': event.get('rerun_policy_sha256')}
        score_rows.append({**common, 'prompt_id': event['parent_prompt_id'], 'prompt_sha256': event['parent_sha256'],
                           'split': 'search', **(event.get('parent_search_objective') or {'J': None, 'status': 'not_evaluated'})})
        feedback_rows.extend({**common, **row} for row in event.get('reflections', []))
        for version in event.get('versions', []):
            versions.append(version)
            sha = version['sha256']
            if sha:
                prompt_names[sha] = version['prompt_id']
            score_rows.append({**common, 'prompt_id': version['prompt_id'], 'prompt_sha256': sha,
                               'split': 'search', **(version['metrics'].get('objective') or {'J': None, 'status':'not_evaluated'})})
            parent_sha = next((key for key,value in prompt_names.items() if value == version.get('parent_prompt_id')), event['parent_sha256'])
            parent = root / 'prompts' / (parent_sha + '.txt')
            candidate = root / 'prompts' / (sha + '.txt')
            if parent.exists() and candidate.exists():
                diffs.append(''.join(difflib.unified_diff(parent.read_text().splitlines(True), candidate.read_text().splitlines(True),
                                                        fromfile=version.get('parent_prompt_id') or event['parent_prompt_id'], tofile=version['prompt_id'])))
        decision = event['decision']; acc = decision.get('acceptance') or {}
        acceptances.append({**common, **decision})
        for prefix, pid, sha in [('parent', event['parent_prompt_id'],event['parent_sha256']),
                                 ('candidate',decision.get('candidate_prompt_id'),decision.get('candidate_sha256'))]:
            if acc.get(prefix + '_objective'):
                score_rows.append({**common,'prompt_id':pid,'prompt_sha256':sha,'split':'acceptance', **acc[prefix+'_objective']})
    heldout = read(root/'heldout/result.json') if (root/'heldout/result.json').is_file() else None
    if heldout:
        score_rows.extend(dict(row,event='heldout') for row in heldout['prompt_scores'])
        target=dest/'heldout';target.mkdir(exist_ok=True)
        atomic_write_json(target/'comparison.json',heldout)
        jsonl(target/'trajectories.jsonl',heldout['tree']['leaves'])
        jsonl(target/'doctor_feedback.jsonl',heldout['doctor_feedback'])
    for sha, pid in prompt_names.items():
        source = root / 'prompts' / (sha + '.txt')
        if source.exists():
            target = dest / 'prompts' / (sha + '.txt'); target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text())
    if events:
        first = next((e for e in events if e['decision'].get('candidate_sha256')), None)
        if first:
            candidate_sha=first['decision']['candidate_sha256']
            source=root/'prompts'/(candidate_sha+'.txt')
            if source.exists():
                (dest/'prompts/P1.txt').write_text(source.read_text())
                atomic_write_json(dest/'prompts/P1_provenance.json',
                    {'alias':'P1','prompt_id':first['decision']['candidate_prompt_id'],
                     'sha256':candidate_sha,'event':first['event'],'accepted':first['decision']['accepted'],
                     'note':'Best candidate in the first completed comparison; existence does not imply acceptance.'})
    jsonl(dest / 'prompts/index.jsonl', [{'prompt_id':pid,'sha256':sha,'file':sha+'.txt'} for sha,pid in prompt_names.items()])
    (dest / 'prompts/prompt_diff.md').write_text('# Prompt changes\n\n' + ''.join('```diff\n'+diff+'\n```\n\n' for diff in diffs))
    dataset_path = Path(manifest['config'].get('data', {}).get('train', ''))
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parents[3] / dataset_path
    dataset = {r['case_id']:r for r in (json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip())} if dataset_path.is_file() else {}
    for case in cases:
        folder = dest / 'cases' / str(case['case_cursor'])
        jsonl(folder / 'trajectories.jsonl', case['tree']['leaves'])
        jsonl(folder / 'agent_beliefs.jsonl', [{'agent_turn_id':n['node_id'], **n['trace']['pre_response_belief']} for n in case['tree']['nodes'] if n['trace'].get('pre_response_belief')])
        jsonl(folder / 'doctor_feedback.jsonl', case['feedback'])
        folder.mkdir(parents=True,exist_ok=True)
        if case['case_id'] in dataset:
            atomic_write_json(folder / 'case.json', dataset[case['case_id']])
        atomic_write_json(folder / 'index.json', {k:case[k] for k in ('case_id','prompt_id','prompt_sha256','policy_sha256')})
        jsonl(folder / 'rubrics.jsonl', case['trajectory_rubrics'])
        retained = {r['rubric_id'] for r in case['selection'] if r['retained']}
        jsonl(folder / 'kept_rubrics.jsonl', [r for r in case['trajectory_rubrics'] if r['rubric_id'] in retained])
        jsonl(folder / 'rubric_selection.jsonl', case['selection'])
        jsonl(folder / 'response_scores.jsonl', case['response_scores'])
        jsonl(folder / 'reruns.jsonl', [dict(r,case_id=case['case_id']) for e in case['effectiveness'] for r in e['runs']])
    # Every candidate's evidence is written when evaluated, not only the winner's.
    for folder in sorted((root / 'evaluations').glob('*/*')):
        if not folder.is_dir(): continue
        records = [read(p) for p in sorted(folder.glob('*.json'))]
        target = dest / 'evaluations' / folder.parent.name / folder.name
        jsonl(target / 'rubrics.jsonl', [{k:r[k] for k in ('evidence_id','case_id','source_outcome','rubric')} for r in records])
        jsonl(target / 'reruns.jsonl', [dict(run,evidence_id=r['evidence_id'],case_id=r['case_id'])
              for r in records if r['effectiveness'] for run in r['effectiveness']['runs']])
        jsonl(target / 'effectiveness.jsonl', records)
        jsonl(target / 'response_scores.jsonl', [dict(score,evidence_id=r['evidence_id'],case_id=r['case_id']) for r in records for score in r.get('source_response_scores',[])])
        atomic_write_json(target / 'effectiveness.json', {'prompt_sha256':folder.parent.name,
            'scores':[row for row in score_rows if row.get('prompt_sha256') == folder.parent.name
                      and sha256_json(row.get('policy_sha256')) == folder.name],
            'evidence_count':len(records), 'note':'Search and acceptance scores remain separate.'})
    jsonl(dest / 'prompt_scores.jsonl', score_rows)
    with (dest / 'prompt_scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['event','prompt_id','prompt_sha256','policy_sha256','split','S_success','S_failure','J','status'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(score_rows)
    jsonl(dest / 'prompt_optimization/feedback.jsonl', feedback_rows)
    jsonl(dest / 'prompt_optimization/candidates.jsonl', versions)
    atomic_write_json(dest / 'prompt_optimization/acceptance.json', acceptances)
    total_runs = [r for c in cases for e in c['effectiveness'] for r in e['runs']]
    ledger = root / 'api_usage.jsonl'
    usage = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    summary = {'cases':len(cases)+(1 if heldout else 0),'optimization_cases':len(cases),'heldout_complete':bool(heldout),'trajectories':sum(len(c['tree']['leaves']) for c in cases)+(len(heldout['tree']['leaves']) if heldout else 0),
               'generated_rubrics':sum(len(c['trajectory_rubrics']) for c in cases),
               'kept_rubrics':sum(r['retained'] for c in cases for r in c['selection']),
               'source_reruns_incomplete':sum(r['status'] != 'complete' for r in total_runs),
               'prompt_events':len(events),'accepted_updates':sum(e['decision']['accepted'] for e in events),
               'prompt_scores':score_rows, 'api_calls_recorded':len(usage),
               'reported_cost_usd':sum(r['cost_usd'] for r in usage if r.get('cost_usd') is not None),
               'calls_without_reported_cost':sum(r.get('cost_usd') is None for r in usage),
               'api_usage_file':str(ledger),'state':read(root/'run_state.json') if (root/'run_state.json').exists() else None}
    atomic_write_json(dest / 'summary.json', summary)
    def fmt(value): return '—' if value is None else f'{value:.4f}'
    lines = ['# Pilot review', '', *(['SYNTHETIC DEMO: no real model or API evaluation.', ''] if manifest['config'].get('synthetic_demo') else []), f"Cases: {summary['cases']} · Trajectories: {summary['trajectories']} · Kept rubrics: {summary['kept_rubrics']}",
             '', '## Prompt scores', '', '| Event | Prompt | Split | S_success | S_failure | J | Status |',
             '|---|---|---|---|---|---|---|']
    for row in score_rows:
        lines.append(f"| {row['event']} | {row['prompt_id']} | {row['split']} | {fmt(row.get('S_success'))} | {fmt(row.get('S_failure'))} | {fmt(row.get('J'))} | {row.get('status','')} |")
    if not score_rows: lines += ['', 'No prompt comparison has run yet.']
    lines += ['', '## Acceptance', '']
    lines += [f"- Event {e['event']}: {e['decision'].get('reason')}; accepted={e['decision']['accepted']}" for e in events]
    lines += ['', 'See [prompt changes](prompts/prompt_diff.md), [summary](summary.json), and [individual scores](prompt_scores.jsonl).',
              '', 'J measures rubric intervention effectiveness; it does not measure improvement in the Agent’s unassisted accuracy.']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    return {'review_dir':str(dest),'cases':summary['cases'],'prompt_score_records':len(score_rows)}
