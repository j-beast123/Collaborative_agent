"""Read-only snapshot of current original trajectories, candidates and recorded scores."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def dialogue(turns, leaf_id=None):
    branch = leaf_id.split('/', 1)[1].split('.') if leaf_id else []
    case_id = leaf_id.split('/', 1)[0] if leaf_id else ''
    result, number = [], 0
    for speaker, stage, turn, text in turns:
        node = None
        if speaker == 'agent':
            number += 1
            if branch:
                node = case_id + '/' + '.'.join(branch[:number])
        result.append({'speaker': speaker, 'stage': stage, 'turn': turn, 'text': text, 'node_id': node})
    return result


def reconstruct(calls, source):
    """Recover original completed paths only, without executing the pipeline."""
    case_id = source['case_id']
    agents = {r['identity']['path']: r['value'] for r in calls if r['kind'] == 'agent'}
    doctors = {r['identity']['path']: r['value']['visible'] for r in calls if r['kind'] == 'doctor_turn'}
    feedback = {r['identity']['leaf']: r for r in calls if r['kind'] == 'feedback'}
    nodes, leaves = {}, []
    for record in calls:
        if record['kind'] != 'final':
            continue
        path, turns = record['identity']['path'], []
        parts = path.split('.')
        for i in range(len(parts)):
            before, here = '.'.join(parts[:i]), '.'.join(parts[:i + 1])
            assert before in doctors and here in agents, (case_id, path, before, here)
            turns.append(['doctor', f'D{i+1}', i+1, doctors[before]])
            node_id = case_id + '/' + here
            nodes[node_id] = {'node_id': node_id, 'depth': i+1, 'prefix': list(turns),
                              'response': agents[here]['completion_text']}
            turns.append(['agent', f'A{i+1}', i+1, agents[here]['completion_text']])
        final = record['value']
        turns.append(['doctor', 'FINAL', 0, final])
        leaf_id = case_id + '/' + path
        fb = feedback.get(leaf_id, {})
        option = re.search(r'Final answer:\s*option\s+([A-Z])', final, re.I)
        leaves.append({'leaf_id': leaf_id, 'dialogue': turns,
                       'outcome': {'success': 1.0, 'failure': 0.0}.get(fb.get('identity', {}).get('outcome')),
                       'final_option': option.group(1).upper() if option else None})
    return {'nodes': list(nodes.values()), 'leaves': sorted(leaves, key=lambda r: r['leaf_id'])}


def csv_dump(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def export(run, out, through_batch=None, data_only=False):
    if out.exists():
        raise ValueError('Choose a new snapshot directory')
    out.mkdir(parents=True)
    now = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    inventory = []
    paths = [run/'run_manifest.json', run/'run_state.json', run/'api_usage.jsonl']
    for batch in sorted((run/'batches').glob('batch_*')):
        if through_batch is not None and int(batch.name.split('_')[1]) > through_batch:
            continue
        paths += [p for p in batch.glob('*.json') if p.is_file()]
        for case in sorted(batch.glob('case_*')):
            paths += [case/name for name in ('calls.jsonl', 'calls.errors.jsonl', 'result.json', 'run_state.json', 'run_manifest.json')]
            paths += list((case/'cases').glob('case_*.json'))
    for src in sorted(set(paths)):
        if not src.is_file():
            continue
        raw = src.read_bytes()
        captured = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
        omitted = 0
        if src.suffix == '.jsonl' and raw and not raw.endswith(b'\n'):
            end = raw.rfind(b'\n') + 1
            omitted = len(raw) - end
            raw = raw[:end]
        dest = out/'raw'/src.relative_to(run)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        inventory.append({'path': str(dest.relative_to(out)), 'source': str(src),
                          'captured_at_kst': captured, 'size_bytes': len(raw),
                          'sha256': hashlib.sha256(raw).hexdigest(), 'incomplete_tail_bytes_omitted': omitted})
    snap = out/'raw'
    manifest = read(snap/'run_manifest.json')
    sources = {c['case_id']: c for c in lines(ROOT/manifest['config']['data']['train'])}
    cases = []
    for batch in sorted((snap/'batches').glob('batch_*')):
        bm = read(batch/'manifest.json')
        for case_dir in sorted(batch.glob('case_*')):
            cursor = int(case_dir.name.split('_')[1])
            assignment_ids = {row['index']: cid for row, cid in zip(bm['assignments'], bm['case_ids'])}
            case_id = assignment_ids[cursor]
            source = sources[case_id]
            calls = lines(case_dir/'calls.jsonl')
            completed_paths = sorted((case_dir/'cases').glob('case_*.json'))
            completed = read(completed_paths[-1]) if completed_paths else None
            tree = completed['tree'] if completed else reconstruct(calls, source)
            fb = {r['identity']['leaf']: r['value'] for r in calls if r['kind'] == 'feedback'}
            if completed:
                tins = {t['leaf_id']: t for t in completed['trajectory_inputs']}
            else:
                tins = {}
            trajectories = []
            for leaf in tree['leaves']:
                lid = leaf['leaf_id']
                reflection = fb.get(lid, {}).get('reflection')
                if reflection is None:
                    reflection = fb.get(lid, {}).get('text') or tins.get(lid, {}).get('feedback')
                trajectories.append({'trajectory_id': lid,
                    'outcome': {1.0: 'success', 0.0: 'failure'}.get(leaf.get('outcome'), 'unknown'),
                    'final_option': leaf.get('final_option'), 'dialogue': dialogue(leaf['dialogue'], lid),
                    'reflection': reflection, 'source': 'saved_case' if completed else 'reconstructed_from_original_call_cache'})
            gate_records = {r['identity']['tin']: r['value'] for r in calls if r['kind'] == 'posthoc_gate'}
            wrappers = {r['rubric_id']: r for r in (completed or {}).get('trajectory_rubrics', [])}
            witness = {r['rubric_id']: r for r in (completed or {}).get('rubric_bank', {}).get('witness', [])}
            generated = []
            for call in calls:
                if call['kind'] != 'rubric_generation':
                    continue
                rid = call['identity']['prompt'][:12] + '-' + call['identity']['tin'][:12]
                g = call['value']
                wrap = wrappers.get(rid, {})
                rubric = wrap.get('rubric') or g.get('rubric') or {}
                provenance = rubric.get('source_provenance') or {}
                analysis = provenance.get('analysis') or {}
                raw_obj = {}
                if not rubric and g.get('raw'):
                    try:
                        raw_obj = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', g['raw'].strip()))
                    except (ValueError, TypeError):
                        pass
                candidate = rubric or raw_obj.get('candidate_rubric') or {}
                gate = wrap.get('gate') if wrap else gate_records.get(call['identity']['tin'])
                check = wrap.get('source_check') or {}
                w = witness.get(rid)
                if w:
                    status = w['decision']
                elif g.get('parse_error'):
                    status = 'GENERATION_INVALID'
                elif check.get('status') == 'SOURCE_INCONSISTENT':
                    status = 'SOURCE_INCONSISTENT'
                elif gate is not None and gate.get('passed') is False:
                    status = 'GATE_REJECTED'
                elif gate is not None and gate.get('passed') is True:
                    status = 'VERIFICATION_PENDING'
                else:
                    status = 'ADMISSION_PENDING' if not completed else 'NOT_ADMITTED'
                generated.append({'rubric_id': rid,
                    'source_trajectory_id': wrap.get('leaf_id') or provenance.get('source_trajectory_id') or raw_obj.get('source_trajectory_id'),
                    'selected_agent_turn_id': analysis.get('selected_agent_turn_id') or raw_obj.get('analysis', {}).get('selected_agent_turn_id'),
                    'source_outcome': provenance.get('source_outcome') or raw_obj.get('source_outcome'),
                    'name': candidate.get('name', '(invalid generation)'),
                    'required_behavior': candidate.get('required_behavior'),
                    'applicability': candidate.get('applicability', {}), 'scores': candidate.get('scores', {}),
                    'status': status, 'problems': wrap.get('problems', []) + ([g['parse_error']] if g.get('parse_error') else []),
                    'gate': gate or {}, 'source_check': check, 'witness': w,
                    'intervention_instruction': wrap.get('intervention_instruction') or g.get('intervention_instruction'),
                    'generator_source_score': analysis.get('source_score'),
                    'source_independent_gate_score': (gate or {}).get('assessment', {}).get('source_level'),
                    'analysis': analysis, 'call_cache_key': call['key'],
                    'raw_record_file': str((case_dir/'calls.jsonl').relative_to(out))})
            nodes = {n['node_id']: n for n in tree['nodes']}
            all_rids = {r['rubric_id']: r for r in generated}
            scores = []
            for score in (completed or {}).get('response_scores', []):
                n = nodes[score['response_node_id']]
                evidence = score.get('scoring_evidence') or {}
                second = evidence.get('stage2') or {}
                first = (evidence.get('applicability') or {}).get('stage1') or {}
                if not score['match']:
                    level = 'NA'
                    assert score['score'] is None
                elif score.get('scoring_status') == 'unscorable' or second.get('level') == 'UNSCORABLE':
                    level = 'UNSCORABLE'
                else:
                    level = second.get('level')
                    if level is None and score.get('score') is not None:
                        level = str(int(round(2 * score['score'])))
                    assert level in ('0', '1', '2'), score
                    assert abs(float(level)/2 - score['score']) < 1e-9
                assert all_rids[score['rubric_id']]['status'] == 'KEEP'
                scores.append({'response_node_id': score['response_node_id'], 'rubric_id': score['rubric_id'],
                    'score_label': level, 'normalized_score': score['score'], 'scoring_status': score.get('scoring_status'),
                    'applicable': score['match'],
                    'rationale': second.get('anchor_basis') or first.get('missing_requirement') or first.get('evidence_basis') or '',
                    'evidence': evidence, 'response': n['response'], 'prefix': dialogue(n['prefix']),
                    'agent_turn': n['depth'], 'source_trajectory_id': score.get('source_leaf_id'),
                    'containing_trajectories': [t['trajectory_id'] for t in trajectories
                        if any(d['node_id'] == n['node_id'] for d in t['dialogue'])]})
            counts = {'trajectories': len(trajectories), 'responses': len(nodes), 'generated': len(generated),
                      'gate_pass': sum(r['gate'].get('passed') is True for r in generated),
                      'gate_reject': sum(r['gate'].get('passed') is False for r in generated),
                      'kept': sum(r['status'] == 'KEEP' for r in generated),
                      'score_records': len(scores), 'scored': sum(r['score_label'] in ('0','1','2') for r in scores),
                      'distribution': dict(Counter(r['score_label'] for r in scores)),
                      'status_distribution': dict(Counter(r['status'] for r in generated))}
            cases.append({'case_number': cursor+1, 'case_id': case_id, 'batch': batch.name,
                          'scenario': source.get('scenario',''), 'question': source['question'],
                          'options': source['options'], 'status': 'POSTPROCESS_COMPLETE' if completed else 'POSTPROCESS_PENDING',
                          'counts': counts, 'trajectories': trajectories, 'rubrics': generated, 'scores': scores})
    summary = {'snapshot_kst': now, 'source_run': str(run), 'through_batch': through_batch,
               'run_state': read(snap/'run_state.json'),
               'counts': {'cases': len(cases), 'postprocess_complete': sum(c['status']=='POSTPROCESS_COMPLETE' for c in cases),
                          **{k:sum(c['counts'][k] for c in cases) for k in ('trajectories','responses','generated','gate_pass','gate_reject','kept','score_records','scored')}},
               'score_distribution': dict(Counter(s['score_label'] for c in cases for s in c['scores'])),
               'notes': ['원문은 요약·번역하지 않고 그대로 보존했습니다. 읽기 화면에는 시스템 프롬프트와 토큰 배열을 숨겼습니다.',
                         '학습 채점은 확정 KEEP 루브릭에 대한 저장된 response_scores만 표시합니다. Gate 원본 점수·생성기 의도 점수와 구분합니다.',
                         'NA는 미적용이며 0점이 아닙니다. 표시 점수는 0/1/2, 내부 정규화 보상은 0/0.5/1입니다.',
                         '후처리 대기 항목은 스냅샷 시점까지 저장된 기록만 포함합니다. 새 API·리런·채점 호출은 하지 않았습니다.',
                         '동일 Agent 노드가 여러 궤적에 공유됩니다. 학습 점수 집계는 응답 노드×루브릭 기준으로 중복 제거했습니다.',
                         '서로 다른 파일은 순차로 읽은 스냅샷입니다. 파일별 저장 시각·해시와 잘린 미완성 JSONL 꼬리 바이트 수는 snapshot_manifest.json에 기록합니다.']}
    data = {'summary': summary, 'cases': cases}
    dump(out/'review.json', data)
    dump(out/'summary.json', summary)
    dump(out/'snapshot_manifest.json', inventory)
    dump(out/'original_trajectories.json', [{'case_number':c['case_number'],'case_id':c['case_id'],
        'question':c['question'],'options':c['options'],**t} for c in cases for t in c['trajectories']])
    dump(out/'rubrics.json', [{'case_number':c['case_number'],'case_id':c['case_id'],**r} for c in cases for r in c['rubrics']])
    score_rows = [{'case_number':c['case_number'],'case_id':c['case_id'],**r} for c in cases for r in c['scores']]
    dump(out/'response_scores.json', score_rows)
    csv_dump(out/'response_scores.csv', score_rows)
    csv_dump(out/'case_summary.csv', [{'case_number':c['case_number'],'case_id':c['case_id'],'status':c['status'],**c['counts']} for c in cases])
    dump(out/'source_cases.json', [sources[c['case_id']] for c in cases])
    if data_only:
        print(json.dumps({'output_dir':str(out),'status':'snapshot_data_saved','summary':summary},ensure_ascii=False,indent=2))
        return
    from scripts.online_rl.render_training_review import render_review
    (out/'review.html').write_text(render_review(data))
    bundle = out.with_suffix('.zip')
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                archive.write(path, str(path.relative_to(out)))
    print(json.dumps({'output_dir':str(out),'zip':str(bundle),'summary':summary},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--out-dir')
    parser.add_argument('--through-batch', type=int)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.out_dir) if args.out_dir else ROOT/'outputs/exports'/('max2_review_'+datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_KST'))
    export(Path(args.run_dir).resolve(), output.resolve(), args.through_batch, args.data_only)
