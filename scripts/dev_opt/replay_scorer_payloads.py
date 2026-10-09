"""Offline replay of the production rubric-scorer requests (no network). For every response node of a finished run, rebuild the
exact Stage-1 / Stage-2 OpenRouter payloads with the production scorer code, feeding back that node's stored decisions, and
compare payloads across responses.

    python scripts/dev_opt/replay_scorer_payloads.py <run_dir> <train_jsonl> <out_json>

Reports (a) identical-text sibling sets whose R_A differs, with full-payload hashes per stage; (b) Stage-1 consistency inside
every sibling group (Stage 1 sees only the pre-reply dialogue, so siblings send byte-identical Stage-1 requests)."""
from __future__ import annotations
import hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.online_rl.tree.agent_state import visible_rubric          # noqa: E402
from src.online_rl.tree.backends import two_stage_structured_scorer  # noqa: E402
from src.online_rl.tree.luna import LunaDecisionClient             # noqa: E402
from src.online_rl.tree.posthoc import case_view                   # noqa: E402
from src.online_rl.tree.records import to_dialogue  # noqa: E402


def h(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def replay(cfg: dict, view: dict, prefix, rubrics: list, reply: str, stored: dict) -> list[dict]:
    """Payloads the scorer sends for this response, answering each stage with the stored decisions."""
    sent = []

    def post(url, payload, key, timeout):
        sent.append(payload); stage = json.loads(payload['messages'][-1]['content'])['stage']
        if stage == 'applicability':
            ans = {rid: stored[rid]['stage1'] for rid in stored if not rid.startswith('_')}
        else:
            ans = {rid: stored[rid]['stage2'] for rid in stored if not rid.startswith('_') and stored[rid]['stage2'] is not None}
        return {'choices': [{'message': {'content': json.dumps({'answers': ans})}}], 'model': payload['model'], 'usage': {}}

    system = (ROOT / cfg['judges']['scorer_prompt_file']).read_text()
    client = LunaDecisionClient(model=cfg['judges']['scorer_model'], key='offline', system=system, post=post)
    scorer = two_stage_structured_scorer(client, observed_context=True)
    scorer(view, prefix, rubrics, reply)
    return sent, scorer.protocol


def main(run_dir: str, train: str, out: str) -> None:
    run = Path(run_dir); cfg = json.load(open(run / 'run_manifest.json'))['config']
    data = {d['case_id']: d for d in map(json.loads, open(train))}
    report = {'identical_text_sets': [], 'stage1_groups': Counter(), 'stage1_inconsistent_examples': [], 'protocol_match': Counter()}
    for f in sorted(run.glob('cases/case_*.json')):
        c = json.load(open(f)); cur = c['case_cursor']
        calls = run / f'batches/batch_{cur // 5 + 1:06d}/case_{cur:06d}/calls.jsonl'
        if not calls.exists():
            report['cases_without_call_log'] = report.get('cases_without_call_log', 0) + 1; continue
        view = case_view(data[c['case_id']])
        by_id = {r['rubric_id']: visible_rubric(r['rubric']) for r in c['training_rubrics']}
        calls_by_set = defaultdict(dict)              # training rubrics may be scored in several calls (P0 / P1 union)
        for line in open(calls):
            d = json.loads(line)
            if d['kind'] == 'rubric_score' and set(d['identity']['rubrics']) <= set(by_id):
                calls_by_set[tuple(sorted(d['identity']['rubrics']))][d['identity']['node']] = (d['identity'], d['value'])
        nodes = {n['node_id']: n for n in c['tree']['nodes']}
        t = c['agent_targets']
        for rset, stored in calls_by_set.items():
            # score_nodes keeps the kept-rubric order; training_rubrics lists them in that order
            rubrics = [by_id[r['rubric_id']] for r in c['training_rubrics'] if r['rubric_id'] in rset]
            payloads = {}
            for nid, (ident, value) in stored.items():
                sent, proto = replay(cfg, view, to_dialogue(nodes[nid]['prefix']), rubrics, nodes[nid]['response'], value)
                report['protocol_match'][proto == ident['protocol']] += 1
                payloads[nid] = {'stage1': h(sent[0]), 'stage2': h(sent[1]) if len(sent) > 1 else None, 'n_requests': len(sent),
                                 'params': {k: v for k, v in sent[0].items() if k != 'messages'},
                                 'applicable': {rid: value[rid]['applicable'] for rid in value if not rid.startswith('_')},
                                 'levels': {rid: value[rid]['hard'] for rid in value if not rid.startswith('_')}}
            for key, ids in t['groups'].items():
                ids = [i for i in ids if i in payloads]
                if len(ids) < 2: continue
                s1 = {payloads[i]['stage1'] for i in ids}
                apps = {json.dumps(payloads[i]['applicable'], sort_keys=True) for i in ids}
                kind = 'stage1_payload_differs' if len(s1) > 1 else ('same_payload_same_decision' if len(apps) == 1 else 'same_payload_different_applicability')
                report['stage1_groups'][kind] += 1
                if kind == 'same_payload_different_applicability' and len(report['stage1_inconsistent_examples']) < 10:
                    report['stage1_inconsistent_examples'].append({'case_id': c['case_id'], 'group': key, 'stage1_payload_sha256': s1.pop(),
                                                                   'applicability_by_sibling': {i: payloads[i]['applicable'] for i in ids}})
                bytext = defaultdict(list)
                for i in ids: bytext[nodes[i]['response']].append(i)
                for same in bytext.values():
                    if len(same) < 2: continue
                    if len({json.dumps([payloads[i]['applicable'], payloads[i]['levels']], sort_keys=True) for i in same}) < 2: continue
                    report['identical_text_sets'].append({
                        'case_id': c['case_id'], 'group': key, 'rubric_call_set': rset, 'nodes': same, 'R_A': [t['nodes'][i]['R_A'] for i in same],
                        'stage1_payload_identical': len({payloads[i]['stage1'] for i in same}) == 1,
                        'stage2_payloads_identical_where_both_reached': len({payloads[i]['stage2'] for i in same if payloads[i]['stage2']}) <= 1,
                        'request_params': payloads[same[0]]['params'],
                        'per_node': {i: {k: payloads[i][k] for k in ('stage1', 'stage2', 'n_requests', 'applicable', 'levels')} for i in same}})
    report['stage1_groups'] = dict(report['stage1_groups']); report['protocol_match'] = {str(k): v for k, v in report['protocol_match'].items()}
    Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report.get(k) for k in ('stage1_groups', 'protocol_match', 'cases_without_call_log')}, indent=1))
    for s in report['identical_text_sets']:
        print(s['case_id'][-8:], s['group'][-8:], 'R_A', s['R_A'], 'stage1 identical', s['stage1_payload_identical'], 'stage2 identical', s['stage2_payloads_identical_where_both_reached'],
              {i[-6:]: (v['applicable'], v['levels']) for i, v in s['per_node'].items()})


if __name__ == '__main__':
    main(*sys.argv[1:4])
