"""Regression diagnostic (no tuning): replay the v2.1 admission flow (locus-aware generation -> source-self-applicability ->
one condition-only repair -> existing quality gate) on audited source-No-Match rubrics of a v2 run. Read-only for the run;
new calls go to --cache. Usage: source_consistency_replay.py --run RUN_DIR --audit AUDIT.json --cache X.jsonl --out Y.json"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts/online_rl'))


def main():
    a = argparse.ArgumentParser(); a.add_argument('--run', required=True); a.add_argument('--audit', required=True); a.add_argument('--cache', required=True)
    a.add_argument('--out', required=True); a.add_argument('--config', default='configs/online_rl/tree_v21_witness.json'); x = a.parse_args()
    import run_tree as RT
    from src.online_rl.cases import load_cases
    from src.online_rl.integrated.records import from_dict
    from src.online_rl.tree import posthoc as PH, rubric_bank as RB, witness as W
    from src.online_rl.tree.backends import api_rubric_generator, generation_settings, posthoc_gates, two_stage_structured_scorer
    from src.online_rl.tree.cache import CallCache
    from src.online_rl.tree.records import DoctorFeedback, Tree
    cfg = json.loads((ROOT / x.config).read_text()); key = RT.load_key(); jm = cfg['judges']; text = lambda rel: (ROOT / rel).read_text()
    cases = {c['case_id']: c for c in load_cases(ROOT / cfg['data']['train'])}
    gen = api_rubric_generator(key, cfg['rubric']['generator_model'], generation_settings(cfg['rubric']))
    scorer = two_stage_structured_scorer(RT.judge_client(jm['scorer_model'], key, text(jm['scorer_prompt_file'])))
    gate = posthoc_gates(RT.judge_client(jm['gate_model'], key, text(jm['gate_prompt_file'])), cases)
    prompt = text(cfg['rubric']['generator_prompt_file']); create_text, _ = RB.prompt_sections(prompt); sha = cfg['rubric']['prompt_sha256']
    cache = CallCache(Path(x.cache)); rows = []
    for item in json.loads(Path(x.audit).read_text()):
        rec = json.loads((Path(x.run) / 'cases' / f"case_{item['case'] - 1:06d}.json").read_text()); case = cases[rec['case_id']]
        tree = from_dict(Tree, rec['tree']); nodes = {n.node_id: n for n in tree.nodes}; leaf = next(l for l in tree.leaves if l.leaf_id == item['leaf'])
        f = next(z for z in rec['feedback'] if z['leaf_id'] == item['leaf'])
        fb = DoctorFeedback(f['leaf_id'], f['outcome'], f['text'], f['prompt_sha256'], f['target_reply_id'], f.get('locus'), f.get('valid', True), f.get('kind'))
        tin = PH.freeze_trajectory_input(case, leaf, fb, include_locus=True)
        ssf = W.source_stage1_for(case, nodes, leaf, tin.target_reply_id, scorer.applicability, cache)
        r = PH.generate_rubric(create_text, sha, tin, gen, cache, gate, ssf)
        sc = r.source_check or {}
        rows.append({'case': item['case'], 'v2_rubric_id': item['rid'], 'leaf': item['leaf'], 'locus': tin.target_reply_id,
                     'original_v2_rubric': item['rub'], 'original_v2_stage1': item['stage1'],
                     'v21_generated': {k: (sc.get('original_rubric') or r.rubric or {}).get(k) for k in ('name', 'applicability_condition', 'required_behavior')},
                     'v21_first_stage1': (sc.get('first') or {}).get('stage1'), 'repaired_condition': r.rubric.get('applicability_condition') if sc.get('status') == 'REPAIRED_PASS' else
                     ((sc.get('repaired_rubric') or {}).get('applicability_condition') if sc.get('status') == 'SOURCE_INCONSISTENT' else None),
                     'v21_second_stage1': (sc.get('second') or {}).get('stage1'), 'source_status': sc.get('status'),
                     'gate': None if r.gate is None else {'passed': r.gate.get('passed'), 'reasons': r.gate.get('reasons')}, 'valid': r.valid, 'problems': list(r.problems)})
        print(json.dumps({k: rows[-1][k] for k in ('case', 'v2_rubric_id', 'locus', 'source_status', 'gate', 'valid')}, ensure_ascii=False), flush=True)
    Path(x.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == '__main__': main()
