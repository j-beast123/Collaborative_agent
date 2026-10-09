"""Compare the DEV-only optimization arms (read-only): training learning health from each arm's case records and DEV behaviour
from its DEV evaluation (theta0 = checkpoint_000000 vs the final checkpoint, same DEV cases, seeds and Doctor openings).
CasiMedicos622 is never read.

    python scripts/dev_opt/compare_arms.py --arm BASE=<run_dir>:<dev_eval_root> [--arm ...] --out <json>

Primary (learning health): option-diverse groups, outcome-varying groups, correct+incorrect siblings, R_A available, zero-kept
cases, zero-advantage / contributing groups, mean |advantage|, KL to theta0, MC target-hit rate, candidate exhaustion, kept
rubrics, optimizer steps, behaviour change on DEV. Secondary (DEV): correctness, Recovery, Preservation, harm, rescue."""
from __future__ import annotations
import argparse, json, re, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.collaboration.generator import option          # noqa: E402
from src.online_rl.tree.reward import _AGREE, _CHALLENGE  # noqa: E402  the run's own interaction heuristics

EPS = 1e-9
OPENING = re.compile(r'answer is \(?([A-D])\b|\b([A-D])[).:]\s', re.I)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 4) if xs else None


def training_health(run: Path) -> dict:
    groups, cases, batches = [], [], {}
    for f in sorted(run.glob('cases/case_*.json')):
        c = json.load(open(f)); t = c['agent_targets']; nodes = {n['node_id']: n for n in c['tree']['nodes']}
        for key, ids in t['groups'].items():
            rs = [t['nodes'][i] for i in ids]; rew = [r['reward'] for r in rs if r['reward'] is not None]
            o = [r['outcome_component'] for r in rs]
            m = st.mean(rew) if rew else 0.0
            groups.append({'option_diverse': len({x for x in (option(nodes[i]['response']) for i in ids) if x}) > 1,
                           'outcome_varies': max(o) - min(o) > EPS, 'correct_and_incorrect': 1.0 in o and 0.0 in o,
                           'R_A_available': sum(r['R_A'] is not None for r in rs) / len(rs),
                           'contributing': len(rew) >= 2 and max(rew) - min(rew) > EPS,
                           'abs_adv': [abs(x - m) for x in rew] if len(rew) >= 2 and max(rew) - min(rew) > EPS else []})
        wit = c['rubric_bank'].get('witness') or []
        tested = [x for w in wit for x in w.get('tested') or []]
        cases.append({'kept': len(c.get('training_rubrics') or []), 'rubrics': len(wit),
                      'exhausted': sum(w['stop_reason'] == 'candidates_exhausted' for w in wit),
                      'reruns': sum(x.get('reruns', 0) for x in tested), 'hits': sum(x.get('positive', 0) for x in tested),
                      'leaf_accuracy': c['interaction_diagnostics'].get('leaf_accuracy')})
        bu = c.get('batch_update') or {}
        batches[c['case_cursor'] // 5] = {'step': bool(bu.get('optimizer_step')), 'kl': bu.get('kl_loss'), 'grad_norm': bu.get('grad_norm'),
                                          'gap': (bu.get('ratio') or {}).get('max_abs_logprob_gap_to_reference')}
    prompt = [json.load(open(p))['decision'] for p in sorted(run.glob('prompt_events/batch_*.json'))]
    n = len(groups); rr = sum(x['reruns'] for x in cases)
    kls = [b['kl'] for _, b in sorted(batches.items()) if b['kl'] is not None]
    return {'cases': len(cases), 'groups': n,
            'option_diverse_groups': mean([g['option_diverse'] for g in groups]),
            'outcome_varying_groups': mean([g['outcome_varies'] for g in groups]),
            'correct_and_incorrect_siblings': mean([g['correct_and_incorrect'] for g in groups]),
            'R_A_available': mean([g['R_A_available'] for g in groups]),
            'zero_kept_cases': mean([x['kept'] == 0 for x in cases]),
            'zero_advantage_groups': mean([not g['contributing'] for g in groups]),
            'contributing_groups': mean([g['contributing'] for g in groups]),
            'mean_abs_advantage': mean([a for g in groups for a in g['abs_adv']]),
            'mc_target_hit_rate': round(sum(x['hits'] for x in cases) / rr, 4) if rr else None,
            'candidate_exhaustion_rate': round(sum(x['exhausted'] for x in cases) / max(1, sum(x['rubrics'] for x in cases)), 4),
            'kept_rubrics_per_case': mean([x['kept'] for x in cases]),
            'training_leaf_accuracy': mean([x['leaf_accuracy'] for x in cases]),
            'optimizer_steps': sum(b['step'] for b in batches.values()), 'batches': len(batches),
            'kl_to_theta0_last_update': kls[-1] if kls else None, 'kl_to_theta0_by_update': kls,
            'grad_norm_mean': mean([b['grad_norm'] for b in batches.values()]),
            'prompt_opt_record_only': {'proposals': len(prompt), 'adoptions': sum(bool(d.get('accepted')) for d in prompt),
                                       'J_zero_parent': sum((d.get('parent') or {}).get('J') == 0 for d in prompt)}}


def eval_rows(root: Path, name: str) -> dict:
    job = root / name
    if not (job / 'cases').exists(): return {}
    return {r['case_id']: r for r in (json.load(open(p)) for p in sorted((job / 'cases').glob('case_*.json')))}


def dev_behavior(rows: dict, gold: dict) -> dict:
    pres, rec, n_ok, agree, chal, quest, length, rep, degen, replies = [], [], [], 0, 0, 0, [], 0, 0, 0
    for cid, r in rows.items():
        d = r['dialogue']; first = next((t for t in d if t.get('speaker') == 'doctor'), None)
        m = OPENING.search(first['text']) if first else None
        init = next((g for g in (m.groups() if m else ()) if g), None)
        ok = r['outcome'] == 1.0; n_ok.append(ok)
        if init: (pres if init == gold[cid] else rec).append(ok)
        prev = None
        for t in d:
            if t.get('speaker') != 'agent': continue
            x = t['text']; replies += 1; length.append(len(x.split()))
            agree += bool(_AGREE.search(x[:200])); chal += bool(_CHALLENGE.search(x)); quest += '?' in x
            rep += prev is not None and x.strip() == prev.strip(); degen += len(re.findall(r'[A-Za-z]', x)) < 20; prev = x
    return {'cases': len(rows), 'correctness': mean(n_ok), 'preservation': mean(pres), 'recovery': mean(rec),
            'initial_correct_to_final_wrong': round(1 - st.mean(pres), 4) if pres else None,
            'initial_wrong_to_final_correct': mean(rec), 'n_initial_correct': len(pres), 'n_initial_wrong': len(rec),
            'agent_agreement_rate': round(agree / replies, 4) if replies else None, 'agent_challenge_rate': round(chal / replies, 4) if replies else None,
            'agent_question_rate': round(quest / replies, 4) if replies else None, 'agent_reply_words': mean(length),
            'repeated_reply_rate': round(rep / replies, 4) if replies else None, 'degenerate_reply_rate': round(degen / replies, 4) if replies else None}


def behavior_change(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    if not common: return {}
    final = [a[c]['final_option'] != b[c]['final_option'] for c in common]
    first = [next((t['text'] for t in a[c]['dialogue'] if t.get('speaker') == 'agent'), '') !=
             next((t['text'] for t in b[c]['dialogue'] if t.get('speaker') == 'agent'), '') for c in common]
    flips = [(a[c]['outcome'], b[c]['outcome']) for c in common]
    return {'paired_cases': len(common), 'final_option_changed': mean(final), 'first_agent_reply_changed': mean(first),
            'wrong_to_right': sum(x == 0.0 and y == 1.0 for x, y in flips), 'right_to_wrong': sum(x == 1.0 and y == 0.0 for x, y in flips)}


def main() -> None:
    a = argparse.ArgumentParser(); a.add_argument('--arm', action='append', required=True); a.add_argument('--out', required=True)
    a.add_argument('--dev-file', default='data/interim/medmcqa/five_gpu_train500/dev100_order400_499.jsonl')
    x = a.parse_args()
    gold = {d['case_id']: d['ground_truth']['correct_option'] for d in map(json.loads, open(ROOT / x.dev_file))}
    table = {}
    for spec in x.arm:
        name, paths = spec.split('=', 1); run, ev = (Path(p) for p in paths.split(':'))
        cases = json.load(open(run / 'run_state.json'))['case_cursor'] if (run / 'run_state.json').exists() else None
        theta0, final = eval_rows(ev, 'checkpoint_000000'), eval_rows(ev, f'checkpoint_{cases:06d}') if cases else {}
        table[name] = {'run': str(run), 'training': training_health(run),
                       'dev_theta0': dev_behavior(theta0, gold) if theta0 else None, 'dev_final': dev_behavior(final, gold) if final else None,
                       'dev_behavior_change_theta0_to_final': behavior_change(theta0, final)}
    Path(x.out).write_text(json.dumps(table, indent=1))
    rows = [('sibling option diversity', 'training', 'option_diverse_groups'), ('outcome-varying %', 'training', 'outcome_varying_groups'),
            ('correct+incorrect siblings', 'training', 'correct_and_incorrect_siblings'), ('R_A available %', 'training', 'R_A_available'),
            ('zero-kept-rubric cases', 'training', 'zero_kept_cases'), ('zero-advantage %', 'training', 'zero_advantage_groups'),
            ('contributing %', 'training', 'contributing_groups'), ('mean |advantage|', 'training', 'mean_abs_advantage'),
            ('MC target-hit rate', 'training', 'mc_target_hit_rate'), ('candidate exhaustion', 'training', 'candidate_exhaustion_rate'),
            ('kept rubrics / case', 'training', 'kept_rubrics_per_case'), ('optimizer steps', 'training', 'optimizer_steps'),
            ('KL (last update)', 'training', 'kl_to_theta0_last_update'), ('DEV correctness', 'dev_final', 'correctness'),
            ('Recovery', 'dev_final', 'recovery'), ('Preservation', 'dev_final', 'preservation'),
            ('init-correct -> final-wrong', 'dev_final', 'initial_correct_to_final_wrong'),
            ('init-wrong -> final-correct', 'dev_final', 'initial_wrong_to_final_correct'),
            ('DEV final option changed vs theta0', 'dev_behavior_change_theta0_to_final', 'final_option_changed')]
    names = list(table)
    print('| metric | ' + ' | '.join(names) + ' |\n|---|' + '---|' * len(names))
    for label, sec, key in rows:
        print(f'| {label} | ' + ' | '.join(str((table[n][sec] or {}).get(key)) for n in names) + ' |')
    t0 = next((table[n]['dev_theta0'] for n in names if table[n]['dev_theta0']), None)
    if t0: print(f"\ntheta0 DEV: correctness {t0['correctness']} recovery {t0['recovery']} preservation {t0['preservation']}")


if __name__ == '__main__':
    main()
