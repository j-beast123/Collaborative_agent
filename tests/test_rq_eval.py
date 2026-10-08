"""RQ evaluation harnesses with injected judge/model callables (no model, no API)."""
import pytest
from src.rq_eval import appropriateness, downstream, fixed_probe, main_table, rq3_analysis, themes

CASES = [{'case_id': f'c{i}', 'question': f'q{i}', 'options': {'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}, 'ground_truth': {'correct_option': 'A'}, 'scenario': ''}
         for i in range(6)]

def fake_decide(state, questions):
    out = {}
    for q, spec in questions.items():
        keys = list(spec['criteria']); v = keys[len(state) % len(keys)]
        out[q] = {'value': v, 'probabilities': {k: (1.0 if k == v else 0.0) for k in keys}}
    return out


def test_themes_tidy_and_bubble():
    items = [{'rubric': {'name': f'n{i}', 'required_behavior': 'x' * i, 'scores': {'2': 'y'}}, 'policy_iteration': i % 2, 'case_id': f'c{i}', 'C': 0.5} for i in range(6)]
    rows = themes.tidy_rows(items, fake_decide, 'on_policy'); assert all(r['theme'] in themes.THEMES for r in rows)
    b = themes.bubble_data(rows); assert abs(sum(x['proportion'] for x in b if x['policy_iteration'] == 0) - 1) < 1e-9
    with pytest.raises(ValueError): themes.tidy_rows(items, fake_decide, 'keywords')


def test_fixed_probe_transitions(tmp_path):
    probes = [{'probe_id': f'p{i}', 'case_id': f'c{i}', 'prefix': [{'speaker': 'doctor', 'text': 'd'}], 'turn': 1, 'theme': 'calibrate_confidence' if i < 2 else 'other'} for i in range(4)]
    assert fixed_probe.freeze_probes(probes, tmp_path / 'probes.json') == fixed_probe.freeze_probes(probes, tmp_path / 'probes.json')
    with pytest.raises(RuntimeError): fixed_probe.freeze_probes(probes[:2], tmp_path / 'probes.json')
    cases = {c['case_id']: c for c in CASES}; rows = []
    order = ('base', 'step_000010', 'step_000020', 'step_000030')
    for j, ck in enumerate(order):
        rows += fixed_probe.evaluate_checkpoint(probes, cases, ck, lambda c, p, t, j=j: f'r{j}', lambda c, p, t, r, s, j=j: {'correct': int(j >= int(c['case_id'][1:]))}, 3, lambda pid, i: i, order)
    tr = fixed_probe.transitions_by_theme(rows, order)
    tot = {k: sum(r[k] for r in tr) for k in ('fail_to_success', 'fail_to_fail', 'success_to_fail')}
    assert tot['fail_to_success'] == 3 and tot['success_to_fail'] == 0


def test_downstream_and_appropriateness_and_tables():
    hs = [{'history_id': f'h{i}', 'case': CASES[i], 'prefix': [{'speaker': 'doctor', 'text': 'd'}], 'turn': 1} for i in range(4)]
    rows = downstream.downstream_rows(hs, lambda h: 'base', lambda h: 'model', lambda c, p, t, r, s: {'correct': int(r == 'model')}, 3, lambda h, i: i, 'integrated_evolving')
    s = downstream.downstream_summary(rows); assert s['paired_success_difference']['mean_difference'] == 1.0
    probes = [{'probe_id': f'p{i}', 'case': CASES[i], 'prefix': [{'speaker': 'doctor', 'text': 'd'}], 'turn': 1} for i in range(4)]
    ar = appropriateness.state_appropriateness_rows(probes, {'base': lambda c, p, t: 'a', 'ours': lambda c, p, t: 'bb'}, fake_decide)
    assert {r['system'] for r in appropriateness.summarize_appropriateness(ar)} == {'base', 'ours'}
    conds = appropriateness.CONDITIONS
    items = [{'item_id': f'i{i}{c}', 'case': CASES[i], 'prefix': [{'speaker': 'doctor', 'text': 'd'}], 'turn': 1, 'claim_option': 'A' if c.startswith('correct') else 'B',
              'claim_correct': c.startswith('correct'), 'evidence_condition': c, 'collaborator_message': f'msg [{c}]'} for i in range(3) for c in conds]
    follows_support = lambda c, p, t: 'Final answer: option A' if 'correct_strong_valid' in p[-1]['text'] else 'Final answer: option C'
    es = appropriateness.sensitivity_summary(appropriateness.evidence_sensitivity(items, {'ours': follows_support}))
    assert es[0]['valid_evidence_sensitivity']['mean_difference'] == 1.0 and es[0]['discrimination_actual_vs_apparent']['mean_difference'] == 1.0
    assert 'incorrect_strong_valid' not in conds and len(conds) == 4
    with pytest.raises(ValueError): appropriateness.evidence_sensitivity([{**items[0], 'evidence_condition': 'incorrect_strong_valid'}], {'ours': follows_support})
    res = {'fixed': [{'case_id': f'c{i}', 'correct': i % 2} for i in range(6)], 'evolving': [{'case_id': f'c{i}', 'correct': 1} for i in range(6)]}
    assert rq3_analysis.task_success_table(res)[-1]['accuracy']['mean_difference'] == 0.5
    conds3 = ('base', 'outcome_only', 'integrated_evolving')
    mt = main_table.table({'base': {'val': res['fixed']}, 'outcome_only': {'val': res['fixed']}, 'integrated_evolving': {'val': res['evolving']}},
                          {c: {'condition': c} for c in conds3})
    assert any(r['condition'] == 'integrated_evolving' and r['minus_outcome_only']['mean_difference'] == 0.5 for r in mt['main'])
    with pytest.raises(ValueError): main_table.table({'static_rubric': {'val': res['fixed']}}, {'static_rubric': {}})
