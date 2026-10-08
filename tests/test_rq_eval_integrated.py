"""RQ1-RQ3 evaluation on integrated runs and the structured-state consumer contract (mocks only)."""
import json
from pathlib import Path
import pytest
import test_integrated_v1 as T
from src.online_rl.integrated.conditions import MAIN, condition_config, prompt_mode, unmatched_keys
from src.online_rl.integrated.records import read_json
from src.online_rl.integrated.rubric import STATE_SCHEMA, STATE_SCHEMA_SHA256, freeze_generator_input
from src.rq_eval import appropriateness, downstream, main_table, rq3_analysis, runs


def _run(tmp, name, steps=24, **kw):
    cfg = condition_config(T.make_cfg(), name)
    tr, pol, _, be = T.make_trainer(tmp / name, cfg=cfg, **kw); tr.run(max_steps=steps)
    return tr, pol, be


def test_conditions_are_matched_and_share_p0(tmp_path):
    base = T.make_cfg(); cfgs = {n: condition_config(base, n) for n in ('integrated_fixed', 'integrated_evolving', 'outcome_only')}
    assert [prompt_mode(c) for c in cfgs.values()] == ['fixed', 'evolving', 'none'] and MAIN == ('base', 'outcome_only', 'integrated_fixed', 'integrated_evolving')
    assert unmatched_keys(cfgs['integrated_fixed'], cfgs['integrated_evolving']) == set() and unmatched_keys(cfgs['outcome_only'], cfgs['integrated_fixed']) == set()
    assert cfgs['integrated_fixed']['rubric'] == cfgs['integrated_evolving']['rubric']
    with pytest.raises(ValueError): condition_config(base, 'static_rubric')
    fixed = runs.load_run(_run(tmp_path, 'integrated_fixed')[0].out); evolving = runs.load_run(_run(tmp_path, 'integrated_evolving')[0].out)
    assert fixed['initial_prompt_sha256'] == evolving['initial_prompt_sha256'] and fixed['state_schema_sha256'] == evolving['state_schema_sha256']
    assert runs.rq3_pair_problems(fixed, evolving) == [] and not (tmp_path / 'integrated_fixed' / 'prompt_events').exists()
    assert len(fixed['accepted_prompts']) == 1 and len(evolving['accepted_prompts']) >= 2


def test_outcome_only_skips_state_and_rubric(tmp_path):
    tr, pol, be = _run(tmp_path, 'outcome_only', steps=4)
    assert 'elicit' not in pol.order and be.gen_inputs == [] and be.scored == []
    for e in T.evidence(tr.out):
        assert e.generator_input.state_status == 'not_elicited' and e.rubric.failure == 'rubric_disabled'
        assert all(b.reward is None or b.reward == pytest.approx(0.4 * b.R_O) for b in e.branches)
    with pytest.raises(ValueError): T.make_trainer(tmp_path / 'bad', cfg={**T.make_cfg(), 'rubric': {**T.make_cfg()['rubric'], 'enabled': False}})


def test_run_loader_resolves_adapter_prompt_and_provenance(tmp_path):
    tr, *_ = _run(tmp_path, 'integrated_evolving')
    p = runs.load_run(tr.out); s = read_json(tr.out / 'run_state.json')
    assert (p['checkpoint'], p['adapter_sha256'], p['prompt_sha256'], p['step']) == (s['checkpoint'], s['adapter_sha256'], s['prompt_sha256'], 24)
    for k in ('base_model', 'checkpoint_sha256', 'seed', 'prompt_id', 'prompt_mode', 'state_schema_version', 'state_schema_sha256', 'reward',
              'advantage_mode', 'config_sha256', 'initial_prompt_sha256'):
        assert p[k] is not None
    assert p['state_schema_sha256'] == STATE_SCHEMA_SHA256 and p['advantage_mode'] == 'mean_center' and p['condition'] == 'integrated_evolving'
    m = read_json(tr.out / 'run_manifest.json'); assert m['state_schema']['consumers'] == STATE_SCHEMA['consumers'] and 'P0' in m['initial_prompt']['role']
    (Path(p['checkpoint']) / 'state.json').write_text('{"version": 999, "steps": 0}')
    with pytest.raises(RuntimeError, match='checkpoint files changed'): runs.load_run(tr.out)


def test_rq3_lineage_and_pair_check(tmp_path):
    fixed = runs.load_run(_run(tmp_path, 'integrated_fixed')[0].out); ev_tr = _run(tmp_path, 'integrated_evolving')[0]
    evolving = runs.load_run(ev_tr.out); lineage = runs.prompt_lineage(ev_tr.out)
    assert lineage and {'event', 'step', 'prompt_id', 'parent_prompt_id', 'sha256', 'status', 'accepted'} <= set(lineage[0])
    acc = [r for r in lineage if r['accepted']]
    assert [r['prompt_id'] for r in acc] == [h['prompt_id'] for h in evolving['accepted_prompts'][1:]]
    task = {'fixed': [{'case_id': 'c', 'correct': 1}], 'evolving': [{'case_id': 'c', 'correct': 1}]}
    out = rq3_analysis.rq3_table(task, fixed, evolving, lineage)
    assert out['provenance']['fixed']['prompt_mode'] == 'fixed' and 'config' not in out['provenance']['evolving']
    with pytest.raises(ValueError): rq3_analysis.rq3_table(task, fixed, {**evolving, 'initial_prompt_sha256': 'other'}, lineage)
    assert runs.onpolicy_rubrics(ev_tr.out) and all('rubric' in r for r in runs.onpolicy_rubrics(ev_tr.out))


def test_main_table_requires_provenance(tmp_path):
    res = {'base': {'val': [{'case_id': 'c', 'correct': 1}]}, 'integrated_fixed': {'val': [{'case_id': 'c', 'correct': 1}]}}
    with pytest.raises(ValueError): main_table.table(res, {'base': runs.base_provenance(T.make_cfg())})
    fixed = runs.load_run(_run(tmp_path, 'integrated_fixed', steps=2)[0].out)
    t = main_table.table(res, {'base': runs.base_provenance(T.make_cfg()), 'integrated_fixed': fixed})
    assert t['provenance']['integrated_fixed']['adapter_sha256'] == fixed['adapter_sha256'] and 'config' not in t['provenance']['integrated_fixed']


def test_rq2_appropriateness_is_independent_of_training_rubrics(tmp_path):
    tr, pol, _ = _run(tmp_path, 'integrated_evolving', steps=4)
    seen = []

    def decide(state, questions):
        seen.append((json.loads(state), questions)); return {q: {'value': '2', 'probabilities': {'0': 0, '1': 0, '2': 1}} for q in questions}
    probe = {'probe_id': 'p', 'case': T.CASES[0], 'prefix': [{'speaker': 'doctor', 'text': 'd'}], 'turn': 1}
    reply = runs.policy_reply_fn(pol, lambda *_: 7)
    appropriateness.state_appropriateness_rows([probe], {'integrated_evolving': reply}, decide)
    state, questions = seen[0]
    assert questions == appropriateness.JUDGE['questions'] and set(state) == {'case_context', 'dialogue_before_target_turn', 'target_turn'}
    rubric_texts = {json.dumps(e.rubric.rubric) for e in T.evidence(tr.out) if e.rubric.rubric}
    assert not any(r in json.dumps(state) for r in rubric_texts) and pol.calls['natural'] >= 1


def test_rq2_downstream_uses_matched_continuations_not_q_cov():
    h = [{'history_id': 'h', 'case': T.CASES[0], 'prefix': [{'speaker': 'doctor', 'text': 'd'}], 'turn': 1}]
    seeds = []
    cont = lambda c, p, t, r, s: (seeds.append((r, s)), {'correct': int(r == 'model' or s == 0)})[1]
    rows = downstream.downstream_rows(h, lambda x: 'base', lambda x: 'model', cont, 3, lambda hid, i: i, 'integrated_evolving')
    assert rows[0]['base_success'] == pytest.approx(1 / 3) and rows[0]['model_success'] == 1.0
    assert [s for r, s in seeds if r == 'base'] == [s for r, s in seeds if r == 'model'] and 'Q_cov' not in rows[0]


def test_structured_state_reaches_only_the_generator(monkeypatch):
    from src.prompt_opt import harness as H
    from src.prompt_opt import harness_v12 as V12
    from src.online_rl.integrated.rubric import real_components
    calls = {}

    def doctor_call(key, model, system, user, *a): calls['generator'] = user; return '{"name": "n"}'

    def gates(client, case, prefix, cand, rid): calls['gates'] = (case, prefix); return {'schema': True}

    def adherence(client, case, prefix, rubric, reply, purpose): calls['adherence'] = (case, prefix, reply); return {'probabilities': {'0': 1, '1': 0, '2': 0}, 'hard': 0}
    monkeypatch.setattr(H, 'doctor_call', doctor_call); monkeypatch.setattr(V12, 'gates', gates); monkeypatch.setattr(H, 'adherence', adherence)
    gen, gate, score = real_components(None, 'k')
    gin = freeze_generator_input(T.CASES[0], T._prefix(2), 2, T.mock_state(T._prefix(2), 2))
    gen('P', gin); gate(T.CASES[0], gin, {'rubric': {}}, 'r'); score(gin.case_view(), gin, {'rubric_id': 'r'}, 'reply')
    assert 'AGENTSTATE' in calls['generator'] and 'latest_doctor_message' in calls['generator']
    for name in ('gates', 'adherence'):
        blob = json.dumps(calls[name], default=str)
        assert 'AGENTSTATE' not in blob and 'clinician_current_position' not in blob                   # history-only judges
    assert 'ground_truth' not in json.dumps(calls['adherence'][0]) and STATE_SCHEMA['consumers'] == {
        'rubric_generator': True, 'gates': False, 'contextual_appropriateness': False, 'adherence': False, 'agent_policy': False}
