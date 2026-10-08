"""TreeTrainer under bounded_witness_v1 (mocks only): every rubric arm uses the same witness verification and equal-weight
R_A; the Evolving prompt event still scores prompts with full member verification (trajectory-level s_j, J unchanged)."""
import copy, json, re
import pytest
import test_tree_v1 as T
from src.online_rl.integrated.records import read_json, sha256_text
from src.online_rl.tree import rubric_bank as RB, prompt_search as PS, witness as W
from src.online_rl.tree.conditions import arm_config
from src.online_rl.tree.team import effectiveness_weighted_r_a
from src.online_rl.tree.trainer import TreeTrainer

P0R = T.P0 + '\n\n[Assign or create]\n\nDECISION RULES\n'


def witness_env(log, w):
    env = T.mock_env(log, w)
    base_fb, base_score = env.feedback_fn, env.scorer

    def feedback(view, dialogue, outcome):
        text = base_fb(view, dialogue, outcome); n = sum(d['speaker'] == 'agent' for d in dialogue)
        log.add('locus'); return {'text': text, 'target_reply_id': n, 'status': 'ok', 'raw': [text]}

    def assign(package, user):
        log.add('assign'); ids = re.findall(r'"rubric_id": "([^"]+)"', user)
        return {'decision': 'REUSE_EXISTING', 'rubric_id': ids[0], 'checks': {c: True for c in RB.CHECKS}, 'rationale': 'same'}

    def scorer(view, prefix, rubrics, reply): return base_score(view, prefix, rubrics, reply)

    def applicability(view, prefix, rubrics):
        log.add('stage1'); depth = sum(d['speaker'] == 'agent' for d in prefix) + 1
        return {r['rubric_id']: {'applicable': r['applicability_condition'] in ('any turn', f'turn {depth}'), 'stage1': {}} for r in rubrics}
    scorer.applicability = applicability
    return T.TreeEnv(env.agent, env.doctor, env.stop_rule, env.final_option, env.outcome_fn, feedback, env.generate_fn, scorer, env.writer,
                     env.continuation, assign_fn=assign)


def wcfg(arm, cases=2):
    c = T.cfg(cases=cases); c['rubric'].update({'reuse': RB.ASSIGNMENT_VERSION, 'prompt_sha256': sha256_text(P0R)})
    c['feedback'] = {'behavioral_locus': True}; c['agent_reward'] = {'w_outcome': 0.5, 'w_rubric': 0.5}
    c['verification'] = {'mode': W.PROTOCOL, 'max_valid_supports': 4}
    c = arm_config(c, arm); c['agent_reward']['r_a_weighting'] = 'equal'
    return c


def make(tmp, arm, cases=2):
    log = T.Log(); w = {'v': 0}; seen = []
    st = T.mock_strategies(w, seen)
    st = type(st)(multi_target=st.multi_target, r_a_aggregator=effectiveness_weighted_r_a, credit_assignment=st.credit_assignment,
                  agent_update=st.agent_update, prompt_unit_value=st.prompt_unit_value)
    tr = TreeTrainer(wcfg(arm, cases), out_dir=tmp, cases=T.CASES, env=witness_env(log, w), strategies=st, search_backend=T.search_backend(log),
                     initial_prompt=P0R, policy_sha=lambda: f'policy-{w["v"]}', protocol_sha='proto', feedback_prompt_sha256='fb', updater=T.Updater(w))
    return tr, log, seen


@pytest.mark.parametrize('arm', ['rubric_only', 'fixed', 'evolving'])
def test_every_rubric_arm_uses_witness_verification(tmp_path, arm):
    tr, log, seen = make(tmp_path, arm); rec = tr.step()
    wit = rec['rubric_bank']['witness']
    assert wit and all(v['decision'] in ('KEEP', 'NOT_VALIDATED') for v in wit)
    assert rec['effectiveness'] == [] and rec['source_path_scores'] == []          # no exhaustive member verification in training
    assert all(t['result'] in ('PASS', 'FAIL', 'no_match', 'no_locus') for v in wit for t in v['tested'])
    assert all(v['num_valid_supports'] <= 4 for v in wit)
    kept = {v['rubric_id'] for v in wit if v['decision'] == 'KEEP'}
    assert set(rec['rubric_weights']) == kept and all(x == 1.0 for x in rec['rubric_weights'].values())
    assert {s['rubric_id'] for s in rec['response_scores']} == kept
    assert all(d['policy'] == W.PROTOCOL for d in rec['selection'])
    assert rec['rubric_bank']['telemetry']['protocol'] == W.PROTOCOL and rec['rubric_bank']['telemetry']['locus'] == {'ok': len(rec['feedback'])}
    assert all(f['target_reply_id'] is not None for f in rec['feedback'])


def test_same_witness_semantics_across_rubric_arms(tmp_path):
    out = {}
    for arm in ('rubric_only', 'fixed', 'evolving'):
        tr, _, _ = make(tmp_path / arm, arm); rec = tr.step()
        out[arm] = [(v['rubric_id'], v['decision'], [t['leaf_id'] for t in v['tested']], v['num_reruns_executed']) for v in rec['rubric_bank']['witness']]
    assert out['rubric_only'] == out['fixed'] == out['evolving']


def test_evolving_prompt_event_keeps_full_trajectory_level_j(tmp_path, monkeypatch):
    calls = {'verify_members': 0, 'witness': 0}
    vm, vw = RB.verify_members, W.verify_witness
    monkeypatch.setattr(RB, 'verify_members', lambda *a, **k: (calls.__setitem__('verify_members', calls['verify_members'] + 1), vm(*a, **k))[1])
    monkeypatch.setattr(W, 'verify_witness', lambda *a, **k: (calls.__setitem__('witness', calls['witness'] + 1), vw(*a, **k))[1])
    tr, log, _ = make(tmp_path, 'evolving', cases=4); tr.run()
    events = sorted((tmp_path / 'prompt_events').glob('event_*.json'))
    assert events and calls['witness'] == 4 and calls['verify_members'] > 0         # training: witness; prompt search: full members
    ev = read_json(events[0]); res = ev['parent_search'] if 'parent_search' in ev else ev
    txt = json.dumps(ev)
    assert '"J"' in txt and 'S_success' in txt and 'S_failure' in txt


def test_outcome_only_has_no_rubric_side_under_the_witness_config(tmp_path):
    tr, log, _ = make(tmp_path, 'outcome_only'); rec = tr.step()
    assert 'rubric_bank' not in rec and log.counts.get('stage1', 0) == 0 and log.counts.get('rerun', 0) == 0 and log.counts.get('feedback', 0) == 0


def test_witness_config_needs_equal_weights_and_locus(tmp_path):
    log = T.Log(); w = {'v': 0}
    for mutate, msg in ((lambda c: c['agent_reward'].pop('r_a_weighting'), 'r_a_weighting'), (lambda c: c['feedback'].pop('behavioral_locus'), 'behavioral_locus')):
        c = wcfg('rubric_only'); mutate(c)
        with pytest.raises(ValueError, match=msg):
            TreeTrainer(c, out_dir=tmp_path / msg, cases=T.CASES, env=witness_env(log, w), strategies=T.mock_strategies(w), search_backend=T.search_backend(log),
                        initial_prompt=P0R, policy_sha=lambda: 'p', protocol_sha='proto', feedback_prompt_sha256='fb', updater=T.Updater(w))
    c = T.cfg(); c['agent_reward'] = {'w_outcome': 0.5, 'w_rubric': 0.5, 'r_a_weighting': 'equal'}
    with pytest.raises(ValueError, match='equal R_A weighting'):
        TreeTrainer(c, out_dir=tmp_path / 'eq', cases=T.CASES, env=T.mock_env(log, w), strategies=T.mock_strategies(w), search_backend=T.search_backend(log),
                    initial_prompt=T.P0, policy_sha=lambda: 'p', protocol_sha='proto', feedback_prompt_sha256='fb', updater=T.Updater(w))


def test_invalid_v4_feedback_is_excluded_from_mapping_witness_and_j_units(tmp_path):
    tr, log, _ = make(tmp_path, 'rubric_only')
    base = tr.env.feedback_fn; bad = set()
    def fb(view, dialogue, outcome):
        v = base(view, dialogue, outcome); key = dialogue[-1]['text'] + str(len(dialogue))
        if len(bad) < 2 and key not in bad: bad.add(key); return {**v, 'valid': False, 'target_reply_id': None, 'status': 'leakage'}
        return v
    import dataclasses
    tr.env = dataclasses.replace(tr.env, feedback_fn=fb); rec = tr.step()
    inv = rec['rubric_bank']['telemetry']['feedback_invalid']
    assert len(inv) >= 1
    mapped = {a['leaf_id'] for a in rec['rubric_bank']['assignments']}
    assert not (set(inv) & mapped) and not (set(inv) & {t['leaf_id'] for t in rec['trajectory_inputs']})
    assert not (set(inv) & {t['leaf_id'] for v in rec['rubric_bank']['witness'] for t in v['tested']})


def test_calibration_positions_replay_main_order_and_training_cannot_reach_them(tmp_path):
    from src.online_rl.integrated.episode import schedule
    from src.online_rl.tree import trainer as TT
    tr, _, _ = make(tmp_path / 'a', 'rubric_only', cases=4); tr.calibration_only = {2, 3}
    ids = sorted(c['case_id'] for c in T.CASES)
    assert [schedule(ids, 0, tr._position(i), 1)[0] for i in range(4)] == [schedule(ids, 0, i, 1)[0] for i in range(4)]
    c = wcfg('rubric_only'); c['schedule']['calibration_only_cursors'] = [2, 3]; c['schedule']['positions'] = [3, 1]
    with pytest.raises(ValueError, match='calibration-only positions'):
        TreeTrainer(c, out_dir=tmp_path / 'b', cases=T.CASES, env=witness_env(T.Log(), {'v': 0}), strategies=T.mock_strategies({'v': 0}),
                    search_backend=T.search_backend(T.Log()), initial_prompt=P0R, policy_sha=lambda: 'p', protocol_sha='proto', feedback_prompt_sha256='fb',
                    updater=T.Updater({'v': 0}))
    tr.state = TT.dataclasses.replace(tr.state, case_cursor=2)
    with pytest.raises(RuntimeError, match='calibration-only'): tr.step()


def test_prompt_acceptance_requires_the_same_trajectory_mask():
    acc = {'bootstrap_seed': 0, 'bootstrap_n': 10, 'delta': 0.0, 'p_min': 0.5, 'minimum_success_trajectories': 1, 'minimum_failure_trajectories': 1}
    par = {'per_trajectory': {'a': 1.0, 'b': 0.0}, 'source_outcome': {'a': 1.0, 'b': 0.0}}
    PS.stratified_paired_bootstrap_J(par, dict(par), acc)
    with pytest.raises(AssertionError, match='same trajectory mask'):
        PS.stratified_paired_bootstrap_J(par, {'per_trajectory': {'a': 1.0}, 'source_outcome': {'a': 1.0}}, acc)


def test_evolving_event_scores_parent_and_candidates_on_valid_feedback_units_only(tmp_path, monkeypatch):
    seen = []
    orig = PS.score_prompt
    monkeypatch.setattr(PS, 'score_prompt', lambda text, sha, units, cases, d: (seen.append(frozenset(u.leaf.leaf_id for u in units)), orig(text, sha, units, cases, d))[1])
    tr, log, _ = make(tmp_path, 'evolving', cases=4)
    base = tr.env.feedback_fn; bad = []
    def fb(view, dialogue, outcome):
        v = base(view, dialogue, outcome)
        if len(bad) < 3: bad.append(1); return {**v, 'valid': False, 'target_reply_id': None, 'status': 'leakage'}
        return v
    import dataclasses
    tr.env = dataclasses.replace(tr.env, feedback_fn=fb); tr.run()
    invalid = {l for f in sorted((tmp_path / 'cases').glob('*.json')) for l in read_json(f)['rubric_bank']['telemetry']['feedback_invalid']}
    assert invalid and seen and all(not (s & invalid) for s in seen)
    rec = read_json(sorted((tmp_path / 'cases').glob('*.json'))[0])
    assert set(rec['rubric_bank']['telemetry']['feedback_invalid_by_stratum']) == {'success', 'failure'}
