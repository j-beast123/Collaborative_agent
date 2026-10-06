"""SHERPA case-tree pipeline: mocks only (no API, no GPU)."""
import hashlib, inspect, json, re
from pathlib import Path
import pytest
from src.online_rl.integrated.prompt_search import SearchBackend
from src.online_rl.integrated.records import read_json, sha256_text
from src.online_rl.tree import effectiveness as EF, posthoc as PH, reward as RW
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies, UnresolvedDecision
from src.online_rl.tree.records import ResponseRubricScore
from src.online_rl.tree.rollout import build_tree, tree_counts
from src.online_rl.tree.trainer import TreeEnv, TreeTrainer

CASES = [{'case_id': f'tc{i}', 'question': f'Synthetic tree item {i}: which mechanism?', 'scenario': '', 'options': {k: f'{k.lower()}tree option {i}' for k in 'ABCD'},
          'ground_truth': {'correct_option': 'A', 'correct_text': f'atree option {i}'}} for i in range(4)]
P0 = Path('prompts/sherpa/trajectory_rubric_generator_P0.txt').read_text() + '\nVARIANT=0'
h = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
CONDS = ('turn 1', 'turn 99', 'any turn')                    # per trajectory: single target / No Match / every turn


def nested_rubric(flat, selected='tc0/0'):
    app = {'user_side_context': flat['applicability_condition'], 'agent_side_state': 'No additional state condition'}
    return {'rubric_id': flat.get('rubric_id', 'r'), 'analysis': {'selected_agent_turn_id': selected},
            'candidate_rubric': {'name': flat['name'], 'applicability': app,
                'scores': {k: {**app, 'behavior': v} for k, v in flat['scores'].items()}}}


@pytest.fixture(autouse=True)
def no_real_api(monkeypatch):
    import src.collaboration.generator as G
    from src.prompt_opt import harness as H
    for mod in (G, H): monkeypatch.setattr(mod, 'doctor_call', lambda *a, **k: (_ for _ in ()).throw(AssertionError('real API call in a test')))


class Log:
    def __init__(self): self.events, self.counts = [], {}

    def add(self, kind, key=''):
        self.events.append((kind, key)); self.counts[kind] = self.counts.get(kind, 0) + 1


class MockAgent:
    def __init__(self, log, w, stop_every=0): self.log, self.w, self.stop_every = log, w, stop_every

    def sample(self, case, prefix, turn, seed):
        self.log.add('agent', f'{turn}:{seed}'); assert 'rubric' not in json.dumps(prefix).lower() and 'feedback' not in json.dumps(prefix).lower()
        stop = self.stop_every and h(f'{seed}') % self.stop_every == 0
        text = f'reply t{turn} q{(seed + self.w["v"]) % 3}' + (' STOP' if stop else '')
        return {'prompt_text': 'P|' + json.dumps(prefix), 'prompt_input_ids': [1, 2], 'completion_ids': [3], 'old_policy_logprobs': [-0.1], 'completion_text': text, 'seed': seed}


class MockDoctor:
    def __init__(self, log): self.log = log

    def turn(self, case, dialogue, r, rounds):
        self.log.add('doctor_turn', str(r)); assert r <= rounds; return f'doctor round {r}'

    def final(self, case, dialogue):
        self.log.add('final', dialogue[-1]['text']); q = int(re.search(r'q(\d)', dialogue[-1]['text']).group(1))
        return f'Final answer: option {"A" if q > 0 else "B"}'


def mock_env(log, w, stop_every=0, k_conds=CONDS, gen_malformed=False):
    def feedback(view, dialogue, outcome):
        assert 'ground_truth' not in view and 'correct_option' not in json.dumps(view) and dialogue[-1]['stage'] == 'FINAL'
        log.add('feedback', dialogue[-1]['text']); return f'feedback for a {outcome}'

    def generate(prompt, tin):
        log.add('rubric', f'{tin.leaf_id}|{sha256_text(prompt)[:8]}'); assert not any('gold' in f or 'correct' in f for f in tin.__dataclass_fields__)
        v = int(re.search(r'VARIANT=(\d)', prompt).group(1)); i = h(tin.leaf_id)
        if gen_malformed and i % 2 == 0: return {'raw': 'nope', 'rubric': None}
        return {'raw': '{}', 'rubric': nested_rubric({'name': f'r{i % 7}', 'applicability_condition': k_conds[i % len(k_conds)],
                'scores': {'0': 'bad', '1': 'partial', '2': f'behave v{v} i{i % 7}'}}, tin.agent_node_ids[0])}

    def scorer(view, prefix, rubrics, reply):
        log.add('score'); depth = sum(d['speaker'] == 'agent' for d in prefix) + 1; q = int(re.search(r'q(\d)', reply).group(1)); out = {}
        for r in rubrics:
            c = r['applicability_condition'].split('[Physician-side context]\n', 1)[-1].split('\n[Pre-response Agent state]', 1)[0]; app = c == 'any turn' or c == f'turn {depth}'
            out[r['rubric_id']] = {'applicable': app, 'probabilities': {'0': 1 - q / 2, '1': 0.0, '2': q / 2} if app else None, 'hard': (2 if q == 2 else 0) if app else None}
        return out

    def writer(case, prefix, turn, rubric, level, seed):
        log.add('write', level); return f'[{level}] {rubric["candidate_rubric"]["scores"][level]["behavior"]} q{2 if level == "2" else 0} seed{seed}'

    def continuation(case, prefix, turn, text, seed):
        log.add('rerun'); good = ('[2]' in text) == (h(f'{text}{seed}') % 3 > 0)
        return {'final_option': 'A' if good else 'B', 'outcome': 1.0 if good else 0.0}
    return TreeEnv(MockAgent(log, w, stop_every), MockDoctor(log), lambda r, d, reply: 'agent_final_answer' if 'STOP' in reply else None,
                   lambda t: re.search(r'option ([A-E])', t).group(1), lambda c, o: float(o == c['ground_truth']['correct_option']),
                   feedback, generate, scorer, writer, continuation)


def mock_strategies(w, seen=None, multi=True):
    m = lambda xs: sum(xs) / len(xs) if xs else 0.0

    def credit(tree, leaf_outcomes, node_r_a):
        if seen is not None: seen.append({'leaf_outcomes': dict(leaf_outcomes), 'node_r_a': dict(node_r_a)})
        return {'n': len(node_r_a)}

    def update(tree, targets):
        w['v'] += 1; return {'optimizer_step': True}
    return Strategies(multi_target=(lambda per_node: m(list(per_node.values()))) if multi else None,
                      r_a_aggregator=lambda scores, weights: m([s.score for s in scores if s.match]), credit_assignment=credit, agent_update=update,
                      prompt_unit_value=lambda res: (res['effectiveness'].R_O or 0.0) if res['effectiveness'] else 0.0)


def cfg(cases=4, rounds=3, branching=2, k=3, every=2, min_traj=4):     # k = reruns per intervention
    return {'name': 't', 'output_dir': 'x', 'data': {'train': 'mock'}, 'policy': {'base_model': 'mock'},
            'rollout': {'branching': branching, 'rounds': rounds, 'budget': 'full_tree'}, 'schedule': {'seed': 0, 'cases': cases},
            'rubric': {'generator_prompt_file': 'mock', 'prompt_id': 'P0', 'prompt_sha256': sha256_text(P0), 'gold_answer_visible': False},
            'intervention': {'reruns_per_intervention': k}, 'selection': {'mode': 'threshold', 'min_R_O': 2 / 3},
            'acceptance': {'delta': 0.0, 'p_min': 0.5, 'minimum_trajectories': 1, 'bootstrap_n': 200, 'bootstrap_seed': 0},
            'prompt_update': {'enabled': True, 'every_n_cases': every, 'minimum_trajectories': min_traj, 'acceptance_fraction': 0.5,
                              'candidate_count': 4, 'retain_top_k': 2, 'refinement_depth': 2}}


class Updater:
    def __init__(self, w): self.w = w

    def save(self, p): p.mkdir(parents=True); (p / 'w.json').write_text(json.dumps(self.w))

    def load(self, p): self.w.update(json.loads((Path(p) / 'w.json').read_text()))


def search_backend(log):
    def feedback(parent, view, n):
        log.add('reflect'); assert all('correct_option' not in json.dumps(v) for v in view); return {'diagnosis': 'd', 'directions': [f'dir{i}' for i in range(n)]}

    def refine(parent, diagnosis, direction, history, note):
        log.add('refine'); return P0.replace('VARIANT=0', f'VARIANT={1 + int(direction[3:]) % 2}') + f' tag {direction} {hashlib.sha256(parent.encode()).hexdigest()[:6]}'
    return SearchBackend(feedback, refine, None, None, None)


def trainer(tmp, c=None, w=None, log=None, strategies=None, stop_every=0, gen_malformed=False, seen=None, env=None):
    log = log or Log(); w = w if w is not None else {'v': 0}
    return TreeTrainer(c or cfg(), out_dir=tmp, cases=CASES, env=env or mock_env(log, w, stop_every, gen_malformed=gen_malformed),
                       strategies=strategies or mock_strategies(w, seen), search_backend=search_backend(log), initial_prompt=P0,
                       policy_sha=lambda: f'policy-{w["v"]}', protocol_sha='proto', feedback_prompt_sha256='fb', updater=Updater(w)), log, w


def tree_for(log, rounds=5, branching=4, stop_every=0, w=None):
    env = mock_env(log, w or {'v': 0}, stop_every)
    return build_tree(CASES[0], max_agent_turns=rounds, branching=branching, run_seed=0, policy_sha256='p', protocol_sha256='proto', agent=env.agent,
                      doctor=env.doctor, stop_rule=env.stop_rule, final_option=env.final_option, outcome_fn=env.outcome_fn, cache=CallCache(None)), env


def case_record(tmp, i=0): return read_json(tmp / 'cases' / f'case_{i:06d}.json')


# ---------- rollout ----------
def test_full_tree_counts_and_max_turn_semantics():
    log = Log(); tree, _ = tree_for(log)
    c = tree_counts(tree)
    assert c['by_depth'] == {1: 4, 2: 16, 3: 64, 4: 256, 5: 1024} and c['agent_nodes'] == 1364 and c['leaves'] == 1024
    assert c['doctor_turns'] == 341 == log.counts['doctor_turn'] and log.counts['final'] == 1024 and log.counts['agent'] == 1364
    assert {l.stop_reason for l in tree.leaves} == {'max_agent_turns'} and max(n.depth for n in tree.nodes) == 5
    assert all(sum(1 for d in l.dialogue if d[0] == 'agent') == 5 and l.dialogue[-1][1] == 'FINAL' for l in tree.leaves)
    kids = {}
    for n in tree.nodes: kids.setdefault(n.parent_id, []).append(n)
    assert all(len(v) == 4 and len({x.prefix for x in v}) == 1 for v in kids.values())
    assert json.loads(Path('configs/online_rl/tree_v1.json').read_text())['rollout']['rounds'] == 5


def test_early_stopping_leaves():
    log = Log(); tree, _ = tree_for(log, rounds=4, branching=4, stop_every=3)
    depths = {}
    for l in tree.leaves: depths.setdefault(l.stop_reason, set()).add(len(l.path))
    assert min(depths['agent_final_answer']) < 4 and depths.get('max_agent_turns') == {4}
    assert all(n.terminal == (n.next_doctor is None) for n in tree.nodes) and len(tree.leaves) == log.counts['final']


def test_descendant_outcome_and_leaf_path_relations():
    log = Log(); tree, _ = tree_for(log, rounds=3, branching=2)
    desc = RW.descendant_outcomes(tree); paths = {l.leaf_id: l.path for l in tree.leaves}
    root_children = [n for n in tree.nodes if n.depth == 1]
    assert sum(len(desc[n.node_id]) for n in root_children) == len(tree.leaves)
    assert all(set(desc[n.node_id]) == {l for l, p in paths.items() if n.node_id in p} for n in tree.nodes)


# ---------- post-hoc order, K candidates, no gold ----------
def test_one_generator_call_and_one_rubric_per_trajectory_no_gold(tmp_path):
    tr, log, _ = trainer(tmp_path, cfg(cases=1, every=99)); tr.run(max_cases=1)
    ev = log.events
    first_fb = next(i for i, e in enumerate(ev) if e[0] == 'feedback'); first_rubric = next(i for i, e in enumerate(ev) if e[0] == 'rubric')
    assert max(i for i, e in enumerate(ev) if e[0] in ('final', 'agent', 'doctor_turn')) < first_fb
    assert max(i for i, e in enumerate(ev) if e[0] == 'feedback') < first_rubric
    rec = case_record(tmp_path); leaves = [t['leaf_id'] for t in rec['trajectory_inputs']]
    gen_calls = [k.split('|')[0] for kind, k in ev if kind == 'rubric']
    assert sorted(gen_calls) == sorted(leaves) and len(set(gen_calls)) == len(gen_calls) == len(rec['tree']['leaves'])     # exactly 1 call each
    assert sorted(r['leaf_id'] for r in rec['trajectory_rubrics']) == sorted(leaves)                                    # exactly 1 rubric each
    assert 'rubric_candidates' not in rec and all('index' not in r for r in rec['trajectory_rubrics'])
    for t in rec['trajectory_inputs']:
        assert 'gold' not in json.dumps(list(t)) and 'ground_truth' not in json.dumps(t) and t['dialogue'][-1][1] == 'FINAL'
    assert json.loads(Path('configs/online_rl/tree_v1.json').read_text())['rubric']['gold_answer_visible'] is False


# ---------- interventions ----------
def _leaf_eval(log, outcome_wanted, k_conds=('turn 1',), reruns=3, multi=True, st=None):    # k_conds: the one rubric's condition
    w = {'v': 0}; tree, env = tree_for(log, rounds=2, branching=2); cache = CallCache(None)
    env = mock_env(log, w, k_conds=k_conds); nodes = {n.node_id: n for n in tree.nodes}
    leaf = next(l for l in tree.leaves if l.outcome == outcome_wanted)
    fb = PH.doctor_feedback(CASES[0], leaf, env.feedback_fn, 'fb', cache); tin = PH.freeze_trajectory_input(CASES[0], leaf, fb)
    rub = PH.generate_rubric(P0, sha256_text(P0), tin, env.generate_fn, cache)
    return leaf, rub, EF.evaluate_rubric(CASES[0], nodes, leaf, rub, scorer=env.scorer, writer=env.writer, continuation=env.continuation,
                                         reruns=reruns, run_seed=0, policy_sha256='p', strategies=st or mock_strategies(w, multi=multi), cache=cache)


@pytest.mark.parametrize('outcome,level', [(0.0, '2'), (1.0, '0')])
def test_intervention_routing_and_k_reruns(outcome, level):
    log = Log(); leaf, _, (_, e) = _leaf_eval(log, outcome)
    assert e.level == level and e.status == 'single_target' and e.target_node_ids == (leaf.path[0],)        # matched response = turn 1
    assert len(e.runs) == 3 and {r.level for r in e.runs} == {level} and all(r.intervention_text.startswith(f'[{level}]') for r in e.runs)
    assert len({r.intervention_seed for r in e.runs}) == 3 and len({r.intervention_text for r in e.runs}) == 3 and log.counts['write'] == 3                       # independent Agent generation for every trial
    target = 1.0 if level == '2' else 0.0
    assert e.R_O == pytest.approx(sum(r.outcome == target for r in e.runs) / 3) == e.per_node_R_O[leaf.path[0]]
    assert all(r.continuation_policy_sha256 == 'p' for r in e.runs)


def test_target_rate_convention():
    assert EF.target_rate('2', [1.0, 1.0, 0.0]) == pytest.approx(2 / 3) and EF.target_rate('0', [1.0, 0.0, None]) is None
    assert EF.target_rate('2', [None, None, 1.0]) is None
    with pytest.raises(ValueError): EF.intervention_level(None)


def test_no_match_and_single_selected_target():
    _, _, (src, nm) = _leaf_eval(Log(), 0.0, k_conds=('turn 99',))
    leaf, _, (_, matched) = _leaf_eval(Log(), 0.0, k_conds=('any turn',), multi=False)
    assert nm.status == 'no_match' and nm.R_O is None and nm.runs == ()
    assert nm.target_node_ids == (leaf.path[0],)
    assert matched.status == 'single_target' and matched.target_node_ids == (leaf.path[0],)
    assert len(matched.runs) == 3 and set(matched.per_node_R_O) == {leaf.path[0]}
    assert any(not score.match and score.score is None for score in src)

# ---------- selection, response-level scoring, reward ----------
def test_selection_scoring_and_provenance(tmp_path):
    tr, _, _ = trainer(tmp_path, cfg(cases=1, every=99)); tr.run(max_cases=1)
    rec = case_record(tmp_path); effs = {e['rubric_id']: e for e in rec['effectiveness']}
    assert {d['rubric_id'] for d in rec['selection']} == set(effs)
    for d in rec['selection']:
        e = effs[d['rubric_id']]
        assert d['retained'] == (e['R_O'] is not None and e['R_O'] >= 2 / 3) and e['status'] in d['reason']
    kept = {d['rubric_id'] for d in rec['selection'] if d['retained']}; cands = {c['rubric_id']: c for c in rec['trajectory_rubrics']}
    assert kept and not any(effs[r]['status'] == 'no_match' for r in kept)
    scores = rec['response_scores']; node_ids = {n['node_id'] for n in rec['tree']['nodes']}
    assert {(s['response_node_id'], s['rubric_id']) for s in scores} == {(n, r) for n in node_ids for r in kept}      # every tree response
    assert all(s['source_leaf_id'] == cands[s['rubric_id']]['leaf_id'] for s in scores)
    assert any(not s['match'] and s['score'] is None for s in scores) and any(s['match'] and s['score'] is not None for s in scores)
    assert all(set(s) == {'response_node_id', 'rubric_id', 'source_leaf_id', 'match', 'score', 'hard', 'probabilities'} for s in scores)


def test_agent_reward_excludes_R_O_and_includes_final_outcome(tmp_path):
    assert list(inspect.signature(RW.agent_targets).parameters) == ['tree', 'scores', 'rubric_weights', 'strategies']
    assert 'effectiveness' not in inspect.getsource(RW).split('"""')[2] and 'R_O' not in inspect.getsource(RW).split('"""')[2]
    seen = []; tr, _, _ = trainer(tmp_path / 'a', cfg(cases=1, every=99), seen=seen); tr.run(max_cases=1)
    rec = case_record(tmp_path / 'a'); outcomes = {l['leaf_id']: l['outcome'] for l in rec['tree']['leaves'] if l['outcome'] is not None}
    assert seen[0]['leaf_outcomes'] == outcomes
    seen2 = []; log2 = Log(); w2 = {'v': 0}; env2 = mock_env(log2, w2)
    env2 = type(env2)(**{**env2.__dict__, 'continuation': lambda case, prefix, turn, text, seed: {'final_option': 'B', 'outcome': 0.0}})
    tr2, _, _ = trainer(tmp_path / 'b', cfg(cases=1, every=99), log=log2, w=w2, seen=seen2, env=env2,
                        strategies=mock_strategies(w2, seen2)); tr2.run(max_cases=1)
    rec2 = case_record(tmp_path / 'b')
    assert [e['R_O'] for e in rec2['effectiveness']] != [e['R_O'] for e in rec['effectiveness']]
    assert seen2[0]['leaf_outcomes'] == seen[0]['leaf_outcomes']


def test_raw_scores_until_aggregation():
    s = [ResponseRubricScore('n1', 'r1', 'L1', True, 1.0, 1.0, {}), ResponseRubricScore('n1', 'r2', 'L2', False, None, None, None)]
    with pytest.raises(UnresolvedDecision): RW.response_r_a(s, {'r1': 1.0}, Strategies())
    got = []; RW.response_r_a(s, {'r1': 1.0}, Strategies(r_a_aggregator=lambda xs, wts: got.append((xs, wts)) or 0.0))
    assert got[0] == (s, {'r1': 1.0})                                                                                        # No Match passed through raw


def test_selection_requires_two_of_three_complete_trials():
    effs = [_fixed_eval(0.0, outcomes, 3)[3] for outcomes in ([0,0,0], [1,0,0], [1,1,0], [1,1,1], [1,1,None])]
    assert [d.retained for d in EF.select(effs, {'mode': 'threshold', 'min_R_O': 2/3})] == [False,False,True,True,False]
    for bad in ({'mode':'CALIBRATION_REQUIRED'}, {'mode':'threshold','min_R_O':'CALIBRATION_REQUIRED'},
                {'mode':'top_k','k':2}, {'mode':'threshold','min_R_O':0.5}):
        with pytest.raises(ValueError): EF.select(effs,bad)

# ---------- prompt optimization ----------
def test_prompt_search_uses_rerun_effectiveness(tmp_path, monkeypatch):
    import src.online_rl.integrated.metrics as M
    monkeypatch.setattr(M, 'q_cov', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Q_cov used')))
    src = ''.join(p.read_text() for p in Path('src/online_rl/tree').glob('*.py'))
    assert not re.search(r'q_cov|Q_cov|informative|state_diagnostics|effect_diagnostics|V_h|r_e\(', src)
    tr, log, _ = trainer(tmp_path, cfg(cases=4, every=2, min_traj=4)); tr.run()
    events = sorted((tmp_path / 'prompt_events').glob('event_*.json'))
    assert events and log.counts.get('refine', 0) > 0
    ev = read_json(events[0]); acc = ev['decision'].get('acceptance') or {}
    assert set(ev['search_leaf_ids']).isdisjoint(ev['acceptance_leaf_ids'])
    assert set(acc['checks']) == {'minimum_trajectories', 'gain', 'bootstrap'} and acc['n'] > 0                     # paired acceptance ran
    s = read_json(tmp_path / 'run_state.json'); assert s['lineage_cursor'] == len(events) and s['case_cursor'] == 4


def test_prompt_unit_value_is_an_open_decision(tmp_path):
    w = {'v': 0}; st = Strategies(**{**mock_strategies(w).__dict__, 'prompt_unit_value': None})
    tr, _, _ = trainer(tmp_path, cfg(cases=4, every=2, min_traj=4), w=w, strategies=st)
    with pytest.raises(UnresolvedDecision, match='prompt_unit_value'): tr.run()


# ---------- resume ----------
@pytest.mark.parametrize('crash_at', ['rerun', 'score', 'refine'])
def test_resume_repeats_no_cached_work(tmp_path, crash_at):
    class Crash(Exception): pass
    ref, ref_log, _ = trainer(tmp_path / 'ref'); ref.run()
    log = Log(); w = {'v': 0}; n = {'k': 0}

    def crashing(fn):
        def g(*a, **k):
            n['k'] += 1
            if n['k'] == 7: raise Crash()
            return fn(*a, **k)
        return g
    env = mock_env(log, w)
    if crash_at in ('rerun', 'score'):
        env = type(env)(**{**env.__dict__, {'rerun': 'continuation', 'score': 'scorer'}[crash_at]: crashing(getattr(env, {'rerun': 'continuation', 'score': 'scorer'}[crash_at]))})
    tr, _, _ = trainer(tmp_path / 'run', log=log, w=w, env=env)
    if crash_at == 'refine': tr.backend = type(tr.backend)(tr.backend.feedback, crashing(tr.backend.refine), None, None, None)
    with pytest.raises(Crash): tr.run()
    before = dict(log.counts); tr2, log2, _ = trainer(tmp_path / 'run', w={'v': 0}); tr2.run()
    gen = [k for kind, k in log.events + log2.events if kind == 'rubric']
    assert len(gen) == len(set(gen))                                                       # one generator call per (trajectory, prompt), ever
    for kind in ('agent', 'doctor_turn', 'final', 'feedback', 'rubric', 'score', 'write', 'rerun', 'reflect', 'refine'):
        assert before.get(kind, 0) + log2.counts.get(kind, 0) == ref_log.counts.get(kind, 0), kind
    a, b = read_json(tmp_path / 'run' / 'run_state.json'), read_json(tmp_path / 'ref' / 'run_state.json')
    assert {k: a[k] for k in ('case_cursor', 'prompt_sha256', 'policy_sha256', 'lineage_cursor')} == {k: b[k] for k in ('case_cursor', 'prompt_sha256', 'policy_sha256', 'lineage_cursor')}


# ---------- missing strategies, malformed, real-component wiring ----------
def test_missing_strategy_stops_the_pipeline(tmp_path):
    w = {'v': 0}; st = Strategies(**{**mock_strategies(w).__dict__, 'credit_assignment': None})
    tr, _, _ = trainer(tmp_path, cfg(cases=1, every=99), w=w, strategies=st)
    with pytest.raises(UnresolvedDecision, match='credit_assignment'): tr.run(max_cases=1)
    assert not (tmp_path / 'cases').exists()
    c = cfg(); c['rubric'] = {**c['rubric'], 'gold_answer_visible': True}
    with pytest.raises(NotImplementedError): trainer(tmp_path / 'g', c)
    shipped = json.loads(Path('configs/online_rl/tree_v1.json').read_text())
    assert 'open_decisions' not in shipped and 'observable' in shipped['notes']['post_hoc_gates']
    with pytest.raises(ValueError, match='unresolved'): trainer(tmp_path / 'u', {**cfg(), 'intervention': {'reruns_per_intervention': 'CALIBRATION_REQUIRED'}})


def test_malformed_rubrics_are_recorded_not_rerun(tmp_path):
    tr, _, _ = trainer(tmp_path, cfg(cases=1, every=99), gen_malformed=True); tr.run(max_cases=1)
    rec = case_record(tmp_path); bad = [c for c in rec['trajectory_rubrics'] if not c['valid']]
    assert bad and all(c['problems'] for c in bad) and len(bad) < len(rec['trajectory_rubrics'])
    assert {e['rubric_id'] for e in rec['effectiveness']} == {c['rubric_id'] for c in rec['trajectory_rubrics'] if c['valid']}


def test_intervention_backend_generates_then_doctor_final_only():
    from src.online_rl.tree import backends as B
    calls=[]
    class Pol:
        def agent_user_prompt(self,case,prefix,t): return 'USER'
        def agent(self,user,seed): calls.append(('agent',user,seed)); return 'fresh reply'
        def continue_from(self,*a): raise AssertionError('No further discussion allowed')
    class Doc:
        def final(self,case,dialogue):
            assert 'ground_truth' not in case and dialogue[-1]['text']=='fresh reply'
            calls.append(('final',dialogue)); return 'Final answer: A'
    iv=B.PolicyInterventions(Pol(),Path('prompts/sherpa/guided_response_template_score0_v1.txt').read_text(),
                            Path('prompts/sherpa/guided_response_template_score2_v2.txt').read_text(),doctor=Doc())
    for level in ('0','2'):
        text=iv.write(CASES[0],[],1,nested_rubric(RUB),level,5)
        result=iv.continue_(CASES[0],[],1,text,9)
        assert result['outcome']==1 and result['additional_discussion_turns']==0
        assert result['termination_reason']=='intervention_doctor_final_only'
    assert [c[0] for c in calls]==['agent','final','agent','final']
    assert 'Score-0' in calls[0][1] and 'Score-2' in calls[2][1]
    with pytest.raises(ValueError): iv.write(CASES[0],[],1,nested_rubric(RUB),'1',0)

def test_two_stage_scorer_maps_no_match(monkeypatch):
    import src.judges.rubric_scoring as RS
    from src.online_rl.tree import backends as B
    def fake(client, *, history_state, target_state, rubrics, purpose):
        assert 'target_turn' not in history_state and target_state['target_turn']['text'] == 'reply'
        return {'applicability_labels': {'r1': 'Y', 'r2': 'N'}, 'behavior_probabilities': {'r1': {'0': 0, '1': 0, '2': 1}}, 'behavior_scores': {'r1': 2}}
    monkeypatch.setattr(RS, 'two_stage_jev_score', fake)
    out = B.two_stage_scorer(None)(PH.case_view(CASES[0]), [], [nested_rubric({**RUB, 'rubric_id': 'r1'}), nested_rubric({**RUB, 'rubric_id': 'r2'})], 'reply')
    assert out == {'r1': {'applicable': True, 'probabilities': {'0': 0, '1': 0, '2': 1}, 'hard': 2}, 'r2': {'applicable': False, 'probabilities': None, 'hard': None}}


def test_feedback_and_generator_backends_without_gold(monkeypatch):
    import src.collaboration.generator as G
    from src.prompt_opt import harness as H
    from src.online_rl.tree import backends as B
    calls = []
    def fake(key,model,system,user,*args):
        calls.append((model,system,user))
        if 'STRUCTURED DISCUSSION DECISION' in system:
            from tests.test_doctor_protocol_repair import decision
            return decision(True)
        if 'VERBALIZE FIXED DECISION' in system:
            return json.dumps({'assessment':'I favor B.','question':'Could you compare the alternatives?'})
        return 'Final answer: option B' if 'final decision' in user.lower() else '{"name": "n"}'
    monkeypatch.setattr(G, 'doctor_call', fake); monkeypatch.setattr(H, 'doctor_call', fake)
    d = B.ApiDoctor('k', 'doc', 'SYS', 64); dlg = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'hi'}]
    d.turn(CASES[0], dlg, 2, 5); d.final(CASES[0], dlg)
    assert 'turn 2 of at most 5' not in calls[0][2] and 'discussion decision' in calls[0][2] and 'final decision' in calls[2][2].lower()
    B.api_feedback('k', 'doc', 'FB', 64)(PH.case_view(CASES[0]), dlg, 'failure')
    log = Log(); tree, _ = tree_for(log, rounds=1, branching=1); leaf = tree.leaves[0]
    tin = PH.freeze_trajectory_input(CASES[0], leaf, PH.DoctorFeedback(leaf.leaf_id, leaf.outcome, 'the Agent ignored my question', 'fb'))
    B.api_rubric_generator('k')('GEN', tin)
    for _, _, user in calls[2:]: assert 'correct_option' not in user and 'atree option 0 is correct' not in user
    assert 'the Agent ignored my question' in calls[4][2] and calls[4][0] == H.SOL


# ---------- team-decided strategies (src/online_rl/tree/team.py) ----------
from src.online_rl.tree import team as T
from src.online_rl.tree import prompt_search as PS
from src.online_rl.tree.prompt_search import unit_values


def test_selected_target_does_not_average_other_matching_turns():
    st = T.team_strategies(None, 1.0, 1.0)
    _, _, (_, nm) = _leaf_eval(Log(), 0.0, k_conds=('turn 99',), st=st)
    leaf, _, (_, matched) = _leaf_eval(Log(), 0.0, k_conds=('any turn',), st=st)
    assert not EF.select([nm], {'mode':'threshold','min_R_O':2/3})[0].retained
    assert len(leaf.path)>1 and matched.target_node_ids == (leaf.path[0],)
    assert matched.R_O == matched.per_node_R_O[leaf.path[0]] and len(matched.runs)==3

def test_prompt_fitness_is_product_of_source_means():
    E = lambda r, status='verified': type('E', (), {'R_O': r, 'status': status})()
    res = {'L1': {'effectiveness': E(1 / 3)}, 'L2': {'effectiveness': E(1.0)}, 'L3': {'effectiveness': E(None, 'incomplete')}, 'L4': {'effectiveness': None}}
    vals = unit_values(res, T.team_strategies(None, 1.0, 1.0))
    assert vals == {'L1': pytest.approx(1 / 3), 'L2': 1.0, 'L4': 0.0}
    outcomes = {'L1':'success','L2':'failure','L3':'failure','L4':'success'}
    assert PS.prompt_fitness(vals, outcomes) is None
    vals['L3'] = 1.0
    assert PS.prompt_fitness(vals, outcomes) == pytest.approx(1 / 6)



def _tree_and_targets(w_outcome=1.0, w_rubric=0.5, r_a=None):
    tree, _ = tree_for(Log(), rounds=3, branching=4)
    return tree, T.node_credit(w_outcome, w_rubric)(tree, RW.leaf_outcomes(tree), r_a or {})


def test_node_outcome_component_is_descendant_leaf_mean():
    tree, t = _tree_and_targets()
    for n in tree.nodes:
        outs = [l.outcome for l in tree.leaves if n.node_id in l.path and l.outcome is not None]
        assert t['nodes'][n.node_id]['outcome_component'] == pytest.approx(sum(outs) / len(outs)) and t['nodes'][n.node_id]['descendant_leaves'] == len(outs)
    leaves = {l.leaf_id for l in tree.leaves}
    assert all(t['nodes'][l]['descendant_leaves'] == 1 for l in leaves)                                         # a leaf node credits its own outcome


def test_four_siblings_are_one_group_and_no_leaf_duplication():
    tree, t = _tree_and_targets(); by_id = {n.node_id: n for n in tree.nodes}
    groups = t['groups']; flat = [nid for g in groups.values() for nid in g]
    assert len(flat) == len(set(flat)) == len(tree.nodes) == 4 + 16 + 64                                        # each response once, not once per leaf
    assert all(len(g) == 4 and len({by_id[n].parent_id for n in g}) == 1 and len({by_id[n].prefix for n in g}) == 1 for g in groups.values())
    seen = []

    class Up:
        def update_groups(self, gs): seen.extend(gs); return {'optimizer_step': True}
    T.sibling_grpo_update(Up())(tree, t)
    assert len(seen) == len(groups) and sum(len(g) for g in seen) == len(tree.nodes)
    assert all(len({r['case_id'] for r in g}) == 1 and len(g) == 4 for g in seen)
    assert sorted(r['trajectory_id'] for g in seen for r in g) == sorted(n.node_id for n in tree.nodes)
    assert all(r['reward'] == t['nodes'][r['trajectory_id']]['reward'] and r['agent_traces'] == [by_id[r['trajectory_id']].trace] for g in seen for r in g)


def test_R_O_is_only_a_weight_inside_R_A():
    S = lambda rid, match, score: ResponseRubricScore('n', rid, 'L', match, score, None, None)
    ss = [S('a', True, 2.0), S('b', True, 0.0), S('c', False, None), S('d', True, 2.0)]
    w = {'a': 1.0, 'b': 1 / 3, 'c': 1.0}                                                                          # d: not retained -> no weight
    assert T.effectiveness_weighted_r_a(ss, w) == pytest.approx((1.0 * 2.0 + (1 / 3) * 0.0) / (1 + 1 / 3))
    assert T.effectiveness_weighted_r_a(ss, {k: 3 * v for k, v in w.items()}) == pytest.approx(T.effectiveness_weighted_r_a(ss, w))   # scale-free
    assert T.effectiveness_weighted_r_a([S('c', False, None)], w) is None and T.effectiveness_weighted_r_a(ss, {'a': 0.0}) is None
    tree, t = _tree_and_targets(2.0, 0.5, r_a={n.node_id: 1.5 for n in tree_for(Log(), rounds=3, branching=4)[0].nodes[:4]})
    for nid, x in t['nodes'].items():
        assert set(x) == {'outcome_component', 'descendant_leaves', 'R_A', 'reward'}                           # no R_O / R_E / V(h) / Q_cov term
        assert x['reward'] == pytest.approx(2.0 * x['outcome_component'] + (0.5 * x['R_A'] if x['R_A'] is not None else 0.0))
    src = inspect.getsource(T) + inspect.getsource(RW)
    assert not re.search(r'q_cov|Q_cov|R_E|V_h|V\(h\)|state_value', src.split('"""', 2)[2])


class GroupUpdater(Updater):
    def __init__(self, w, calls): super().__init__(w); self.calls = calls

    def update_groups(self, groups):
        self.calls.append([[r['trajectory_id'] for r in g] for g in groups]); self.w['v'] += 1
        return {'optimizer_step': True, 'groups_used': len(groups)}


def _team_trainer(tmp, calls, w=None, env=None, log=None):
    w = w if w is not None else {'v': 0}; log = log or Log(); up = GroupUpdater(w, calls)
    st = T.team_strategies(up, 1.0, 0.5)
    return TreeTrainer(cfg(cases=4, every=2, min_traj=4), out_dir=tmp, cases=CASES, env=env or mock_env(log, w), strategies=st, search_backend=search_backend(log),
                       initial_prompt=P0, policy_sha=lambda: f'policy-{w["v"]}', protocol_sha='proto', feedback_prompt_sha256='fb', updater=up), log


def test_team_pipeline_end_to_end_and_resume(tmp_path):
    calls = []; tr, log = _team_trainer(tmp_path / 'ref', calls); tr.run()
    rec = case_record(tmp_path / 'ref'); t = rec['agent_targets']; kept = {d['rubric_id'] for d in rec['selection'] if d['retained']}
    assert set(rec['rubric_weights']) == kept and all(rec['rubric_weights'][r] == next(e['R_O'] for e in rec['effectiveness'] if e['rubric_id'] == r) for r in kept)
    assert rec['update']['optimizer_step'] and len(calls) == 4 and all(len(g) == 2 for g in calls[0])           # branching 2 in the mock config
    assert any(x['R_A'] is not None for x in t['nodes'].values()) and read_json(tmp_path / 'ref' / 'run_state.json')['lineage_cursor'] > 0
    class Crash(Exception): pass
    n = {'k': 0}; log2 = Log(); w2 = {'v': 0}; env = mock_env(log2, w2)
    def cont(*a):
        n['k'] += 1
        if n['k'] == 40: raise Crash()
        return env.continuation(*a)
    env = type(env)(**{**env.__dict__, 'continuation': cont}); calls2 = []
    tr2, _ = _team_trainer(tmp_path / 'run', calls2, w=w2, env=env, log=log2)
    with pytest.raises(Crash): tr2.run()
    before = dict(log2.counts); tr3, log3 = _team_trainer(tmp_path / 'run', calls2, w={'v': 0}); tr3.run()
    for kind in ('agent', 'feedback', 'rubric', 'score', 'write', 'rerun', 'reflect', 'refine'):
        assert before.get(kind, 0) + log3.counts.get(kind, 0) == log.counts.get(kind, 0), kind
    for i in range(4):
        a, b = case_record(tmp_path / 'run', i), case_record(tmp_path / 'ref', i)
        assert a['agent_targets'] == b['agent_targets'] and a['rubric_weights'] == b['rubric_weights']


# ---------- post-hoc gates ----------
def _gate_answers(monkeypatch, **vals):
    import src.judges.rubric_scoring as RS
    seen = []
    def fake(client, state, questions, *, purpose, max_rubrics_per_call):
        seen.append((state, set(questions))); return {k: {'value': vals.get(k, 'N')} for k in questions}, []
    monkeypatch.setattr(RS, '_batched_jev_decisions', fake); return seen


def _tin():
    tree, _ = tree_for(Log(), rounds=2, branching=1); leaf = tree.leaves[0]
    return PH.freeze_trajectory_input(CASES[0], leaf, PH.DoctorFeedback(leaf.leaf_id, leaf.outcome, 'fb', 'fb'))


RUB = {'name': 'Ask for the reason', 'applicability_condition': 'the physician states a conclusion', 'required_behavior': 'ask what supports it',
       'scores': {'0': 'no question', '1': 'vague question', '2': 'specific question about the stated evidence'}, 'rubric_id': 'r'}


def test_posthoc_gates_subset(monkeypatch):
    from src.online_rl.tree import backends as B
    tin = _tin(); cases = {CASES[0]['case_id']: CASES[0]}
    seen = _gate_answers(monkeypatch, observable='N')
    g = B.posthoc_gates(None, cases)(tin, nested_rubric(RUB, tin.agent_node_ids[0]))
    assert g['passed'] and g['observable'] is False and g['unresolved'] == ['observable']                        # observable recorded, not blocking
    assert seen[0][1] == {'answer_presupposition', 'new_domain_premise', 'observable'}                          # no pre-action questions
    st = json.dumps(seen[0][0]); assert 'correct_option' not in st and 'Final answer' not in st and 'FINAL' not in st
    _gate_answers(monkeypatch, new_domain_premise='Y'); assert not B.posthoc_gates(None, cases)(tin, nested_rubric(RUB, tin.agent_node_ids[0]))['passed']
    _gate_answers(monkeypatch, answer_presupposition='Y'); assert not B.posthoc_gates(None, cases)(tin, nested_rubric(RUB, tin.agent_node_ids[0]))['passed']
    _gate_answers(monkeypatch)
    g = B.posthoc_gates(None, cases)(tin, nested_rubric({**RUB, 'scores': {**RUB['scores'], '2': 'endorse option A'}}, tin.agent_node_ids[0])); assert not g['passed'] and g['leakage_deterministic']
    g = B.posthoc_gates(None, cases)(tin, nested_rubric({**RUB, 'scores': {**RUB['scores'], '2': 'say atree option 0'}}, tin.agent_node_ids[0])); assert 'gold_option_text' in g['leakage_deterministic']


def test_gate_failure_makes_rubric_invalid_and_is_cached():
    tin = _tin(); cache = CallCache(None); n = {'gate': 0, 'gen': 0}
    def gen(p, t): n['gen'] += 1; return {'raw': '{}', 'rubric': nested_rubric(RUB, t.agent_node_ids[0])}
    def gate(t, r): n['gate'] += 1; return {'passed': False}
    r = PH.generate_rubric(P0, sha256_text(P0), tin, gen, cache, gate)
    assert not r.valid and r.problems == () and r.gate == {'passed': False}
    assert EF.evaluate_rubric(CASES[0], {}, _leaf_of(tin), r, scorer=None, writer=None, continuation=None, reruns=3, run_seed=0, policy_sha256='p',
                              strategies=Strategies(), cache=cache) == ([], None)                                # not verified, no reruns
    PH.generate_rubric(P0, sha256_text(P0), tin, gen, cache, gate); assert n == {'gate': 1, 'gen': 1}            # cache hit, no repeated call


def _leaf_of(tin):
    return type('L', (), {'leaf_id': tin.leaf_id, 'dialogue': tin.dialogue, 'outcome': float(tin.outcome == 'success'), 'path': tin.agent_node_ids})()


# ---------- one rubric per trajectory: verification semantics ----------
def _fixed_eval(outcome_wanted, rerun_outcomes, k):
    log = Log(); w = {'v': 0}; tree, _ = tree_for(log, rounds=2, branching=2); cache = CallCache(None)
    env = mock_env(log, w, k_conds=('turn 1',)); nodes = {n.node_id: n for n in tree.nodes}; it = iter(rerun_outcomes)
    cont = lambda *a: (lambda o: {'final_option': 'A' if o == 1.0 else ('B' if o == 0.0 else ''), 'outcome': o})(next(it))
    leaf = next(l for l in tree.leaves if l.outcome == outcome_wanted); n0 = log.counts.get('rubric', 0)
    fb = PH.doctor_feedback(CASES[0], leaf, env.feedback_fn, 'fb', cache); tin = PH.freeze_trajectory_input(CASES[0], leaf, fb)
    rub = PH.generate_rubric(P0, sha256_text(P0), tin, env.generate_fn, cache)
    _, e = EF.evaluate_rubric(CASES[0], nodes, leaf, rub, scorer=env.scorer, writer=env.writer, continuation=cont, reruns=k, run_seed=0,
                              policy_sha256='p', strategies=T.team_strategies(None, 1.0, 1.0), cache=cache)
    return log, log.counts['rubric'] - n0, rub, e


def test_failure_s_is_successes_over_three():
    _, _, _, e = _fixed_eval(0.0, [1.0, 0.0, 1.0], 3)
    assert e.level == '2' and e.R_O == pytest.approx(2/3)
    _, _, _, incomplete = _fixed_eval(0.0, [1.0, None, 1.0], 3)
    assert incomplete.status == 'incomplete' and incomplete.R_O is None

def test_success_s_is_failures_over_three():
    _, _, _, e = _fixed_eval(1.0, [0.0, 1.0, 0.0], 3)
    assert e.level == '0' and e.R_O == pytest.approx(2/3)
    _, _, _, incomplete = _fixed_eval(1.0, [0.0, None, 0.0], 3)
    assert incomplete.status == 'incomplete' and incomplete.R_O is None

@pytest.mark.parametrize('k', [1, 3, 7])
def test_rerun_count_is_exactly_three_not_rubric_count(k):
    if k != 3:
        with pytest.raises(ValueError, match='exactly 3'): _fixed_eval(0.0, [1.0]*k, k)
        return
    log, gen_calls, rub, e = _fixed_eval(0.0, [1.0]*3, 3)
    assert gen_calls == 1 and e.rubric_id == rub.rubric_id
    assert log.counts['write']==3 and len(e.runs)==3
    assert len({r.intervention_seed for r in e.runs})==3
    shipped=json.loads(Path('configs/online_rl/tree_v1.json').read_text())
    assert shipped['intervention']['reruns_per_intervention']==3
    assert shipped['selection']=={'mode':'threshold','min_R_O':2/3}

def test_keep_remove_is_one_decision_per_trajectory_rubric(tmp_path):
    tr, _, _ = trainer(tmp_path, cfg(cases=1, every=99)); tr.run(max_cases=1)
    rec = case_record(tmp_path); per_leaf = {}
    for d in rec['selection']: per_leaf.setdefault(d['leaf_id'], []).append(d)
    rub = {r['leaf_id']: r['rubric_id'] for r in rec['trajectory_rubrics']}
    assert per_leaf and all(len(v) == 1 and v[0]['rubric_id'] == rub[l] and v[0]['policy'] == 'threshold' for l, v in per_leaf.items())
    assert not hasattr(PH, 'generate_candidates') and not hasattr(EF, 'evaluate_candidates')


# ---------- prompt-optimizer reflection input ----------
from src.online_rl.tree.prompt_search import EvidenceUnit, ScoringDeps, reflection_view, score_prompt


def _units_and_deps(log, k_conds, gate_fn=None, continuation=None, scorer=None, reruns=3):
    w = {'v': 0}; tree, _ = tree_for(log, rounds=2, branching=2); env = mock_env(log, w, k_conds=k_conds); cache = CallCache(None)
    nodes = {n.node_id: n for n in tree.nodes}; units = []
    for leaf in tree.leaves:
        fb = PH.doctor_feedback(CASES[0], leaf, env.feedback_fn, 'fb', cache)
        units.append(EvidenceUnit(CASES[0]['case_id'], leaf, tuple(nodes[n] for n in leaf.path), PH.freeze_trajectory_input(CASES[0], leaf, fb), 'p'))
    deps = ScoringDeps(reruns, 0, env.generate_fn, gate_fn, scorer or env.scorer, env.writer, continuation or env.continuation, 'p',
                       T.team_strategies(None, 1.0, 1.0), cache)
    return units, deps, cache


def test_reflection_view_includes_full_original_trajectory():
    log = Log(); units, deps, _ = _units_and_deps(log, ('any turn',))
    view = reflection_view(units, score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps))
    assert all(u.tin.dialogue[-1][1] == 'FINAL' for u in units)                                         # the stored trajectory keeps it
    for row, u in zip(view, units):
        assert row['full_trajectory'] == PH.to_dialogue(u.tin.dialogue)
        assert u.tin.dialogue[-1][3] in json.dumps(row)
        assert len(row['dialogue']) == len(u.tin.dialogue) - 1


def test_reflection_view_from_trainer_includes_original_final(tmp_path):
    views = []; tr, log, _ = trainer(tmp_path, cfg(cases=4, every=2, min_traj=4))
    fb = tr.backend.feedback; tr.backend = type(tr.backend)(lambda p, v, n: views.append(v) or fb(p, v, n), tr.backend.refine, None, None, None); tr.run()
    assert views and all('full_trajectory' in row for view in views for row in view)


def test_gate_failure_reasons_are_exposed_without_gold():
    gate = lambda tin, r: {'passed': False, 'schema': True, 'leakage_deterministic': ['mentions_option_letter', 'gold_option_text'],
                           'answer_presupposition': True, 'new_domain_premise': True, 'observable': False,
                           'reasons': ["deterministic leakage: ['mentions_option_letter', 'gold_option_text']", 'answer presupposition', 'new domain premise']}
    log = Log(); units, deps, _ = _units_and_deps(log, ('any turn',), gate_fn=gate)
    view = reflection_view(units, score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps))
    for row in view:
        assert row['valid'] is False and row['problems'] == ['deterministic leakage: option letter', 'answer copy: option text', 'answer_presupposition', 'new_domain_premise']
        assert 'gold' not in json.dumps(row) and row['verification'] == 'not_verified' and row['interventions'] == [] and row['R_O'] is None
    log2 = Log(); units, deps, _ = _units_and_deps(log2, ('any turn',), gate_fn=lambda t, r: {'passed': False, 'schema': False, 'reasons': ["schema: ['bad']"]})
    assert reflection_view(units, score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps))[0]['problems'] == ["schema: ['bad']"]
    tin = _tin(); bad = PH.generate_rubric(P0, sha256_text(P0), tin, lambda p, t: {'raw': 'x', 'rubric': None}, CallCache(None))
    row = reflection_view([EvidenceUnit('tc0', _leaf_of(tin), (), tin, 'p')], {PS.evidence_id(EvidenceUnit('tc0', _leaf_of(tin), (), tin, 'p')): {'rubric': bad, 'effectiveness': None}})[0]
    assert row['problems'] == ['no rubric object']                                                    # parse / format


@pytest.mark.parametrize('kind', ['gate', 'no_match'])
def test_unverified_rubrics_get_no_rerun_and_count_zero(kind):
    log = Log()
    units, deps, _ = _units_and_deps(log, ('turn 99',) if kind == 'no_match' else ('any turn',), gate_fn=(lambda t, r: {'passed': False}) if kind == 'gate' else None)
    res = score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps)
    assert log.counts.get('write', 0) == 0 and log.counts.get('rerun', 0) == 0
    vals = unit_values(res, deps.strategies)
    assert vals == {PS.evidence_id(u): 0.0 for u in units} and PS.prompt_fitness(vals, {PS.evidence_id(u): u.tin.outcome for u in units}) == 0.0
    assert all(r['verification'] == ('no_match' if kind == 'no_match' else 'not_verified') for r in reflection_view(units, res))


@pytest.mark.parametrize('where', ['continuation', 'scorer', 'generator'])
def test_operational_failure_is_an_error_not_zero(where):
    class ApiDown(RuntimeError): pass
    def down(*a, **k): raise ApiDown('timeout')
    log = Log(); units, deps, cache = _units_and_deps(log, ('any turn',), continuation=down if where == 'continuation' else None,
                                                     scorer=down if where == 'scorer' else None)
    if where == 'generator': deps = ScoringDeps(**{**deps.__dict__, 'generate_fn': down})
    with pytest.raises(ApiDown): score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps)
    kind = {'continuation': 'rerun', 'scorer': 'rubric_score', 'generator': 'rubric_generation'}[where]
    assert cache.calls.get(kind, 0) == 0                                                               # nothing cached -> resume retries it


def test_selected_target_provenance_is_kept():
    units,deps,_=_units_and_deps(Log(), ('any turn',))
    res=score_prompt(P0,sha256_text(P0),units,{CASES[0]['case_id']:CASES[0]},deps)
    for row,u in zip(reflection_view(units,res),units):
        eff=res[PS.evidence_id(u)]['effectiveness']; iv=row['interventions']
        assert row['verification']=='single_target' and len(iv)==1
        assert iv[0]['response_node_id']==u.leaf.path[0] and iv[0]['agent_turn']==1
        assert len(iv[0]['rerun_outcomes'])==3 and iv[0]['level']==eff.level
        assert row['R_O']==pytest.approx(EF.target_rate(eff.level,iv[0]['rerun_outcomes']))

LEAKY = {**RUB, 'scores': {**RUB['scores'], '2': 'insist on atree option 0 and option A'}}

@pytest.mark.parametrize('gate,redacted', [
    ({'passed': False, 'schema': True, 'leakage_deterministic': ['gold_option_text'], 'answer_presupposition': False, 'new_domain_premise': False}, True),
    ({'passed': False, 'schema': True, 'leakage_deterministic': [], 'answer_presupposition': True, 'new_domain_premise': False}, True),
    ({'passed': False, 'schema': True, 'leakage_deterministic': [], 'answer_presupposition': False, 'new_domain_premise': True}, False),
    ({'passed': False, 'schema': False, 'reasons': ["schema: ['bad']"]}, False),
    ({'passed': True, 'schema': True, 'leakage_deterministic': [], 'answer_presupposition': False, 'new_domain_premise': False}, False)])
def test_answer_exposing_rubric_text_is_withheld_from_reflection(gate, redacted):
    log = Log(); units, deps, _ = _units_and_deps(log, ('any turn',), gate_fn=lambda t, r: gate)
    deps = ScoringDeps(**{**deps.__dict__, 'generate_fn': lambda p, t: {'raw': json.dumps(LEAKY), 'rubric': nested_rubric(LEAKY, t.agent_node_ids[0])}})
    res = score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps); view = reflection_view(units, res)
    for row in view:
        text = json.dumps(row)
        if redacted:
            assert row['rubric'] is None and 'atree option 0' not in text and row['analysis'] is None and row['problems'] and row['valid'] is False
        else:
            assert row['rubric']['scores']['2']['behavior'] == LEAKY['scores']['2']
    assert all(r['rubric'] is not None for r in reflection_view(*_valid_view()))       # ordinary valid rubrics stay visible


def _valid_view():
    log = Log(); units, deps, _ = _units_and_deps(log, ('any turn',))
    return units, score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps)


def test_approved_prompt_hashes_are_pinned_and_checked():
    import importlib.util
    spec = importlib.util.spec_from_file_location('run_tree', 'scripts/online_rl/run_tree.py'); RT = importlib.util.module_from_spec(spec); spec.loader.exec_module(RT)
    shipped = json.loads(Path('configs/online_rl/tree_v1.json').read_text())
    assert len(RT.PROMPT_HASHES) == 6 and shipped['intervention']['score2_template_sha256'] == 'd4284095d47215b827b9db06345a193db74b7f769269ae2c8c3c358af344b3e0'
    for sec, f, h in RT.PROMPT_HASHES:
        assert shipped[sec][h] == sha256_text(Path(shipped[sec][f]).read_text()) == RT.file_sha(shipped[sec][f])
    assert not [b for b in RT.blockers(shipped) if 'differs' in b]
    for sec, f, h in RT.PROMPT_HASHES:
        bad = json.loads(json.dumps(shipped)); bad[sec][h] = '0' * 64
        assert f'{sec}.{f} differs from {sec}.{h}' in RT.blockers(bad)                     # any edited file blocks the launch


# ---------- intervention fidelity and Doctor feedback quality ----------
S2_V1, S2_V2 = Path('prompts/prompt_opt/guided_response_template_v1.txt'), Path('prompts/sherpa/guided_response_template_score2_v2.txt')


def test_score2_v2_requires_performing_the_behavior_not_a_conclusion():
    t = S2_V2.read_text()
    assert 'in the reply itself' in t and 'without performing that behavior does not meet Score-2' in t
    assert all(w in t for w in ('test, compare, verify, reassess', 'ask for support', 'Invent no case facts', 'do not mention the rubric'))
    assert t.count('{rubric_json}') == 1 and '{' not in t.replace('{rubric_json}', '')
    for gold in ('correct option', 'gold', 'ground truth'): assert gold not in t.lower()



def test_configured_score2_template_receives_behavior_projection():
    from src.online_rl.tree import backends as B
    seen=[]
    class Pol:
        def agent_user_prompt(self,case,prefix,t): return 'USER'
        def agent(self,user,seed): seen.append(user); return 'reply'
    B.PolicyInterventions(Pol(),'S0 {rubric_json}',S2_V2.read_text(),doctor=object()).write(CASES[0],[],1,nested_rubric(RUB),'2',3)
    assert 'without performing that behavior does not meet Score-2' in seen[0]
    assert '"target_behavior": "specific question about the stated evidence"' in seen[0]
    assert '"scores"' not in seen[0] and '"required_behavior"' not in seen[0]
    assert 'selected_agent_turn_id' not in seen[0] and 'analysis' not in seen[0]

def test_feedback_transcript_has_stable_agent_reply_labels():
    dlg = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'start'}, {'speaker': 'agent', 'stage': 'A1', 'round': 1, 'text': 'first'},
           {'speaker': 'doctor', 'stage': 'D2', 'round': 2, 'text': 'more'}, {'speaker': 'agent', 'stage': 'A2', 'round': 2, 'text': 'second'},
           {'speaker': 'doctor', 'stage': 'FINAL', 'round': 0, 'text': 'Final answer: option B'}]
    msg = PH.feedback_message(PH.case_view(CASES[0]), dlg, 'failure')
    assert 'Agent reply 1: first' in msg and 'Agent reply 2: second' in msg and 'Agent reply 3' not in msg and 'gave 2 replies' in msg
    assert 'Doctor (final decision): Final answer: option B' in msg and 'correct_option' not in msg
    assert PH.feedback_label_problems('Agent reply 2 echoed me; the third reply and Agent reply 4 were vague', 2) == ['reply 3', 'reply 4']
    assert PH.feedback_label_problems('Agent reply 1 challenged me; the second response compared evidence', 2) == []


def test_feedback_retries_once_on_a_nonexistent_reply_label(monkeypatch):
    import src.collaboration.generator as G
    from src.online_rl.tree import backends as B
    dlg = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'start'}, {'speaker': 'agent', 'stage': 'A1', 'round': 1, 'text': 'x'},
           {'speaker': 'doctor', 'stage': 'FINAL', 'round': 0, 'text': 'Final answer: option B'}]
    for replies, expect_calls, expect in ((['the third reply misled me', 'Agent reply 1 echoed my view'], 2, 'Agent reply 1 echoed my view'),
                                          (['Agent reply 1 echoed my view'], 1, 'Agent reply 1 echoed my view'),
                                          (['the third reply misled me', 'the fourth reply too'], 2, 'the fourth reply too')):
        calls = []; it = iter(replies)
        monkeypatch.setattr(G, 'doctor_call', lambda key, model, system, user, *a: calls.append(user) or next(it))
        assert B.api_feedback('k', 'doc', 'FB', 64)(PH.case_view(CASES[0]), dlg, 'failure') == expect and len(calls) == expect_calls
        if expect_calls == 2: assert 'reply 3' in calls[1] and 'only 1 replies' in calls[1]


def test_feedback_v2_contract():
    t = Path('prompts/sherpa/doctor_feedback_v2.txt').read_text()
    assert 'Agent reply 1' in t and 'never mention a reply label that is not in the transcript' in t and 'Do not count turns yourself' in t
    assert 'Do not explain or judge medical facts' in t and 'Describe collaborative behavior only' in t
    assert all(w in t for w in ('accepted', 'challenged', 'compared the evidence', 'asked for support', 'revised', 'echoed', 'failed to'))
    assert 'separate gold-answer field' in t and 'Do not state, copy or encode a correct option' in t
    shipped = json.loads(Path('configs/online_rl/tree_v1.json').read_text())
    assert shipped['feedback']['prompt_file'] == 'prompts/sherpa/doctor_feedback_v2.txt' and shipped['feedback']['prompt_sha256'] == '2cec09b39cd08e1f56661ae2188e73405cd30cc3c60116a0a59de3ad7dacc2b4'
    smoke = Path('configs/online_rl/tree_smoke.json')
    if smoke.exists(): assert json.loads(smoke.read_text())['feedback']['prompt_file'] == 'prompts/sherpa/doctor_feedback_v1.txt'


# ---------- transcript labels shared by Doctor feedback and Rubric Generator ----------
def test_generator_transcript_uses_the_feedback_reply_labels(tmp_path):
    log = Log(); tree, _ = tree_for(log, rounds=2, branching=1); leaf = tree.leaves[0]
    tin = PH.freeze_trajectory_input(CASES[0], leaf, PH.DoctorFeedback(leaf.leaf_id, leaf.outcome, 'Agent reply 2 echoed me', 'fb'))
    gen, fb = PH.trajectory_message(tin), PH.feedback_message(PH.case_view(CASES[0]), PH.to_dialogue(leaf.dialogue), 'failure')
    labels = lambda m: re.findall(r'^(Agent reply \d+|Doctor(?: \(final decision\))?)(?: \[turn_id=[^]]+\])?:', m, re.M)
    assert labels(gen) == labels(fb) == ['Doctor', 'Agent reply 1', 'Doctor', 'Agent reply 2', 'Doctor (final decision)']
    for turn in leaf.dialogue: assert turn[3] in gen                                                   # content unchanged, same order
    assert [gen.index(t[3]) for t in leaf.dialogue] == sorted(gen.index(t[3]) for t in leaf.dialogue)
    assert 'correct_option' not in gen and 'ground_truth' not in gen
    assert hashlib.sha256(Path('prompts/sherpa/trajectory_rubric_generator_P0.txt').read_bytes()).hexdigest() == \
        json.loads(Path('configs/online_rl/tree_v1.json').read_text())['rubric']['prompt_sha256']


def test_transcript_format_is_part_of_the_post_hoc_cache_keys(monkeypatch):
    log = Log(); tree, env = tree_for(log, rounds=2, branching=1); leaf = tree.leaves[0]; cache = CallCache(None)
    fb = PH.doctor_feedback(CASES[0], leaf, env.feedback_fn, 'fb', cache); tin = PH.freeze_trajectory_input(CASES[0], leaf, fb)
    PH.generate_rubric(P0, sha256_text(P0), tin, env.generate_fn, cache)
    keys = {r['kind']: r['identity'].get('transcript') for r in cache.store.values()}
    assert keys['feedback'] == keys['rubric_generation'] == PH.TRANSCRIPT_FORMAT
    monkeypatch.setattr(PH, 'TRANSCRIPT_FORMAT', 'unlabelled_v0'); n = log.counts['feedback']
    PH.doctor_feedback(CASES[0], leaf, env.feedback_fn, 'fb', cache); assert log.counts['feedback'] == n + 1     # a format change misses the old cache


def test_score2_v2_is_pinned_and_v1_untouched():
    shipped = json.loads(Path('configs/online_rl/tree_v1.json').read_text())['intervention']
    assert shipped['score2_template_file'] == str(S2_V2) and shipped['score2_template_sha256'] == 'd4284095d47215b827b9db06345a193db74b7f769269ae2c8c3c358af344b3e0'
    assert hashlib.sha256(S2_V2.read_bytes()).hexdigest() == shipped['score2_template_sha256']
    assert hashlib.sha256(S2_V1.read_bytes()).hexdigest() == 'b98732fa33d09dc8eeb9df7a2be0fa0d6d5bbefa041d648f9628d6cbe3fa8d5b'
    smoke = Path('configs/online_rl/tree_smoke.json')                                                   # local provenance of the completed smoke
    if smoke.exists(): assert json.loads(smoke.read_text())['intervention']['score2_template_file'] == str(S2_V1)


def test_prompt_event_defers_one_sided_sources_and_keeps_evidence(tmp_path):
    log = Log(); w = {'v': 0}; env = mock_env(log, w)
    env = type(env)(**{**env.__dict__, 'outcome_fn': lambda *args: 1.0})
    tr, _, _ = trainer(tmp_path, cfg(cases=3, every=1), log=log, w=w, env=env)
    tr.run(max_cases=1)
    event = read_json(tmp_path / 'prompt_events/event_0001.json')
    assert event['decision']['status'] == 'deferred'
    assert tr.state.last_event_cursor == 0 and tr.state.prompt_sha256 == sha256_text(P0)
    assert log.counts.get('reflect', 0) == 0
    assert tr._units()


def test_score0_intervention_excludes_ideal_behavior():
    from src.online_rl.tree import backends as B
    rubric=nested_rubric(RUB)
    rubric['candidate_rubric']['scores']['2']['behavior']='IDEAL_BEHAVIOR_SENTINEL'
    instruction=B.PolicyInterventions(object(),'TARGET {rubric_json}','IDEAL {rubric_json}',doctor=object()).instruction(rubric,'0')
    assert 'IDEAL_BEHAVIOR_SENTINEL' not in instruction
    assert '"target_level": "0"' in instruction
    assert rubric['candidate_rubric']['scores']['0']['behavior'] in instruction
