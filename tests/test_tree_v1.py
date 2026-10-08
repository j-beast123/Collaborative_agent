"""SHERPA case-tree pipeline: mocks only (no API, no GPU)."""
import collections, hashlib, inspect, json, re
from pathlib import Path
import pytest
from src.online_rl.integrated.prompt_search import SearchBackend
from src.online_rl.integrated.records import read_json, sha256_text
from src.online_rl.tree import effectiveness as EF, posthoc as PH, reward as RW
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.decisions import Strategies, UnresolvedDecision
from src.online_rl.tree.records import ResponseRubricScore
from src.online_rl.tree.rollout import build_tree, tree_counts
from src.online_rl.tree.scoring import score_nodes
from src.online_rl.tree.trainer import TreeEnv, TreeTrainer

CASES = [{'case_id': f'tc{i}', 'question': f'Synthetic tree item {i}: which mechanism?', 'scenario': '', 'options': {k: f'{k.lower()}tree option {i}' for k in 'ABCD'},
          'ground_truth': {'correct_option': 'A', 'correct_text': f'atree option {i}'}} for i in range(4)]
P0 = 'Rubric generator for completed discussions. Return JSON with "name", "applicability_condition", "required_behavior" and "scores". VARIANT=0'
h = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
CONDS = ('turn 1', 'turn 99', 'any turn')                    # per trajectory: single target / No Match / every turn


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
        return {'raw': '{}', 'rubric': {'name': f'r{i % 7}', 'applicability_condition': k_conds[i % len(k_conds)], 'required_behavior': f'behave v{v} i{i % 7}',
                                        'scores': {'0': 'bad', '1': 'partial', '2': 'good'}}}

    def scorer(view, prefix, rubrics, reply):
        log.add('score'); depth = sum(d['speaker'] == 'agent' for d in prefix) + 1; q = int(re.search(r'q(\d)', reply).group(1)); out = {}
        for r in rubrics:
            c = r['applicability_condition']; app = c == 'any turn' or c == f'turn {depth}'
            out[r['rubric_id']] = {'applicable': app, 'probabilities': {'0': 1 - q / 2, '1': 0.0, '2': q / 2} if app else None, 'hard': (2 if q == 2 else 0) if app else None}
        return out

    def writer(case, prefix, turn, rubric, level, seed):
        log.add('write', level); return f'[{level}] {rubric["required_behavior"]} q{2 if level == "2" else 0}'

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
            'intervention': {'reruns_per_intervention': k}, 'selection': {'mode': 'threshold', 'min_R_O': 0.5},
            'acceptance': {'rule': 'stratified_paired_bootstrap_J', 'delta': 0.0, 'p_min': 0.5, 'minimum_success_trajectories': 1, 'minimum_failure_trajectories': 1, 'bootstrap_n': 200, 'bootstrap_seed': 0},
            'prompt_update': {'enabled': True, 'every_n_cases': every, 'minimum_trajectories': min_traj, 'reruns_per_intervention': 3, 'acceptance_fraction': 0.5,
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
    assert len({r.intervention_text for r in e.runs}) == 1 and log.counts['write'] == 1                       # one fixed reply, K continuations
    target = 1.0 if level == '2' else 0.0
    assert e.R_O == pytest.approx(sum(r.outcome == target for r in e.runs) / 3) == e.per_node_R_O[leaf.path[0]]
    assert all(r.continuation_policy_sha256 == 'p' for r in e.runs)


def test_target_rate_convention():
    assert EF.target_rate('2', [1.0, 1.0, 0.0]) == pytest.approx(2 / 3) and EF.target_rate('0', [1.0, 0.0, None]) == pytest.approx(1 / 3)
    assert EF.target_rate('2', [None, None, 1.0]) == pytest.approx(1 / 3)
    with pytest.raises(ValueError): EF.intervention_level(None)


def test_no_match_and_multi_target():
    log = Log(); _, _, (src, nm) = _leaf_eval(log, 0.0, k_conds=('turn 99',)); leaf, _, (_, mt) = _leaf_eval(Log(), 0.0, k_conds=('any turn',))
    assert nm.status == 'no_match' and nm.R_O is None and nm.runs == () and nm.target_node_ids == ()
    assert mt.status == 'multi_target' and mt.target_node_ids == leaf.path and set(mt.per_node_R_O) == set(leaf.path)
    assert len(mt.runs) == 3 * len(leaf.path)                                                                  # every matched response, no pick
    with pytest.raises(UnresolvedDecision, match='multi_target'): _leaf_eval(Log(), 0.0, k_conds=('any turn',), multi=False)
    assert any(not s.match and s.score is None for s in src)


# ---------- selection, response-level scoring, reward ----------
def test_selection_scoring_and_provenance(tmp_path):
    tr, _, _ = trainer(tmp_path, cfg(cases=1, every=99)); tr.run(max_cases=1)
    rec = case_record(tmp_path); effs = {e['rubric_id']: e for e in rec['effectiveness']}
    assert {d['rubric_id'] for d in rec['selection']} == set(effs)
    for d in rec['selection']:
        e = effs[d['rubric_id']]
        assert d['retained'] == (e['R_O'] is not None and e['R_O'] >= 0.5) and e['status'] in d['reason']
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


def test_selection_requires_explicit_calibration_values():
    e = [type('E', (), {'rubric_id': f'r{i}', 'leaf_id': 'L', 'R_O': i / 3, 'status': 'single_target'})() for i in range(3)]
    assert [d.retained for d in EF.select(e, {'mode': 'threshold', 'min_R_O': 0.5})] == [False, False, True]
    for bad in ({'mode': 'CALIBRATION_REQUIRED'}, {'mode': 'threshold', 'min_R_O': 'CALIBRATION_REQUIRED'}, {'mode': 'top_k', 'k': 2}):
        with pytest.raises(ValueError): EF.select(e, bad)


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
    assert acc['dJ'] == pytest.approx(acc['candidate']['J'] - acc['parent']['J']) == pytest.approx(acc['observed_dJ'])
    assert set(acc['checks']) == {'minimum_success_trajectories', 'minimum_failure_trajectories', 'delta', 'bootstrap'}
    assert {'p_gain', 'n_success', 'n_failure', 'bootstrap_n', 'bootstrap_seed'} <= set(acc)
    for scores in (ev['parent_search'], acc['parent'], acc['candidate'], *(v['metrics']['search'] for v in ev['versions'] if v['metrics']['search'])):
        assert {'S_success', 'S_failure', 'J', 'n_success', 'n_failure', 'invalid', 'no_match', 'per_trajectory'} <= set(scores)
        assert scores['J'] == pytest.approx(scores['S_success'] * scores['S_failure']) and scores['n_success'] and scores['n_failure']
    assert ev['reruns_per_intervention'] == 3
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


def test_intervention_backends_reuse_the_guided_precedent(monkeypatch):
    from src.prompt_opt import harness as H
    from src.online_rl.tree import backends as B
    calls = []

    class Pol:
        def agent_user_prompt(self, case, prefix, t): calls.append(('prompt', t)); return 'USER'
        def agent(self, user, seed): calls.append(('agent', user, seed)); return 'reply'
        def continue_from(self, case, prefix, t, text, seed): calls.append(('continue', t, text, seed)); return {'final_option': 'A', 'correct': 1.0, 'termination_reason': 'x', 'agent_turns': 3}
    pol = Pol(); rubric = {'name': 'n', 'applicability_condition': 'c', 'required_behavior': 'b', 'scores': {'0': 'z', '1': 'o', '2': 't'}, 'rubric_id': 'r'}
    guided = []; real_guided = H.Env.guided
    monkeypatch.setattr(H.Env, 'guided', lambda self, *a: guided.append(self) or real_guided(self, *a))
    iv = B.PolicyInterventions(pol, Path('prompts/sherpa/guided_response_template_score0_v1.txt').read_text())
    iv.write(CASES[0], [], 1, rubric, '2', 5); iv.write(CASES[0], [], 1, rubric, '0', 6)
    assert guided == [pol] and 'Score-2' in calls[1][1] and 'Score-0' in calls[3][1] and 'Score-2' not in calls[3][1]       # Env.guided as-is for 2
    assert iv.continue_(CASES[0], [], 1, 'reply', 9) == {'final_option': 'A', 'outcome': 1.0, 'termination_reason': 'x', 'agent_turns': 3}
    assert calls[-1] == ('continue', 1, 'reply', 9)                                                              # same policy continues
    with pytest.raises(ValueError): iv.write(CASES[0], [], 1, rubric, '1', 0)
    class Sampler:
        def sample_completions(self, case, prefix, turn, seeds):
            return [{'prompt_text': 'p', 'prompt_input_ids': [1], 'completion_ids': [2], 'old_policy_logprobs': [-1.0], 'completion_text': 'x', 'seed': seeds[0], 'stage': 'A1'}]
    assert set(B.PolicyAgent(Sampler()).sample(CASES[0], [], 1, 7)) == {'prompt_text', 'prompt_input_ids', 'completion_ids', 'old_policy_logprobs', 'completion_text', 'seed'}


def test_two_stage_scorer_maps_no_match(monkeypatch):
    import src.judges.rubric_scoring as RS
    from src.online_rl.tree import backends as B
    def fake(client, *, history_state, target_state, rubrics, purpose):
        assert 'target_turn' not in history_state and target_state['target_turn']['text'] == 'reply'
        return {'applicability_labels': {'r1': 'Y', 'r2': 'N'}, 'behavior_probabilities': {'r1': {'0': 0, '1': 0, '2': 1}}, 'behavior_scores': {'r1': 2}}
    monkeypatch.setattr(RS, 'two_stage_jev_score', fake)
    out = B.two_stage_scorer(None)(PH.case_view(CASES[0]), [], [{'rubric_id': 'r1'}, {'rubric_id': 'r2'}], 'reply')
    assert out == {'r1': {'applicable': True, 'probabilities': {'0': 0, '1': 0, '2': 1}, 'hard': 2}, 'r2': {'applicable': False, 'probabilities': None, 'hard': None}}


def test_feedback_and_generator_backends_without_gold(monkeypatch):
    import src.collaboration.generator as G
    from src.prompt_opt import harness as H
    from src.online_rl.tree import backends as B
    calls = []
    fake = lambda key, model, system, user, *a: calls.append((model, system, user)) or ('Final answer: option B' if 'final decision' in user.lower() else '{"name": "n"}')
    monkeypatch.setattr(G, 'doctor_call', fake); monkeypatch.setattr(H, 'doctor_call', fake)
    d = B.ApiDoctor('k', 'doc', 'SYS', 64); dlg = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'hi'}]
    d.turn(CASES[0], dlg, 2, 5); d.final(CASES[0], dlg)
    assert 'turn 2 of at most 5' in calls[0][2] and 'final decision' in calls[1][2].lower()
    B.api_feedback('k', 'doc', 'FB', 64)(PH.case_view(CASES[0]), dlg, 'failure')
    log = Log(); tree, _ = tree_for(log, rounds=1, branching=1); leaf = tree.leaves[0]
    tin = PH.freeze_trajectory_input(CASES[0], leaf, PH.DoctorFeedback(leaf.leaf_id, leaf.outcome, 'the Agent ignored my question', 'fb'))
    B.api_rubric_generator('k', 'google/gemini-3.8-flash-20260902')('GEN', tin)
    for _, _, user in calls[2:]: assert 'correct_option' not in user and 'atree option 0 is correct' not in user
    assert 'the Agent ignored my question' in calls[3][2] and calls[3][0] == 'google/gemini-3.8-flash-20260902'


# ---------- team-decided strategies (src/online_rl/tree/team.py) ----------
from src.online_rl.tree import team as T
from src.online_rl.tree import prompt_search as PS
from src.online_rl.tree.prompt_search import unit_values


def test_multi_target_R_O_is_mean_of_per_node_R_O():
    assert T.mean_target_R_O({'a': 1.0, 'b': 0.0, 'c': 2 / 3}) == pytest.approx(5 / 9)
    st = T.team_strategies(None, 1.0, 1.0)
    _, _, (_, nm) = _leaf_eval(Log(), 0.0, k_conds=('turn 99',), st=st); leaf, _, (_, mt) = _leaf_eval(Log(), 0.0, k_conds=('any turn',), st=st)
    assert nm.status == 'no_match' and nm.R_O is None and not EF.select([nm], {'mode': 'threshold', 'min_R_O': 0.0})[0].retained   # removed
    assert mt.status == 'multi_target' and len(mt.per_node_R_O) == len(leaf.path) > 1
    assert mt.R_O == pytest.approx(sum(mt.per_node_R_O.values()) / len(mt.per_node_R_O))


def _unit(leaf_id, outcome):
    return PS.EvidenceUnit('c', type('L', (), {'leaf_id': leaf_id, 'outcome': outcome})(), (), None, 'p')


def test_directional_prompt_scores_and_J():
    E = lambda r, status='single_target': type('E', (), {'R_O': r, 'status': status})()
    units = [_unit('S1', 1.0), _unit('S2', 1.0), _unit('F1', 0.0), _unit('F2', 0.0), _unit('F3', 0.0)]
    res = {'S1': {'effectiveness': E(1.0)}, 'S2': {'effectiveness': E(None, 'no_match')}, 'F1': {'effectiveness': E(2 / 3)},
           'F2': {'effectiveness': None}, 'F3': {'effectiveness': E(1 / 3)}}
    d = PS.directional_scores(units, res, T.team_strategies(None, 1.0, 1.0))
    assert d['S_success'] == pytest.approx(0.5) and d['S_failure'] == pytest.approx(1 / 3) and d['J'] == pytest.approx(1 / 6)   # invalid / No Match = 0
    assert (d['n_success'], d['n_failure'], d['invalid'], d['no_match']) == (2, 3, 1, 1) and d['per_trajectory']['F2'] == 0.0
    one_sided = PS.directional_scores(units[:2], {k: res[k] for k in ('S1', 'S2')}, T.team_strategies(None, 1.0, 1.0))
    assert one_sided['S_failure'] is None and one_sided['J'] is None                                     # a missing stratum is not imputed
    J = lambda ss, sf: ss * sf
    assert J(0.6, 0.6) > J(0.9, 0.2) and J(0.9, 0.2) == pytest.approx(0.18) and J(0.6, 0.6) == pytest.approx(0.36)
    src = inspect.getsource(PS) + inspect.getsource(T)
    assert not re.search(r'prompt_fitness|paired_acceptance|mean_value|mean_candidate|candidates_per_trajectory|generate_candidates', src)


def test_event_compares_prompts_on_same_trajectories_policy_and_seeds(tmp_path, monkeypatch):
    calls = []; real = PS.score_prompt

    def spy(text, sha, units, cases, d):
        out = real(text, sha, units, cases, d); calls.append((sha, [u.leaf.leaf_id for u in units], d.policy_sha256, d.reruns, out)); return out
    monkeypatch.setattr(PS, 'score_prompt', spy)
    tr, _, _ = trainer(tmp_path, cfg(cases=4, every=2, min_traj=4, k=2)); tr.run()               # training K = 2, prompt evaluation K = 3
    ev = read_json(sorted((tmp_path / 'prompt_events').glob('event_*.json'))[0])
    first = [c for c in calls if c[1] and set(c[1]) <= set(ev['search_leaf_ids']) | set(ev['acceptance_leaf_ids'])]
    by_split = {}
    for sha, ids, pol, k, out in first: by_split.setdefault(tuple(ids), []).append((sha, pol, k, out))
    assert by_split and all(len({s for s, *_ in v}) >= 2 for v in by_split.values())                    # parent + candidate(s)
    seeds = {}
    for ids, runs in by_split.items():
        assert len({pol for _, pol, _, _ in runs}) == 1 and {k for _, _, k, _ in runs} == {3}                           # same rerun policy, K = 3
        for sha, _, _, out in runs:
            for leaf_id, r in out.items():
                e = r['effectiveness']
                if e is None or not e.runs: continue
                assert e.level == ('0' if e.runs[0].source_outcome == 1.0 else '2')                        # success -> Score-0, failure -> Score-2
                for t in e.target_node_ids:
                    rs = [x for x in e.runs if x.node_id == t]; assert len(rs) == 3
                    seeds.setdefault(t, set()).add(tuple(x.seed for x in rs))
    assert seeds and all(len(v) == 1 for v in seeds.values())                                          # matched rerun seeds across prompts
    assert ev['rerun_policy_sha256'] and ev['evidence_policy_sha256']


def test_event_waits_for_both_strata(tmp_path, monkeypatch):
    monkeypatch.setattr(PS, 'ready', lambda *a: False)
    import src.online_rl.tree.trainer as TR; monkeypatch.setattr(TR, 'ready', lambda *a: False)
    tr, log, _ = trainer(tmp_path, cfg(cases=4, every=2, min_traj=4)); tr.run()
    assert not (tmp_path / 'prompt_events').exists() and log.counts.get('reflect', 0) == 0                 # deferred, never scored as 0
    one_sided = [_unit('S1', 1.0), _unit('S2', 1.0)]
    assert PS.strata(one_sided) == {'success': 2, 'failure': 0}
    with pytest.raises(RuntimeError, match='both splits'):
        PS.run_event(1, 'P0', 'x', one_sided, {'c': CASES[0]}, None, None, None, cfg(), 0, 0, [])


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
    g = B.posthoc_gates(None, cases)(tin, RUB)
    assert g['passed'] and g['observable'] is False and g['unresolved'] == ['observable']                        # observable recorded, not blocking
    assert seen[0][1] == {'answer_presupposition', 'new_domain_premise', 'observable'}                          # no pre-action questions
    st = json.dumps(seen[0][0]); assert 'correct_option' not in st and 'Final answer' not in st and 'FINAL' not in st
    _gate_answers(monkeypatch, new_domain_premise='Y'); assert not B.posthoc_gates(None, cases)(tin, RUB)['passed']
    _gate_answers(monkeypatch, answer_presupposition='Y'); assert not B.posthoc_gates(None, cases)(tin, RUB)['passed']
    _gate_answers(monkeypatch)
    g = B.posthoc_gates(None, cases)(tin, {**RUB, 'required_behavior': 'endorse option A'}); assert not g['passed'] and g['leakage_deterministic']
    g = B.posthoc_gates(None, cases)(tin, {**RUB, 'required_behavior': 'say atree option 0'}); assert 'gold_option_text' in g['leakage_deterministic']


def test_gate_failure_makes_rubric_invalid_and_is_cached():
    tin = _tin(); cache = CallCache(None); n = {'gate': 0, 'gen': 0}
    def gen(p, t): n['gen'] += 1; return {'raw': '{}', 'rubric': dict(RUB)}
    def gate(t, r): n['gate'] += 1; return {'passed': False}
    r = PH.generate_rubric(P0, sha256_text(P0), tin, gen, cache, gate)
    assert not r.valid and r.problems == () and r.gate == {'passed': False}
    assert EF.evaluate_rubric(CASES[0], {}, _leaf_of(tin), r, scorer=None, writer=None, continuation=None, reruns=3, run_seed=0, policy_sha256='p',
                              strategies=Strategies(), cache=cache) == ([], None)                                # not verified, no reruns
    PH.generate_rubric(P0, sha256_text(P0), tin, gen, cache, gate); assert n == {'gate': 1, 'gen': 1}            # cache hit, no repeated call


def _leaf_of(tin):
    return type('L', (), {'leaf_id': tin.leaf_id})()


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


def test_failure_s_is_successes_over_K():
    _, _, _, e = _fixed_eval(0.0, [1.0, 0.0, 1.0, 1.0, None], 5)
    assert e.level == '2' and [r.outcome for r in e.runs] == [1.0, 0.0, 1.0, 1.0, None] and e.R_O == pytest.approx(3 / 5)


def test_success_s_is_failures_over_K():
    _, _, _, e = _fixed_eval(1.0, [1.0, 0.0, 1.0, 1.0, None], 5)
    assert e.level == '0' and e.R_O == pytest.approx(1 / 5)              # only the parsed failure counts; None is not a flip


@pytest.mark.parametrize('k', [1, 3, 7])
def test_K_is_the_rerun_count_not_a_rubric_count(k):
    log, gen_calls, rub, e = _fixed_eval(0.0, [1.0] * k, k)
    assert gen_calls == 1 and isinstance(rub, PH.TrajectoryRubric) and e.rubric_id == rub.rubric_id
    assert log.counts['write'] == 1 and len(e.runs) == k == len({r.seed for r in e.runs})
    assert 'candidates_per_trajectory' not in json.dumps(json.loads(Path('configs/online_rl/tree_v1.json').read_text())['rubric'])
    shipped = json.loads(Path('configs/online_rl/tree_v1.json').read_text())
    assert set(shipped['intervention']) >= {'reruns_per_intervention'} and 'reruns' not in shipped['intervention'] and shipped['rollout']['branching'] == 4
    assert shipped['selection'] == {'mode': 'threshold', 'min_R_O': 2 / 3} and shipped['intervention']['reruns_per_intervention'] == 3   # frozen 2026-10-07


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


def test_reflection_view_has_no_doctor_final():
    log = Log(); units, deps, _ = _units_and_deps(log, ('any turn',))
    view = reflection_view(units, score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps))
    assert all(u.tin.dialogue[-1][1] == 'FINAL' for u in units)                                         # the stored trajectory keeps it
    for row, u in zip(view, units):
        assert u.tin.dialogue[-1][3] not in json.dumps(row) and 'Final answer' not in json.dumps(row) and 'option A' not in json.dumps(row)
        assert len(row['dialogue']) == len(u.tin.dialogue) - 1


def test_reflection_view_from_the_trainer_has_no_final(tmp_path):
    views = []; tr, log, _ = trainer(tmp_path, cfg(cases=4, every=2, min_traj=4))
    fb = tr.backend.feedback; tr.backend = type(tr.backend)(lambda p, v, n: views.append(v) or fb(p, v, n), tr.backend.refine, None, None, None); tr.run()
    assert views and not any('Final answer' in json.dumps(v) for v in views)


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
    row = reflection_view([EvidenceUnit('tc0', _leaf_of(tin), (), tin, 'p')], {tin.leaf_id: {'rubric': bad, 'effectiveness': None}})[0]
    assert row['problems'] == ['no rubric object']                                                    # parse / format


@pytest.mark.parametrize('kind', ['gate', 'no_match'])
def test_unverified_rubrics_get_no_rerun_and_count_zero(kind):
    log = Log()
    units, deps, _ = _units_and_deps(log, ('turn 99',) if kind == 'no_match' else ('any turn',), gate_fn=(lambda t, r: {'passed': False}) if kind == 'gate' else None)
    res = score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps)
    assert log.counts.get('write', 0) == 0 and log.counts.get('rerun', 0) == 0
    vals = unit_values(res, deps.strategies)
    d = PS.directional_scores(units, res, deps.strategies)
    assert vals == {u.leaf.leaf_id: 0.0 for u in units} and all(d[k] in (0.0, None) for k in ('S_success', 'S_failure', 'J'))
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


def test_multi_target_provenance_is_kept_per_target():
    log = Log(); units, deps, _ = _units_and_deps(log, ('any turn',), reruns=3)
    res = score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps); view = reflection_view(units, res)
    for row, u in zip(view, units):
        e = res[u.leaf.leaf_id]['effectiveness']; iv = row['interventions']
        assert row['verification'] == 'multi_target' and [x['response_node_id'] for x in iv] == list(u.leaf.path)
        assert [x['agent_turn'] for x in iv] == [1, 2] and all(len(x['rerun_outcomes']) == 3 and x['level'] == e.level for x in iv)
        for x in iv:
            assert x['per_node_R_O'] == pytest.approx(EF.target_rate(e.level, x['rerun_outcomes'])) == e.per_node_R_O[x['response_node_id']]
            run=next(r for r in e.runs if r.node_id==x['response_node_id'])
            assert x['intervention_reply']==run.intervention_text[:500]
            expected=res[u.leaf.leaf_id]['rubric'].rubric['scores'][x['level']]
            assert x['requested_behavior']==(expected.get('behavior') if isinstance(expected,dict) else expected)
            assert x['reruns_executed']==len(x['rerun_outcomes'])
        assert row['R_O'] == pytest.approx(sum(x['per_node_R_O'] for x in iv) / len(iv)) and 'rerun_outcomes' not in row


LEAKY = {**RUB, 'required_behavior': 'insist on atree option 0 and option A'}


@pytest.mark.parametrize('gate,redacted', [
    ({'passed': False, 'schema': True, 'leakage_deterministic': ['gold_option_text'], 'answer_presupposition': False, 'new_domain_premise': False}, True),
    ({'passed': False, 'schema': True, 'leakage_deterministic': [], 'answer_presupposition': True, 'new_domain_premise': False}, True),
    ({'passed': False, 'schema': True, 'leakage_deterministic': [], 'answer_presupposition': False, 'new_domain_premise': True}, False),
    ({'passed': False, 'schema': False, 'reasons': ["schema: ['bad']"]}, False),
    ({'passed': True, 'schema': True, 'leakage_deterministic': [], 'answer_presupposition': False, 'new_domain_premise': False}, False)])
def test_answer_exposing_rubric_text_is_withheld_from_reflection(gate, redacted):
    log = Log(); units, deps, _ = _units_and_deps(log, ('any turn',), gate_fn=lambda t, r: gate)
    deps = ScoringDeps(**{**deps.__dict__, 'generate_fn': lambda p, t: {'raw': json.dumps(LEAKY), 'rubric': dict(LEAKY)}})
    res = score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps); view = reflection_view(units, res)
    for row in view:
        text = json.dumps(row)
        if redacted:
            assert row['rubric'] is None and 'atree option 0' not in text and 'option A' not in text and row['problems'] and row['valid'] is False
        else:
            assert row['rubric']['required_behavior'] == LEAKY['required_behavior']
    assert all(r['rubric'] is not None for r in reflection_view(*_valid_view()))       # ordinary valid rubrics stay visible


def _valid_view():
    log = Log(); units, deps, _ = _units_and_deps(log, ('any turn',))
    return units, score_prompt(P0, sha256_text(P0), units, {CASES[0]['case_id']: CASES[0]}, deps)


def test_approved_prompt_hashes_are_pinned_and_checked():
    import importlib.util
    spec = importlib.util.spec_from_file_location('run_tree', 'scripts/online_rl/run_tree.py'); RT = importlib.util.module_from_spec(spec); spec.loader.exec_module(RT)
    shipped = json.loads(Path('configs/online_rl/tree_v1.json').read_text())
    assert len(RT.PROMPT_HASHES) == 8 and shipped['intervention']['score2_template_sha256'] == 'd4284095d47215b827b9db06345a193db74b7f769269ae2c8c3c358af344b3e0'
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



def test_configured_score2_template_uses_the_env_guided_call_shape(monkeypatch):
    from src.prompt_opt import harness as H
    from src.online_rl.tree import backends as B
    seen = []

    class Pol:
        def agent_user_prompt(self, case, prefix, t): return 'USER'
        def agent(self, user, seed): seen.append(user); return 'reply'
    rubric = {'name': 'n', 'applicability_condition': 'c', 'required_behavior': 'b', 'scores': {'0': 'z', '1': 'o', '2': 't'}, 'rubric_id': 'r'}
    H.Env.guided(Pol(), CASES[0], [], 1, rubric, 3)
    B.PolicyInterventions(Pol(), 'S0 {rubric_json}', S2_V1.read_text()).write(CASES[0], [], 1, rubric, '2', 3)
    assert seen[0] == seen[1]                                                       # v1 file through the config = Env.guided exactly
    B.PolicyInterventions(Pol(), 'S0 {rubric_json}', S2_V2.read_text()).write(CASES[0], [], 1, rubric, '2', 3)
    assert 'without performing that behavior does not meet Score-2' in seen[2] and '"required_behavior": "b"' in seen[2]


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
    labels = lambda m: re.findall(r'^(Agent reply \d+|Doctor(?: \(final decision\))?):', m, re.M)
    assert labels(gen) == labels(fb) == ['Doctor', 'Agent reply 1', 'Doctor', 'Agent reply 2', 'Doctor (final decision)']
    for turn in leaf.dialogue: assert turn[3] in gen                                                   # content unchanged, same order
    assert [gen.index(t[3]) for t in leaf.dialogue] == sorted(gen.index(t[3]) for t in leaf.dialogue)
    assert 'correct_option' not in gen and 'ground_truth' not in gen
    assert hashlib.sha256(Path('prompts/sherpa/trajectory_rubric_generator_P0_aoc_v2.txt').read_bytes()).hexdigest() == \
        json.loads(Path('configs/online_rl/tree_v1.json').read_text())['rubric']['prompt_sha256'] == 'bd0a2378753ed123cdb20e3601a73be251a5fab33990f8f813724348257a9aba'
        # P0 + gold-free applicability / answer-neutral behavior rules + the [Assign or create] decision section (P0aoc2)


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


# ---------- stratified paired bootstrap on dJ ----------
def _scores(succ, fail, prefix=''):
    """Directional score record with the given per-trajectory s(tau) for source-success and source-failure trajectories."""
    pt = {f'S{i}': v for i, v in enumerate(succ)} | {f'F{i}': v for i, v in enumerate(fail)}
    out = {f'{k}': (1.0 if k[0] == 'S' else 0.0) for k in pt}
    S_s, S_f = sum(succ) / len(succ), sum(fail) / len(fail)
    return {'per_trajectory': pt, 'source_outcome': out, 'S_success': S_s, 'S_failure': S_f, 'J': S_s * S_f}


ACC = {'rule': 'stratified_paired_bootstrap_J', 'delta': 0.0, 'p_min': 0.9, 'minimum_success_trajectories': 2, 'minimum_failure_trajectories': 2,
       'bootstrap_n': 300, 'bootstrap_seed': 0}


def test_bootstrap_resamples_strata_separately_and_keeps_pairs(monkeypatch):
    seen = []; real = PS.random.Random

    class Spy(real):
        def choices(self, population, k):
            seen.append((list(population), k)); return super().choices(population, k=k)
    monkeypatch.setattr(PS.random, 'Random', Spy)
    p, c = _scores([0.9, 0.8, 0.7], [0.2, 0.1]), _scores([0.8, 0.8, 0.6], [0.5, 0.4])
    r = PS.stratified_paired_bootstrap_J(p, c, {**ACC, 'bootstrap_n': 5})
    assert len(seen) == 10 and [k for _, k in seen] == [3, 2] * 5                                        # success then failure, every replicate
    assert all(pop == [(0.9, 0.8), (0.8, 0.8), (0.7, 0.6)] for pop, k in seen[0::2])                     # (parent, candidate) pairs of one trajectory
    assert all(pop == [(0.2, 0.5), (0.1, 0.4)] for pop, k in seen[1::2])
    assert r['n_success'] == 3 and r['n_failure'] == 2 and r['bootstrap_n'] == 5 and r['bootstrap_seed'] == 0


def test_bootstrap_tests_J_not_the_mean_s():
    p, c = _scores([0.5, 0.5], [0.5, 0.5]), _scores([1.0, 1.0], [0.1, 0.1])        # mean s rises 0.5 -> 0.55, J falls 0.25 -> 0.10
    r = PS.stratified_paired_bootstrap_J(p, c, ACC)
    assert r['observed_dJ'] == pytest.approx(-0.15) and r['p_gain'] == 0.0 and not r['accepted']
    vals = [0.9, 0.85, 0.95, 0.9]; succ_p = vals; fail_p = [0.2, 0.25, 0.15, 0.2]
    p, c = _scores(succ_p, fail_p), _scores([v - 0.05 for v in succ_p], [v + 0.4 for v in fail_p])        # success side slightly worse, J better
    r = PS.stratified_paired_bootstrap_J(p, c, ACC)
    assert c['S_success'] < p['S_success'] and r['observed_dJ'] > 0 and r['p_gain'] >= 0.9 and r['accepted']


def test_bootstrap_checks_and_missing_stratum():
    p, c = _scores([0.5, 0.6], [0.2, 0.3]), _scores([0.5, 0.6], [0.3, 0.4])
    assert PS.stratified_paired_bootstrap_J(p, c, ACC)['accepted']
    r = PS.stratified_paired_bootstrap_J(p, c, {**ACC, 'minimum_failure_trajectories': 3})
    assert not r['accepted'] and r['checks']['minimum_failure_trajectories'] is False and 'minimum_failure_trajectories' in r['reason']
    r = PS.stratified_paired_bootstrap_J(p, c, {**ACC, 'delta': 0.5}); assert not r['accepted'] and r['checks']['delta'] is False
    one = {'per_trajectory': {'S0': 0.5}, 'source_outcome': {'S0': 1.0}, 'S_success': 0.5, 'S_failure': None, 'J': None}
    r = PS.stratified_paired_bootstrap_J(one, one, ACC); assert not r['accepted'] and r['reason'] == 'missing outcome stratum'
    other = _scores([0.5, 0.6], [0.2, 0.3]); other['per_trajectory'] = {k + 'x': v for k, v in other['per_trajectory'].items()}
    with pytest.raises(RuntimeError, match='same trajectories'): PS.j_acceptance(p, other, ACC)


def test_research_acceptance_config_leaves_research_choices_open():
    acc = json.loads(Path('configs/online_rl/tree_v1.json').read_text())['acceptance']
    assert acc['rule'] == 'stratified_paired_bootstrap_J' and acc['rule'] in PS.ACCEPTANCE_RULES
    assert all(acc[k] == 'CALIBRATION_REQUIRED' for k in ('delta', 'p_min', 'minimum_success_trajectories', 'minimum_failure_trajectories'))
    assert json.loads(Path('configs/online_rl/tree_v1.json').read_text())['prompt_update']['reruns_per_intervention'] == 3


# ---------- checkpoint retention and milestone evaluation ----------
from src.online_rl.tree.evaluation import MilestoneEvaluator, eval_cases


class CountingEval:
    """Real MilestoneEvaluator with a fake episode; records the trainer state at every call."""
    def __init__(self, tr_ref, fail_at=None, mutate=None):
        self.calls, self.tr_ref, self.fail_at, self.mutate, self.plays = [], tr_ref, fail_at, mutate, 0
        self.inner = MilestoneEvaluator(self.play, lambda c, o: float(o == c['ground_truth']['correct_option']), CASES, 'val_mock', 'm', 'proto', 0)

    def play(self, case, seed):
        self.plays += 1; return {'final_option': 'A' if seed % 2 else 'B', 'termination_reason': 'x', 'num_discussion_rounds': 2}

    def __call__(self, **kw):
        tr = self.tr_ref[0]; self.calls.append((kw['milestone'], tr.state.case_cursor, len(list((tr.out / 'cases').glob('*.json'))), tr.policy_sha()))
        if self.fail_at == kw['milestone']: self.fail_at = None; raise RuntimeError('eval API down')
        if self.mutate: self.mutate()
        return self.inner(**kw)


def eval_trainer(tmp, every=2, cases=5, fail_at=None, mutate=None, w=None, log=None):
    c = {**cfg(cases=cases, every=99), 'evaluation': {'checkpoint_every_cases': every, 'split': 'val_mock', 'cases_file': 'x', 'seed': 0}}
    ref = []; ev = CountingEval(ref, fail_at, mutate)
    log = log or Log(); w = w if w is not None else {'v': 0}
    tr = TreeTrainer(c, out_dir=tmp, cases=CASES, env=mock_env(log, w), strategies=mock_strategies(w), search_backend=search_backend(log),
                     initial_prompt=P0, policy_sha=lambda: f'policy-{w["v"]}', protocol_sha='proto', feedback_prompt_sha256='fb',
                     updater=Updater(w), evaluator=ev)
    ref.append(tr); return tr, ev, w


def ckpts(tmp): return sorted(p.name for p in (tmp / 'checkpoints').iterdir())


def test_rolling_and_milestone_checkpoints(tmp_path):
    tr, ev, _ = eval_trainer(tmp_path, every=2, cases=5); tr.run()
    assert ckpts(tmp_path) == ['checkpoint_000000', 'checkpoint_000002', 'checkpoint_000004', 'rolling_000005']
    s = read_json(tmp_path / 'run_state.json')
    assert s['checkpoint'].endswith('rolling_000005') and s['case_cursor'] == 5 and s['eval_cursor'] == 4
    tr.run(max_cases=0)                                                                             # nothing pending: no new eval
    assert [c[0] for c in ev.calls] == [2, 4]


def test_old_rolling_checkpoint_is_removed_only_after_the_commit(tmp_path, monkeypatch):
    tr, _, w = eval_trainer(tmp_path, every=3, cases=5); tr.run(max_cases=1)
    assert ckpts(tmp_path) == ['checkpoint_000000', 'rolling_000001']
    real = tr._write
    def crash(s):
        if s.case_cursor == 2: raise RuntimeError('crash before run_state')
        real(s)
    monkeypatch.setattr(tr, '_write', crash)
    with pytest.raises(RuntimeError, match='crash before'): tr.run(max_cases=1)
    assert ckpts(tmp_path) == ['checkpoint_000000', 'rolling_000001', 'rolling_000002']                 # old one kept: run_state still points to it
    assert read_json(tmp_path / 'run_state.json')['checkpoint'].endswith('rolling_000001')
    tr2, _, _ = eval_trainer(tmp_path, every=3, cases=5, w={'v': 0})
    assert ckpts(tmp_path) == ['checkpoint_000000', 'rolling_000001'] and tr2.state.case_cursor == 1       # uncommitted checkpoint dropped
    tr2.run(); assert ckpts(tmp_path) == ['checkpoint_000000', 'checkpoint_000003', 'rolling_000005']


def test_eval_runs_at_the_milestone_before_the_next_case_and_is_read_only(tmp_path):
    tr, ev, w = eval_trainer(tmp_path, every=2, cases=5); tr.run()
    assert [(m, cursor, n_cases) for m, cursor, n_cases, _ in ev.calls] == [(2, 2, 2), (4, 4, 4)]       # right after case N, before case N+1
    r = read_json(tmp_path / 'eval' / 'checkpoint_000002' / 'result.json')
    assert r['milestone'] == 2 and r['status'] == 'complete' and r['split'] == 'val_mock' and len(r['per_case']) == len(CASES)
    assert r['policy_sha256'] == ev.calls[0][3] and r['checkpoint'].endswith('checkpoint_000002') and r['prompt_sha256'] == sha256_text(P0)
    assert {'checkpoint_sha256', 'eval_protocol_sha256', 'manifest_sha256', 'accuracy', 'completed_at'} <= set(r)
    assert not (tmp_path / 'cases' / 'eval').exists() and all('rubric' not in json.dumps(x) for x in r['per_case'])
    tr2, _, _ = eval_trainer(tmp_path / 'm', every=2, cases=3, mutate=lambda: None)
    tr2.evaluator.mutate = lambda: tr2.updater.w.__setitem__('v', 99)
    with pytest.raises(RuntimeError, match='changed the policy'): tr2.run()


def test_failed_eval_blocks_training_and_is_retried_once_on_resume(tmp_path):
    tr, ev, w = eval_trainer(tmp_path, every=2, cases=4, fail_at=2)
    with pytest.raises(RuntimeError, match='eval API down'): tr.run()
    s = read_json(tmp_path / 'run_state.json')
    assert s['case_cursor'] == 2 and s['eval_cursor'] == 0 and not (tmp_path / 'cases' / 'case_000002.json').exists()
    assert (tmp_path / 'checkpoints' / 'checkpoint_000002').exists()
    tr2, ev2, _ = eval_trainer(tmp_path, every=2, cases=4, w={'v': 0}); tr2.run()
    assert [c[:2] for c in ev2.calls] == [(2, 2), (4, 4)] and read_json(tmp_path / 'run_state.json')['eval_cursor'] == 4


def test_milestone_evaluator_resumes_cases_and_never_repeats_a_finished_eval(tmp_path):
    ev = MilestoneEvaluator(lambda c, s: {'final_option': 'A'}, lambda c, o: float(o == c['ground_truth']['correct_option']), CASES, 'v', 'm', 'p', 0)
    plays = []; ev.play = lambda c, s: plays.append(c['case_id']) or {'final_option': 'A'}
    prov = {'milestone': 2, 'checkpoint': 'ck', 'checkpoint_sha256': 'h', 'policy_sha256': 'p', 'prompt_id': 'P0', 'prompt_sha256': 's', 'config_sha256': 'c'}
    out = tmp_path / 'e'; out.mkdir()
    key = {'checkpoint_sha256': 'h', 'eval_protocol_sha256': ev.info['eval_protocol_sha256']}
    (out / 'cases.jsonl').write_text(json.dumps({'case_id': 'tc0', 'correct': 1.0, 'final_option': 'A', 'parsed': True, 'termination_reason': None,
                                                 'agent_turns': None, **key}) + '\n')
    r = ev(milestone=2, checkpoint=Path('ck'), out_dir=out, provenance=prov)
    assert plays == ['tc1', 'tc2', 'tc3'] and r['accuracy'] == 1.0 and [x['case_id'] for x in r['per_case']] == ['tc0', 'tc1', 'tc2', 'tc3']
    assert ev(milestone=2, checkpoint=Path('ck'), out_dir=out, provenance=prov) == r and len(plays) == 3          # finished eval is not rerun
    with pytest.raises(RuntimeError, match='another checkpoint'): ev(milestone=2, checkpoint=Path('ck'), out_dir=out, provenance={**prov, 'checkpoint_sha256': 'x'})


def test_eval_cases_must_be_the_complete_protected_split(tmp_path):
    from src.online_rl.guards import question_sha256
    f = tmp_path / 'val.jsonl'; f.write_text(''.join(json.dumps({**c, 'correct_option': 'A'}) + '\n' for c in CASES[:2]))
    m = {'splits': {'v': {'ids': [], 'question_sha256': [question_sha256(c['question']) for c in CASES[:2]]}}}
    assert [c['case_id'] for c in eval_cases(f, 'v', m)] == ['tc0', 'tc1']
    with pytest.raises(ValueError, match='complete'): eval_cases(f, 'v', {'splits': {'v': {'ids': [], 'question_sha256': [question_sha256(c['question']) for c in CASES[:3]]}}})
    with pytest.raises(ValueError, match='not in protected split'): eval_cases(f, 'v', {'splits': {'v': {'ids': ['tc0'], 'question_sha256': []}}})


def test_research_config_keeps_the_interval_and_eval_file_open():
    ev = json.loads(Path('configs/online_rl/tree_v1.json').read_text())['evaluation']
    assert ev['checkpoint_every_cases'] == 50 and ev['cases_file'] == 'data/interim/eval/val50.jsonl' and ev['split'] == 'val50'
    src = inspect.getsource(__import__('src.online_rl.tree.trainer', fromlist=['x']))
    assert '% 50' not in src and '50' not in src.split('def _commit')[1].split('def _units')[0]


# ---------- RQ adapters on tree runs ----------
from src.rq_eval import runs as RUNS
from src.rq_eval.main_table import table as rq1_table
from src.rq_eval.rq3_analysis import rq3_table


def _tree_run(tmp, enabled=True):
    c = {**cfg(cases=4, every=2, min_traj=4), 'evaluation': {'checkpoint_every_cases': 2, 'split': 'val_mock', 'cases_file': 'x', 'seed': 0}}
    c['prompt_update'] = {**c['prompt_update'], 'enabled': enabled}
    ref = []; ev = CountingEval(ref); log = Log(); w = {'v': 0}
    tr = TreeTrainer(c, out_dir=tmp, cases=CASES, env=mock_env(log, w), strategies=mock_strategies(w), search_backend=search_backend(log),
                     initial_prompt=P0, policy_sha=lambda: f'policy-{w["v"]}', protocol_sha='proto', feedback_prompt_sha256='fb', updater=Updater(w), evaluator=ev)
    ref.append(tr); tr.run(); return tmp


def test_tree_run_loader_and_rq1_milestones(tmp_path):
    run = _tree_run(tmp_path / 'evolving')
    prov = RUNS.load_tree_run(run)
    assert prov['format'] == 'tree' and prov['prompt_mode'] == 'evolving' and prov['case_cursor'] == 4 and prov['finished']
    assert prov['checkpoint_sha256'] == RUNS.dir_sha256(Path(prov['checkpoint'])) and [m['milestone'] for m in prov['milestones']] == [2, 4]
    results, mprov = RUNS.rq1_milestone_inputs(run, 'tree_evolving')
    out = rq1_table(results, mprov, references=(), conditions=list(results))
    assert [r['condition'] for r in out['main']] == ['tree_evolving@000002', 'tree_evolving@000004'] and all(r['split'] == 'val_mock' for r in out['main'])
    with pytest.raises(ValueError, match='unknown conditions'): rq1_table(results, mprov)                     # legacy condition list unchanged
    with pytest.raises(RuntimeError, match='not a case-tree run'):
        m = read_json(run / 'run_manifest.json'); (run / 'run_manifest.json').write_text(json.dumps({**m, 'version': 'x'})); RUNS.load_tree_run(run)


def test_tree_prompt_lineage_and_rq3_pairing(tmp_path):
    evo, fix = _tree_run(tmp_path / 'evolving'), _tree_run(tmp_path / 'fixed', enabled=False)
    rows = RUNS.tree_prompt_lineage(evo)
    assert rows and all({'S_success', 'S_failure', 'J'} <= set(r['search']) for r in rows if r['sha256'])
    scored = [r for r in rows if r['acceptance']]
    assert scored and all(r['acceptance']['observed_dJ'] is not None and r['acceptance']['p_gain'] is not None and r['acceptance']['n_success'] for r in scored)
    assert all(r['parent_sha256'] for r in rows if r['depth'] == 1)
    f, e = RUNS.load_tree_run(fix), RUNS.load_tree_run(evo)
    assert RUNS.rq3_pair_problems(f, e) == [] and len(f['accepted_prompts']) == 1
    out = rq3_table({'fixed': [{'case_id': 'a', 'correct': 1.0}], 'evolving': [{'case_id': 'a', 'correct': 1.0}]}, f, e, rows)
    assert 'state schema' not in out['differs_only_in']
    assert 'arms are not fixed/evolving' in RUNS.rq3_pair_problems(e, f)


def test_rq2_tree_policy_loading_is_verified_and_reply_is_precommit(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch'); tf = pytest.importorskip('transformers'); peft = pytest.importorskip('peft')
    from src.online_rl.integrated.continuation import adapter_sha256
    import src.online_rl.runtime as RT_

    def tiny():
        torch.manual_seed(0)
        base = tf.LlamaForCausalLM(tf.LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                                  num_key_value_heads=2, max_position_embeddings=64))
        return peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], lora_dropout=0.0))
    trained = tiny()
    with torch.no_grad():
        for n_, p in trained.named_parameters():
            if 'lora_B' in n_: p.normal_(0, 0.5)
    trained.save_pretrained(tmp_path / 'ck' / 'adapter'); sha = adapter_sha256(trained)

    class Pol:
        def __init__(self, cfg, key): self.model, self.calls, self.precommit = tiny(), [], True
        def agent_text(self, case, prefix, turn, seed): self.calls.append((case['case_id'], turn)); return 'visible reply'
    monkeypatch.setattr(RT_, 'SharedPolicy', Pol)
    cfg = {'rollout': {'agent_protocol': 'precommit_v1'}}
    pol = RUNS.load_tree_policy(cfg, tmp_path / 'ck', sha)
    assert adapter_sha256(pol.model) == sha
    reply = RUNS.policy_reply_fn(pol, lambda cid, h: 7)(CASES[0], [{'text': 'hi'}], 1)
    assert reply == 'visible reply' and pol.calls == [('tc0', 1)]                                             # precommit_v1 visible reply, no rubric argument
    with pytest.raises(RuntimeError, match='adapter hash differs'): RUNS.load_tree_policy(cfg, tmp_path / 'ck', 'other')
    with pytest.raises(RuntimeError, match='precommit_v1'): RUNS.load_tree_policy({}, tmp_path / 'ck', sha)


# ---------- final main setup: 500-case pool, one pass, three arms ----------
from src.online_rl.tree.conditions import ARMS, arm_config
from src.online_rl.integrated.episode import schedule
RESEARCH = json.loads(Path('configs/online_rl/tree_v1.json').read_text())
TRAIN = Path(RESEARCH['data']['train'])
needs_train = pytest.mark.skipif(not TRAIN.exists(), reason='the training pool is not redistributed')


@needs_train
def test_train_pool_is_the_verified_500_case_file():
    from src.online_rl.cases import load_cases
    from src.online_rl.guards import assert_not_protected, question_sha256
    raw = TRAIN.read_bytes(); cases = load_cases(TRAIN); man = json.loads(Path(RESEARCH['data']['train_manifest']).read_text())
    assert hashlib.sha256(raw).hexdigest() == RESEARCH['data']['train_sha256'] == man['case_sha256']
    assert len(cases) == 500 == man['count'] and len({c['case_id'] for c in cases}) == 500 and len({question_sha256(c['question']) for c in cases}) == 500
    assert man['dataset'] == 'openlifescienceai/medmcqa' and man['split'] == 'train' and man['seed'] == 20261006
    assert [c['source_row_index'] for c in cases] == man['selected_source_row_indices']
    assert all(set(c['options']) == set('ABCD') and c['options'][c['ground_truth']['correct_option']] == c['ground_truth']['correct_text'] for c in cases)
    assert_not_protected(cases, 'test')
    m = json.loads(Path('data/splits/protected_eval_v1.json').read_text())
    prot = set().union(*(set(v['question_sha256']) for v in m['splits'].values())) | set().union(*(set(v['ids']) for v in m['splits'].values()))
    assert not ({question_sha256(c['question']) for c in cases} | {c['case_id'] for c in cases} | {c['source_question_id'] for c in cases}) & prot


def test_one_pass_schedule_over_500_cases():
    assert RESEARCH['schedule']['cases'] == 500 and RESEARCH['evaluation']['checkpoint_every_cases'] == 50
    ids = [f'c{i:03d}' for i in range(500)]
    order = [schedule(ids, RESEARCH['schedule']['seed'], k, 1)[0] for k in range(500)]
    assert len(set(order)) == 500 and order == [schedule(ids, RESEARCH['schedule']['seed'], k, 1)[0] for k in range(500)]   # deterministic
    assert [m for m in range(1, 501) if m % 50 == 0] == [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def arm_trainer(tmp, arm, cases=4, every=2, rounds=3, branching=2, log=None, calls=None):
    base = {**cfg(cases=cases, every=2, min_traj=4, rounds=rounds, branching=branching),
            'evaluation': {'checkpoint_every_cases': every, 'split': 'val_mock', 'cases_file': 'x', 'seed': 0}}
    c = arm_config(base, arm); log = log or Log(); w = {'v': 0}; calls = calls if calls is not None else []
    ref = []; up = GroupUpdater(w, calls); ev = CountingEval(ref)
    tr = TreeTrainer(c, out_dir=tmp, cases=CASES, env=mock_env(log, w), strategies=T.team_strategies(up, 1.0, 0.5), search_backend=search_backend(log),
                     initial_prompt=P0, policy_sha=lambda: f'policy-{w["v"]}', protocol_sha='proto', feedback_prompt_sha256='fb', updater=up, evaluator=ev)
    ref.append(tr); return tr, log, calls, ev


def test_outcome_only_arm_skips_the_whole_rubric_side(tmp_path):
    tr, log, calls, _ = arm_trainer(tmp_path, 'outcome_only'); tr.run()
    assert all(log.counts.get(k, 0) == 0 for k in ('feedback', 'rubric', 'score', 'write', 'rerun', 'reflect', 'refine'))
    assert not (tmp_path / 'prompt_events').exists() and len(calls) == 4                                       # GRPO every case
    for i in range(4):
        r = case_record(tmp_path, i)
        assert r['trajectory_rubrics'] == [] and r['response_scores'] == [] and r['rubric_weights'] == {}
        assert all(x['R_A'] is None and x['reward'] == pytest.approx(x['outcome_component']) for x in r['agent_targets']['nodes'].values() if x['reward'] is not None)
    with pytest.raises(ValueError, match='needs the rubric pipeline'):
        TreeTrainer({**arm_config(cfg(), 'outcome_only'), 'prompt_update': {**cfg()['prompt_update'], 'enabled': True}}, out_dir=tmp_path / 'x', cases=CASES,
                    env=mock_env(Log(), {'v': 0}), strategies=mock_strategies({'v': 0}), search_backend=search_backend(Log()), initial_prompt=P0,
                    policy_sha=lambda: 'p', protocol_sha='proto', feedback_prompt_sha256='fb', updater=Updater({'v': 0}))


def test_fixed_and_evolving_arms(tmp_path):
    tr, log, _, _ = arm_trainer(tmp_path / 'fixed', 'fixed'); tr.run()
    assert log.counts['feedback'] and log.counts['rubric'] and log.counts['rerun'] and not (tmp_path / 'fixed' / 'prompt_events').exists()
    s = read_json(tmp_path / 'fixed' / 'run_state.json'); assert s['prompt_sha256'] == sha256_text(P0) and len(s['history']) == 1
    tr, log, _, _ = arm_trainer(tmp_path / 'evolving', 'evolving'); tr.run()
    ev = read_json(sorted((tmp_path / 'evolving' / 'prompt_events').glob('event_*.json'))[0])
    assert log.counts['reflect'] and ev['decision']['acceptance']['n_success'] and 'observed_dJ' in ev['decision']['acceptance']


def test_500_case_run_keeps_exactly_ten_milestones(tmp_path):
    tr, _, _, ev = arm_trainer(tmp_path, 'outcome_only', cases=500, every=50, rounds=1, branching=2); tr.run()
    assert ckpts(tmp_path) == ['checkpoint_000000'] + [f'checkpoint_{m:06d}' for m in range(50, 501, 50)]
    assert [c[0] for c in ev.calls] == list(range(50, 501, 50)) and all(c[0] == c[1] for c in ev.calls)
    assert sorted(p.name for p in (tmp_path / 'eval').iterdir()) == [f'checkpoint_{m:06d}' for m in range(50, 501, 50)]


def test_rq1_all_arms_and_rq3_fixed_vs_evolving_only(tmp_path):
    runs = {}
    for arm in ARMS:
        tr, *_ = arm_trainer(tmp_path / arm, arm); tr.run(); runs[arm] = RUNS.load_tree_run(tmp_path / arm)
    assert {a: runs[a]['prompt_mode'] for a in ARMS} == {a: a for a in ARMS}
    results, prov = {}, {}
    for arm in ARMS:
        r, p = RUNS.rq1_milestone_inputs(tmp_path / arm, arm); results.update(r); prov.update(p)
    out = rq1_table(results, prov, references=(), conditions=sorted(results))
    assert {row['condition'].split('@')[0] for row in out['main']} == set(ARMS)
    assert {row['condition'].split('@')[0] for row in out['main'] if row['rq1_role'] == 'auxiliary_reward_ablation'} == {'rubric_only'}
    assert {tuple(sorted(x['case_id'] for x in v['val_mock'])) for v in results.values()} == {tuple(sorted(c['case_id'] for c in CASES))}
    assert RUNS.rq3_pair_problems(runs['fixed'], runs['evolving']) == []
    assert 'arms are not fixed/evolving' in RUNS.rq3_pair_problems(runs['outcome_only'], runs['evolving'])
    assert 'arms are not fixed/evolving' in RUNS.rq3_pair_problems(runs['rubric_only'], runs['evolving'])            # Rubric-only is not an RQ3 arm
    assert 'arms are not fixed/evolving' in RUNS.rq3_pair_problems(runs['fixed'], runs['rubric_only'])



def test_feedback_model_is_part_of_the_cache_key():
    log = Log(); tree, env = tree_for(log, rounds=2, branching=1); leaf = tree.leaves[0]; cache = CallCache(None)
    a = PH.doctor_feedback(CASES[0], leaf, env.feedback_fn, 'fb', cache, 'google/gemma-3-12b-it'); n = log.counts['feedback']
    PH.doctor_feedback(CASES[0], leaf, env.feedback_fn, 'fb', cache, 'google/gemma-3-12b-it'); assert log.counts['feedback'] == n   # same model: cached
    b = PH.doctor_feedback(CASES[0], leaf, env.feedback_fn, 'fb', cache, 'anthropic/claude-sonnet-5.5-20260928')
    assert log.counts['feedback'] == n + 1 and a.leaf_id == b.leaf_id                                                       # Gemma feedback not reused
    assert {r['identity']['model'] for r in cache.store.values() if r['kind'] == 'feedback'} == {'google/gemma-3-12b-it', 'anthropic/claude-sonnet-5.5-20260928'}
    shipped = json.loads(Path('configs/online_rl/tree_v1.json').read_text())
    assert shipped['feedback']['model'] == 'anthropic/claude-4.5-haiku-20251001' and shipped['rollout']['doctor_model'].startswith('ollama/gemma3:12b-it-q8_0@')
    assert shipped['rollout']['doctor_backend']['source_model'] == 'google/gemma-3-12b-it' and shipped['feedback']['model'] != shipped['rollout']['doctor_model']
    assert shipped['rubric']['generator_model'] == 'google/gemini-3.8-flash-20260902'
    assert shipped['judges']['gate_model'] == shipped['judges']['scorer_model'] == 'openai/gpt-5.6-luna'
    assert shipped['judges']['gate_prompt_file'] != shipped['judges']['scorer_prompt_file']                       # separate role prompts
    assert shipped['prompt_update']['reflection_model'] == shipped['prompt_update']['refiner_model'] == 'anthropic/claude-4.5-haiku-20251001'



def test_role_model_ids_isolate_every_cache():
    from src.online_rl.tree.prompt_search import cached_backend
    from src.online_rl.integrated.prompt_search import SearchBackend
    log = Log(); tree, env = tree_for(log, rounds=2, branching=1); leaf = tree.leaves[0]; cache = CallCache(None)
    fb = PH.doctor_feedback(CASES[0], leaf, env.feedback_fn, 'fb', cache, 'anthropic/claude-sonnet-5.5-20260928')
    tin = PH.freeze_trajectory_input(CASES[0], leaf, fb); calls = collections.Counter()

    def tagged(fn, model, kind):
        def f(*a): calls[(kind, model)] += 1; return fn(*a)
        f.model = model; return f
    gen = lambda p, t: {'raw': '{}', 'rubric': dict(RUB)}; gate = lambda t, r: {'passed': True}
    for g_model, j_model in (('openai/gpt-6.1-sol', 'typesafe/jev-1.13'), ('google/gemini-3.8-flash-20260902', 'typesafe/jev-1.13'),
                             ('google/gemini-3.8-flash-20260902', 'typesafe/jev-1.13')):
        PH.generate_rubric(P0, sha256_text(P0), tin, tagged(gen, g_model, 'gen'), cache, tagged(gate, j_model, 'gate'))
    assert calls[('gen', 'openai/gpt-6.1-sol')] == 1 and calls[('gen', 'google/gemini-3.8-flash-20260902')] == 1    # Sol rubric not reused for Gemini
    rub = PH.generate_rubric(P0, sha256_text(P0), tin, tagged(gen, 'google/gemini-3.8-flash-20260902', 'gen'), cache)
    nodes = [tree.nodes[0]]
    for s_model in ('typesafe/jev-1.13', 'openai/gpt-5.6-luna', 'typesafe/jev-1.13'):
        score_nodes(CASES[0], nodes, [rub], tagged(env.scorer, s_model, 'score'), cache)
    assert calls[('score', 'typesafe/jev-1.13')] == 1 and calls[('score', 'openai/gpt-5.6-luna')] == 1                   # scorer model in the key
    PH.doctor_feedback(CASES[0], leaf, tagged(env.feedback_fn, 'anthropic/claude-4.5-haiku-20251001', 'fb'), 'fb', cache)
    assert calls[('fb', 'anthropic/claude-4.5-haiku-20251001')] == 1                                                    # Sonnet/Gemma feedback not reused
    refl = lambda parent, view, n: {'diagnosis': 'd', 'directions': ['x'] * n}; refi = lambda *a: 'new prompt'
    for r_model in ('openai/gpt-6.1-sol', 'anthropic/claude-4.5-haiku-20251001', 'anthropic/claude-4.5-haiku-20251001'):
        b = cached_backend(SearchBackend(tagged(refl, r_model, 'reflect'), tagged(refi, r_model, 'refine'), None, None, None), cache)
        b.feedback('P', [{'v': 1}], 1); b.refine('P', 'd', 'x', [], None)
    for k in ('reflect', 'refine'):
        assert calls[(k, 'openai/gpt-6.1-sol')] == 1 and calls[(k, 'anthropic/claude-4.5-haiku-20251001')] == 1
    same = {'parent': 'x', 'model': 'anthropic/claude-4.5-haiku-20251001'}                                              # three Haiku roles never collide
    for kind in ('feedback', 'reflect', 'refine'): cache.call(kind, same, lambda kind=kind: kind)
    assert [cache.call(kind, same, lambda: 'other') for kind in ('feedback', 'reflect', 'refine')] == ['feedback', 'reflect', 'refine']


# ---------- Luna judges: hard labels, same-model repair, gate and scorer ----------
from src.common.config import IntegrationUnavailable
from src.online_rl.tree.luna import LunaDecisionClient


def luna(replies, system='S'):
    """LunaDecisionClient whose HTTP post returns the given contents in order; records every payload."""
    sent, it = [], iter(replies)
    def post(url, payload, key, timeout):
        sent.append(payload); return {'model': payload['model'], 'usage': {}, 'choices': [{'message': {'content': next(it)}}]}
    return LunaDecisionClient(model='openai/gpt-5.6-luna', key='k', system=system, post=post), sent


Q = {'a': {'type': 'choice', 'instructions': 'i', 'criteria': {'Y': 'y', 'N': 'n'}}, 'b': {'type': 'choice', 'instructions': 'j', 'criteria': {'0': 'z', '1': 'o', '2': 't'}}}


def test_luna_client_alignment_and_one_same_model_repair():
    c, sent = luna(['{"answers": {"a": "y", "b": 2}}']); ans, rec = c.decide('{"s": 1}', Q, 'p')
    assert {k: v['value'] for k, v in ans.items()} == {'a': 'Y', 'b': '2'} and not rec['repaired'] and ans['a']['probabilities'] is None
    assert sent[0]['response_format']['json_schema']['strict'] and sent[0]['response_format']['json_schema']['schema']['properties']['answers']['properties']['b']['enum'] == ['0', '1', '2']
    c, sent = luna(['not json', '{"answers": {"a": "N", "b": "1"}}']); ans, rec = c.decide('{}', Q, 'p')
    assert rec['repaired'] and rec['repair_ok'] and len(sent) == 2 and sent[1]['model'] == sent[0]['model'] == 'openai/gpt-5.6-luna'
    assert sent[1]['messages'][:2] == sent[0]['messages'] and 'not valid' in sent[1]['messages'][-1]['content']        # same input + schema fix
    for bad in (['{"answers": {"a": "Y", "b": "3"}}', '{"answers": {"a": "Y", "b": "3"}}'], ['{"answers": {"a": "Y"}}', '[]']):
        c, sent = luna(bad)
        with pytest.raises(IntegrationUnavailable, match='after one repair'): c.decide('{}', Q, 'p')
        assert len(sent) == 2 and c.log[-1]['repaired'] and not c.log[-1]['repair_ok']                                 # '3' is never mapped to '2'


def test_luna_gate_rejects_semantic_failures_and_keeps_deterministic_checks():
    from src.online_rl.tree import backends as B
    tin = _tin(); cases = {CASES[0]['case_id']: CASES[0]}
    ok = '{"answers": {"answer_presupposition": "N", "new_domain_premise": "N", "observable": "N"}}'
    c, sent = luna([ok], system='GATE'); g = B.posthoc_gates(c, cases)(tin, RUB)
    assert g['passed'] and g['decisions'][0]['raw'] == [ok] and 'GATE' == sent[0]['messages'][0]['content']
    for flag in ('answer_presupposition', 'new_domain_premise'):
        c, _ = luna([ok.replace(f'"{flag}": "N"', f'"{flag}": "Y"')]); assert not B.posthoc_gates(c, cases)(tin, RUB)['passed']
    c, sent = luna([ok]); g = B.posthoc_gates(c, cases)(tin, {**RUB, 'required_behavior': 'endorse option A'})
    assert not g['passed'] and g['leakage_deterministic']                                                              # deterministic check unchanged
    c, sent = luna([]); g = B.posthoc_gates(c, cases)(tin, {**RUB, 'scores': {'0': 'x'}})
    assert not g['passed'] and g['schema'] is False and sent == []                                                     # schema failure: no Luna call
    c, sent = luna(['oops', ok]); assert B.posthoc_gates(c, cases)(tin, RUB)['decisions'][0]['repaired']
    c, _ = luna(['oops', 'oops'])
    with pytest.raises(IntegrationUnavailable): B.posthoc_gates(c, cases)(tin, RUB)


def test_luna_hard_scorer_levels_and_no_match(tmp_path):
    from src.online_rl.tree import backends as B
    rubrics = [{**RUB, 'rubric_id': f'R{i}', 'name': f'r{i}'} for i in range(4)]
    app = json.dumps({'answers': {'R0': 'Y', 'R1': 'Y', 'R2': 'Y', 'R3': 'N'}}); beh = json.dumps({'answers': {'R0': '0', 'R1': '1', 'R2': '2'}})
    c, sent = luna([app, beh], system='SCORER'); scorer = B.two_stage_hard_scorer(c)
    log = Log(); tree, _ = tree_for(log, rounds=1, branching=1)
    cands = [PH.TrajectoryRubric(r['rubric_id'], 'L', 'p', None, r, True, (), None) for r in rubrics]
    scores = {s.rubric_id: s for s in score_nodes(CASES[0], [tree.nodes[0]], cands, scorer, CallCache(None))}
    assert [scores[f'R{i}'].score for i in range(3)] == [0.0, 0.5, 1.0] and all(scores[f'R{i}'].probabilities is None for i in range(3))
    assert scores['R3'].match is False and scores['R3'].score is None                                                   # No Match excluded
    st = [json.loads(x['messages'][1]['content'])['state'] for x in sent]
    assert 'target_turn' not in st[0] and st[1]['target_turn']['text'] == tree.nodes[0].response                    # applicability never sees the reply
    assert set(json.loads(sent[1]['messages'][1]['content'])['questions']) == {'R0', 'R1', 'R2'}                     # stage 2 only for applicable
    c, _ = luna([app, json.dumps({'answers': {'R0': '0', 'R1': '1', 'R2': '5'}}), json.dumps({'answers': {'R0': '0', 'R1': '1', 'R2': '5'}})])
    with pytest.raises(IntegrationUnavailable): score_nodes(CASES[0], [tree.nodes[0]], cands, B.two_stage_hard_scorer(c), CallCache(None))


def test_r_a_with_hard_scores():
    S = lambda rid, match, level: ResponseRubricScore('n', rid, 'L', match, None if level is None else level / 2, level, None)
    w = {'a': 1.0, 'b': 1 / 3, 'c': 0.5}
    assert T.effectiveness_weighted_r_a([S('a', True, 2)], w) == 1.0
    assert T.effectiveness_weighted_r_a([S('a', True, 2), S('b', True, 0)], w) == pytest.approx((1.0 * 1.0 + 1 / 3 * 0.0) / (4 / 3))
    assert T.effectiveness_weighted_r_a([S('a', True, 1), S('c', True, 2), S('b', False, None)], w) == pytest.approx((0.5 + 0.5) / 1.5)
    assert T.effectiveness_weighted_r_a([S('b', False, None), S('c', False, None)], w) is None
    vals = [T.effectiveness_weighted_r_a([S('a', True, x), S('c', True, y)], w) for x in (0, 1, 2) for y in (0, 1, 2)]
    assert min(vals) == 0.0 and max(vals) == 1.0                                                                       # R_A stays in [0, 1]


def test_luna_gate_and_scorer_caches_are_isolated():
    tin = _tin(); cache = CallCache(None); n = collections.Counter()
    gen = lambda p, t: {'raw': '{}', 'rubric': dict(RUB)}

    def gate_fn(model, protocol):
        def g(t, r): n[('gate', model, protocol)] += 1; return {'passed': True}
        g.model, g.protocol = model, protocol; return g
    for model, protocol in (('typesafe/jev-1.13', None), ('openai/gpt-5.6-luna', 'gate-proto'), ('openai/gpt-5.6-luna', 'gate-proto'),
                            ('openai/gpt-5.6-luna', 'scorer-proto')):
        PH.generate_rubric(P0, sha256_text(P0), tin, gen, cache, gate_fn(model, protocol))
    assert n == {('gate', 'typesafe/jev-1.13', None): 1, ('gate', 'openai/gpt-5.6-luna', 'gate-proto'): 1, ('gate', 'openai/gpt-5.6-luna', 'scorer-proto'): 1}
    assert {r['kind'] for r in cache.store.values()} == {'rubric_generation', 'posthoc_gate'}                       # scorer results live under rubric_score
    shipped = json.loads(Path('configs/online_rl/tree_v1.json').read_text())['judges']
    assert hashlib.sha256(Path(shipped['gate_prompt_file']).read_bytes()).hexdigest() != hashlib.sha256(Path(shipped['scorer_prompt_file']).read_bytes()).hexdigest()



def test_max_cases_stops_after_the_milestone_eval(tmp_path):
    tr, _, calls, ev = arm_trainer(tmp_path, 'outcome_only', cases=8, every=2, rounds=1, branching=2)
    tr.run(max_cases=2)
    assert tr.state.case_cursor == 2 and tr.state.eval_cursor == 2 and [c[0] for c in ev.calls] == [2]          # case 2's eval ran before stopping



def test_luna_stage2_receives_required_behavior_but_not_the_condition():
    from src.online_rl.tree import backends as B
    rub = {**RUB, 'rubric_id': 'R0', 'required_behavior': 'ask what supports the stated conclusion', 'applicability_condition': 'COND-TEXT'}
    c, sent = luna([json.dumps({'answers': {'R0': 'Y'}}), json.dumps({'answers': {'R0': '1'}})]); scorer = B.two_stage_hard_scorer(c)
    log = Log(); tree, _ = tree_for(log, rounds=1, branching=1)
    s = score_nodes(CASES[0], [tree.nodes[0]], [PH.TrajectoryRubric('R0', 'L', 'p', None, rub, True, (), None)], scorer, CallCache(None))
    q1, q2 = (json.loads(x['messages'][1]['content'])['questions']['R0'] for x in sent)
    assert 'COND-TEXT' in q1['instructions'] and 'ask what supports' not in q1['instructions']                       # stage 1 unchanged
    assert 'Required behavior: ask what supports the stated conclusion' in q2['instructions'] and 'COND-TEXT' not in q2['instructions']
    assert 'remain authoritative' in q2['instructions'] and q2['criteria'] == rub['scores'] and s[0].score == 0.5
    assert scorer.protocol and scorer.protocol != c.protocol_sha256                                                  # new protocol hash -> new cache key
