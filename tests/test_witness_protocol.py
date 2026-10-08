"""bounded_witness_v1 (src/online_rl/tree/witness.py): feedback behavioral locus, outcome-blind topology support order,
2-of-3 short-circuit, KEEP at the first witness / NOT_VALIDATED after 4 valid supports, Stage-1-only applicability, equal
R_A weights, untouched prompt-search J, protocol hash separation, identical parallel strata. Fakes only; no network."""
import json, re
import pytest
from src.online_rl.tree import rubric_bank as RB, witness as W, posthoc as PH
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import AgentNode, DoctorFeedback, Leaf, ResponseRubricScore
from src.online_rl.tree.team import effectiveness_weighted_r_a

CASE = {'case_id': 'c1', 'question': 'Which structure?', 'scenario': '', 'options': {k: f'structure {k}' for k in 'ABCD'},
        'ground_truth': {'correct_option': 'B'}}


# ---------- feedback locus ----------
@pytest.mark.parametrize('text,n,want', [
    ('Good work.\nBehavioral locus: Agent reply 2', 3, ('Good work.', 2, None)),
    ('Good.\n**Behavioral locus:** Agent reply 1', 1, ('Good.', 1, None)),
    ('No locus here.', 3, ('No locus here.', None, 'missing')),
    ('x\nBehavioral locus: Agent reply 5', 3, ('x', None, 'out_of_range')),
    ('Behavioral locus: Agent reply 1\nmore text', 3, (None, None, 'not_last')),
    ('a\nBehavioral locus: Agent reply 1\nBehavioral locus: Agent reply 2', 3, (None, None, 'multiple'))])
def test_split_locus(text, n, want):
    clean, rid, problem = PH.split_locus(text, n)
    assert (rid, problem) == want[1:] and (want[0] is None or clean == want[0])


def test_feedback_locus_single_bounded_repair(monkeypatch):
    import src.collaboration.generator as G
    from src.online_rl.tree.backends import api_feedback
    answers = iter(['Helpful reply.', 'Helpful reply.\nBehavioral locus: Agent reply 2'])
    seen = []; monkeypatch.setattr(G, 'doctor_call', lambda key, model, system, user, mt: (seen.append(user), next(answers))[1])
    dlg = [{'speaker': 'doctor', 'text': 'D1'}, {'speaker': 'agent', 'text': 'a1'}, {'speaker': 'doctor', 'text': 'D2'}, {'speaker': 'agent', 'text': 'a2'},
           {'speaker': 'doctor', 'stage': 'FINAL', 'text': 'Final answer: B'}]
    out = api_feedback('k', 'm', 'sys', 100, locus=True)(PH.case_view(CASE), dlg, 'success')
    assert out == {'text': 'Helpful reply.', 'target_reply_id': 2, 'status': 'repaired', 'raw': ['Helpful reply.', 'Helpful reply.\nBehavioral locus: Agent reply 2']}
    assert len(seen) == 2 and 'Behavioral locus' in seen[1]
    answers2 = iter(['x\nBehavioral locus: Agent reply 9', 'y\nBehavioral locus: Agent reply 7'])
    monkeypatch.setattr(G, 'doctor_call', lambda *a: next(answers2))
    out = api_feedback('k', 'm', 'sys', 100, locus=True)(PH.case_view(CASE), dlg, 'failure')
    assert out['target_reply_id'] is None and out['status'] == 'out_of_range' and len(out['raw']) == 2     # bounded: no third call


def test_doctor_feedback_carries_locus_and_text_without_the_line():
    leaf = Leaf('c1/0', 'c1', ('c1/0',), (('doctor', 'D1', 1, 'd'), ('agent', 'A1', 1, 'a'), ('doctor', 'FINAL', 0, 'Final answer: option B')),
                'Final answer: option B', 'B', 1.0, 'max_agent_turns')
    fb = PH.doctor_feedback(CASE, leaf, lambda v, d, o: {'text': 'ok', 'target_reply_id': 1, 'status': 'ok', 'raw': ['ok\nBehavioral locus: Agent reply 1']},
                            'sha', CallCache(None), 'm')
    assert fb.text == 'ok' and fb.target_reply_id == 1 and fb.locus['status'] == 'ok'
    assert PH.freeze_trajectory_input(CASE, leaf, fb).feedback == 'ok'


# ---------- witness core on a small synthetic tree ----------
def tree(paths, outcome=0.0):
    """paths: leaf paths like '0.1.2'; every prefix is an Agent node. Returns nodes, leaves (all with `outcome`)."""
    nodes, leaves = {}, {}
    for p in paths:
        parts = p.split('.')
        for d in range(1, len(parts) + 1):
            nid = 'c1/' + '.'.join(parts[:d])
            nodes.setdefault(nid, AgentNode(nid, None, 'c1', d, (('doctor', 'D1', 1, 'lean B?'),), f'reply {nid}', {}, 'pol', 0, None, False, None))
        lid = 'c1/' + p
        leaves[lid] = Leaf(lid, 'c1', tuple('c1/' + '.'.join(parts[:d]) for d in range(1, len(parts) + 1)),
                           (('doctor', 'D1', 1, 'd'), ('agent', 'A1', 1, 'a'), ('doctor', 'FINAL', 0, 'Final answer: option B')), 'F', 'B', outcome, 'max_agent_turns')
    return nodes, leaves


def rubric(rid='R'):
    from src.online_rl.tree.records import TrajectoryRubric
    r = {'rubric_id': rid, 'name': 'test the leaning', 'applicability_condition': 'physician leans', 'required_behavior': 'probe it',
         'scores': {'0': 'no', '1': 'some', '2': 'yes'}}
    return TrajectoryRubric(rid, 'c1/0.0', 'p', '{}', r, True, (), None)


def assignments(leaves, rid='R', source=None):
    ls = sorted(leaves); source = source or ls[0]
    return [RB.RubricAssignment(l, 'c1', 'failure', i, rid, 'CREATED_NEW' if l == source else 'REUSED_EXISTING', None, '', 'h') for i, l in enumerate(ls)]


class Fx:
    def __init__(self, outcomes, applicable=lambda node: True):
        self.outcomes, self.applicable, self.cont, self.app_calls, self.full_calls = outcomes, applicable, [], [], []

    def applicability(self, view, prefix, rubrics):
        self.app_calls.append(rubrics[0]['rubric_id']); return {r['rubric_id']: {'applicable': self.applicable(self._node), 'stage1': {}} for r in rubrics}

    def writer(self, case, prefix, turn, rub, level, seed): return f'iv {level} {seed}'

    def continuation(self, case, prefix, turn, text, seed):
        self.cont.append(seed); o = self.outcomes(seed); return {'final_option': 'B' if o else 'A', 'outcome': o}


def run(fx, nodes, leaves, loci, source=None, max_valid=4, rid='R', cache=None, support_sampling='topology'):
    rub = rubric(rid); A = assignments(leaves, rid, source)
    orig = W.stage1_applicable
    def stage1(case, node, r, applicability, cache):
        fx._node = node.node_id; return orig(case, node, r, applicability, cache)
    W.stage1_applicable = stage1
    try:
        return W.verify_witness(CASE, nodes, leaves, {rid: rub}, A, loci, applicability=fx.applicability, writer=fx.writer, continuation=fx.continuation,
                                reruns=3, run_seed=0, policy_sha256='pol', cache=cache or CallCache(None), max_valid_supports=max_valid,
                                support_sampling=support_sampling)[rid]
    finally: W.stage1_applicable = orig


def seq_outcomes(seq):
    """Outcomes by call order (failure source: 1.0 = target reached)."""
    it = iter(seq); return lambda seed: next(it)


def test_support_order_uses_topology_only():
    _, leaves = tree(['0.0.0', '0.0.1', '0.1.0', '1.0.0', '1.1.1', '2.0.0'])
    order = [l.leaf_id for l in W.support_order(list(leaves.values()), 'c1/0.0.1')]
    assert order[0] == 'c1/0.0.1'                                    # source first
    assert order[1:4] == ['c1/1.0.0', 'c1/2.0.0', 'c1/0.1.0']        # other root branches first (shared prefix 0), then depth-2 split
    assert set(order) == set(leaves)


@pytest.mark.parametrize('seq,reruns,passed', [([1.0, 1.0], 2, True), ([0.0, 0.0], 2, False), ([1.0, 0.0, 1.0], 3, True), ([0.0, 1.0, 0.0], 3, False)])
def test_two_of_three_short_circuit(seq, reruns, passed):
    nodes, leaves = tree(['0.0']); fx = Fx(seq_outcomes(seq))
    v = run(fx, nodes, leaves, {'c1/0.0': 2}, max_valid=1)
    assert v.num_reruns_executed == reruns == len(fx.cont) and (v.decision == 'KEEP') == passed
    assert v.tested[0]['node_id'] == 'c1/0.0' and v.tested[0]['reruns'] == reruns


def test_keep_stops_at_first_witness_and_not_validated_after_four_valid_supports():
    nodes, leaves = tree(['0.0', '1.0', '2.0', '3.0', '0.1', '1.1'])
    loci = {l: 2 for l in leaves}
    fx = Fx(seq_outcomes([0.0, 0.0, 1.0, 1.0] + [1.0] * 20))          # 1st support fails, 2nd passes
    v = run(fx, nodes, leaves, loci)
    assert v.decision == 'KEEP' and v.num_valid_supports == 2 and v.num_reruns_executed == 4 and v.stop_reason == 'witness_found'
    assert v.validated_by_trajectory_id == v.tested[1]['leaf_id'] and len(v.tested) == 2
    fx = Fx(lambda seed: 0.0)
    v = run(fx, nodes, leaves, loci)
    assert v.decision == 'NOT_VALIDATED' and v.num_valid_supports == 4 and v.num_reruns_executed == 8 and v.stop_reason == 'max_valid_supports'


def test_no_match_and_missing_locus_are_skipped_and_not_counted():
    nodes, leaves = tree(['0.0', '1.0', '2.0'])
    fx = Fx(lambda seed: 0.0, applicable=lambda node: node != 'c1/1.0')
    v = run(fx, nodes, leaves, {'c1/0.0': 2, 'c1/1.0': 2, 'c1/2.0': None})
    res = {t['leaf_id']: t['result'] for t in v.tested}
    assert res == {'c1/0.0': 'FAIL', 'c1/1.0': 'no_match', 'c1/2.0': 'no_locus'}
    assert v.num_valid_supports == 1 and v.decision == 'NOT_VALIDATED' and v.stop_reason == 'candidates_exhausted'
    assert len(fx.app_calls) == 2                                    # Stage-1 only for members with a locus; no level call exists here


def test_all_same_id_members_reach_fifth_support_and_stop_at_first_witness():
    nodes,leaves=tree(['0.0','1.0','2.0','3.0','4.0','5.0'])
    fx=Fx(seq_outcomes([0.,0.]*4+[1.,1.]))
    v=run(fx,nodes,leaves,{l:2 for l in leaves},max_valid=None)
    assert v.decision=='KEEP' and v.num_valid_supports==5 and len(v.tested)==5
    assert v.num_reruns_executed==10 and v.protocol==W.ALL_MEMBERS_PROTOCOL
    v=run(Fx(lambda seed:None),nodes,leaves,{l:2 for l in leaves},max_valid=None)
    assert v.decision=='NOT_VALIDATED' and v.num_valid_supports==6 and v.stop_reason=='candidates_exhausted'


def test_random10_uses_reproducible_uniform_sample_and_caps_trajectories():
    nodes, leaves = tree([f'{i}.0' for i in range(12)])
    loci = {l: 2 for l in leaves}
    sampling = 'uniform_random_without_replacement_v1'
    fx = Fx(lambda seed: 0.0)
    v = run(fx, nodes, leaves, loci, max_valid=10, support_sampling=sampling)
    assert v.protocol == W.RANDOM10_PROTOCOL
    assert v.decision == 'NOT_VALIDATED' and v.num_valid_supports == 10
    assert len(v.candidate_order) == len(v.tested) == 10 and len(set(v.candidate_order)) == 10
    assert v.stop_reason == 'max_valid_supports' and v.num_reruns_executed == 20
    again = run(Fx(lambda seed: 0.0), nodes, leaves, loci, max_valid=10, support_sampling=sampling)
    assert again.candidate_order == v.candidate_order
    assert v.candidate_order != tuple(W.support_order(list(leaves.values()), 'c1/0.0')[i].leaf_id for i in range(10))


def test_order_and_targets_ignore_outcomes():
    nodes, leaves = tree(['0.0', '1.0', '2.0', '3.0']); loci = {l: 1 for l in leaves}
    a = run(Fx(lambda s: 0.0), nodes, leaves, loci); b = run(Fx(lambda s: 1.0), nodes, leaves, loci)
    assert a.candidate_order == b.candidate_order and [t['node_id'] for t in b.tested] == [t['node_id'] for t in a.tested][:len(b.tested)]


def test_witness_reruns_are_reused_by_full_member_verification(tmp_path):
    """Prompt-search s_j (rubric_bank.verify_members, every Applicable reply, K reruns) reuses the witness reruns."""
    from src.online_rl.tree.decisions import Strategies
    nodes, leaves = tree(['0.0']); cache = CallCache(tmp_path / 'calls.jsonl'); fx = Fx(lambda seed: 1.0)
    v = run(fx, nodes, leaves, {'c1/0.0': 2}, max_valid=1, cache=cache)
    assert v.num_reruns_executed == 2 and len(fx.cont) == 2
    scorer = lambda view, prefix, rubrics, reply: {r['rubric_id']: {'applicable': reply.endswith('0.0'), 'probabilities': None, 'hard': 2} for r in rubrics}
    _, member = RB.verify_members(CASE, nodes, leaves, {'R': rubric()}, assignments(leaves), scorer=scorer, writer=fx.writer, continuation=fx.continuation,
                                  reruns=3, run_seed=0, policy_sha256='pol', strategies=Strategies(multi_target=lambda d: sum(d.values()) / len(d)), cache=cache)
    assert len(fx.cont) == 3 and member['c1/0.0'].R_O == 1.0         # only rerun index 2 was new; s_j is the full K = 3 value


def test_equal_weight_r_a_is_the_plain_mean_of_levels():
    s = [ResponseRubricScore('n', r, 'l', True, x, x, None) for r, x in (('a', 1.0), ('b', 0.5), ('c', 0.0))]
    assert effectiveness_weighted_r_a(s, {'a': 1.0, 'b': 1.0, 'c': 1.0}) == pytest.approx(0.5)
    assert effectiveness_weighted_r_a(s, {'a': 0.75, 'b': 1.0, 'c': 1.0}) != pytest.approx(0.5)


# ---------- protocol hash / config ----------
def test_protocol_hash_unchanged_for_v1_and_different_for_witness(monkeypatch):
    from pathlib import Path
    from src.online_rl.integrated import continuation as C
    root = Path(__file__).resolve().parents[1]
    v1 = json.loads((root / 'configs/online_rl/tree_v1.json').read_text()); v2 = json.loads((root / 'configs/online_rl/tree_v2_witness.json').read_text())
    from src.online_rl.doctor import OllamaDoctor
    monkeypatch.setattr(OllamaDoctor, '__init__', lambda self, *a, **k: setattr(self, 'protocol', {'doctor': 'fixed'}))
    h1, h2 = C.protocol_sha256(v1, root), C.protocol_sha256(v2, root)
    assert h1 != h2
    v1m = dict(v1); v1m.pop('verification', None)
    assert C.protocol_sha256(v1m, root) == h1                         # v1 has no method block: its hash does not change
    assert v2['verification']['mode'] == W.PROTOCOL and v2['agent_reward']['r_a_weighting'] == 'equal' and v2['feedback']['behavioral_locus'] is True


# ---------- parallel strata ----------
def test_parallel_strata_identical_to_sequential():
    import test_rubric_reuse as T
    nodes, lv, tins = T.tree()
    reuse = lambda leaf, bank: bank[0] if leaf in ('L1', 'L3') else None
    seq = RB.build_banks(T.CASE, lv, tins, T.PROMPT, 'p', T.Fakes(reuse).generate_fn, T.Fakes(reuse).assign_fn, T.Fakes(reuse).gate_fn, CallCache(None))
    f = T.Fakes(reuse)
    par = RB.build_banks(T.CASE, lv, tins, T.PROMPT, 'p', f.generate_fn, f.assign_fn, f.gate_fn, CallCache(None), parallel_strata=True)
    assert list(seq[0]) == list(par[0]) and [x.rubric_id for x in seq[0].values()] == [x.rubric_id for x in par[0].values()]
    assert seq[1] == par[1]


# ---------- feedback v4: behavior observer, no answer path ----------
CASE4 = {'case_id': 'c4', 'question': 'Wickham striae are seen in?', 'scenario': '',
         'options': {'A': 'Psoriasis', 'B': 'Pityriasis rosea', 'C': 'Lichen planus', 'D': 'None of the above'}, 'ground_truth': {'correct_option': 'C'}}
DLG4 = [{'speaker': 'doctor', 'stage': 'D1', 'text': 'I lean to B.'}, {'speaker': 'agent', 'stage': 'A1', 'text': 'Consider C as well.'},
        {'speaker': 'doctor', 'stage': 'D2', 'text': 'Why?'}, {'speaker': 'agent', 'stage': 'A2', 'text': 'Because of the striae.'},
        {'speaker': 'doctor', 'stage': 'FINAL', 'text': 'Final answer: option B'}]


def test_v4_writer_sees_only_the_dialogue_before_final():
    msg, n = PH.behavior_message(DLG4)
    assert n == 2 and 'Agent reply 2: Because of the striae.' in msg and 'Final answer' not in msg
    for banned in ('Wickham striae are seen in?', 'Psoriasis', 'Lichen planus', 'success', 'failure', 'correct'):
        assert banned not in msg


@pytest.mark.parametrize('text,hit', [
    ('The Agent challenged the premise and the physician hesitated.', False),
    ('The Agent pushed option C.', True), ('It argued for (C).', True), ('It mentioned lichen planus.', True),
    ('The physician was wrong to hold on.', True), ('The Agent correctly flagged it.', True), ('The answer is clear.', True),
    ('Neither position was settled; none of the above options were named.', False)])
def test_leakage_guard(text, hit):
    assert bool(PH.leakage_problems(text, PH.case_view(CASE4))) == hit


def test_v4_repair_then_valid_and_persistent_leak_is_invalid(monkeypatch):
    import src.collaboration.generator as G
    from src.online_rl.tree.backends import api_behavior_feedback
    seen = []
    def fake(answers):
        it = iter(answers)
        return lambda key, model, system, user, mt: (seen.append(user), next(it))[1]
    ok = 'The Agent raised an alternative and asked for the basis.\nBehavioral locus: Agent reply 1'
    monkeypatch.setattr(G, 'doctor_call', fake(['The Agent pushed lichen planus, the correct one.\nBehavioral locus: Agent reply 1', ok]))
    out = api_behavior_feedback('k', 'm', 'sys', 100)(PH.case_view(CASE4), DLG4, 'failure')
    assert out['valid'] and out['status'] == 'repaired' and out['target_reply_id'] == 1 and out['leakage']['first'] and not out['leakage']['final']
    assert out['text'] == 'The Agent raised an alternative and asked for the basis.' and len(out['raw']) == 2
    for u in seen: assert 'Wickham' not in u and 'Psoriasis' not in u and 'failure' not in u and 'Final answer' not in u
    monkeypatch.setattr(G, 'doctor_call', fake(['It favoured option C.\nBehavioral locus: Agent reply 2', 'Still option C.\nBehavioral locus: Agent reply 2']))
    out = api_behavior_feedback('k', 'm', 'sys', 100)(PH.case_view(CASE4), DLG4, 'success')
    assert not out['valid'] and out['target_reply_id'] is None and out['status'] == 'leakage' and out['text'] == 'Still option C.'   # no redaction


def test_behavior_feedback_protocol_v5_is_recorded(monkeypatch):
    import src.collaboration.generator as G
    from src.online_rl.tree.backends import api_behavior_feedback
    answer=('The physician repeated the concern; the Agent addressed it; the physician maintained their position, '
            'and the dialogue does not show whether the concern was resolved.\nBehavioral locus: Agent reply 1')
    monkeypatch.setattr(G, 'doctor_call', lambda *a: answer)
    out = api_behavior_feedback('k', 'm', 'sys', 100, 'behavior_observer_v5')(PH.case_view(CASE4), DLG4, 'failure')
    assert out['kind'] == 'behavior_observer_v5' and out['target_reply_id'] == 1


def test_v4_feedback_kind_changes_the_generator_header_and_tin_hash_only_for_v4():
    leaf = Leaf('c4/0', 'c4', ('c4/0',), (('doctor', 'D1', 1, 'd'), ('agent', 'A1', 1, 'a'), ('doctor', 'FINAL', 0, 'Final answer: option B')),
                'Final answer: option B', 'B', 0.0, 'max_agent_turns')
    v2 = PH.freeze_trajectory_input(CASE4, leaf, DoctorFeedback('c4/0', 0.0, 'fb', 'h'))
    v4 = PH.freeze_trajectory_input(CASE4, leaf, DoctorFeedback('c4/0', 0.0, 'fb', 'h', 1, None, True, 'behavior_observer_v4'))
    v5 = PH.freeze_trajectory_input(CASE4, leaf, DoctorFeedback('c4/0', 0.0, 'fb', 'h', 1, None, True, 'behavior_observer_v5'))
    assert v2.sha256 != v4.sha256 and 'Physician feedback after the decision' in PH.trajectory_message(v2)
    assert 'Observer feedback' in PH.trajectory_message(v4) and 'Physician feedback after' not in PH.trajectory_message(v4)
    assert v5.sha256 != v4.sha256 and 'visible request' in PH.trajectory_message(v5)


def test_protocol_hash_changes_with_the_feedback_protocol(monkeypatch):
    from pathlib import Path
    from src.online_rl.integrated import continuation as C
    from src.online_rl.doctor import OllamaDoctor
    monkeypatch.setattr(OllamaDoctor, '__init__', lambda self, *a, **k: setattr(self, 'protocol', {'doctor': 'fixed'}))
    root = Path(__file__).resolve().parents[1]; v2 = json.loads((root / 'configs/online_rl/tree_v2_witness.json').read_text())
    other = json.loads(json.dumps(v2)); other['feedback'].pop('protocol')
    assert C.protocol_sha256(v2, root) != C.protocol_sha256(other, root) and v2['feedback']['protocol'] == 'behavior_observer_v4'


@pytest.mark.parametrize('source,seq,reruns,passed,positive', [
    (1.0,[None,None],2,False,0),
    (1.0,[None,0.0,1.0],3,False,1),
    (1.0,[None,0.0,0.0],3,True,2),
    (1.0,[0.0,0.0],2,True,2),
    (0.0,[None,None],2,False,0),
    (0.0,[None,1.0,0.0],3,False,1),
    (0.0,[None,1.0,1.0],3,True,2),
    (0.0,[1.0,1.0],2,True,2),
])
def test_unparsed_final_never_counts_as_flip(source,seq,reruns,passed,positive):
    import dataclasses
    nodes,leaves=tree(['0.0'])
    leaves={key:dataclasses.replace(leaf,outcome=source) for key,leaf in leaves.items()}
    fx=Fx(seq_outcomes(seq))
    v=run(fx,nodes,leaves,{'c1/0.0':2},max_valid=1)
    assert (v.decision=='KEEP')==passed and v.num_reruns_executed==reruns
    assert v.tested[0]['positive']==positive
    assert v.tested[0]['outcomes'].count(None)==v.tested[0]['unparsed']
    assert W.witness_telemetry({'R':v})['unparsed_reruns']==seq.count(None)


@pytest.mark.parametrize('level,outcomes,rate',[
    ('0',[None,0.0,1.0],1/3), ('0',[None,None,0.0],1/3),
    ('0',[None,None,None],0), ('0',[0.0,0.0,None],2/3),
    ('2',[None,0.0,1.0],1/3), ('2',[1.0,1.0,None],2/3),
])
def test_full_verification_and_prompt_objective_count_only_parsed_flips(level,outcomes,rate):
    from src.online_rl.tree.effectiveness import target_rate
    assert target_rate(level,outcomes)==pytest.approx(rate)


def test_rerun_outcome_policy_changes_protocol_hash(monkeypatch):
    from pathlib import Path
    from src.online_rl.integrated.continuation import protocol_sha256
    import src.online_rl.tree.effectiveness as effectiveness
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/online_rl/tree_v21_five_gpu.json').read_text())
    current=protocol_sha256(cfg,root)
    monkeypatch.setattr(effectiveness,'RERUN_OUTCOME_POLICY','legacy_none_as_failure')
    assert protocol_sha256(cfg,root)!=current


def test_rerun_continuation_protocol_is_separate_from_tree_protocol_hash(monkeypatch):
    from pathlib import Path
    from src.online_rl.integrated.continuation import protocol_sha256
    import src.online_rl.tree.effectiveness as effectiveness
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/online_rl/tree_v21_five_gpu.json').read_text())
    current=protocol_sha256(cfg,root)
    monkeypatch.setattr(effectiveness,'RERUN_CONTINUATION_PROTOCOL','legacy_discussion_continuation')
    assert protocol_sha256(cfg,root)==current


def test_old_rerun_interpretation_cannot_resume(tmp_path):
    import test_tree_v1 as F
    from src.online_rl.tree.effectiveness import RERUN_OUTCOME_POLICY
    trainer,_,_=F.trainer(tmp_path,F.cfg(cases=1))
    path=tmp_path/'run_manifest.json'
    manifest=json.loads(path.read_text())
    assert manifest['rerun_outcome_policy']==RERUN_OUTCOME_POLICY
    manifest.pop('rerun_outcome_policy')
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='rerun outcome policy'):
        trainer._resume()


def test_outcome_only_does_not_depend_on_rerun_policy(monkeypatch):
    from pathlib import Path
    from src.online_rl.tree.conditions import arm_config
    from src.online_rl.integrated.continuation import protocol_sha256
    import src.online_rl.tree.effectiveness as effectiveness
    root=Path(__file__).resolve().parents[1]
    cfg=arm_config(json.loads((root/'configs/online_rl/tree_v21_five_gpu.json').read_text()),'outcome_only')
    current=protocol_sha256(cfg,root)
    monkeypatch.setattr(effectiveness,'RERUN_OUTCOME_POLICY','legacy_none_as_failure')
    assert protocol_sha256(cfg,root)==current
