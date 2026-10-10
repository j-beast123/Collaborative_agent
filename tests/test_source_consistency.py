"""Rubric protocol v2.1 (rubric.source_consistency = source_locus_v1): the generator sees the behavioral locus; a new rubric must
be Stage-1 Applicable at its own source locus before the quality gate (one repair, then SOURCE_INCONSISTENT); everything after
admission is v2. Fakes only; no network."""
import json, re
import pytest
import test_tree_v1 as T
import test_witness_trainer as WT
from src.online_rl.tree import posthoc as PH, witness as W
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import AgentNode, DoctorFeedback, Leaf

CASE = {'case_id': 'c9', 'question': 'Which nerve?', 'scenario': '', 'options': {k: f'nerve {k}' for k in 'ABCD'}, 'ground_truth': {'correct_option': 'C'}}
DLG = (('doctor', 'D1', 1, 'I lean to A; what typical features should I check?'), ('agent', 'A1', 1, 'Consider B as well.'),
       ('doctor', 'D2', 2, 'Why B?'), ('agent', 'A2', 2, 'Because of the deficit pattern.'), ('doctor', 'FINAL', 0, 'Final answer: option A'))
LEAF = Leaf('c9/0.1', 'c9', ('c9/0', 'c9/0.1'), DLG, 'Final answer: option A', 'A', 0.0, 'max_agent_turns')
NODES = {'c9/0': AgentNode('c9/0', None, 'c9', 1, DLG[:1], DLG[1][3], {}, 'pol', 0, None, False, None),
         'c9/0.1': AgentNode('c9/0.1', 'c9/0', 'c9', 2, DLG[:3], DLG[3][3], {}, 'pol', 0, None, True, None)}
FB = DoctorFeedback('c9/0.1', 0.0, 'The Agent raised an alternative and then justified it when asked.', 'h', 2, {'status': 'ok'}, True, 'behavior_observer_v4')
RUB = {'name': 'justify', 'applicability_condition': 'the physician asks why', 'required_behavior': 'give the basis', 'scores': {'0': 'no', '1': 'some', '2': 'yes'}}


def tin(locus=True): return PH.freeze_trajectory_input(CASE, LEAF, FB, include_locus=locus)


class Gen:
    def __init__(self, repaired=None): self.calls, self.repairs, self.repaired = [], [], repaired
    def __call__(self, prompt_text, t): self.calls.append(PH.trajectory_message(t)); return {'raw': json.dumps(RUB), 'rubric': dict(RUB)}
    def repair(self, prompt_text, t, rubric, stage1):
        self.repairs.append((t.target_reply_id, rubric['rubric_id'], stage1)); r = self.repaired or RUB
        return {'raw': json.dumps(r), 'rubric': dict(r)}


def stage1(results, seen):
    it = iter(results)
    def check(rubric): seen.append(rubric['rubric_id']); return {'applicable': next(it), 'stage1': {'missing_requirement': 'not asked'}, 'node_id': 'c9/0.1'}
    return check


def gate_ok(t, r): return {'passed': True, 'reasons': []}


# 1
def test_generator_input_has_locus_id_text_and_context_and_v2_input_is_unchanged():
    m = PH.trajectory_message(tin(True))
    assert 'Behavioral locus: Agent reply 2' in m and 'Locus reply (Agent reply 2):\nBecause of the deficit pattern.' in m
    assert 'Conversation before the locus reply:' in m and 'Agent reply 1: Consider B as well.' in m and PH.LOCUS_PRINCIPLE in m
    v2 = PH.trajectory_message(tin(False))
    assert 'Behavioral locus' not in v2 and PH.LOCUS_PRINCIPLE not in v2 and tin(False).sha256 != tin(True).sha256


# 2
def test_applicable_at_source_locus_is_admitted_without_repair():
    g = Gen(); seen = []; gates = []
    r = PH.generate_rubric('P', 'p' * 64, tin(), g, CallCache(None), lambda t, rb: (gates.append(rb['rubric_id']), gate_ok(t, rb))[1], stage1([True], seen))
    assert r.valid and r.source_check['status'] == 'PASS' and not g.repairs and gates == [r.rubric_id] and not r.rubric_id.endswith('-r1')


# 3 + 4
def test_no_match_gets_exactly_one_repair_then_admission():
    g = Gen(repaired={**RUB, 'applicability_condition': 'the physician asks for the basis of a raised alternative'}); seen = []; gates = []
    r = PH.generate_rubric('P', 'p' * 64, tin(), g, CallCache(None), lambda t, rb: (gates.append(rb['rubric_id']), gate_ok(t, rb))[1], stage1([False, True], seen))
    assert len(g.repairs) == 1 and r.valid and r.source_check['status'] == 'REPAIRED_PASS'
    orig = r.source_check['original_rubric']
    assert r.rubric_id == orig['rubric_id'] and gates == [r.rubric_id] and orig['applicability_condition'] == RUB['applicability_condition']
    assert r.rubric['applicability_condition'] == 'the physician asks for the basis of a raised alternative'


def test_repair_changes_only_the_applicability_condition():
    rewritten = {'name': 'NEW NAME', 'applicability_condition': 'the physician asks for the basis of a raised alternative',
                 'required_behavior': 'something broader', 'scores': {'0': 'x', '1': 'y', '2': 'z'}}
    g = Gen(repaired=rewritten); seen = []
    r = PH.generate_rubric('P', 'p' * 64, tin(), g, CallCache(None), gate_ok, stage1([False, True], seen))
    assert {k: r.rubric[k] for k in ('name', 'required_behavior', 'scores')} == {k: RUB[k] for k in ('name', 'required_behavior', 'scores')}
    assert r.rubric['applicability_condition'] == rewritten['applicability_condition'] and r.rubric['source_repair'] == 1
    assert seen[0] == seen[1] == r.rubric_id                                              # same rubric identity, checked twice


def test_source_pass_is_never_causal_evidence_keep_needs_the_witness(tmp_path):
    """A rubric admitted by the source check whose counterfactual reruns all fail is NOT_VALIDATED; the source support still runs reruns."""
    nodes, leaves = {k: v for k, v in NODES.items()}, {LEAF.leaf_id: LEAF}
    from src.online_rl.tree import rubric_bank as RB
    from src.online_rl.tree.records import TrajectoryRubric
    rub = TrajectoryRubric('R', LEAF.leaf_id, 'p', '{}', {**RUB, 'rubric_id': 'R'}, True, (), None, {'status': 'PASS'})
    A = [RB.RubricAssignment(LEAF.leaf_id, 'c9', 'failure', 0, 'R', 'CREATED_NEW', None, '', 'h')]
    cont = []
    v = W.verify_witness(CASE, nodes, leaves, {'R': rub}, A, {LEAF.leaf_id: 2}, applicability=lambda view, prefix, rs: {'R': {'applicable': True, 'stage1': {}}},
                         writer=lambda *a: 'iv', continuation=lambda case, prefix, turn, text, seed: (cont.append(seed), {'final_option': 'A', 'outcome': 0.0})[1],
                         reruns=3, run_seed=0, policy_sha256='pol', cache=CallCache(None))['R']
    assert v.decision == 'NOT_VALIDATED' and v.tested[0]['result'] == 'FAIL' and v.num_reruns_executed == 2 and len(cont) == 2


# 5
def test_still_no_match_after_repair_is_source_inconsistent_and_skips_the_gate():
    g = Gen(); seen = []; gates = []
    r = PH.generate_rubric('P', 'p' * 64, tin(), g, CallCache(None), lambda t, rb: (gates.append(1), gate_ok(t, rb))[1], stage1([False, False], seen))
    assert not r.valid and r.source_check['status'] == 'SOURCE_INCONSISTENT' and len(g.repairs) == 1 and gates == []
    assert any('source_inconsistent' in p for p in r.problems)


# 6
def test_repair_keeps_the_same_locus_and_node():
    g = Gen(repaired={**RUB, 'applicability_condition': 'the physician asks for the basis'}); cache = CallCache(None); asked = []
    def applicability(view, prefix, rubrics):
        asked.append((len(prefix), rubrics[0]['rubric_id'])); return {rubrics[0]['rubric_id']: {'applicable': False, 'stage1': {}}}
    ssf = W.source_stage1_for(CASE, NODES, LEAF, tin().target_reply_id, applicability, cache)
    r = PH.generate_rubric('P', 'p' * 64, tin(), g, cache, gate_ok, ssf)
    assert [a[0] for a in asked] == [3, 3] and g.repairs[0][0] == 2                    # both checks at the reply-2 prefix; repair told locus 2
    assert r.source_check['node_id'] == 'c9/0.1' and r.source_check['second']['node_id'] == 'c9/0.1'
    assert W.source_stage1_for(CASE, NODES, LEAF, None, applicability, cache)(RUB) is None


# 7
def test_repair_request_forbids_answer_specific_content_and_shows_no_gold(monkeypatch):
    from src.online_rl.tree import backends as B
    sent = []
    monkeypatch.setattr(B, '_openrouter_json', lambda key, model, sys_, user, settings, schema, name, role: (sent.append(user), (json.dumps(RUB), {}))[1])
    g = B.api_rubric_generator('k', 'm', {'max_tokens': 10, 'temperature': 0, 'reasoning_effort': None, 'response_format': None})
    g.repair('P', tin(), {**RUB, 'rubric_id': 'x'}, {'missing_requirement': 'the physician did not ask why'})
    u = sent[0]
    assert PH.REPAIR_PRINCIPLE in u and 'Behavioral locus: Agent reply 2' in u and 'the physician did not ask why' in u
    assert 'correct_option' not in u and 'ground_truth' not in u


# 8
def test_existing_answer_presupposition_gate_still_rejects_after_source_pass():
    g = Gen(); seen = []
    r = PH.generate_rubric('P', 'p' * 64, tin(), g, CallCache(None), lambda t, rb: {'passed': False, 'reasons': ['answer presupposition']}, stage1([True], seen))
    assert not r.valid and r.source_check['status'] == 'PASS' and r.gate['reasons'] == ['answer presupposition']


# 9 + 11: trainer under v2.1: admission is the only change; REUSE / witness / K / supports / equal R_A as in v2; caches separate
def v21_env(log, w, fix=lambda cond: 'any turn'):
    env = WT.witness_env(log, w); base = env.generate_fn
    def gen(prompt, t): return base(prompt, t)
    def repair(prompt, t, rubric, stage1):
        log.add('repair'); r = {k: rubric[k] for k in ('name', 'required_behavior', 'scores')}
        return {'raw': '{}', 'rubric': {**r, 'applicability_condition': fix(rubric['applicability_condition'])}}
    gen.repair = repair
    import dataclasses
    return dataclasses.replace(env, generate_fn=gen)


def make(tmp, arm, v21=True, fix=lambda cond: 'any turn', cases=2):
    log = T.Log(); w = {'v': 0}
    c = WT.wcfg(arm, cases)
    if v21: c['rubric']['source_consistency'] = 'source_locus_v1'
    st = T.mock_strategies(w)
    from src.online_rl.tree.team import effectiveness_weighted_r_a
    st = type(st)(multi_target=st.multi_target, r_a_aggregator=effectiveness_weighted_r_a, credit_assignment=st.credit_assignment,
                  agent_update=st.agent_update, prompt_unit_value=st.prompt_unit_value)
    from src.online_rl.tree.trainer import TreeTrainer
    tr = TreeTrainer(c, out_dir=tmp, cases=T.CASES, env=v21_env(log, w, fix), strategies=st, search_backend=T.search_backend(log), initial_prompt=WT.P0R,
                     policy_sha=lambda: f'policy-{w["v"]}', protocol_sha='proto', feedback_prompt_sha256='fb', updater=T.Updater(w))
    return tr, log


@pytest.mark.parametrize('fix,expect', [(lambda c: 'any turn', 'REPAIRED_PASS'), (lambda c: c, 'SOURCE_INCONSISTENT')])
def test_trainer_v21_admission_and_unchanged_downstream(tmp_path, fix, expect):
    tr, log = make(tmp_path, 'rubric_only', fix=fix); rec = tr.step()
    sc = rec['rubric_bank']['telemetry']['source_check']
    assert expect in sc or 'PASS' in sc
    for r in rec['trajectory_rubrics']:
        chk = r['source_check']
        if chk is None: continue
        if chk['status'] in ('SOURCE_INCONSISTENT', 'NO_SOURCE_LOCUS'): assert not r['valid']
        if chk['status'] == 'REPAIRED_PASS': assert r['rubric_id'] == chk['original_rubric']['rubric_id'] and r['rubric']['source_repair'] == 1
    valid = {r['rubric_id'] for r in rec['trajectory_rubrics'] if r['valid']}
    assert {v['rubric_id'] for v in rec['rubric_bank']['witness']} == valid                 # only admitted rubrics are verified
    assert all(v['num_valid_supports'] <= 4 for v in rec['rubric_bank']['witness'])
    assert all(t.get('reruns', 0) in (0, 2, 3) for v in rec['rubric_bank']['witness'] for t in v['tested'])   # K = 3 with 2-of-3 short-circuit
    assert all(x == 1.0 for x in rec['rubric_weights'].values())                            # equal-weight R_A
    assert all(t['target_reply_id'] is not None for t in rec['trajectory_inputs'])
    assert log.counts.get('repair', 0) <= sum(1 for r in rec['trajectory_rubrics'])         # at most one repair per generated rubric


def test_v2_and_v21_generation_caches_never_share_keys(tmp_path):
    a, _ = make(tmp_path / 'v2', 'rubric_only', v21=False); ra = a.step()
    b, _ = make(tmp_path / 'v21', 'rubric_only', v21=True); rb = b.step()
    ka = {json.loads(l)['key'] for l in open(tmp_path / 'v2' / 'calls.jsonl') if json.loads(l)['kind'] in ('rubric_generation', 'rubric_match', 'posthoc_gate')}
    kb = {json.loads(l)['key'] for l in open(tmp_path / 'v21' / 'calls.jsonl') if json.loads(l)['kind'] in ('rubric_generation', 'rubric_match', 'posthoc_gate')}
    assert ka and kb and not (ka & kb)
    assert {t['sha256'] for t in ra['trajectory_inputs']}.isdisjoint({t['sha256'] for t in rb['trajectory_inputs']})


# 10
def test_outcome_only_path_and_hashes(tmp_path, monkeypatch):
    tr, log = make(tmp_path, 'outcome_only'); rec = tr.step()
    assert 'rubric_bank' not in rec and not log.counts.get('repair') and not log.counts.get('stage1') and not log.counts.get('feedback')
    from pathlib import Path
    from src.online_rl.integrated import continuation as C
    from src.online_rl.tree.conditions import arm_config
    root = Path(__file__).resolve().parents[1]
    v2 = json.loads((root / 'configs/online_rl/tree_v2_witness.json').read_text()); v21 = json.loads((root / 'configs/online_rl/tree_v21_witness.json').read_text())
    h2 = {a: C.protocol_sha256(arm_config(v2, a), root) for a in ('outcome_only', 'rubric_only')}
    h21 = {a: C.protocol_sha256(arm_config(v21, a), root) for a in ('outcome_only', 'rubric_only')}
    assert h2['outcome_only'] == '5b77544a79d30533710abaf3cc215911420654c8b22f23ca4866a0ed0efbe444'  # no rubric reruns
    assert h2['rubric_only'] != h2['outcome_only']  # parsed-only flip policy prevents old rubric-run resumes
    assert h21['rubric_only'] != h2['rubric_only'] and v21['rubric']['source_consistency'] == 'source_locus_v1'
