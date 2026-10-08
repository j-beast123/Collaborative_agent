"""Same-reason REUSE vs NEW rubric generation (src/online_rl/tree/rubric_bank.py), member-level verification, existential
KEEP with R_O(R) = max_j s_j, single scoring per rubric, trajectory-level J, scorer v2 and the Gemini output fix. Fakes
only; no network."""
import json, re
import pytest
from src.online_rl.integrated.continuation import derive_seed
from src.online_rl.tree import rubric_bank as RB
from src.online_rl.tree import posthoc as PH
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import AgentNode, DoctorFeedback, Leaf
from src.online_rl.tree.team import team_strategies

CASE = {'case_id': 'c1', 'question': 'Which structure is involved?', 'scenario': '', 'options': {k: f'structure {k}' for k in 'ABCD'},
        'ground_truth': {'correct_option': 'B'}}
PROMPT = 'GENERATION RULES\n\n[Assign or create]\n\nDECISION RULES\n'
STATUS = {'L0': 0.0, 'L1': 0.0, 'L2': 1.0, 'L3': 0.0, 'L4': 1.0}          # leaf -> outcome (failure 0.0 / success 1.0)


def tree():
    nodes, leaves = {}, []
    for lid, o in STATUS.items():
        nid = lid
        prefix = (('doctor', 'D1', 1, 'I lean to one option; what do you think?'),)
        nodes[nid] = AgentNode(nid, None, 'c1', 1, prefix, f'reply of {lid}', {}, 'p', 0, 'max_agent_turns', True, None)
        dlg = prefix + (('agent', 'A1', 1, f'reply of {lid}'), ('doctor', 'FINAL', 0, 'Final answer: option B'))
        leaves.append(Leaf(nid, 'c1', (nid,), dlg, 'Final answer: option B', 'B', o, 'max_agent_turns'))
    tins = {l.leaf_id: PH.freeze_trajectory_input(CASE, l, DoctorFeedback(l.leaf_id, l.outcome, f'feedback for {l.leaf_id}', 'h')) for l in leaves}
    return nodes, leaves, tins


class Fakes:
    """generate_fn / assign_fn / gate_fn / scorer / writer / continuation with call logs. reuse(leaf, bank_ids) -> decision."""
    def __init__(self, reuse=lambda leaf, bank: None, gate_pass=lambda leaf: True, reruns=None, applicable=lambda node: True):
        self.reuse, self.gate_pass, self.reruns, self.applicable = reuse, gate_pass, reruns or {}, applicable
        self.calls = {'generate': [], 'assign': [], 'gate': [], 'score': [], 'continue': []}
        self.generate_fn.__func__.model = 'gen-model'

    def generate_fn(self, prompt_text, tin):
        self.calls['generate'].append((tin.leaf_id, prompt_text))
        r = {'name': f'behavior of {tin.leaf_id}', 'applicability_condition': 'the physician states a leaning', 'required_behavior': 'test the leaning',
             'scores': {'0': 'no test', '1': 'partial test', '2': 'explicit test'}}
        return {'raw': json.dumps(r), 'rubric': r, 'parse_error': None}

    def assign_fn(self, package, user):
        leaf = re.search(r'feedback for (L\d)', user).group(1); bank = re.findall(r'"rubric_id": "([^"]+)"', user)
        self.calls['assign'].append((leaf, package, tuple(bank)))
        target = self.reuse(leaf, bank)
        checks = {c: True for c in RB.CHECKS}
        if isinstance(target, tuple): target, failed = target; checks[failed] = False
        return {'decision': 'REUSE_EXISTING' if target else 'CREATE_NEW', 'rubric_id': target or '', 'checks': checks, 'rationale': 'r'}

    def gate_fn(self, tin, rubric):
        self.calls['gate'].append(tin.leaf_id); return {'passed': self.gate_pass(tin.leaf_id), 'reasons': [] if self.gate_pass(tin.leaf_id) else ['new domain premise']}

    def scorer(self, view, prefix, rubrics, reply):
        self.calls['score'].append((reply, tuple(r['rubric_id'] for r in rubrics)))
        node = re.search(r'reply of (L\d)', reply).group(1)
        return {r['rubric_id']: {'applicable': self.applicable(node), 'probabilities': None, 'hard': 2 if self.applicable(node) else None} for r in rubrics}

    def writer(self, case, prefix, turn, rubric, level, seed): return f'intervention level {level} seed {seed}'

    def continuation(self, case, prefix, turn, text, seed):
        node = next(n for n in STATUS if derive_seed(0, n, 'guided', 0) == int(text.rsplit(' ', 1)[1]))
        self.calls['continue'].append((node, text))
        k = sum(1 for n, _ in self.calls['continue'] if n == node) - 1
        o = self.reruns.get(node, [0.0, 0.0, 0.0])[k]
        return {'final_option': 'B' if o == 1.0 else 'C', 'outcome': o}


def banks(f, cache=None):
    nodes, leaves, tins = tree()
    return (*RB.build_banks(CASE, leaves, tins, PROMPT, 'sha', f.generate_fn, f.assign_fn, f.gate_fn, cache or CallCache(None)), nodes, leaves)


SAME = lambda leaf, bank: bank[0] if bank else None                          # every later trajectory reuses the first rubric


# ---------- REUSE vs NEW ----------
def test_separate_strata_first_creates_then_same_reason_reuses():
    f = Fakes(reuse=SAME); canon, asg, *_ = banks(f)
    by = {a.leaf_id: a for a in asg}
    assert [a.leaf_id for a in asg] == ['L0', 'L1', 'L3', 'L2', 'L4']                                    # failure stratum first, leaf-id order
    assert by['L0'].assignment == by['L2'].assignment == 'CREATED_NEW' and by['L0'].decision is None
    assert by['L1'].assignment == by['L3'].assignment == by['L4'].assignment == 'REUSED_EXISTING'
    assert by['L1'].canonical_rubric_id == by['L3'].canonical_rubric_id == by['L0'].canonical_rubric_id != by['L2'].canonical_rubric_id
    assert by['L4'].canonical_rubric_id == by['L2'].canonical_rubric_id                                  # never across strata
    assert all(bank == () or set(bank) <= {by['L0'].canonical_rubric_id} for leaf, _, bank in f.calls['assign'] if STATUS[leaf] == 0.0)
    assert [g[0] for g in f.calls['generate']] == ['L0', 'L2'] and len(canon) == 2                       # no rubric generated for reuses
    assert all(p == 'GENERATION RULES\n' for _, p in f.calls['generate'])                                 # CREATE: generation section only
    assert all(p == PROMPT for _, p, _ in f.calls['assign'])                                              # decision: full package
    assert sorted(a.leaf_id for a in asg) == sorted(STATUS) and len({a.leaf_id for a in asg}) == len(asg)


@pytest.mark.parametrize('failed', RB.CHECKS)
def test_any_failed_check_or_new_reason_generates_a_new_rubric(failed):
    f = Fakes(reuse=lambda leaf, bank: (bank[0], failed) if bank else None); canon, asg, *_ = banks(f)
    assert all(a.assignment == 'CREATED_NEW' for a in asg) and len(canon) == 5 and 'treated as uncertain' in asg[1].note
    f = Fakes(reuse=lambda leaf, bank: None); canon, asg, *_ = banks(f)
    assert all(a.assignment == 'CREATED_NEW' for a in asg) and len(f.calls['generate']) == 5
    f = Fakes(reuse=lambda leaf, bank: 'no-such-id' if bank else None); _, asg, *_ = banks(f)
    assert all(a.assignment == 'CREATED_NEW' for a in asg) and 'unknown rubric id' in asg[1].note


def test_gate_runs_only_for_newly_generated_rubrics_and_failures_stay_out_of_the_bank():
    f = Fakes(reuse=SAME); banks(f); assert f.calls['gate'] == ['L0', 'L2']
    f = Fakes(reuse=SAME, gate_pass=lambda leaf: leaf != 'L0'); canon, asg, *_ = banks(f)
    by = {a.leaf_id: a for a in asg}
    assert by['L0'].assignment == 'CREATE_REJECTED' and 'new domain premise' in by['L0'].note
    assert by['L1'].decision is None and by['L1'].assignment == 'CREATED_NEW'                             # empty bank again: L1 generates
    assert by['L3'].assignment == 'REUSED_EXISTING' and by['L3'].canonical_rubric_id == by['L1'].canonical_rubric_id


def test_decisions_are_cached_separately_and_keyed_by_the_bank(tmp_path):
    f = Fakes(reuse=SAME); c = CallCache(tmp_path / 'calls.jsonl'); banks(f, c)
    kinds = [json.loads(l)['kind'] for l in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert kinds.count('rubric_match') == 3 and kinds.count('rubric_generation') == 2
    g = Fakes(reuse=SAME); banks(g, CallCache(tmp_path / 'calls.jsonl')); assert g.calls['assign'] == g.calls['generate'] == []   # exact resume
    rec = [json.loads(l) for l in (tmp_path / 'calls.jsonl').read_text().splitlines() if json.loads(l)['kind'] == 'rubric_match'][0]
    assert {'tin', 'prompt', 'version', 'stratum', 'bank', 'model', 'settings', 'transcript'} == set(rec['identity'])


def test_old_per_trajectory_cache_cannot_satisfy_reuse_or_new_generation_settings():
    from src.online_rl.tree.backends import api_rubric_generator
    nodes, leaves, tins = tree(); c = CallCache(None); f = Fakes()
    PH.generate_rubric(PROMPT, 'sha', tins['L0'], f.generate_fn, c, None)                               # legacy identity (no settings)
    gen = api_rubric_generator('k', 'gen-model', {'max_tokens': 4096, 'temperature': 0.7, 'reasoning_effort': 'low', 'response_format': 'json_schema_strict'})
    calls = []
    gen2 = lambda p, t: calls.append(t.leaf_id) or f.generate_fn(p, t)
    gen2.model, gen2.settings = 'gen-model', gen.settings
    PH.generate_rubric(PROMPT, 'sha', tins['L0'], gen2, c, None); assert calls == ['L0']


# ---------- member verification, KEEP, R_O ----------
def verify(f):
    canon, asg, nodes, leaves = banks(f)
    _, member = RB.verify_members(CASE, nodes, {l.leaf_id: l for l in leaves}, canon, asg, scorer=f.scorer, writer=f.writer,
                                  continuation=f.continuation, reruns=3, run_seed=0, policy_sha256='p', strategies=team_strategies(None, 1.0, 1.0),
                                  cache=CallCache(None))
    return canon, asg, member


def test_every_member_is_verified_on_its_own_path_in_its_own_direction():
    f = Fakes(reuse=SAME, reruns={'L0': [0.0, 0.0, 0.0], 'L1': [1.0, 1.0, 0.0], 'L3': [1.0, 1.0, 1.0], 'L2': [0.0, 1.0, 1.0], 'L4': [1.0, 1.0, 1.0]})
    canon, asg, member = verify(f)
    assert set(member) == set(STATUS) and all(member[l].leaf_id == l for l in STATUS)
    assert {member[l].level for l in ('L0', 'L1', 'L3')} == {'2'} and {member[l].level for l in ('L2', 'L4')} == {'0'}   # failure->2, success->0
    assert {member[l].rubric_id for l in ('L0', 'L1', 'L3')} == {asg[0].canonical_rubric_id}
    assert [member[l].R_O for l in ('L0', 'L1', 'L3', 'L2', 'L4')] == [0.0, 2 / 3, 1.0, 1 / 3, 0.0]
    assert {s[0] for s in f.calls['score']} >= {f'reply of {l}' for l in STATUS}                          # Stage-1 of R on each member path
    assert len(f.calls['continue']) == 5 * 3


def test_existential_keep_and_r_o_is_the_member_maximum():
    f = Fakes(reuse=SAME, reruns={'L0': [0.0, 0.0, 0.0], 'L1': [1.0, 1.0, 0.0], 'L3': [0.0, 0.0, 0.0], 'L2': [1.0, 1.0, 0.0], 'L4': [1.0, 1.0, 0.0]})
    canon, asg, member = verify(f); sup = RB.canonical_support(canon, asg, member, 2 / 3)
    fail_r, succ_r = asg[0].canonical_rubric_id, asg[3].canonical_rubric_id
    assert sup[fail_r].retained and sup[fail_r].weight == pytest.approx(2 / 3) and sup[fail_r].max_s == pytest.approx(2 / 3)
    assert sup[fail_r].member_s == {'L0': 0.0, 'L1': pytest.approx(2 / 3), 'L3': 0.0} and sup[fail_r].supported_member_count == 1
    assert not sup[succ_r].retained and sup[succ_r].weight is None             # success members: s = failed reruns / K = 1/3 each -> DROP
    assert sup[succ_r].max_s == pytest.approx(1 / 3)


def test_member_count_does_not_multiply_the_weight_or_the_scoring():
    from src.online_rl.tree.scoring import score_nodes
    from src.online_rl.tree.team import effectiveness_weighted_r_a
    one = Fakes(reuse=lambda leaf, bank: None, reruns={'L0': [1.0, 1.0, 1.0]})
    many = Fakes(reuse=SAME, reruns={'L0': [1.0, 1.0, 1.0], 'L1': [1.0, 1.0, 1.0], 'L3': [1.0, 1.0, 1.0]})
    c1, a1, m1 = verify(one); c3, a3, m3 = verify(many)
    s1, s3 = RB.canonical_support(c1, a1, m1, 2 / 3), RB.canonical_support(c3, a3, m3, 2 / 3)
    r = a3[0].canonical_rubric_id
    assert s3[r].member_count == 3 and s3[r].weight == s1[a1[0].canonical_rubric_id].weight == 1.0
    kept = [c3[cid] for cid, sp in s3.items() if sp.retained]; f = Fakes()
    scores = score_nodes(CASE, list(tree()[0].values()), kept, f.scorer, CallCache(None))
    assert all(ids.count(r) == 1 for _, ids in f.calls['score'])                                          # [R1], never [R1, R1, R1]
    w = {cid: s3[cid].weight for cid in [k.rubric_id for k in kept]}
    assert effectiveness_weighted_r_a([s for s in scores if s.response_node_id == 'L0'], w) == 1.0


def test_operational_failure_is_not_a_zero():
    class Boom(Fakes):
        def continuation(self, *a): raise RuntimeError('Doctor unreachable')
    with pytest.raises(RuntimeError, match='unreachable'): verify(Boom(reuse=SAME))


def test_j_stays_trajectory_level():
    from src.online_rl.tree.prompt_search import EvidenceUnit, ScoringDeps, directional_scores, score_prompt
    f = Fakes(reuse=SAME, reruns={'L0': [0.0, 0.0, 0.0], 'L1': [1.0, 1.0, 0.0], 'L3': [1.0, 1.0, 1.0], 'L2': [0.0, 1.0, 1.0], 'L4': [1.0, 1.0, 1.0]})
    nodes, leaves, tins = tree()
    units = [EvidenceUnit('c1', l, (nodes[l.leaf_id],), tins[l.leaf_id], 'p') for l in leaves]
    deps = ScoringDeps(3, 0, f.generate_fn, f.gate_fn, f.scorer, f.writer, f.continuation, 'p', team_strategies(None, 1.0, 1.0), CallCache(None), f.assign_fn)
    res = score_prompt(PROMPT, 'sha', units, {'c1': CASE}, deps); d = directional_scores(units, res, deps.strategies)
    assert d['per_trajectory'] == pytest.approx({'L0': 0.0, 'L1': 2 / 3, 'L3': 1.0, 'L2': 1 / 3, 'L4': 0.0})
    assert d['S_failure'] == pytest.approx((0 + 2 / 3 + 1) / 3) and d['S_success'] == pytest.approx(1 / 6) and d['J'] == pytest.approx(d['S_failure'] * d['S_success'])
    assert res['L1']['assignment'].assignment == 'REUSED_EXISTING' and res['L1']['members'] == ['L0', 'L1', 'L3']


def test_outcome_only_config_has_no_reuse_side():
    import pathlib
    cfg = json.loads((pathlib.Path(__file__).resolve().parents[1] / 'configs/online_rl/tree_v1.json').read_text())
    from src.online_rl.tree.conditions import arm_config
    oo = arm_config(cfg, 'outcome_only'); fx, ev = arm_config(cfg, 'fixed'), arm_config(cfg, 'evolving')
    assert not oo['rubric']['enabled'] and fx['rubric'] == ev['rubric'] and fx['rubric']['reuse'] == 'assign_or_create_v1'
    assert {k: v for k, v in fx.items() if k not in ('prompt_update', 'condition', 'output_dir')} == {k: v for k, v in ev.items() if k not in ('prompt_update', 'condition', 'output_dir')}


# ---------- scorer v2 ----------
class Client:
    model, protocol_sha256 = 'luna', 'p'

    def __init__(self): self.log, self.seen = [], []

    def judge(self, stage, state, items, fields, purpose):
        self.seen.append((stage, state, items, fields))
        if stage == 'applicability': return {i: {'applicable': True, 'evidence_basis': 'x', 'missing_requirement': ''} for i in items}, {}
        return {i: {'level': '1', 'anchor_basis': 'y'} for i in items}, {}


RUB = {'rubric_id': 'r1', 'name': 'N', 'applicability_condition': 'COND', 'required_behavior': 'REQ', 'scores': {'0': 'S0', '1': 'S1', '2': 'S2'}}


def test_scorer_v2_stage_inputs_are_separated():
    from src.online_rl.tree.backends import two_stage_structured_scorer
    cl = Client(); sc = two_stage_structured_scorer(cl)
    out = sc({'case_id': 'c1', 'question': 'Q', 'options': {'A': 'a'}}, [{'speaker': 'doctor', 'stage': 'D1', 'text': 'prev'}], [RUB], 'TARGET REPLY')
    (s1, st1, it1, _), (s2, st2, it2, _) = cl.seen
    assert s1 == 'applicability' and 'TARGET REPLY' not in json.dumps(st1) + json.dumps(it1)
    assert 'REQ' not in json.dumps(it1) and 'S2' not in json.dumps(it1) and 'COND' in json.dumps(it1)
    assert s2 == 'level' and st2['target_agent_reply'] == 'TARGET REPLY' and 'COND' not in json.dumps(it2['r1']['score_anchors']) + it2['r1']['required_behavior']
    assert it2['r1']['score_anchors'] == {'0': 'S0', '1': 'S1', '2': 'S2'} and it2['r1']['required_behavior'] == 'REQ'
    blob = json.dumps([st1, st2]).lower()
    assert all(k not in blob for k in ('final', 'outcome', 'ground_truth', 'correct_option', 'r_o'))
    assert out['r1']['applicable'] and out['r1']['hard'] == 1 and out['r1']['stage1']['evidence_basis'] == 'x'


def test_luna_structured_output_gets_exactly_one_repair():
    from src.online_rl.tree.luna import LunaDecisionClient
    from src.common.config import IntegrationUnavailable
    from src.online_rl.tree.backends import STAGE1_FIELDS
    good = json.dumps({'answers': {'r1': {'applicable': False, 'evidence_basis': '', 'missing_requirement': 'no leaning stated'}}})
    for script, ok in (([good], True), (['{"answers": {"r1": {"applicable": "yes"}}}', good], True), (['bad', 'still bad'], False)):
        sent = []
        post = lambda url, payload, key, t, s=list(script): sent.append(payload) or {'choices': [{'message': {'content': s.pop(0)}}]}
        c = LunaDecisionClient(model='luna', key='k', system='S', post=post)
        if ok:
            out, rec = c.judge('applicability', {}, {'r1': {}}, STAGE1_FIELDS, 'p'); assert out['r1']['applicable'] is False and len(sent) == len(script)
        else:
            with pytest.raises(IntegrationUnavailable): c.judge('applicability', {}, {'r1': {}}, STAGE1_FIELDS, 'p')
            assert len(sent) == 2


# ---------- Gemini output fix ----------
def test_truncated_or_malformed_generation_is_operational(monkeypatch):
    from src.online_rl.tree import backends as B
    from src.prompt_opt import harness as H
    nodes, leaves, tins = tree(); settings = {'max_tokens': 4096, 'temperature': 0.7, 'reasoning_effort': 'low', 'response_format': 'json_schema_strict'}
    sent = []

    def fake(key, model, system, user, max_tokens, temperature, extra=None, meta=None):
        sent.append(extra); meta.update(finish_reason=fake.finish); return fake.text
    monkeypatch.setattr(H, 'doctor_call', fake)
    gen = B.api_rubric_generator('k', 'google/gemini-3.8-flash-20260902', settings); c = CallCache(None)
    fake.finish, fake.text = 'length', '{"name": "cut'
    with pytest.raises(B.RubricOutputIncomplete, match='finish_reason=length'): PH.generate_rubric('P', 'sha', tins['L0'], gen, c, None)
    fake.finish, fake.text = 'stop', 'not json'
    with pytest.raises(B.RubricOutputIncomplete, match='did not parse'): PH.generate_rubric('P', 'sha', tins['L0'], gen, c, None)
    assert c.store == {}                                                                                  # nothing cached, nothing scored
    assert sent[0]['reasoning'] == {'effort': 'low'} and sent[0]['response_format']['json_schema']['strict'] is True
    fake.text = json.dumps({'name': 'n', 'applicability_condition': 'a', 'required_behavior': 'b', 'scores': {'0': 'x', '1': 'y', '2': 'z'}})
    assert PH.generate_rubric('P', 'sha', tins['L0'], gen, c, None).valid
    m = B.api_rubric_matcher('k', 'google/gemini-3.8-flash-20260902', settings)
    fake.text = json.dumps({'decision': 'REUSE_EXISTING', 'rubric_id': 'x', 'checks': {k: True for k in B.MATCH_CHECKS}})
    with pytest.raises(B.RubricOutputIncomplete, match='schema'): m('P', 'U')
