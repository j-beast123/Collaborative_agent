"""A member whose Stage-1 applicability stays unresolved gives the witness no evidence: it is not a No Match, not a support,
and it neither relaxes nor replaces the original PASS requirement (two real target hits on an Applicable member)."""
from dataclasses import dataclass
from types import SimpleNamespace as NS
from src.online_rl.tree import witness as W


@dataclass
class Run:
    outcome: float
    level: str = '2'
    intervention_text: str = 'x'
    final_option: str = 'A'


def verify(monkeypatch, verdicts, results):
    """verdicts: leaf_id -> True / False / None (unresolved); results: leaf_id -> witness_test result (True / False)."""
    tested_calls = []
    monkeypatch.setattr(W, 'resolve_member_intervention', lambda rubric, leaf, member_spec=None, node=None: {'selected_agent_turn_id': 1})
    monkeypatch.setattr(W, 'stage1_decision', lambda case, node, rubric, applicability, cache:
                        {'applicable': verdicts[node.node_id], 'stage1': {}, 'provenance': {}})
    def witness_test(case, leaf, node, rub, **kw):
        tested_calls.append(leaf.leaf_id); hit = results[leaf.leaf_id]
        return hit, [Run(1.0 if hit else 0.0), Run(1.0 if hit else 0.0)]
    monkeypatch.setattr(W, 'witness_test', witness_test)
    monkeypatch.setattr(W, 'reaches_target', lambda level, outcome: outcome == 1.0)
    ids = sorted(verdicts)
    leaves = {i: NS(leaf_id=i, path=(i,), outcome=0.0) for i in ids}
    nodes = {i: NS(node_id=i, prefix=()) for i in ids}
    canon = {'r': NS(rubric_id='r', valid=True, rubric={'rubric_id': 'r'}, leaf_id=ids[0])}
    assignments = [NS(canonical_rubric_id='r', leaf_id=i, assignment='CREATED_NEW' if k == 0 else 'REUSED_EXISTING',
                      intervention_spec=None, stratum='failure') for k, i in enumerate(ids)]
    out = W.verify_witness({'case_id': 'c'}, nodes, leaves, canon, assignments, {}, applicability=None, writer=None, continuation=None,
                           reruns=3, run_seed=0, policy_sha256='p', cache=None, max_valid_supports=5,
                           support_sampling=W.RANDOM5_SAMPLING)['r']
    return out, tested_calls


def test_unresolved_member_is_not_no_match_and_not_a_support(monkeypatch):
    out, calls = verify(monkeypatch, {'a': None}, {})
    row = out.tested[0]
    assert row['result'] == 'unresolved_abstain' and row['stage1_applicable'] is None and row['reruns'] == 0
    assert calls == [] and out.num_valid_supports == 0 and out.num_reruns_executed == 0
    assert out.decision == 'NOT_VALIDATED' and out.validated_by_trajectory_id is None       # no evidence is never a KEEP
    assert W.witness_telemetry({'r': out})['no_match'] == 0


def test_keep_still_needs_a_passing_applicable_member(monkeypatch):
    out, calls = verify(monkeypatch, {'a': None, 'b': True}, {'b': True})
    assert out.decision == 'KEEP' and out.validated_by_trajectory_id == 'b' and calls == ['b'] and out.num_valid_supports == 1
    assert {t['leaf_id']: t['result'] for t in out.tested}.get('a', 'unresolved_abstain') == 'unresolved_abstain'


def test_unresolved_member_does_not_relax_a_failed_witness(monkeypatch):
    out, calls = verify(monkeypatch, {'a': None, 'b': True}, {'b': False})
    assert out.decision == 'NOT_VALIDATED' and out.validated_by_trajectory_id is None and calls == ['b'] and out.num_valid_supports == 1
    assert [t['result'] for t in out.tested if t['leaf_id'] == 'b'] == ['FAIL']


def test_false_verdict_is_still_recorded_as_no_match(monkeypatch):
    out, calls = verify(monkeypatch, {'a': False}, {})
    assert out.tested[0]['result'] == 'no_match' and calls == [] and out.decision == 'NOT_VALIDATED'
