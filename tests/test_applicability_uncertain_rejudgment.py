"""Uncertain Stage-1 applicability: two fresh re-judgments, then abstain on that rubric / state pair only (2026-10-11)."""
from dataclasses import replace
import pytest
from src.online_rl.tree import applicability as A
from src.online_rl.tree import posthoc, scoring
from src.online_rl.tree.records import ResponseRubricScore
from src.online_rl.tree.team import effectiveness_weighted_r_a

UNCERTAIN = {'applicable': None, 'applicability_status': 'uncertain', 'provenance': {'cache_key': 'k0'}}


def judge(*verdicts):
    calls = []
    def rejudge(n):
        calls.append(n); v = verdicts[n - 1]
        return {'applicable': v, 'applicability_status': 'uncertain' if v is None else 'resolved'}
    return rejudge, calls


def test_uncertain_then_true_is_used_normally():
    rejudge, calls = judge(True)
    row, audit = A.resolve_uncertain(dict(UNCERTAIN), rejudge)
    assert row['applicable'] is True and 'unresolved_abstain' not in row and calls == [1]
    assert audit == {'initial': 'uncertain', 'retries': [{'retry': 1, 'result': 'true'}], 'final': 'retry_resolved_true'}


def test_uncertain_then_false_is_used_normally():
    rejudge, calls = judge(None, False)
    row, audit = A.resolve_uncertain(dict(UNCERTAIN), rejudge)
    assert row['applicable'] is False and 'unresolved_abstain' not in row and calls == [1, 2]
    assert audit['final'] == 'retry_resolved_false' and [r['result'] for r in audit['retries']] == ['uncertain', 'false']


def test_three_uncertain_verdicts_abstain_without_raising():
    rejudge, calls = judge(None, None)
    row, audit = A.resolve_uncertain(dict(UNCERTAIN), rejudge)
    assert row['applicable'] is None and row['unresolved_abstain'] is True and calls == [1, 2]
    assert audit['final'] == 'final_unresolved_abstain' and [r['result'] for r in audit['retries']] == ['uncertain', 'uncertain']


def test_unresolved_source_pair_is_skipped_not_judged_inconsistent():
    class Tin: target_reply_id = 1
    check, rubric, raw, problems = posthoc._source_consistency('p', 'sha', Tin(), None, None,
        lambda r: {'applicable': None, 'target_reply_id': 1, 'node_id': 'n'}, {'rubric_id': 'r'}, 'raw', 'r')
    assert check['status'] == 'APPLICABILITY_UNRESOLVED' and problems and problems[0].startswith('source_unresolved')
    assert all('source_inconsistent' not in p for p in problems)


def test_unresolved_pair_gets_no_score_and_the_rest_is_scored(monkeypatch):
    class Node:
        node_id = 'c/0'; policy_sha256 = 'pol'; prefix = (); response = 'reply'; trace = {}
    class Rubric:
        def __init__(self, rid): self.rubric_id = rid; self.leaf_id = 'c/0'; self.rubric = {'rubric_id': rid, 'name': rid}
    class Scorer:
        model = 'm'; protocol = 'p'; review_identity = None
        def review_state(self, *a): raise AssertionError('applicability comes from state_applicability')
        def score_behavior(self, view, prefix, rubrics, reply, matched=None):
            return {r['rubric_id']: {'applicable': True, 'hard': 2, 'probabilities': None, 'scoring_status': 'scored'} for r in rubrics}
    class Cache:
        def call(self, kind, identity, fn): return fn()
    verdict = {'unresolved': None, 'resolved': True}
    monkeypatch.setattr(scoring, 'state_applicability', lambda case, n, r, scorer, cache: {'applicable': verdict[r['rubric_id']]})
    monkeypatch.setattr(scoring, 'visible_rubric', lambda rubric: dict(rubric))
    monkeypatch.setattr(scoring, 'case_view', lambda case: {'case_id': 'c'})
    monkeypatch.setattr(scoring, 'to_dialogue', lambda prefix: [])
    rows = scoring.score_nodes({'case_id': 'c'}, [Node()], [Rubric('resolved'), Rubric('unresolved')], Scorer(), Cache())
    by = {r.rubric_id: r for r in rows}
    assert by['unresolved'].scoring_status == 'applicability_unresolved' and by['unresolved'].score is None and by['unresolved'].match is False
    assert by['resolved'].scoring_status == 'scored' and by['resolved'].score == 1.0
    assert effectiveness_weighted_r_a(rows, {'resolved': 1.0, 'unresolved': 1.0}) == 1.0     # the abstained pair adds no reward or penalty


def test_abstained_pair_alone_gives_no_rubric_reward():
    row = ResponseRubricScore('c/0', 'r', 'c/0', False, None, None, None, 'applicability_unresolved')
    assert effectiveness_weighted_r_a([row], {'r': 1.0}) is None
