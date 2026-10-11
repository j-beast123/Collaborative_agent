"""EXECUTION_UNRESOLVED: a judgment whose response is still execution-invalid after the local retry budget has no
verdict. It never stops the run, is never read as FALSE / NA / 0, and never relaxes the admission or witness rules."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from scripts.online_rl.run_tree import make_scorer
from src.common.config import IntegrationUnavailable
from src.online_rl.tree import applicability as A
from src.online_rl.tree import instruction_audit as IA
from src.online_rl.tree import offline_chunk_bank as OCB
from src.online_rl.tree import posthoc, witness as W
from src.online_rl.tree.backends import RubricOutputIncomplete
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.experiment import load_config
from src.online_rl.tree.luna import LunaDecisionClient
from src.online_rl.tree.records import ResponseRubricScore
from src.online_rl.tree.scoring import score_nodes
from src.online_rl.tree.team import effectiveness_weighted_r_a
from tests.test_compact_runtime_v7 import CFG, Transport
from tests.test_scoring_batch import CASE, KEPT, RUBRIC, node


def scorer(broken, attempts=3):
    """broken(packet) -> True: that request's response is truncated (finish_reason = length), every time."""
    cfg = load_config(CFG)
    if attempts: cfg['judges']['execution_retry_attempts'] = attempts
    good = Transport()
    def post(url, payload, key, timeout):
        body = good(url, payload, key, timeout)
        if broken(json.loads(payload['messages'][1]['content'])): body['choices'][0]['finish_reason'] = 'length'
        return body
    client = LunaDecisionClient(model=cfg['judges']['scorer_model'], key='OFFLINE', system=Path(cfg['judges']['scorer_prompt_file']).read_text(),
                                max_tokens=4096, post=post)
    return good, make_scorer(cfg, client)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    import time; monkeypatch.setattr(time, 'sleep', lambda s: None)


def is_applicability(p): return 'responses' not in p


def unresolved_log(path): return [json.loads(x) for x in Path(path).with_suffix('.execution_unresolved.jsonl').read_text().splitlines()]


# ---------------------------------------------------------------- applicability (source / node x rubric)
def test_applicability_invalid_after_budget_is_terminal_and_logged(tmp_path):
    t, s = scorer(is_applicability); cache = CallCache(tmp_path / 'calls.jsonl')
    row = A.state_applicability(CASE, node(0), RUBRIC, s, cache)
    assert row['execution_unresolved'] is True and row['applicable'] is None and row['applicability_status'] == 'execution_unresolved'
    assert row['total_attempts'] == 3 and len(t.calls) == 3                      # the configured budget, no extra uncertain re-judgment
    assert 'uncertain_rejudgment' not in row and 'unresolved_abstain' not in row  # kept apart from scientific UNCERTAIN
    assert [r['status'] for r in row['reasons_by_attempt']] == ['execution_incomplete'] * 3
    again = A.state_applicability(CASE, node(1), {**RUBRIC, 'rubric_id': 'alias'}, s, cache)
    assert again['execution_unresolved'] is True and len(t.calls) == 3          # terminal for this run: not re-sent
    log = unresolved_log(cache.path)
    assert log[0]['final_status'] == 'EXECUTION_UNRESOLVED' and log[0]['stage'] == 'applicability' and log[0]['total_attempts'] == 3
    assert log[0]['case_id'] == CASE['case_id'] and log[0]['rubric_id'] == RUBRIC['rubric_id'] and log[0]['raw_reference']
    assert not Path(cache.path).with_suffix('.applicability_uncertain.jsonl').exists()


def test_without_a_retry_budget_the_original_error_is_kept():
    _, s = scorer(is_applicability, attempts=None)
    with pytest.raises(IntegrationUnavailable): A.state_applicability(CASE, node(0), RUBRIC, s, CallCache(None))


def test_source_check_does_not_admit_and_is_not_source_inconsistent():
    class Tin: target_reply_id = 1
    check, rubric, raw, problems = posthoc._source_consistency('p', 'sha', Tin(), None, None,
        lambda r: {'applicable': None, 'execution_unresolved': True, 'target_reply_id': 1, 'node_id': 'n'}, {'rubric_id': 'r'}, 'raw', 'r')
    assert check['status'] == 'EXECUTION_UNRESOLVED' and problems and problems[0].startswith('execution_unresolved')
    assert all('source_inconsistent' not in p and 'source_unresolved' not in p for p in problems)


def test_node_rubric_applicability_unresolved_is_excluded_from_r_a():
    _, s = scorer(is_applicability)
    rows = score_nodes(CASE, [replace(node(i), response=f'Observed response {i}') for i in range(2)], [KEPT], s, CallCache(None))
    assert [r.scoring_status for r in rows] == ['execution_unresolved'] * 2
    assert all(r.score is None and r.hard is None and r.match is False for r in rows)
    assert effectiveness_weighted_r_a(rows, {KEPT.rubric_id: 1.0}) is None       # no zero, no penalty


# ---------------------------------------------------------------- behavior scoring
def test_only_the_invalid_behavior_pair_is_left_unscored(tmp_path):
    t, s = scorer(lambda p: 'responses' in p and p['responses'][0]['reply'].endswith('1'))
    nodes = [replace(node(i), response=f'Observed response {v}') for i, v in enumerate([0, 1, 2, 2])]
    cache = CallCache(tmp_path / 'calls.jsonl')
    rows = score_nodes(CASE, nodes, [KEPT], s, cache)
    assert [r.scoring_status for r in rows] == ['scored', 'execution_unresolved', 'scored', 'scored']
    assert [r.score for r in rows] == [0, None, 1, 1] and rows[1].hard is None and rows[1].match is True
    weights = {KEPT.rubric_id: 1.0}
    assert [effectiveness_weighted_r_a([r], weights) for r in rows] == [0, None, 1, 1]
    other = ResponseRubricScore(rows[1].response_node_id, 'other', 'leaf', True, 1.0, 2, None, 'scored')
    assert effectiveness_weighted_r_a([rows[1], other], {**weights, 'other': 1.0}) == 1.0     # the remaining valid judgment decides R_A
    behavior_calls = [json.loads(c['messages'][1]['content']) for c in t.calls]
    assert sum('responses' in p and p['responses'][0]['reply'].endswith('1') for p in behavior_calls) == 3
    log = unresolved_log(cache.path)
    assert [(e['stage'], e['node_id'], e['total_attempts']) for e in log] == [('behavior_scoring', nodes[1].node_id, 3)]


def test_unresolved_row_with_a_number_is_rejected():
    from src.online_rl.tree.scoring import _score_record, ScoringContractError
    with pytest.raises(ScoringContractError):
        _score_record(node(0), 'r', 'leaf', True, {'applicable': True, 'hard': 0, 'probabilities': None, 'scoring_status': 'execution_unresolved'})


# ---------------------------------------------------------------- PEER re-confirmation
def test_peer_recheck_invalid_after_budget_confirms_nobody():
    calls = []
    def request(): calls.append(1); raise RubricOutputIncomplete('matcher JSON schema parse error')
    decisions, n, reasons = OCB.recheck_peers(request, 4)
    assert decisions is None and n == 4 and len(calls) == 4 and len(reasons) == 4
    assert OCB.PEER_EXECUTION_UNRESOLVED.startswith('EXECUTION_UNRESOLVED')


def test_peer_recheck_recovers_within_budget_and_keeps_original_behaviour_without_one():
    state = {'n': 0}
    def flaky():
        state['n'] += 1
        if state['n'] < 3: raise RubricOutputIncomplete('truncated')
        return {'p': 'decision'}
    assert OCB.recheck_peers(flaky, 12) == ({'p': 'decision'}, 3, ['truncated', 'truncated'])
    def bad(): raise RubricOutputIncomplete('truncated')
    with pytest.raises(RubricOutputIncomplete): OCB.recheck_peers(bad, None)


# ---------------------------------------------------------------- witness / intervention audit
def invalid(message='Invalid instruction audit evidence after one correction: quote'):
    exc = IntegrationUnavailable(message); exc.audit = {'requests': [], 'decisions': [], 'validation_errors': [message]}; return exc


def test_audit_wrapper_turns_only_invalid_responses_into_no_verdict():
    seen = []
    def auditor(case, prefix, rubric, member_spec): raise invalid()
    auditor.settings = {'s': 1}; auditor.model = 'm'; auditor.protocol = IA.PROTOCOL
    audit = IA.execution_unresolved_auditor(auditor, 12, lambda rubric, spec, exc: seen.append(rubric['rubric_id']))
    out = audit({}, [], {'rubric_id': 'r'}, None)
    assert out['execution_unresolved'] is True and 'passed' not in out and out['total_attempts'] == 12 and seen == ['r']
    assert audit.settings == {'s': 1}
    def offline(case, prefix, rubric, member_spec): raise IntegrationUnavailable('HTTP 401: no credentials')
    with pytest.raises(IntegrationUnavailable): IA.execution_unresolved_auditor(offline, 12)({}, [], {'rubric_id': 'r'}, None)


def verify(monkeypatch, verdicts, results, audits):
    """verdicts: leaf -> True / 'exec' (stage 1); audits: leaf -> 'pass' / 'exec'; results: leaf -> witness_test hit."""
    from tests.test_witness_unresolved_member import Run
    tested_calls = []
    monkeypatch.setattr(W, 'resolve_member_intervention', lambda rubric, leaf, member_spec=None, node=None:
                        {'selected_agent_turn_id': 1, 'leaf': leaf.leaf_id})
    monkeypatch.setattr(W, 'stage1_decision', lambda case, node, rubric, applicability, cache:
        {'applicable': None, 'stage1': {}, 'provenance': {}, 'execution_unresolved': True} if verdicts[node.node_id] == 'exec'
        else {'applicable': verdicts[node.node_id], 'stage1': {}, 'provenance': {}})
    def witness_test(case, leaf, node, rub, **kw):
        tested_calls.append(leaf.leaf_id); hit = results[leaf.leaf_id]
        return hit, [Run(1.0 if hit else 0.0), Run(1.0 if hit else 0.0)]
    monkeypatch.setattr(W, 'witness_test', witness_test)
    monkeypatch.setattr(W, 'reaches_target', lambda level, outcome: outcome == 1.0)
    monkeypatch.setattr(W, 'case_view', lambda case: case); monkeypatch.setattr(W, 'to_dialogue', lambda prefix: [])
    monkeypatch.setattr(W, 'visible_rubric', lambda rubric: dict(rubric))
    ids = sorted(verdicts)
    def audit(case, prefix, rubric, member_spec):
        return ({'execution_unresolved': True, 'protocol': IA.PROTOCOL} if audits.get(member_spec['leaf']) == 'exec'
                else {'passed': True, 'reasons': []})
    leaves = {i: NS(leaf_id=i, path=(i,), outcome=0.0) for i in ids}
    nodes = {i: NS(node_id=i, prefix=()) for i in ids}
    canon = {'r': NS(rubric_id='r', valid=True, rubric={'rubric_id': 'r'}, leaf_id=ids[0])}
    assignments = [NS(canonical_rubric_id='r', leaf_id=i, assignment='CREATED_NEW' if k == 0 else 'REUSED_EXISTING',
                      intervention_spec=None, stratum='failure') for k, i in enumerate(ids)]
    out = W.verify_witness({'case_id': 'c'}, nodes, leaves, canon, assignments, {}, applicability=None, writer=None, continuation=None,
                           reruns=3, run_seed=0, policy_sha256='p', cache=None, max_valid_supports=5,
                           support_sampling=W.RANDOM5_SAMPLING, intervention_audit=audit)['r']
    return out, tested_calls


def test_stage1_execution_unresolved_member_is_no_evidence(monkeypatch):
    out, calls = verify(monkeypatch, {'a': 'exec'}, {}, {})
    assert out.tested[0]['result'] == 'execution_unresolved' and calls == [] and out.num_valid_supports == 0
    assert out.decision == 'NOT_VALIDATED' and W.witness_telemetry({'r': out})['no_match'] == 0


def test_audit_execution_unresolved_runs_no_rerun_and_never_keeps(monkeypatch):
    out, calls = verify(monkeypatch, {'a': True}, {'a': True}, {'a': 'exec'})
    assert out.tested[0]['result'] == 'execution_unresolved' and calls == [] and out.num_reruns_executed == 0
    assert out.decision == 'NOT_VALIDATED' and out.validated_by_trajectory_id is None


def test_keep_still_requires_a_fully_judged_passing_member(monkeypatch):
    out, calls = verify(monkeypatch, {'a': True, 'b': True}, {'a': True, 'b': True}, {'a': 'exec', 'b': 'pass'})
    assert out.decision == 'KEEP' and out.validated_by_trajectory_id == 'b' and calls == ['b']
    out, calls = verify(monkeypatch, {'a': True, 'b': True}, {'a': True, 'b': False}, {'a': 'exec', 'b': 'pass'})
    assert out.decision == 'NOT_VALIDATED' and calls == ['b']                    # the unresolved member relaxes nothing
