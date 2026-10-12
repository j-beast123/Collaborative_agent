"""Source, witness and scoring share singleton applicability; no live API calls."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.applicability import APPLICABILITY_CONTRACT, ScoringContractError, state_applicability
from src.online_rl.tree.backends import two_stage_structured_scorer
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import Leaf
from src.online_rl.tree.scoring import score_nodes
from src.online_rl.tree import witness as W
from tests.test_scoring_batch import BatchClient, CASE, KEPT, RUBRIC, node
from tests.test_witness_protocol import assignments, rubric, tree


def test_source_witness_and_scoring_share_a_singleton_decision_across_nodes_and_bundles(tmp_path):
    client = BatchClient()
    scorer = two_stage_structured_scorer(client)
    cache = CallCache(tmp_path / 'calls.jsonl')
    source, sibling = node(0), node(1)
    leaf = Leaf('source', 'c1', (source.node_id,), (), '', '', 0.0, 'max_agent_turns')
    source_check = W.source_stage1_for(CASE, {source.node_id: source}, leaf, 1, scorer.applicability, cache)(RUBRIC)
    decision = W.stage1_decision(CASE, sibling, RUBRIC, scorer.applicability, cache)
    second = replace(KEPT, rubric_id='r2', rubric={**RUBRIC, 'rubric_id': 'r2'})
    scores = score_nodes(CASE, [sibling], [second, KEPT], scorer, cache)

    app_calls = [call for call in client.calls if call[0] == 'applicability']
    assert [list(call[2]) for call in app_calls] == [['r1'], ['r2']]
    scored = next(s for s in scores if s.rubric_id == 'r1')
    assert scored.scoring_evidence['applicability'] == decision
    assert source_check['applicability_provenance'] == decision['provenance']
    identity = decision['provenance']['identity']
    assert identity['contract'] == APPLICABILITY_CONTRACT
    assert set(identity) == {'contract', 'state_sha256', 'policy', 'rubric_id', 'rubric_sha256', 'model', 'protocol', 'review_identity'}
    assert decision['provenance']['cache_key'] == sha256_json(['rubric_state_applicability', identity])

    def forbidden(*args):
        pytest.fail('Reloaded decisions must not call the judge')
    client.judge = forbidden
    loaded = CallCache(cache.path)
    other = replace(sibling, node_id='other-node', response='entirely different target',
                    trace={'pre_response_belief': {'different': 'hidden'}}, next_doctor='future outcome')
    assert W.stage1_decision({**CASE, 'correct_option': 'B', 'future': 'unseen'}, other,
                            {**RUBRIC, 'source_provenance': {'leaf_id': 'unseen'}}, scorer.applicability, loaded) == decision
    assert cache.path.read_text() == loaded.path.read_text()


def test_scoring_first_is_reused_by_witness():
    client = BatchClient(); scorer = two_stage_structured_scorer(client); cache = CallCache(None)
    scores = score_nodes(CASE, [node(0)], [KEPT], scorer, cache)
    decision = W.stage1_decision(CASE, node(9), RUBRIC, scorer.applicability, cache)
    assert decision == scores[0].scoring_evidence['applicability']
    assert len([c for c in client.calls if c[0] == 'applicability']) == 1


def test_noop_source_repair_reuses_the_same_negative_decision():
    from tests.test_source_consistency import CASE as SOURCE_CASE, Gen, LEAF, NODES, gate_ok, tin
    from src.online_rl.tree.posthoc import generate_rubric
    cache, calls = CallCache(None), []
    def review(view, prefix, rubrics):
        calls.append(rubrics)
        return {rubrics[0]['rubric_id']: {'applicable': False}}
    source_check = W.source_stage1_for(SOURCE_CASE, NODES, LEAF, 2, review, cache)
    result = generate_rubric('P', 'p' * 64, tin(), Gen(), cache, gate_ok, source_check)
    assert result.source_check['status'] == 'SOURCE_INCONSISTENT'
    assert len(calls) == 1
    assert result.source_check['first']['applicability_provenance'] == result.source_check['second']['applicability_provenance']


def test_decisions_partition_changed_visible_content_prefix_model_protocol_and_policy():
    client = BatchClient(); scorer = two_stage_structured_scorer(client); cache = CallCache(None)
    def decide(n=node(0), rub=RUBRIC, case=CASE):
        return W.stage1_decision(case, n, rub, scorer.applicability, cache)['provenance']['cache_key']
    keys = [decide()]
    keys.append(decide(rub={**RUBRIC, 'required_behavior': 'A changed behavior under the same ID.'}))
    keys.append(decide(n=replace(node(0), prefix=(('doctor', 'D1', 1, 'A changed request.'),))))
    keys.append(decide(case={**CASE, 'question': 'A changed visible question?'}))
    keys.append(decide(n=replace(node(0), policy_sha256='different-policy')))
    scorer.applicability.model = 'different-model'; keys.append(decide())
    scorer.applicability.protocol = 'different-review-protocol'; keys.append(decide())
    assert len(set(keys)) == len(keys) == 7
    assert len(client.calls) == 7


def test_old_node_or_bundle_cache_records_are_not_reused():
    cache = CallCache(None)
    cache.call('rubric_applicability', {'node': node(0).node_id}, lambda: {'r1': {'applicable': False}})
    cache.call('rubric_state_applicability', {'state_sha256': 'old', 'rubrics': 'old-bundle'},
               lambda: {'r1': {'applicable': False}})
    client = BatchClient(); scorer = two_stage_structured_scorer(client)
    assert W.stage1_decision(CASE, node(0), RUBRIC, scorer.applicability, cache)['applicable'] is True
    assert len(client.calls) == 1


def test_policy_boundaries_are_not_collapsed_by_scoring_state_groups():
    client = BatchClient(); scorer = two_stage_structured_scorer(client); cache = CallCache(None)
    nodes = [node(0), replace(node(1), policy_sha256='policy-b')]
    scores = score_nodes(CASE, nodes, [KEPT], scorer, cache, workers=2)
    assert len([c for c in client.calls if c[0] == 'applicability']) == 2
    assert {s.scoring_evidence['applicability']['provenance']['identity']['policy'] for s in scores} == {'policy-sha', 'policy-b'}


@pytest.mark.parametrize('bad', [
    {}, [], {'r1': {'applicable': 'false'}}, {'r1': {'applicable': 1}},
    {'r1': {'applicable': None}}, {'r1': {'applicable': True, 'error': 'judge failed'}},
    {'r1': {'applicable': True}, 'extra': {'applicable': True}},
    {'r1': {'applicable': True, 'stage1': []}},
    {'r1': {'applicable': True, 'stage1': {'unexpected': 'malformed'}}},
    {'r1': {'applicable': True, 'stage1': {'applicable': 'false', 'evidence_basis': '', 'missing_requirement': ''}}},
    {'r1': {'applicable': True, 'stage1': {'applicable': False, 'evidence_basis': '', 'missing_requirement': ''}}},
    {'r1': {'applicable': True, 'stage1': {'applicable': True}}},
])
def test_invalid_applicability_never_enters_cache_and_retry_is_fresh(bad):
    cache = CallCache(None)
    with pytest.raises(ScoringContractError):
        W.stage1_decision(CASE, node(0), RUBRIC, lambda *args: bad, cache)
    assert cache.store == {} and cache._inflight == {}
    assert W.stage1_decision(CASE, node(1), RUBRIC,
                            lambda *args: {'r1': {'applicable': False}}, cache)['applicable'] is False


def test_loaded_cache_values_are_revalidated(tmp_path):
    cache = CallCache(tmp_path / 'calls.jsonl')
    value = state_applicability(CASE, node(0), RUBRIC, lambda *args: {'r1': {'applicable': True}}, CallCache(None))
    cache.call('rubric_state_applicability', value['provenance']['identity'], lambda: {'r1': {'applicable': 'yes'}})
    with pytest.raises(ScoringContractError):
        state_applicability(CASE, node(1), RUBRIC, lambda *args: pytest.fail('must inspect cached value'), CallCache(cache.path))


def test_simultaneous_source_witness_and_scoring_requests_are_single_flight(tmp_path):
    started, release, follower, source_started = Event(), Event(), Event(), Event()
    calls = []
    def review(view, prefix, rubrics):
        calls.append(rubrics)
        started.set()
        assert release.wait(5)
        return {'r1': {'applicable': False}}
    review.model = 'test'; review.protocol = 'shared-test'
    scorer = SimpleNamespace(applicability=review, review_state=review,
                             score_behavior=lambda *args: pytest.fail('NA must skip behavior'))
    cache = CallCache(tmp_path / 'calls.jsonl')
    def score():
        follower.set()
        return score_nodes(CASE, [node(1)], [KEPT], scorer, cache)
    def source():
        n = node(2)
        leaf = Leaf('source', 'c1', (n.node_id,), (), '', '', 0.0, 'max_agent_turns')
        source_started.set()
        return W.source_stage1_for(CASE, {n.node_id: n}, leaf, 1, review, cache)(RUBRIC)
    with ThreadPoolExecutor(3) as pool:
        witness = pool.submit(W.stage1_decision, CASE, node(0), RUBRIC, review, cache)
        assert started.wait(5)
        scoring = pool.submit(score)
        source_check = pool.submit(source)
        assert follower.wait(5)
        assert source_started.wait(5)
        release.set()
        decision, scored, checked = witness.result(), scoring.result()[0], source_check.result()
        assert decision['applicable'] is scored.match is checked['applicable'] is False
        assert decision['provenance'] == scored.scoring_evidence['applicability']['provenance'] == checked['applicability_provenance']
    assert len(calls) == 1
    assert cache.calls == {'rubric_state_applicability': 1}


def _verify(audit, *, members=2, outcomes=0.0, sampling='topology'):
    nodes, leaves = tree([f'{i}.0' for i in range(members)])
    cache = CallCache(None)
    result = W.verify_witness(CASE, nodes, leaves, {'R': rubric()}, assignments(leaves),
        {leaf_id: 2 for leaf_id in leaves},
        applicability=lambda *args: {'R': {'applicable': True}},
        writer=lambda *args: 'intervention', continuation=lambda *args: {'final_option': 'A', 'outcome': outcomes},
        reruns=3, run_seed=0, policy_sha256='pol', cache=cache, intervention_audit=audit,
        max_valid_supports=5 if sampling == W.RANDOM5_SAMPLING else 4, support_sampling=sampling)['R']
    return result, cache


def test_witness_persists_shared_decision_and_instruction_rejection_does_not_reject_other_members():
    answers = iter([False, True])
    result, cache = _verify(lambda *args: {'passed': next(answers)}, outcomes=1.0)
    assert result.decision == 'KEEP'
    assert [row['result'] for row in result.tested] == ['instruction_rejected', 'PASS']
    assert result.num_valid_supports == 1 and result.num_reruns_executed == 2
    assert result.tested[0]['applicability_provenance'] == result.tested[1]['applicability_provenance']
    assert cache.calls['rubric_state_applicability'] == 1


def test_random5_rejected_instructions_do_not_resample_or_execute_reruns():
    result, cache = _verify(lambda *args: {'passed': False}, members=12, sampling=W.RANDOM5_SAMPLING)
    assert len(result.candidate_order) == len(result.tested) == 5
    assert result.num_valid_supports == result.num_reruns_executed == 0
    assert result.decision == 'NOT_VALIDATED'
    assert all(row['result'] == 'instruction_rejected' for row in result.tested)
    assert cache.calls == {'rubric_state_applicability': 1}


@pytest.mark.parametrize('bad', [None, {}, {'passed': 'false'}, {'passed': True, 'error': 'unavailable'}])
def test_malformed_intervention_audit_is_an_operational_error(bad):
    with pytest.raises(ScoringContractError, match='instruction audit'):
        _verify(lambda *args: bad)


def test_unavailable_intervention_audit_skips_support_without_crashing_worker():
    from src.common.config import IntegrationUnavailable

    def unavailable(*args):
        error = IntegrationUnavailable('invalid exact quote')
        error.audit = {'raw': 'bad evidence'}
        raise error

    result, cache = _verify(unavailable)
    assert result.decision == 'INCOMPLETE'
    assert result.num_valid_supports == result.num_reruns_executed == 0
    assert all(row['result'] == 'instruction_audit_error' for row in result.tested)
    assert all(row['error_type'] == 'IntegrationUnavailable' for row in result.tested)
    assert all(row['audit'] == {'raw': 'bad evidence'} for row in result.tested)
    telemetry = W.witness_telemetry({'R': result})
    assert telemetry['instruction_audit_errors'] == len(result.tested)
    assert telemetry['incomplete_supports'] == len(result.tested)
