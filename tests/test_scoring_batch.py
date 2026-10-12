import json
from dataclasses import replace
from types import SimpleNamespace
import pytest

from src.online_rl.tree.backends import two_stage_structured_scorer
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import AgentNode, TrajectoryRubric
from src.online_rl.tree.scoring import ScoringContractError, score_nodes


CASE = {'case_id': 'c1', 'scenario': 's', 'question': 'Q?', 'options': {'A': 'a', 'B': 'b'},
        'correct_option': 'A'}
RUBRIC = {'rubric_id': 'r1', 'name': 'clarity', 'applicability_condition': 'The doctor asks for an explanation.',
          'required_behavior': 'Explain the reasoning.', 'scores': {'0': 'No explanation', '1': 'Partial explanation',
                                                                      '2': 'Clear explanation'}}
KEPT = TrajectoryRubric('r1', 'source', 'prompt', None, RUBRIC, True, (), None)


def node(i, prefix=None):
    return AgentNode(f'c1/{i}', None, 'c1', 1,
                     tuple(prefix or (('doctor', 'D1', 1, 'Please explain your thinking.'),)),
                     f'agent reply {i}', {'pre_response_belief': {'state': {'belief': 'same'}}},
                     'policy-sha', i, None, True, None)


class BatchClient:
    model, protocol_sha256 = 'test-model', 'test-protocol'

    def __init__(self): self.calls = []

    def judge(self, stage, state, items, fields, purpose):
        self.calls.append((stage, state, items, purpose))
        assert stage == 'applicability'
        assert 'target_agent_reply' not in state
        return {rid: {'applicable': True, 'evidence_basis': 'request is present', 'missing_requirement': ''}
                for rid in items}, {}

    def judge_batch(self, stage, state, responses, items, fields, purpose):
        self.calls.append((stage, state, responses, purpose))
        assert stage == 'level'
        assert 'latest_physician_message' not in state  # matches the established Stage-2 input contract
        return {row['response_id']: {rid: {'level': str((int(row['response_id'].split('/')[-1]) % 3)),
                                                   'anchor_basis': row['target_agent_reply']}
                                            for rid in items}
                for row in responses}, {}


def test_same_state_applicability_once_and_behavior_scored_in_batches():
    client = BatchClient()
    scorer = two_stage_structured_scorer(client, response_batch_size=4)
    nodes = [node(i) for i in range(5)]

    cache = CallCache(None)
    scores = score_nodes(CASE, nodes, [KEPT], scorer, cache)

    assert len(client.calls) == 3
    assert [call[0] for call in client.calls] == ['applicability', 'level', 'level']
    assert [len(call[2]) for call in client.calls[1:]] == [4, 1]
    assert [s.hard for s in scores] == [0, 0.5, 1, 0, 0.5]
    assert all(s.match and s.score == s.hard for s in scores)
    behavior_keys = [row['identity']['rubrics'] for row in cache.store.values() if row['kind'] == 'rubric_behavior_batch']
    assert all(isinstance(key, str) and len(key) == 64 for key in behavior_keys)


def test_same_rubric_id_with_changed_content_does_not_reuse_behavior_score_cache():
    client = BatchClient(); scorer = two_stage_structured_scorer(client, response_batch_size=4)
    cache = CallCache(None); one = [node(1)]
    score_nodes(CASE, one, [KEPT], scorer, cache)
    changed = dict(RUBRIC, required_behavior='Use a different specific behavior.')
    changed_kept = TrajectoryRubric('r1', 'source', 'prompt', None, changed, True, (), None)
    score_nodes(CASE, one, [changed_kept], scorer, cache)
    records = [r for r in cache.store.values() if r['kind'] == 'rubric_behavior_batch']
    assert len(records) == 2
    assert records[0]['identity']['rubrics'] != records[1]['identity']['rubrics']


def test_different_pre_reply_states_do_not_share_applicability():
    client = BatchClient()
    scorer = two_stage_structured_scorer(client, response_batch_size=4)
    nodes = [node(0), node(1, prefix=(('doctor', 'D1', 1, 'A different request.'),))]

    score_nodes(CASE, nodes, [KEPT], scorer, CallCache(None))

    assert [call[0] for call in client.calls] == ['applicability', 'level', 'applicability', 'level']


def test_parallel_states_overlap_without_review_barrier_or_duplicate_calls(monkeypatch, tmp_path):
    from contextlib import contextmanager
    import threading
    from src.common import usage

    ledger = tmp_path / 'usage.jsonl'
    monkeypatch.setenv(usage.ENV, str(ledger))

    class OverlappingClient(BatchClient):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.active = self.peak = 0
            self.second_review_started = threading.Event()
            self.first_behavior_done = threading.Event()

        @contextmanager
        def request(self):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                yield
            finally:
                with self.lock:
                    self.active -= 1
                usage.record('scoring-test', self.model, {})

        def judge(self, stage, state, items, fields, purpose):
            with self.request():
                text = state['dialogue_before_target_reply'][0]['text']
                if text == 'state 0':
                    assert self.second_review_started.wait(5), 'Separate states must review concurrently'
                elif text == 'state 1':
                    self.second_review_started.set()
                    assert self.first_behavior_done.wait(5), 'Ready state must score before all reviews finish'
                return super().judge(stage, state, items, fields, purpose)

        def judge_batch(self, stage, state, responses, items, fields, purpose):
            with self.request():
                result = super().judge_batch(stage, state, responses, items, fields, purpose)
                if state['dialogue_before_target_reply'][0]['text'] == 'state 0':
                    self.first_behavior_done.set()
                return result

    # Interleave sibling IDs to check that output order follows nodes, not tasks.
    nodes = [node(4 * state + sibling, prefix=(('doctor', 'D1', 1, f'state {state}'),))
             for sibling in range(4) for state in range(3)]
    client = OverlappingClient()
    scorer = two_stage_structured_scorer(client, response_batch_size=4)
    cache = CallCache(None)
    with usage.case(CASE['case_id']):
        parallel = score_nodes(CASE, nodes, [KEPT], scorer, cache, workers=2)
    assert client.peak == 2
    assert len(client.calls) == 6
    assert cache.calls == {'rubric_state_applicability': 3, 'rubric_behavior_batch': 3}
    assert [s.response_node_id for s in parallel] == [n.node_id for n in nodes]
    assert all(len(call[2]) == 4 for call in client.calls if call[0] == 'level')
    ledger_rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(ledger_rows) == 6 and all(row['case_id'] == CASE['case_id'] for row in ledger_rows)

    serial_cache = CallCache(None)
    serial = score_nodes(CASE, nodes, [KEPT],
                         two_stage_structured_scorer(BatchClient(), response_batch_size=4), serial_cache, workers=1)
    assert parallel == serial
    assert cache.store == serial_cache.store  # Scheduling never changes request/cache identities.
    assert score_nodes(CASE, nodes, [KEPT], scorer, cache, workers=3) == parallel
    assert len(client.calls) == 6


def test_luna_batch_judge_preserves_response_and_rubric_ids():
    from src.online_rl.tree.luna import LunaDecisionClient

    sent = []

    def post(url, payload, key, timeout):
        sent.append(payload)
        packet = json.loads(payload['messages'][1]['content'])
        answers = {row['response_id']: {rid: {'level': '2', 'anchor_basis': 'reply-specific evidence'}
                                       for rid in packet['rubric_items']} for row in packet['responses']}
        return {'model': 'test-model', 'usage': {'prompt_tokens': 12, 'completion_tokens': 5},
                'choices': [{'message': {'content': json.dumps({'answers': answers})}}]}

    client = LunaDecisionClient(model='test-model', key='k', system='Score independently.', post=post)
    response_rows = [
        {'response_id': 'node-a', 'target_agent_reply': 'first answer'},
        {'response_id': 'node-b', 'target_agent_reply': 'second answer'},
    ]
    items = {'r1': {'rubric': 'clarity', 'required_behavior': 'explain',
                    'score_anchors': {'0': 'x', '1': 'y', '2': 'z'}}}
    fields = {'level': {'type': 'string', 'enum': ['0', '1', '2']}, 'anchor_basis': {'type': 'string'}}

    parsed, record = client.judge_batch('level', {'dialogue_before_target_reply': []}, response_rows, items, fields, 'test')

    assert set(parsed) == {'node-a', 'node-b'}
    assert parsed['node-a']['r1']['level'] == parsed['node-b']['r1']['level'] == '2'
    assert record['repaired'] is False and len(sent) == 1
    request = json.loads(sent[0]['messages'][1]['content'])
    assert len(request['responses']) == 2
    assert request['shared_pre_reply_state'] == {'dialogue_before_target_reply': []}
    assert request['rubric_items'] == items
    assert all('items' not in row for row in request['responses'])


def contract_scorer(behavior, applicable=True):
    return SimpleNamespace(
        review_state=lambda view, prefix, rubrics: {
            r['rubric_id']: {'applicable': applicable} for r in rubrics},
        score_behavior=lambda *args: pytest.fail('batch scorer expected'),
        score_behavior_batch=behavior, response_batch_size=4,
        model='contract-test', protocol='contract-test-v1')


def test_fused_state_applicability_and_behavior_use_one_call_per_state():
    nodes = [node(i) for i in range(4)]
    kept = [replace(KEPT, rubric_id=f'r{i}', rubric={**RUBRIC, 'rubric_id': f'r{i}'})
            for i in range(13)]
    calls = {'app': 0, 'behavior': 0}

    def review(view, prefix, rubrics):
        calls['app'] += 1
        return {r['rubric_id']: {'applicable': True} for r in rubrics}

    def behavior(view, prefix, rubrics, responses):
        calls['behavior'] += 1
        return {row['response_id']: {
            r['rubric_id']: {'hard': 2, 'scoring_status': 'scored'} for r in rubrics
        } for row in responses}

    scorer = contract_scorer(behavior)
    scorer.review_state_batch = review
    scorer.state_batch_contract = 'test-fused-v1'
    cache = CallCache(None)
    scores = score_nodes(CASE, nodes, kept, scorer, cache)
    assert len(scores) == 52 and all(row.score == 1.0 for row in scores)
    assert calls == {'app': 1, 'behavior': 1}
    assert cache.calls == {
        'rubric_state_applicability_bundle': 1,
        'rubric_behavior_batch': 1,
    }


@pytest.mark.parametrize('bad', [
    {},
    {'c1/0': {}},
    {'c1/0': {'r1': {'hard': None, 'applicable': True}}},
    {'c1/0': {'r1': {'hard': 0, 'error': 'judge failed'}}},
    {'c1/0': {'r1': {'hard': 0, 'applicable': False}}},
    {'c1/0': {'r1': {'hard': True}}},
    {'c1/0': {'r1': {'hard': 0.5}}},
    {'c1/0': {'r1': {'hard': 2, 'scoring_status': 'rubric_unscorable'}}},
    {'c1/0': {'r1': {'hard': 2}}, 'unrequested-response': {}},
    [('c1/0', {'r1': {'hard': 2}}), ('c1/0', {'r1': {'hard': 0}})],
])
def test_incomplete_or_invalid_behavior_never_becomes_na_or_enters_cache(bad):
    cache = CallCache(None)
    scorer = contract_scorer(lambda *args: bad)
    failed = score_nodes(CASE, [node(0)], [KEPT], scorer, cache)
    assert len(failed) == 1
    assert failed[0].scoring_status == 'behavior_error'
    assert failed[0].score is None and failed[0].match is True
    assert not any(r['kind'] == 'rubric_behavior_batch' for r in cache.store.values())
    # A retry after a transient malformed response must actually call the scorer.
    scorer.score_behavior_batch = lambda *args: {'c1/0': {'r1': {'hard': 0}}}
    assert score_nodes(CASE, [node(0)], [KEPT], scorer, cache)[0].score == 0.0


def test_parallel_state_error_does_not_emit_partial_scores_or_cache_bad_rows():
    nodes = [node(0), node(1, prefix=(('doctor', 'D1', 1, 'Another state.'),))]
    cache = CallCache(None)
    calls = []
    malformed = True

    def behavior(view, prefix, rubrics, responses):
        ids = tuple(row['response_id'] for row in responses)
        calls.append(ids)
        if malformed and ids == ('c1/0',):
            return {}
        return {rid: {'r1': {'hard': 2}} for rid in ids}

    scorer = contract_scorer(behavior)
    first = score_nodes(CASE, nodes, [KEPT], scorer, cache, workers=2)
    assert sorted((s.response_node_id, s.scoring_status) for s in first) == [
        ('c1/0', 'behavior_error'), ('c1/1', 'scored')]
    assert cache.calls == {'rubric_state_applicability': 2, 'rubric_behavior_batch': 1}
    malformed = False
    scores = score_nodes(CASE, nodes, [KEPT], scorer, cache, workers=2)
    assert [s.score for s in scores] == [1.0, 1.0]
    assert calls.count(('c1/0',)) == 2 and calls.count(('c1/1',)) == 1


@pytest.mark.parametrize('duplicate', ['node', 'rubric'])
def test_duplicate_input_ids_are_rejected_before_any_call(duplicate):
    scorer = contract_scorer(lambda *args: pytest.fail('must validate IDs before scoring'))
    cache = CallCache(None)
    with pytest.raises(ScoringContractError, match='Duplicate'):
        score_nodes(CASE, [node(0)] * (2 if duplicate == 'node' else 1),
                    [KEPT] * (2 if duplicate == 'rubric' else 1), scorer, cache)
    assert cache.store == {}


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('workers', [1, 3])
def test_unscorable_masks_only_that_rubric_for_the_entire_shared_state(batch_size, workers):
    from src.online_rl.tree.team import effectiveness_weighted_r_a
    other_rubric = replace(KEPT, rubric_id='r2', rubric={**RUBRIC, 'rubric_id': 'r2'})
    nodes = [node(0), node(1), node(2, prefix=(('doctor', 'D1', 1, 'Another state.'),))]
    def behavior(view, prefix, rubrics, responses):
        values = {}
        for row in responses:
            values[row['response_id']] = {
                rid: {'hard': 0 if rid == 'r2' else 2, 'scoring_status': 'scored'}
                for rid in ('r1', 'r2')}
            if row['response_id'] == 'c1/0':
                values[row['response_id']]['r1'] = {
                    'hard': None, 'scoring_status': 'rubric_unscorable',
                    'stage2': {'level': 'UNSCORABLE', 'anchor_basis': 'The anchors omit this situation.'}}
        return values
    scorer = contract_scorer(behavior)
    scorer.response_batch_size = batch_size
    scores = score_nodes(CASE, nodes, [KEPT, other_rubric], scorer, CallCache(None), workers=workers)
    by_key = {(s.response_node_id, s.rubric_id): s for s in scores}
    for nid in ('c1/0', 'c1/1'):
        masked = by_key[nid, 'r1']
        assert masked.match is True and masked.scoring_status == 'group_unscorable'
        assert masked.score is masked.hard is masked.probabilities is None
        assert effectiveness_weighted_r_a([masked], {'r1': 1.0}) is None
        assert by_key[nid, 'r2'].score == 0.0
        assert effectiveness_weighted_r_a([masked, by_key[nid, 'r2']], {'r1': 1.0, 'r2': 1.0}) == 0.0
    assert by_key['c1/0', 'r1'].individual_assessment['scoring_status'] == 'rubric_unscorable'
    assert by_key['c1/1', 'r1'].individual_assessment['score'] == 1.0
    assert by_key['c1/0', 'r1'].individual_assessment['evidence']['stage2']['level'] == 'UNSCORABLE'
    assert by_key['c1/2', 'r1'].scoring_status == 'scored' and by_key['c1/2', 'r1'].score == 1.0


def test_true_na_and_zero_stay_distinct_and_old_records_load():
    from src.online_rl.integrated.records import from_dict, to_dict
    from src.online_rl.tree.records import ResponseRubricScore
    zero = score_nodes(CASE, [node(0)], [KEPT],
        contract_scorer(lambda *args: {'c1/0': {'r1': {'hard': 0}}}), CallCache(None))[0]
    na = score_nodes(CASE, [node(0)], [KEPT],
        contract_scorer(lambda *args: pytest.fail('NA must not request behavior scoring'), applicable=False), CallCache(None))[0]
    assert (zero.match, zero.scoring_status, zero.score) == (True, 'scored', 0.0)
    assert (na.match, na.scoring_status, na.score) == (False, 'not_applicable', None)
    old = to_dict(zero)
    for field in ('scoring_status', 'scoring_evidence', 'individual_assessment'): old.pop(field)
    assert from_dict(ResponseRubricScore, old).scoring_status == 'scored'


def test_legacy_scorer_missing_row_is_an_error_and_not_cached():
    cache = CallCache(None)
    with pytest.raises(ScoringContractError, match='IDs differ'):
        score_nodes(CASE, [node(0)], [KEPT], lambda *args: {}, cache)
    assert cache.store == {}
