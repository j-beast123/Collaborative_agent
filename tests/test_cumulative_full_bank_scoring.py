"""Full-bank applicability without changing legacy retrieval or judge contracts."""
import copy
from dataclasses import replace
import json

import pytest

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import cumulative_bank as CB, cumulative_run
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.member_intervention import MEMBER_INTERVENTION_VERSION
from src.online_rl.tree.scoring import ScoringContractError, score_cumulative_nodes
from src.online_rl.tree.team import team_strategies
from tests.test_cumulative_run import transaction, collector
from tests.test_cumulative_trainer import snapshot, match_any
from tests.test_scoring_batch import CASE, node


class StateScorer:
    model, protocol = 'fixed-judge', 'unchanged-protocol'
    response_batch_size = 4

    def __init__(self, applicable=(), unscorable=False, error=False):
        self.applicable_ids = set(applicable)
        self.unscorable, self.error = unscorable, error
        self.app_calls, self.behavior_calls = [], []

    def review_state(self, view, prefix, rubrics):
        assert 'correct_option' not in view and 'target_agent_reply' not in view
        assert all('agent reply' not in turn['text'] for turn in prefix)
        assert len(rubrics) == 1  # Existing state-applicability singleton contract.
        rid = rubrics[0]['rubric_id']
        self.app_calls.append(rid)
        return {rid: {'applicable': rid in self.applicable_ids}}

    def score_behavior(self, *args):
        pytest.fail('Four sibling responses should use the existing batch scorer')

    def score_behavior_batch(self, view, prefix, rubrics, responses):
        ids = [rubric['rubric_id'] for rubric in rubrics]
        assert set(ids) <= self.applicable_ids
        self.behavior_calls.append((ids, [response['response_id'] for response in responses]))
        if self.error:
            return {}  # An operational error must not silently become NA.
        rows = {}
        for i, response in enumerate(responses):
            rows[response['response_id']] = {
                rid: ({'applicable': True, 'hard': None, 'probabilities': None,
                       'scoring_status': 'rubric_unscorable'} if self.unscorable and i == 0 else
                      {'applicable': True, 'hard': i % 3}) for rid in ids}
        return rows


def nine_rubrics():
    return snapshot(*(f'criterion{i}' for i in range(9)))


def test_full_bank_checks_all_nine_once_per_state_and_scores_only_applicable(monkeypatch):
    bank = nine_rubrics()
    ids = [entry['bank_id'] for entry in bank['entries']]
    scorer = StateScorer(ids[:2])
    monkeypatch.setattr(CB, 'shortlist', lambda *a, **kw: pytest.fail('No scoring shortlist in all mode'))
    rows, metadata = score_cumulative_nodes(CASE, [node(i) for i in range(4)], bank,
        scorer, CallCache(None), top_k=1, scoring_scope='all')
    assert scorer.app_calls == ids
    assert scorer.behavior_calls == [(ids[:2], ['c1/0', 'c1/1', 'c1/2', 'c1/3'])]
    assert len(rows) == 36
    assert sum(row.scoring_status == 'scored' for row in rows) == 8
    assert sum(row.scoring_status == 'not_applicable' for row in rows) == 28
    assert not any(row.scoring_status == 'not_retrieved' for row in rows)
    assert all(row.score is None for row in rows if not row.match)
    assert any(row.score == 0 and row.match for row in rows)
    assert metadata[0]['retrieval_ids'] == ids
    assert metadata[0]['scoring_scope'] == 'all'
    assert metadata[0]['candidate_selection'] == 'full_cumulative_bank'
    assert metadata[0]['top_k'] is None


def test_full_bank_reuses_existing_eight_singleton_judgements_and_behavior_cache():
    bank = nine_rubrics()
    nodes = [node(i) for i in range(4)]
    query = json.dumps({'case': {key: value for key, value in CASE.items() if key != 'correct_option'},
                        'prefix': [{'speaker': 'doctor', 'stage': 'D1', 'round': 1,
                                    'text': 'Please explain your thinking.'}]}, sort_keys=True, ensure_ascii=False)
    selected = [entry['bank_id'] for entry in CB.shortlist(bank, query, top_k=8)]
    # Only one selected criterion applies; the ninth receives a real negative.
    scorer = StateScorer(selected[:1]); cache = CallCache(None)
    old_rows, old_metadata = score_cumulative_nodes(CASE, nodes, bank, scorer, cache)
    assert len(scorer.app_calls) == 8
    assert len(scorer.behavior_calls) == 1
    new_rows, _ = score_cumulative_nodes(CASE, nodes, bank, scorer, cache, scoring_scope='all')
    assert len(scorer.app_calls) == 9 and len(set(scorer.app_calls)) == 9
    assert len(scorer.behavior_calls) == 1
    assert sum(row.scoring_status == 'not_retrieved' for row in old_rows) == 4
    assert not any(row.scoring_status == 'not_retrieved' for row in new_rows)
    assert old_metadata[0]['top_k'] == 8 and 'scoring_scope' not in old_metadata[0]


def test_full_bank_preserves_unscorable_group_mask_and_propagates_scoring_errors():
    bank = nine_rubrics(); rid = bank['entries'][0]['bank_id']
    nodes = [node(i) for i in range(4)]
    rows, _ = score_cumulative_nodes(CASE, nodes, bank, StateScorer([rid], unscorable=True),
        CallCache(None), scoring_scope='all')
    assert all(row.scoring_status == 'group_unscorable' and row.score is None
               for row in rows if row.rubric_id == rid)
    assert sum(row.scoring_status == 'not_applicable' for row in rows) == 32
    with pytest.raises(ScoringContractError, match='IDs differ'):
        score_cumulative_nodes(CASE, nodes, bank, StateScorer([rid], error=True),
            CallCache(None), scoring_scope='all')


def test_empty_full_bank_never_calls_judge_or_fabricates_zero():
    scorer = StateScorer()
    rows, metadata = score_cumulative_nodes(CASE, [node(0)], CB.empty_snapshot(),
        scorer, CallCache(None), scoring_scope='all')
    assert rows == [] and scorer.app_calls == scorer.behavior_calls == []
    assert metadata[0]['bank_size'] == 0 and metadata[0]['retrieval_ids'] == []


@pytest.mark.parametrize('scope', [None, False, 'all_candidates', 0])
def test_invalid_scope_rejected_without_judge_calls(transaction, scope):
    cfg, root, _, _ = transaction
    cfg['rubric']['cumulative_bank']['scoring_scope'] = scope
    with pytest.raises(ValueError, match='scoring_scope'):
        cumulative_run.config_identity(cfg, root)
    scorer = StateScorer()
    with pytest.raises(ScoringContractError, match='scoring_scope'):
        score_cumulative_nodes(CASE, [node(0)], CB.empty_snapshot(), scorer,
            CallCache(None), scoring_scope=scope)
    assert not scorer.app_calls


def test_legacy_identity_is_exactly_unchanged_and_all_mode_is_distinct(transaction):
    cfg, root, _, _ = transaction
    spec = cfg['rubric']['cumulative_bank']
    expected = {'protocol': CB.VERSION, 'retrieval_top_k': 8,
        'matcher_prompt_sha256': spec['matcher_prompt_sha256'], 'initial_snapshot_sha256': None,
        'matcher_model': 'existing-generator', 'matcher_generation': {},
        'reuse_effect_verification': 'historical_witness_no_reverification',
        'batch_scoring': 'freeze_full_bank_after_all_five_verifications',
        'weighting': 'equal_unique_criterion', 'retrieval': 'deterministic_prefix_only_state_shortlist_v1'}
    assert cumulative_run.config_identity(cfg, root) == expected
    legacy_hash = sha256_json(expected)
    spec['scoring_scope'] = 'all'
    changed = cumulative_run.config_identity(cfg, root)
    assert changed['scoring_scope'] == 'all' and changed['retrieval_top_k'] == 8
    assert sha256_json(changed) != legacy_hash
    del spec['scoring_scope']
    assert sha256_json(cumulative_run.config_identity(cfg, root)) == legacy_hash


def test_transaction_rejects_partial_or_legacy_scoring_in_full_bank_mode(transaction):
    cfg, root, out, state = transaction
    cfg['rubric']['cumulative_bank']['scoring_scope'] = 'all'
    with pytest.raises(RuntimeError, match='full cumulative bank'):
        cumulative_run.round_records(cfg, root, out, state, 1, out/'batch', collector([]))


def test_transaction_records_full_scope_and_cannot_reuse_legacy_batch(transaction):
    cfg, root, out, state = transaction
    cumulative_run.round_records(cfg, root, out, state, 1, out/'legacy', collector([]))
    cfg['rubric']['cumulative_bank']['scoring_scope'] = 'all'
    with pytest.raises(RuntimeError, match='Pending batch'):
        cumulative_run.round_records(cfg, root, out, state, 1, out/'legacy',
                                     lambda *args: pytest.fail('Do not change a pending scoring scope'))
    def full_collect(phase, payload):
        records = collector([])(phase, payload)
        if phase == 'bank_score':
            for record in records:
                record['cumulative_bank_scoring']['scoring_scope'] = 'all'
        return records
    cumulative_run.round_records(cfg, root, out, state, 1, out/'full', full_collect)
    assert json.loads((out/'full/bank_input.json').read_text())['scoring_scope'] == 'all'
    assert json.loads((out/'full/bank_metrics.json').read_text())['scoring_scope'] == 'all'
    assert 'scoring_scope' not in json.loads((out/'legacy/bank_metrics.json').read_text())


def test_trainer_full_scoring_keeps_generation_reuse_top_k_separate(tmp_path, monkeypatch):
    from tests.test_witness_trainer import make
    trainer, log, _ = make(tmp_path, 'fixed', cases=1)
    trainer.cumulative_bank = {'protocol': CB.VERSION, 'retrieval_top_k': 1, 'scoring_scope': 'all'}
    trainer.member_intervention_version = MEMBER_INTERVENTION_VERSION
    trainer.env = replace(trainer.env, bank_matcher=match_any)
    trainer.st = team_strategies(None, .5, .5)
    bank = nine_rubrics(); original = copy.deepcopy(bank)
    calls = []
    old_shortlist = CB.shortlist
    def shortlist(snapshot, query, top_k):
        calls.append(top_k)
        return old_shortlist(snapshot, query, top_k)
    monkeypatch.setattr(CB, 'shortlist', shortlist)
    record = trainer.step(score_responses=False, commit=False, apply_update=False,
                          defer_verification=True, cumulative_snapshot=bank)
    assert calls and set(calls) == {1}
    record = trainer.finish_deferred_verification(record)
    assert all(log.counts.get(kind, 0) == 0 for kind in ('rubric', 'assign', 'write', 'rerun'))
    # The fake tree uses different prefix text; its replies still must never enter applicability.
    scorer = StateScorer([entry['bank_id'] for entry in bank['entries']])
    trainer.env = replace(trainer.env, scorer=scorer)
    monkeypatch.setattr(CB, 'shortlist', lambda *args, **kwargs: pytest.fail('Scoring must not shortlist'))
    scored = trainer.score_cumulative(record, bank)
    assert scored['cumulative_bank_scoring']['scoring_scope'] == 'all'
    assert all(row['scoring_scope'] == 'all' and len(row['retrieval_ids']) == 9
               for row in scored['cumulative_retrieval'])
    assert set(scored['rubric_weights'].values()) == {1.0}
    assert not any(row['scoring_status'] == 'not_retrieved' for row in scored['response_scores'])
    assert bank == original and not scored['update']['optimizer_step']
