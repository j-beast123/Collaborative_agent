"""Cumulative bank wiring: CPU-only reuse, shared-state scoring and phase barriers."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.online_rl.integrated.records import sha256_json, sha256_text, to_dict
from src.online_rl.tree import cumulative_bank as CB
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.member_intervention import MEMBER_INTERVENTION_VERSION
from src.online_rl.tree.scoring import score_cumulative_nodes
from src.online_rl.tree.team import team_strategies
from tests.test_scoring_batch import CASE, BatchClient, node
from src.online_rl.tree.backends import two_stage_structured_scorer


def source_record(label='explain'):
    rid = 'source_' + label
    app = {'user_side_context': 'The doctor requests an explanation.',
           'agent_side_state': 'No explanation has yet been supplied.'}
    rubric = {'rubric_id': rid, 'name': label, 'applicability': app,
              'applicability_condition': 'The doctor requests an explanation.',
              'required_behavior': f'{label} the requested relation.',
              'scores': {level: {**app, 'behavior': text} for level, text in
                         [('0', 'No contribution'), ('1', 'Partial explanation with missing relation'),
                          ('2', f'Complete {label} of the relation')]},
              'source_provenance': {'source_case_id': 'source', 'source_trajectory_id': 'source/leaf',
                                    'source_outcome': 'success'}}
    raw = {'rubric_id': rid, 'leaf_id': 'source/leaf', 'prompt_sha256': 'p0', 'raw': '{}',
           'rubric': rubric, 'valid': True, 'problems': [],
           'gate': {'protocol': 'evidence_gate_v2', 'passed': True, 'observable': True, 'score_boundaries_clear': True,
                    'source_anchor': {'matches': True}},
           'source_check': {'status': 'PASS', 'first': {'applicable': True}},
           'intervention_instruction': 'Omit the requested explanation.'}
    runs = [{'outcome': 0.0, 'rubric_id': rid, 'leaf_id': 'source/leaf',
             'continuation_policy_sha256': 'source-policy', 'level': '0',
             'seed': seed, 'agent_seed': seed, 'rerun_index': i} for i, seed in enumerate((10, 11))]
    witness = {'decision': 'KEEP', 'rubric_id': rid, 'stratum': 'success',
               'validated_by_trajectory_id': 'source/leaf',
               'tested': [{'leaf_id': 'source/leaf', 'result': 'PASS', 'stage1_applicable': True,
                           'outcomes': [0.0, 0.0], 'rerun_records': runs, 'reruns': 2,
                           'intervention_audit': {'protocol': 'intervention_instruction_audit_v1', 'passed': True,
                               'answer': {'instruction_aligned': True, 'unsupported_context_premise': False,
                                          'conclusion_prescribed': False}}}]}
    return {'case_id': 'source', 'case_cursor': 0, 'policy_sha256': 'source-policy', 'prompt_sha256': 'p0',
            'trajectory_rubrics': [raw], 'selection': [{'rubric_id': rid, 'retained': True}],
            'rubric_bank': {'witness': [witness]}}


def snapshot(*labels):
    return CB.extend_snapshot(CB.empty_snapshot(), [source_record(label) for label in (labels or ('explain',))])


def match_any(payload):
    agent = next(row for row in payload['dialogue'] if row['speaker'] == 'agent')
    doctor = next(row for row in payload['dialogue'] if row['speaker'] == 'doctor')
    return {'reuse': True, 'bank_id': payload['candidates'][0]['bank_id'], 'selected_agent_turn_id': 1,
            'doctor_quote': doctor['text'], 'agent_quote': agent['text'],
            'checks': {key: True for key in CB.CHECKS}, 'rationale': 'Equivalent observed criterion and boundaries.'}


def test_four_siblings_share_retrieval_and_applicability_without_response_leakage(monkeypatch):
    bank = snapshot('explain', 'compare')
    client = BatchClient(); scorer = two_stage_structured_scorer(client, response_batch_size=4, observed_context=True)
    calls = []
    original = CB.shortlist
    def shortlist(bank, query, top_k):
        assert 'agent reply' not in query and 'correct_option' not in query
        calls.append(query)
        return original(bank, query, top_k)
    monkeypatch.setattr(CB, 'shortlist', shortlist)
    rows, retrieval = score_cumulative_nodes(CASE, [node(i) for i in range(4)], bank, scorer,
                                            CallCache(None), top_k=1)
    assert len(calls) == 1 and len(retrieval) == 1
    assert [call[0] for call in client.calls] == ['applicability', 'level']
    assert len(rows) == 8
    omitted = [row for row in rows if row.scoring_status == 'not_retrieved']
    assert len(omitted) == 4 and all(row.score is None and not row.match for row in omitted)
    assert any(row.score == 0 and row.match for row in rows)
    assert len(retrieval[0]['retrieval_content_sha256']) == 1
    assert retrieval[0]['snapshot_sha256'] == bank['sha256']
    # The alias is only scorer-facing; original source identities/guides survive.
    assert bank['entries'][0]['rubric']['leaf_id'] == 'source/leaf'
    assert CB.scoring_rubrics(bank)[0].leaf_id == 'source/leaf'


def test_cumulative_reuse_skips_every_generation_gate_and_rerun(tmp_path):
    from tests.test_witness_trainer import make
    trainer, log, _ = make(tmp_path, 'fixed', cases=1)
    trainer.cumulative_bank = {'protocol': CB.VERSION, 'retrieval_top_k': 8}
    trainer.member_intervention_version = MEMBER_INTERVENTION_VERSION
    trainer.env = replace(trainer.env, bank_matcher=match_any)
    trainer.st = team_strategies(None, 0.5, 0.5)
    bank = snapshot()
    original = copy.deepcopy(bank)
    prepared = trainer.step(score_responses=False, commit=False, apply_update=False,
                            defer_verification=True, cumulative_snapshot=bank)
    assert prepared['cumulative_reused_trajectories'] == len(prepared['trajectory_inputs']) > 0
    assert not prepared['trajectory_rubrics'] and not prepared['rubric_bank']['assignments']
    assert prepared['feedback'] and prepared['trajectory_inputs']
    trainer.validate_member_interventions(prepared)
    verified = trainer.finish_deferred_verification(prepared)
    assert verified['rubric_bank']['witness'] == []
    assert all(log.counts.get(kind, 0) == 0 for kind in ('rubric', 'assign', 'write', 'rerun'))
    assert bank == original
    # Cumulative reuse adds no fake source record or regenerated member guide.
    assert all(row['sourceproof'][0]['policy_sha256'] == 'source-policy'
               for row in verified['cumulative_reuse'])
    scorer = two_stage_structured_scorer(BatchClient(), response_batch_size=4, observed_context=True)
    # Test-tree IDs are nonnumeric; use an independent structured scoring fixture.
    def behavior(view, prefix, rubrics, responses):
        return {response['response_id']: {r['rubric_id']: {'applicable': True, 'hard': 2}
                                          for r in rubrics} for response in responses}
    scorer.score_behavior_batch = behavior
    trainer.env = replace(trainer.env, scorer=scorer)
    scored = trainer.score_cumulative(verified, bank)
    assert scored['response_scores'] and all(value == 1 for value in scored['rubric_weights'].values())
    assert scored['cumulative_bank_scoring']['snapshot_sha256'] == bank['sha256']
    assert not scored['update']['optimizer_step']
    assert trainer.state.case_cursor == 0


@pytest.mark.parametrize('corruption', ['snapshot', 'missing_reuse', 'invented_quote', 'duplicate_reuse'])
def test_saved_reuse_cannot_bypass_member_coverage(tmp_path, corruption):
    from tests.test_witness_trainer import make
    trainer, _, _ = make(tmp_path, 'fixed', cases=1)
    trainer.cumulative_bank = {'protocol': CB.VERSION, 'retrieval_top_k': 8}
    trainer.member_intervention_version = MEMBER_INTERVENTION_VERSION
    trainer.env = replace(trainer.env, bank_matcher=match_any)
    record = trainer.step(score_responses=False, commit=False, apply_update=False,
                          defer_verification=True, cumulative_snapshot=snapshot())
    if corruption == 'snapshot': record['cumulative_bank_input']['sha256'] = 'changed'
    elif corruption == 'missing_reuse': record['cumulative_reuse'].pop()
    elif corruption == 'invented_quote': record['cumulative_reuse'][0]['doctor_quote'] = 'This was never said.'
    else: record['cumulative_reuse'].append(record['cumulative_reuse'][0])
    with pytest.raises((ValueError, RuntimeError)):
        trainer.finish_deferred_verification(record)


def test_worker_phases_freeze_same_scoring_bank_for_five_cases(tmp_path, monkeypatch):
    from src.online_rl.tree import five_gpu_worker as worker, batch_prompt
    from scripts.online_rl import run_tree
    events = []
    initial = CB.empty_snapshot(); frozen = snapshot()
    class Trainer:
        state = SimpleNamespace(policy_sha256='policy')
        def __init__(self, case): self.case = case
        def validate_member_interventions(self, record): pass
        def step(self, **kwargs):
            assert kwargs['defer_verification'] and not kwargs['score_responses'] and not kwargs['apply_update']
            assert kwargs['cumulative_snapshot'] == initial
            events.append(('prepare', self.case['case_id']))
            return {'case_id': self.case['case_id'], 'policy_sha256': 'policy', 'prompt_sha256': sha256_text('P0'),
                    'cumulative_bank_input': initial, 'rubric_bank': {'verification_pending': True}}
        def finish_deferred_verification(self, record):
            events.append(('verify', self.case['case_id']))
            record['rubric_bank']['verification_pending'] = False
            return record
        def score_cumulative(self, record, bank):
            assert bank == frozen
            events.append(('score', self.case['case_id']))
            return {**record, 'cumulative_bank_scoring': {'snapshot': bank, 'snapshot_sha256': bank['sha256'], 'retrieval': []}}
    monkeypatch.setattr(worker, '_CFG', {'schedule': {'cases': 120}})
    monkeypatch.setattr(worker, '_POLICY', SimpleNamespace())
    monkeypatch.setattr(run_tree, 'build', lambda cfg, **kwargs: Trainer(kwargs['cases'][0]))
    monkeypatch.setattr(batch_prompt, 'evaluate_prompt_record', lambda *a: pytest.fail('Prompt evaluation in fixed bank pipeline'))
    jobs = [{'index': i, 'worker': i, 'lane': 0, 'case': {'case_id': f'case{i}'}, 'out': str(tmp_path / str(i))} for i in range(5)]
    for job in jobs: __import__('pathlib').Path(job['out']).mkdir()
    prompt = {'id': 'P0', 'text': 'P0', 'sha256': sha256_text('P0'), 'snapshot': initial}
    for phase in ('bank_prepare', 'bank_verify', 'bank_score'):
        current = {**prompt, 'snapshot': frozen if phase == 'bank_score' else initial}
        records = [worker.run_case(job, phase, current) for job in jobs]
    assert [event[0] for event in events] == ['prepare'] * 5 + ['verify'] * 5 + ['score'] * 5
    assert {r['cumulative_bank_scoring']['snapshot_sha256'] for r in records} == {frozen['sha256']}
    for job in jobs: worker.run_case(job, 'bank_score', {**prompt, 'snapshot': frozen})
    assert len(events) == 15
    with pytest.raises(RuntimeError, match='snapshot changed'):
        worker.run_case(jobs[0], 'bank_score', {**prompt, 'snapshot': initial})
