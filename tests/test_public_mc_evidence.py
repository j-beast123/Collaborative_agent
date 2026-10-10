"""Offline contract tests; synthetic judge answers are not semantic accuracy gold."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from src.common.config import IntegrationUnavailable
from src.online_rl.integrated.records import sha256_text
from src.online_rl.tree.applicability import (condition_evidence_reviewer, condition_items,
    condition_state, state_applicability, validate_condition_evidence, ScoringContractError)
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.cumulative_bank import content_hash
from src.online_rl.tree.pilot_evidence_scorer import (evidence_scorer, PUBLIC_EVIDENCE_CONTRACT,
    anchor_text, validate_public_evidence, public_evidence_fields, public_behavior_items,
    PUBLIC_FIELDS, LEDGER_SCHEMA_REVISION)
from src.online_rl.tree.quality_gate import (public_mc_posthoc_gate, PUBLIC_GATE_PROTOCOL,
    QUALITY_CATEGORIES, PROBE_TYPES, public_gate_state, validate_public_quality)
from src.online_rl.tree.records import to_dialogue
from tests.test_quality_gate_v2 import rubric as legacy_rubric, tin, CASE, REPLY
from tests.test_scoring_batch import node


def rubric():
    return {**legacy_rubric(), 'criterion_version': 'public_behavior_mc_v1',
            'minimum_performance': 'Explain at least one onset timing.',
            'completion_condition': 'Explain both onset timings and explicitly compare them.'}


def ledger(level='2', definition=None):
    definition = definition or rubric()
    rows = []
    for key, met, quotes in (
        ('minimum_performance', level in ('1', '2'), ['Its onset is immediate.']),
        ('completion_condition', level == '2', ['The alternative develops later;', 'their timing differs.'])):
        rows.append({'condition': key, 'definition_quote': definition[key], 'met': met,
                     'evidence_type': 'response_spans' if met else 'absence',
                     'response_quotes': quotes if met else [],
                     'entailment': 'The cited spans perform this operation.' if met else 'This operation is absent.'})
    return {'level': level, 'anchor_quote': anchor_text(definition, level) if level != 'UNSCORABLE' else '',
            'anchor_basis': 'The actual response is assessed against the written criterion.',
            'requirements': rows, 'remaining_gap': '' if level == '2' else 'The required comparison remains absent.',
            'criterion_defect': 'The written anchors do not cover this response.' if level == 'UNSCORABLE' else ''}


def condition_answer(item, state, applicable=True):
    turns = state['dialogue_before_target_reply']
    doctor = next(row for row in turns if row['speaker'] == 'doctor')
    return {'applicable': applicable, 'evidence_basis': 'A visible Doctor request is pending.' if applicable else '',
            'missing_requirement': '' if applicable else 'The Doctor condition is unmet.',
            'doctor_condition': {'condition': item['doctor_condition'],
                'status': 'supported' if applicable else 'unmet',
                'quotes': [{k: doctor[k] for k in ('turn_id', 'speaker')} | {'quote': doctor['text']}],
                'inspected_turn_ids': [row['turn_id'] for row in turns],
                'entailment': 'The earlier request remains available in the prefix.',
                'unmet_condition': '' if applicable else 'The specified request is absent.'},
            'agent_condition': {'condition': item['agent_condition'], 'status': 'not_required',
                'quotes': [], 'inspected_turn_ids': [row['turn_id'] for row in turns],
                'entailment': 'This criterion requires no prior Agent behavior.', 'unmet_condition': ''}}


def quality(definition=None):
    definition = definition or rubric()
    return {'checks': {name: {'passed': True, 'rubric_quotes': [definition['required_behavior']],
                'dialogue_evidence': [{'turn_id': 'source', 'speaker': 'agent', 'quote': 'Its onset is immediate.'}],
                'reason': 'This synthetic test answer reports a supported independent category.'}
            for name in QUALITY_CATEGORIES}, 'score_boundaries_clear': True,
            'boundary_probes': [{'kind': 'synthetic_probe', 'probe_type': name,
                'response': 'Explicitly synthetic diagnostic response ' + name,
                'matching_levels': [level], 'anchor_quotes': [{'level': level, 'quote': anchor_text(definition, level)}],
                'boundary_problem': 'none', 'reason': 'One written level covers this synthetic diagnostic.'}
                for name, level in zip(PROBE_TYPES, ('0', '1', '2', '1', '2', '2'))]}


class Client:
    def __init__(self, *, result=None, app=True, mutate=None, **settings):
        self.model, self.system, self.max_tokens, self.timeout_s = 'fixed-model', 'Neutral judge system.', 4096, 180
        for key, value in settings.items():
            setattr(self, key, value)
        self.protocol_sha256 = sha256_text(self.system)
        self.result, self.app, self.mutate, self.calls, self.log = result or ledger(), app, mutate, [], []

    def judge(self, stage, state, items, fields, purpose):
        call = deepcopy({'stage': stage, 'state': state, 'items': items, 'fields': fields, 'purpose': purpose})
        self.calls.append(call)
        values = {rid: condition_answer(item, state, self.app) if stage == 'applicability'
                  else quality() if stage == PUBLIC_GATE_PROTOCOL else deepcopy(self.result)
                  for rid, item in items.items()}
        if self.mutate:
            self.mutate(stage, values)
        record = {'model': self.model, 'parsed': deepcopy(values), 'raw': [json.dumps(values)]}
        self.log.append(record)
        return values, record

    def judge_batch(self, stage, state, responses, items, fields, purpose):
        self.calls.append(deepcopy({'stage': stage, 'state': state, 'responses': responses, 'items': items}))
        values = {r['response_id']: {rid: deepcopy(self.result) for rid in items} for r in responses}
        return values, {'parsed': deepcopy(values)}


def scorer(client=None, **kwargs):
    return evidence_scorer(client or Client(), evidence_contract=PUBLIC_EVIDENCE_CONTRACT, **kwargs)


@pytest.mark.parametrize('level', ['0', '1', '2', 'UNSCORABLE'])
def test_public_ledger_preserves_numeric_levels_and_unscorable(level):
    row = scorer(Client(result=ledger(level))).score_behavior(CASE, [], [rubric()], REPLY)['candidate-id']
    assert row['hard'] == (None if level == 'UNSCORABLE' else int(level))
    assert row['scoring_status'] == ('rubric_unscorable' if level == 'UNSCORABLE' else 'scored')
    assert row['evidence_validation']['semantic_entailment'] == 'model_assessment_not_independent_proof'


def test_multiple_response_spans_do_not_certify_semantic_truth():
    answer = ledger()
    answer['requirements'][1]['entailment'] = 'An intentionally unverified model interpretation.'
    validate_public_evidence(answer, REPLY, rubric())
    assert len(answer['requirements'][1]['response_quotes']) == 2


@pytest.mark.parametrize('mutation', [
    lambda a: a.update(level='1', anchor_quote=anchor_text(rubric(), '1'), remaining_gap='Missing comparison.'),
    lambda a: a['requirements'][0].update(response_quotes=['Chronologically impossible.']),
    lambda a: a['requirements'][0].update(definition_quote='A hidden new requirement.'),
    lambda a: a['requirements'][1].update(evidence_type='absence', response_quotes=[]),
    lambda a: a.update(remaining_gap='A required behavior remains absent.'),
])
def test_inconsistent_public_ledgers_fail_instead_of_becoming_na(mutation):
    answer = ledger(); mutation(answer)
    with pytest.raises(ValueError):
        validate_public_evidence(answer, REPLY, rubric())


def test_legacy_declaration_one_is_honored_without_inventing_a_new_minimum():
    definition = legacy_rubric()
    definition['scores']['1']['behavior'] = 'Declares that the plan is appropriate.'
    reply = 'The plan is appropriate.'
    answer = {'level': '1', 'anchor_quote': anchor_text(definition, '1'),
        'anchor_basis': 'The legacy anchor explicitly allows the declaration.',
        'requirements': [{'condition': 'written_anchor', 'definition_quote': anchor_text(definition, '1'),
            'met': True, 'evidence_type': 'response_spans', 'response_quotes': [reply],
            'entailment': 'This declaration meets the literal old one-point anchor.'}],
        'remaining_gap': 'The full comparison required by 2 is absent.', 'criterion_defect': ''}
    assert scorer(Client(result=answer)).score_behavior(CASE, [], [definition], reply)['candidate-id']['hard'] == 1


def test_blind_behavior_preserves_public_fields_and_excludes_hidden_or_outcome_data():
    client = Client(); score = scorer(client)
    score.score_behavior(CASE, to_dialogue(tin().dialogue[:3]), [rubric()], REPLY)
    item = client.calls[0]['items']['candidate-id']
    for key in ('required_behavior', 'minimum_performance', 'completion_condition', 'criterion_version'):
        assert item[key] == rubric()[key]
    packet = json.dumps(client.calls)
    for hidden in ('PRIVATE_', 'source_outcome', 'source_score', 'source_provenance', 'expected_level',
                   'boundary_probes', 'applicability'):
        assert hidden not in packet


def test_stage1_first_agent_and_earlier_doctor_request_needs_no_invented_prior_agent():
    client = Client(); review = condition_evidence_reviewer(client)
    prefix = [{'speaker': 'doctor', 'stage': 'D1', 'round': 1, 'text': 'Compare onset timing.'},
              {'speaker': 'doctor', 'stage': 'D2', 'round': 2, 'text': 'Also consider the context.'}]
    assert review(CASE, prefix, [rubric()])['candidate-id']['applicable']
    packet = json.dumps(client.calls)
    for hidden in ('minimum_performance', 'completion_condition', 'score_anchors', 'target_agent_reply', 'PRIVATE_'):
        assert hidden not in packet
    evidence = client.log[0]['parsed']['candidate-id']['doctor_condition']['quotes'][0]
    assert evidence['turn_id'] == 'prefix:0' and evidence['speaker'] == 'doctor'


def test_absence_is_an_inspected_scope_without_fabricated_quotes():
    state = condition_state(CASE, to_dialogue(tin().dialogue[:3])); item = condition_items([rubric()])['candidate-id']
    answer = condition_answer(item, state)
    answer['agent_condition']['status'] = 'absent'
    validate_condition_evidence(answer, state, item)
    answer['agent_condition']['inspected_turn_ids'] = ['prefix:2']
    with pytest.raises(ScoringContractError, match='full inspected prefix'):
        validate_condition_evidence(answer, state, item)


@pytest.mark.parametrize('mutation', [
    lambda a: a['doctor_condition']['quotes'][0].update(speaker='agent'),
    lambda a: a['doctor_condition']['quotes'][0].update(quote='UNOBSERVED_DOCTOR_REQUEST'),
    lambda a: a.update(applicable=False, missing_requirement='Contradictory condition'),
])
def test_bad_applicability_evidence_remains_operational_error(mutation):
    def mutate(stage, values):
        for answer in values.values(): mutation(answer)
    client = Client(mutate=mutate)
    with pytest.raises(IntegrationUnavailable):
        condition_evidence_reviewer(client)(CASE, to_dialogue(tin().dialogue[:1]), [rubric()])
    assert len(client.calls) == 2


def test_na_skips_behavior_and_stays_distinct_from_zero_and_unscorable():
    client = Client(app=False)
    row = scorer(client)(CASE, to_dialogue(tin().dialogue[:1]), [rubric()], REPLY)['candidate-id']
    assert row['applicable'] is False and row['hard'] is None and row['scoring_status'] == 'not_applicable'
    assert [call['stage'] for call in client.calls] == ['applicability']


def test_canonical_app_cache_aliases_bind_full_criterion_policy_state_but_never_send_hidden_belief():
    client = Client(); review = condition_evidence_reviewer(client); cache = CallCache(None)
    n = replace(node(0), prefix=tin().dialogue[:3], trace={'pre_response_belief': {'hidden': 'HIDDEN_BELIEF_CANARY'}})
    first = state_applicability(CASE, n, rubric(), review, cache)
    alias = state_applicability(CASE, replace(n, response='A different response'), {**rubric(), 'rubric_id': 'alias'}, review, cache)
    assert first == alias and len(client.calls) == 1
    variants = [state_applicability(CASE, replace(n, trace={'pre_response_belief': {'changed': True}}), rubric(), review, cache),
                state_applicability(CASE, n, {**rubric(), 'criterion_version': 'revision-2'}, review, cache),
                state_applicability(CASE, replace(n, policy_sha256='another-policy'), rubric(), review, cache)]
    assert len({row['provenance']['cache_key'] for row in [first] + variants}) == 4
    assert 'HIDDEN_BELIEF_CANARY' not in json.dumps(client.calls)


@pytest.mark.parametrize('setting,value', [('model', 'different-model'), ('max_tokens', 3072),
    ('timeout_s', 90), ('system', 'Different system.')])
def test_model_and_request_settings_partition_both_new_protocols(setting, value):
    old, changed = scorer(Client()), scorer(Client(**{setting: value}))
    assert old.protocol != changed.protocol and old.applicability.protocol != changed.applicability.protocol
    assert changed.applicability.review_identity['client']['decoding']['temperature'] == 'not_sent_provider_default'


def test_batch_quote_cannot_be_borrowed_from_another_sibling():
    client = Client(); score = scorer(client, response_batch_size=4)
    with pytest.raises(IntegrationUnavailable):
        score.score_behavior_batch(CASE, [], [rubric()], [{'response_id': 'a', 'reply': REPLY},
            {'response_id': 'b', 'reply': 'I have no comparison.'}])
    assert len(client.calls) == 2


def test_batch_four_keeps_independent_ids_and_validates_every_ledger():
    score = scorer(response_batch_size=4)
    rows = [{'response_id': str(i), 'reply': REPLY} for i in range(4)]
    result = score.score_behavior_batch(CASE, [], [rubric()], rows)
    assert set(result) == set('0123')
    assert all(row['candidate-id']['hard'] == 2 for row in result.values())


@pytest.mark.parametrize('levels,problem', [([], 'gap'), (['0', '1'], 'overlap')])
def test_gate_true_flag_cannot_hide_probe_gap_or_overlap(levels, problem):
    def mutate(stage, values):
        if stage == PUBLIC_GATE_PROTOCOL:
            probe = values['candidate']['boundary_probes'][0]
            probe.update(matching_levels=levels, boundary_problem='none', anchor_quotes=[
                {'level': level, 'quote': anchor_text(rubric(), level)} for level in (levels or ['0'])])
    client = Client(mutate=mutate)
    result = public_mc_posthoc_gate(client, {CASE['case_id']: CASE})(tin(), rubric())
    assert not result['passed'] and not result['score_boundaries_clear']
    assert 'boundary_probe_' + problem in result['reasons']
    assert 'boundary_flag_contradicts_evidence' in result['reasons']
    assert len(client.calls) == 2  # No revote to erase a real contradiction.


def test_gate_failure_source_one_is_preserved_then_rejected_by_unchanged_anchor():
    client = Client(result=ledger('1'))
    result = public_mc_posthoc_gate(client, {CASE['case_id']: CASE})(tin('failure'), rubric())
    assert not result['passed'] and result['source_anchor']['observed_level'] == '1'
    assert result['source_anchor']['expected_level'] == '0'
    assert 'source_anchor_mismatch' in result['reasons']


def test_public_gate_proof_binds_content_and_blind_source_call_is_separate():
    client = Client(); result = public_mc_posthoc_gate(client, {CASE['case_id']: CASE})(tin(), rubric())
    assert result['passed'] and result['observable'] and result['score_boundaries_clear']
    assert result['protocol'] == PUBLIC_GATE_PROTOCOL and result['criterion_sha256'] == content_hash(rubric())
    assert result['public_criterion_version'] == rubric()['criterion_version']
    assert result['evidence']['synthetic_probes_are_independent_gold'] is False
    assert result['evidence']['exact_quotes_are_semantic_proof'] is False
    assert [call['stage'] for call in client.calls] == [PUBLIC_GATE_PROTOCOL, 'level']
    source = json.dumps(client.calls[1])
    for hidden in ('PRIVATE_', 'boundary_probes', 'matching_levels', 'expected_level', 'source_outcome', 'source_score'):
        assert hidden not in source


def test_probe_diagnostics_must_be_explicitly_synthetic():
    answer = quality(); answer['boundary_probes'][0]['kind'] = 'observed_dialogue'
    with pytest.raises(ValueError, match='Synthetic probe kind'):
        validate_public_quality(answer, public_gate_state(tin(), rubric()))


@pytest.mark.parametrize('category', ['unsupported_premise', 'source_claim_strength'])
def test_gate_premise_and_claim_strength_have_distinct_blocking_categories(category):
    def mutate(stage, values):
        if stage == PUBLIC_GATE_PROTOCOL:
            values['candidate']['checks'][category]['passed'] = False
    client = Client(mutate=mutate)
    result = public_mc_posthoc_gate(client, {CASE['case_id']: CASE})(tin(), rubric())
    assert not result['passed'] and result['reasons'] == [category]
    assert result['source_anchor']['matches']
    assert result['assessment']['checks']['source_grounding']['passed']


@pytest.mark.parametrize('definitions,labels', [
    ([legacy_rubric()], ['written_anchor']),
    ([rubric()], ['minimum_performance', 'completion_condition']),
    ([legacy_rubric(), rubric()], ['minimum_performance', 'completion_condition', 'written_anchor']),
])
def test_typed_ledger_schema_is_local_and_preserves_mixed_rubric_validation(definitions, labels):
    original = deepcopy(PUBLIC_FIELDS)
    fields = public_evidence_fields(definitions)
    assert fields['requirements']['items']['properties']['condition']['enum'] == labels
    fields['requirements']['items']['properties']['condition']['enum'].clear()
    assert PUBLIC_FIELDS == original


def test_sent_schema_and_item_kind_enforce_legacy_contract_without_anchor_changes():
    definition = legacy_rubric()
    answer = {'level': '2', 'anchor_quote': anchor_text(definition, '2'),
        'anchor_basis': 'Both timings and the explicit contrast occur in the response.',
        'requirements': [{'condition': 'written_anchor', 'definition_quote': anchor_text(definition, '2'),
            'met': True, 'evidence_type': 'response_spans', 'response_quotes': [REPLY],
            'entailment': 'The full reply supplies the timing contrast.'}],
        'remaining_gap': '', 'criterion_defect': ''}
    client = Client(result=answer)
    row = scorer(client).score_behavior(CASE, [], [definition], REPLY)['candidate-id']
    request = client.calls[0]
    assert request['items']['candidate-id']['evidence_kind'] == 'legacy_written_anchors'
    assert request['items']['candidate-id']['score_anchors'] == {str(i): anchor_text(definition, i) for i in range(3)}
    assert request['fields']['requirements']['items']['properties']['condition']['enum'] == ['written_anchor']
    assert row['evidence_validation']['ledger_schema_revision'] == LEDGER_SCHEMA_REVISION
    assert row['hard'] == 2 and 'minimum_performance' not in request['items']['candidate-id']


def test_public_item_kind_and_schema_use_only_explicit_mc_definitions():
    client = Client(); scorer(client).score_behavior(CASE, [], [rubric()], REPLY)
    request = client.calls[0]
    assert request['items']['candidate-id']['evidence_kind'] == 'public_mc_definitions'
    assert request['fields']['requirements']['items']['properties']['condition']['enum'] == [
        'minimum_performance', 'completion_condition']


def test_typed_ledger_schema_revision_partitions_protocol(monkeypatch):
    import src.online_rl.tree.pilot_evidence_scorer as module
    current = scorer().protocol
    monkeypatch.setattr(module, 'LEDGER_SCHEMA_REVISION', 'another-typed-ledger-revision')
    assert scorer().protocol != current


def test_gate_literal_quote_enum_cannot_emit_field_label_wrappers():
    client = Client(); gate = public_mc_posthoc_gate(client, {CASE['case_id']: CASE},
        rubric_quote_protocol='visible_field_enum_v1')
    assert gate(tin(), rubric())['passed']
    fields = client.calls[0]['fields']
    catalog = fields['checks']['properties']['mc_anchor_alignment']['properties']['rubric_quotes']['items']['enum']
    assert rubric()['minimum_performance'] in catalog
    assert 'Minimum performance: "' + rubric()['minimum_performance'] + '"' not in catalog


def test_public_gate_launch_guard_and_identity_match_actual_adapter():
    from src.online_rl.tree.experiment import load_config
    from scripts.online_rl.run_tree import gate_protocol_problems, evidence_gate_identity, make_gate
    from src.online_rl.tree.luna import LunaDecisionClient
    from pathlib import Path
    cfg = load_config('configs/experiments/quality_public_mc_isolated.json')
    jm = cfg['judges']
    client = LunaDecisionClient(model=jm['gate_model'], key='unused',
        system=Path(jm['gate_prompt_file']).read_text(), max_tokens=jm['gate_max_tokens'])
    assert gate_protocol_problems(cfg) == []
    assert evidence_gate_identity(cfg) == make_gate(cfg, client, []).settings
