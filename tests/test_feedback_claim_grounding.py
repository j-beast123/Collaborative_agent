"""Grounded reflection audits are opt-in and never rewrite the Doctor record."""
import copy
import json
from dataclasses import replace

import pytest

from src.online_rl.tree import backends as B
from src.online_rl.tree.feedback_grounding import PROTOCOL, REFLECTION_KIND, extend_schema
from rubric_test_fixtures import _output, _tin


def fixture():
    reflection = {
        'helpful': [{'agent_reply_id': 1, 'experience': 'Reply 1 explained the later onset.'}],
        'unresolved': [{'agent_reply_id': 2, 'experience': 'I also wanted severity and urgency distinguished.'}],
    }
    tin = replace(_tin(), feedback_kind=REFLECTION_KIND, feedback=json.dumps(reflection),
        dialogue=(('doctor', 'DISCUSS', 1, 'Please compare these two considerations.'),
                  ('agent', 'DISCUSS', 1, 'I will compare the stated considerations.'),
                  ('doctor', 'DISCUSS', 2, 'Was any blood in the airway?'),
                  ('agent', 'DISCUSS', 2, 'The later onset distinguishes the alternative.'),
                  ('doctor', 'DISCUSS', 3, 'Now I would also distinguish urgency.'),
                  ('doctor', 'FINAL', 3, 'My decision is A.')))
    obj = _output(tin)
    obj['intervention_instruction'] = 'Compare the stated considerations.'
    obj['analysis']['feedback_check'] = {
        'claim': 'The retrospective helpful claim is misattributed; the extra request is not earlier context.',
        'assessment': 'partly_supported', 'dialogue_basis': 'Agent reply 2 supplied the claimed explanation.',
        'claim_reviews': [
            {'feedback_kind': 'helpful', 'feedback_index': 0, 'reported_agent_turn_id': 1,
             'claim': reflection['helpful'][0]['experience'], 'decision': 'partly_accept',
             'accepted_part': 'The later onset explanation is present in Agent reply 2.',
             'rejected_part': 'The attribution to Agent reply 1 is incorrect.',
             'reason': 'The exact explanation occurs in Agent reply 2, not Agent reply 1.',
             'actual_agent_turn_id': 2,
             'agent_evidence': [{'speaker': 'agent', 'turn_id': 2, 'quote': 'The later onset distinguishes the alternative.'}],
             'prior_context': [{'speaker': 'doctor', 'turn_id': 2, 'quote': 'Was any blood in the airway?'}],
             'retrospective_only': False},
            {'feedback_kind': 'unresolved', 'feedback_index': 0, 'reported_agent_turn_id': 2,
             'claim': reflection['unresolved'][0]['experience'], 'decision': 'reject', 'accepted_part': None,
             'rejected_part': 'No earlier severity or urgency request is available.',
             'reason': 'A later desire cannot establish a requirement before Agent reply 2.',
             'actual_agent_turn_id': None, 'agent_evidence': [], 'prior_context': [], 'retrospective_only': True},
        ]}
    return tin, obj


def normalize(obj, tin, **kw):
    return B.normalize_trajectory_analysis_output(obj, None, tin, require_intervention=True,
        feedback_check_protocol=PROTOCOL, **kw)


def reviews(obj):
    return obj['analysis']['feedback_check']['claim_reviews']


def test_corrected_attribution_preserves_original_claim_and_actual_evidence():
    tin, obj = fixture()
    before = copy.deepcopy(obj)
    normalized, err = normalize(obj, tin)
    assert err is None
    assert obj == before
    provenance = normalized['source_provenance']
    assert provenance['feedback_check_protocol'] == PROTOCOL
    for view in ('analysis', 'model_reported_analysis'):
        assert provenance[view]['feedback_check'] == obj['analysis']['feedback_check']
    assert reviews(obj)[0]['reported_agent_turn_id'] == 1
    assert reviews(obj)[0]['actual_agent_turn_id'] == 2


def test_corrected_attribution_cannot_be_fully_accepted():
    tin, obj = fixture()
    reviews(obj)[0].update(decision='accept', rejected_part=None)
    normalized, err = normalize(obj, tin)
    assert normalized is None and 'changed Agent attribution cannot be fully accepted' in err
    reviews(obj)[0].update(decision='partly_accept', rejected_part='The original Agent reply ID is wrong.')
    assert normalize(obj, tin)[1] is None
    reviews(obj)[0].update(decision='reject', accepted_part=None,
                          rejected_part='The attributed contribution is unsupported.')
    assert normalize(obj, tin)[1] is None


@pytest.mark.parametrize('change, message', [
    (lambda r: r.update(actual_agent_turn_id=1), 'same actual Agent reply'),
    (lambda r: r['agent_evidence'][0].update(turn_id=1), 'quote exactly'),
    (lambda r: r.update(reported_agent_turn_id=2), 'original reflection attribution'),
    (lambda r: r.update(claim='Corrected claim substituted for the Doctor wording.'), 'exactly preserve'),
    (lambda r: r['agent_evidence'][0].update(quote='A fabricated explanation.'), 'quote exactly'),
])
def test_swapped_attribution_or_rewritten_source_is_not_silently_accepted(change, message):
    tin, obj = fixture()
    change(reviews(obj)[0])
    normalized, err = normalize(obj, tin)
    assert normalized is None and message in err


@pytest.mark.parametrize('span, message', [
    ({'speaker': 'doctor', 'turn_id': 3, 'quote': 'Now I would also distinguish urgency.'}, 'strictly before'),
    ({'speaker': 'doctor', 'turn_id': 4, 'quote': 'My decision is A.'}, 'non-FINAL'),
    ({'speaker': 'agent', 'turn_id': 2, 'quote': 'The later onset distinguishes the alternative.'}, 'strictly before'),
])
def test_future_current_and_final_context_cannot_ground_prior_need(span, message):
    tin, obj = fixture()
    reviews(obj)[0]['prior_context'] = [span]
    normalized, err = normalize(obj, tin)
    assert normalized is None and message in err


def test_earlier_doctor_id_equal_to_agent_id_is_allowed_by_actual_chronology():
    tin, obj = fixture()
    # Doctor 2 occurs before Agent 2; speaker-local number comparisons are wrong.
    assert normalize(obj, tin)[1] is None


@pytest.mark.parametrize('mutation, message', [
    (lambda rs: rs.pop(), 'cover every reflection claim'),
    (lambda rs: rs.append(copy.deepcopy(rs[0])), 'duplicates'),
    (lambda rs: rs[0].update(feedback_index=1), 'supplied reflection claim'),
    (lambda rs: rs[0].update(feedback_index=True), 'integer feedback_index'),
    (lambda rs: rs[0].update(extra='unsupported'), 'exactly'),
    (lambda rs: rs[0].update(actual_agent_turn_id=True), 'existing non-FINAL'),
    (lambda rs: rs[0].update(reason='  '), 'substantive'),
    (lambda rs: rs[0].update(agent_evidence=[]), 'exact Agent evidence'),
])
def test_complete_typed_claim_coverage_is_mandatory(mutation, message):
    tin, obj = fixture()
    mutation(reviews(obj))
    normalized, err = normalize(obj, tin)
    assert normalized is None and message in err


def test_unresolved_acceptance_requires_prior_observed_context():
    tin, obj = fixture()
    unresolved = reviews(obj)[1]
    unresolved.update(decision='accept', accepted_part='An observed prior question remained unresolved.',
        rejected_part=None, retrospective_only=False, actual_agent_turn_id=2,
        agent_evidence=copy.deepcopy(reviews(obj)[0]['agent_evidence']))
    assert 'observed prior context' in normalize(obj, tin)[1]
    unresolved['prior_context'] = copy.deepcopy(reviews(obj)[0]['prior_context'])
    assert normalize(obj, tin)[1] is None
    # Structural validation cannot determine whether this broad interpretation
    # is semantically warranted; that remains explicitly a generator judgment.


@pytest.mark.parametrize('decision, accepted, rejected', [
    ('accept', 'Accepted', 'Rejected'), ('reject', 'Accepted', 'Rejected'),
    ('partly_accept', 'Accepted', None), ('partly_accept', None, 'Rejected'),
])
def test_decision_parts_must_be_consistent(decision, accepted, rejected):
    tin, obj = fixture()
    reviews(obj)[0].update(decision=decision, accepted_part=accepted, rejected_part=rejected)
    assert 'inconsistent with decision' in normalize(obj, tin)[1]


def test_retrospective_need_cannot_be_accepted_and_rejected_evidence_still_checked():
    tin, obj = fixture()
    reviews(obj)[0]['retrospective_only'] = True
    assert 'retrospective-only' in normalize(obj, tin)[1]
    tin, obj = fixture()
    reviews(obj)[1]['agent_evidence'] = [{'speaker': 'agent', 'turn_id': 2, 'quote': 'invented'}]
    assert 'temporal scope' in normalize(obj, tin)[1]
    reviews(obj)[1]['actual_agent_turn_id'] = 2
    assert 'quote exactly' in normalize(obj, tin)[1]


def test_empty_reflection_and_legacy_contract_remain_distinct():
    tin, obj = fixture()
    obj['analysis']['feedback_check'].update(claim=None, dialogue_basis=None,
        assessment='not_provided', claim_reviews=[])
    assert 'not_provided' in normalize(obj, tin)[1]
    tin = replace(tin, feedback=json.dumps({'helpful': [], 'unresolved': []}))
    assert normalize(obj, tin)[1] is None
    assert 'must contain exactly claim, assessment, dialogue_basis' in B.normalize_trajectory_analysis_output(obj, None, tin)[1]
    del obj['analysis']['feedback_check']['claim_reviews']
    assert B.normalize_trajectory_analysis_output(obj, None, tin)[1] is None
    assert 'claim_reviews' in normalize(obj, tin)[1]
    assert 'feedback_kind=' in normalize(fixture()[1], replace(fixture()[0], feedback_kind='observer'))[1]


def test_opt_in_settings_and_schema_do_not_change_legacy_cache_identity():
    expected = {'max_tokens': 1200, 'temperature': 0.7, 'reasoning_effort': None, 'response_format': None,
        'output_contract': None, 'local_schema_file': None, 'local_schema_sha256': None,
        'selected_turn_policy': 'feedback_locus_v1'}
    assert B.generation_settings({}) == expected
    assert B.generation_settings({'generation': {'feedback_check_protocol': None}}) == expected
    assert B.generation_settings({'generation': {'feedback_check_protocol': PROTOCOL}}) == {
        **expected, 'feedback_check_protocol': PROTOCOL}
    before = copy.deepcopy(B.FULL_TRAJECTORY_RUBRIC_WITH_INTERVENTION_JSON_SCHEMA)
    schema = extend_schema(before)
    assert B.FULL_TRAJECTORY_RUBRIC_WITH_INTERVENTION_JSON_SCHEMA == before
    assert 'claim_reviews' not in before['properties']['analysis']['properties']['feedback_check']['required']
    assert 'claim_reviews' in schema['properties']['analysis']['properties']['feedback_check']['required']
    __import__('jsonschema').Draft202012Validator.check_schema(schema)
    __import__('jsonschema').validate(fixture()[1], schema)
    for value in ('unsupported', '', False, [], {}):
        with pytest.raises(ValueError, match='feedback_check_protocol'):
            B.generation_settings({'generation': {'feedback_check_protocol': value}})
    with pytest.raises(ValueError, match='trajectory analysis'):
        B.api_rubric_generator('', 'model', {'feedback_check_protocol': PROTOCOL})


def test_generator_bounded_repair_and_provenance_cache_roundtrip(monkeypatch, tmp_path):
    tin, obj = fixture()
    invalid = copy.deepcopy(obj)
    reviews(invalid).pop()
    sent = []
    def call(key, model, system, user, settings, schema, name, role):
        sent.append({'user': user, 'settings': settings, 'schema': schema})
        return json.dumps(invalid if len(sent) == 1 else obj), {'finish_reason': 'stop'}
    monkeypatch.setattr(B, '_openrouter_json', call)
    generator = B.api_rubric_generator('key', 'model', {
        'max_tokens': 8192, 'temperature': 0.7, 'response_format': None,
        'output_contract': 'trajectory_analysis_candidate_v2_with_intervention',
        'feedback_check_protocol': PROTOCOL, 'selected_turn_policy': 'generator_v1'})
    result = generator('fixed prompt', tin)
    assert result['parse_error'] is None and len(result['raw_attempts']) == 2
    assert len(sent) == 2 and 'cover every reflection claim' in sent[1]['user']
    assert result['intervention_instruction'] == obj['intervention_instruction']
    from src.online_rl.tree.cache import CallCache
    from src.online_rl.tree.posthoc import generate_rubric
    cache = CallCache(tmp_path / 'calls.jsonl')
    cached_generator = lambda *_: result
    cached_generator.settings = generator.settings
    rubric = generate_rubric('fixed prompt', 'prompt-hash', tin, cached_generator, cache)
    assert rubric.valid
    feedback = rubric.rubric['source_provenance']['analysis']['feedback_check']
    assert feedback == obj['analysis']['feedback_check']
    from src.online_rl.tree.agent_state import visible_rubric
    assert 'claim_reviews' not in json.dumps(visible_rubric(rubric.rubric))
    stored = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert any(PROTOCOL in json.dumps(item) for item in stored)


def test_invalid_audit_remains_parse_error_after_one_repair(monkeypatch):
    tin, obj = fixture()
    reviews(obj).pop()
    calls = []
    def call(*args):
        calls.append(args)
        return json.dumps(obj), {}
    monkeypatch.setattr(B, '_openrouter_json', call)
    generator = B.api_rubric_generator('', 'model', {'max_tokens': 8192, 'temperature': 0.7,
        'output_contract': 'trajectory_analysis_candidate_v2_with_intervention',
        'feedback_check_protocol': PROTOCOL, 'response_format': None})
    result = generator('prompt', tin)
    assert len(calls) == 2
    assert 'cover every reflection claim' in result['parse_error']
    assert result['rubric'] is None and result['intervention_instruction'] is None
