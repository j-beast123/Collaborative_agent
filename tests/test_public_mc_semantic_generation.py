"""Offline request/normalization regressions, not a claim of LLM semantic accuracy.

The earlier model outputs demonstrated real semantic failures. These tests pin
the corrective request and prove the runtime no longer forces a partial source
claim to the outcome-derived anchor; real semantic assessment remains a pilot.
"""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree import backends as B
from src.online_rl.tree.agent_state import visible_rubric
from src.online_rl.tree.instruction_audit import ANALYSIS_PROTOCOL
from src.online_rl.tree.feedback_grounding import PROTOCOL as FEEDBACK_PROTOCOL, REFLECTION_KIND
from test_public_mc_contract import public_output, PROTOCOL


BASE = Path('prompts/sherpa')
PROMPT = BASE / 'trajectory_rubric_generator_v2_5_public_mc_semantic.txt'
SCHEMA = BASE / 'trajectory_rubric_generator_v2_5_public_mc_semantic.schema.json'
PREVIOUS_PROMPT = BASE / 'trajectory_rubric_generator_v2_4_public_mc.txt'
PREVIOUS_SCHEMA = BASE / 'trajectory_rubric_generator_v2_4_public_mc.schema.json'


def example(outcome='failure', claimed=1):
    source, obj = public_output(outcome)
    source = replace(source, feedback_kind=REFLECTION_KIND, feedback=json.dumps({'helpful': [], 'unresolved': []}))
    obj['analysis']['feedback_check']['claim_reviews'] = []
    obj['analysis']['source_score'] = claimed
    obj['candidate_rubric']['criterion_version'] = B.SEMANTIC_CRITERION_VERSION
    if claimed != (2 if outcome == 'success' else 0):
        obj['analysis']['limitations'] = [
            'Synthetic source claim differs from the outcome-derived expected anchor; independent Gate review is pending.'
        ]
    return source, obj


def settings(path=SCHEMA):
    return B.generation_settings({'generation': {
        'max_tokens': 4096, 'temperature': 0.7, 'response_format': None,
        'output_contract': 'trajectory_analysis_candidate_v2_with_intervention',
        'public_criterion_protocol': PROTOCOL,
        'generation_semantic_protocol': B.GENERATION_SEMANTIC_PROTOCOL,
        'instruction_alignment_protocol': ANALYSIS_PROTOCOL,
        'feedback_check_protocol': FEEDBACK_PROTOCOL,
        'selected_turn_policy': 'generator_v1',
        'local_schema_file': str(path),
        'local_schema_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }})


def normalize(source, obj, **overrides):
    options = {'require_intervention': True,
               'instruction_alignment_protocol': ANALYSIS_PROTOCOL,
               'feedback_check_protocol': FEEDBACK_PROTOCOL,
               'public_criterion_protocol': PROTOCOL,
               'generation_semantic_protocol': B.GENERATION_SEMANTIC_PROTOCOL,
               **overrides}
    return B.normalize_trajectory_analysis_output(obj, None, source, **options)


@pytest.mark.parametrize('outcome', ['success', 'failure'])
@pytest.mark.parametrize('claimed', [0, 1, 2])
def test_observed_claim_survives_and_expectation_is_separate(outcome, claimed):
    source, obj = example(outcome, claimed)
    before = copy.deepcopy(obj)
    rubric, error = normalize(source, obj)
    assert error is None and obj == before
    prov = rubric['source_provenance']
    expected = 2 if outcome == 'success' else 0
    assert prov['analysis']['source_score'] == claimed
    assert prov['model_reported_analysis']['source_score'] == claimed
    assert prov['deterministic_adjustments'] == {}
    assert prov['source_anchor_contract_mismatch'] is (claimed != expected)
    assert prov['source_anchor_declaration'] == {
        'expected_score_from_outcome': expected,
        'generator_claimed_source_score': claimed,
        'matches_expected': claimed == expected,
        'independent_measurement': False,
    }
    assert prov['generation_semantic_protocol'] == B.GENERATION_SEMANTIC_PROTOCOL
    # This declaration is provenance, not an answer/score hint exposed to judges.
    assert 'source_anchor' not in json.dumps(visible_rubric(rubric))
    # Rerun target remains outcome-derived, not chosen from claimed source level.
    assert prov['analysis']['score_construction_check']['instruction_alignment']['target_level'] == (
        0 if outcome == 'success' else 2)


@pytest.mark.parametrize('bad', [None, True, 1.0, '1', -1, 3])
def test_new_observation_claim_is_an_integer_level_not_a_coerced_value(bad):
    source, obj = example()
    obj['analysis']['source_score'] = bad
    rubric, error = normalize(source, obj)
    assert rubric is None and 'integer 0, 1 or 2' in error


def test_previous_contract_bytes_and_source_anchor_rules_are_unchanged():
    assert hashlib.sha256(PREVIOUS_PROMPT.read_bytes()).hexdigest() == (
        'b79afb38306aaaba63ae8916982168027405202b1d34a5b18e8a968d3d9396c5')
    assert hashlib.sha256(PREVIOUS_SCHEMA.read_bytes()).hexdigest() == (
        'd49e1b795ed5de19c31bfea71fd11938e07ec10931fc2af3e53c542e0b76dd44')
    source, old = public_output('failure')
    legacy_kwargs = dict(require_intervention=True, instruction_alignment_protocol=ANALYSIS_PROTOCOL,
                         public_criterion_protocol=PROTOCOL)
    assert B.normalize_trajectory_analysis_output(old, None, source, **legacy_kwargs)[1] is None
    old['analysis']['source_score'] = 1
    assert 'integer 0 or 2' in B.normalize_trajectory_analysis_output(old, None, source, **legacy_kwargs)[1]
    assert 'generation_semantic_protocol' not in B.generation_settings({})


def test_new_version_and_schema_are_required_not_an_upgrade_of_existing_criteria():
    source, obj = example()
    obj['candidate_rubric']['criterion_version'] = PROTOCOL
    assert 'criterion_version differs' in normalize(source, obj)[1]
    with pytest.raises(ValueError, match='observed source_score'):
        B.api_rubric_generator('unused', 'synthetic', settings(PREVIOUS_SCHEMA))
    with pytest.raises(ValueError, match='generation semantic protocol'):
        B.generation_settings({'generation': {'public_criterion_protocol': PROTOCOL,
                                             'generation_semantic_protocol': 'unknown'}})


def test_request_preserves_partial_claim_without_outcome_retry_or_provider_schema(monkeypatch):
    source, obj = example()
    calls = []

    def fake(key, model, system, user, config, schema, name, role):
        calls.append({'system': system, 'user': user, 'settings': config, 'schema': schema})
        return json.dumps(obj), {}

    monkeypatch.setattr(B, '_openrouter_json', fake)
    generator = B.api_rubric_generator('unused', 'synthetic', settings())
    result = generator(PROMPT.read_text(), source)
    assert result['parse_error'] is None and len(calls) == 1
    assert len(result['raw_attempts']) == 1
    assert result['rubric']['source_provenance']['analysis']['source_score'] == 1
    assert result['rubric']['source_provenance']['source_anchor_contract_mismatch'] is True
    assert result['intervention_instruction'] == obj['intervention_instruction']
    schema = json.loads(SCHEMA.read_text())
    jsonschema.validate(obj, schema)
    assert calls[0]['schema'] == schema
    assert calls[0]['settings']['response_format'] is None
    assert calls[0]['settings']['generation_semantic_protocol'] == B.GENERATION_SEMANTIC_PROTOCOL
    assert calls[0]['settings']['effective_output_schema_sha256'] == sha256_json(schema)
    assert calls[0]['system'] == PROMPT.read_text()
    # These are request-contract checks, not mocked evidence of semantic success.
    prompt = calls[0]['system']
    assert 'a contradictory conclusion fails C and receives 1 while the comparison remains' in prompt
    assert 'allowing a supported fit, a supported lack of fit, or a specifically justified information limit' in prompt
    assert 'unlikely, less typical, or less consistent has not thereby asserted impossible' in prompt
    assert 'an expressed numerical/directional boundary, and verification against a stated reference are distinct' in prompt
    assert 'It must not require agreement with the Doctor' in prompt
    assert 'source_score must be the integer 2 for success and 0 for failure' not in prompt


def test_schema_correction_does_not_rewrite_partial_behavior_into_desired_anchor(monkeypatch):
    source, good = example()
    bad = copy.deepcopy(good)
    bad['analysis'].pop('limitations')
    calls = []

    def fake(key, model, system, user, config, schema, name, role):
        calls.append(user)
        return json.dumps(bad if len(calls) == 1 else good), {}

    monkeypatch.setattr(B, '_openrouter_json', fake)
    result = B.api_rubric_generator('unused', 'synthetic', settings())(PROMPT.read_text(), source)
    assert result['parse_error'] is None and len(calls) == 2
    assert 'Preserve a supported partial score 1' in calls[1]
    assert 'For failure, source_score must be integer 0' not in calls[1]
    assert result['rubric']['source_provenance']['analysis']['source_score'] == 1
    assert result['rubric']['source_provenance']['analysis']['score_construction_check'] == (
        good['analysis']['score_construction_check'])
    assert result['rubric']['source_provenance']['deterministic_adjustments'] == {}


def test_gate_dispatch_forwards_new_and_previous_protocol_without_changing_default(monkeypatch):
    from src.online_rl.tree import quality_gate
    seen = []
    def fake(client, cases, **kwargs):
        seen.append(kwargs)
        return 'synthetic gate'
    monkeypatch.setattr(quality_gate, 'public_mc_posthoc_gate', fake)
    for protocol in ('public_mc_gate_v1', 'public_mc_gate_v2'):
        assert B.posthoc_gates(object(), {}, protocol=protocol,
                              rubric_quote_protocol='visible_field_enum_v1') == 'synthetic gate'
    assert [row['protocol'] for row in seen] == ['public_mc_gate_v1', 'public_mc_gate_v2']
