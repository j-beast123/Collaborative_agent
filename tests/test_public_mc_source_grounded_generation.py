"""Offline v3 generation contract regressions; real semantic fidelity needs LLM review."""
import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from src.online_rl.tree import backends as B
from src.online_rl.integrated.records import sha256_json
from test_public_mc_semantic_generation import example, settings, normalize


BASE = Path('prompts/sherpa')
PROMPT = BASE / 'trajectory_rubric_generator_v2_6_public_mc_source_grounded.txt'
SCHEMA = BASE / 'trajectory_rubric_generator_v2_6_public_mc_source_grounded.schema.json'
PROTOCOL = 'public_mc_generation_v3'
CRITERION = 'public_behavior_mc_semantic_v3'


def v3_example():
    source, obj = example('failure', 1)
    obj['candidate_rubric']['criterion_version'] = CRITERION
    return source, obj


def v3_settings():
    cfg = settings(SCHEMA)
    cfg['generation_semantic_protocol'] = PROTOCOL
    return B.generation_settings({'generation': cfg})


def test_v3_request_is_source_first_and_preserves_failure_partial_claim(monkeypatch):
    source, obj = v3_example()
    calls = []
    def fake(key, model, system, user, config, schema, name, role):
        calls.append({'system': system, 'settings': config, 'schema': schema})
        return json.dumps(obj), {}
    monkeypatch.setattr(B, '_openrouter_json', fake)
    result = B.api_rubric_generator('unused', 'synthetic', v3_settings())(PROMPT.read_text(), source)
    assert result['parse_error'] is None and len(calls) == 1
    prov = result['rubric']['source_provenance']
    assert prov['analysis']['source_score'] == 1
    assert prov['source_anchor_declaration']['expected_score_from_outcome'] == 0
    assert prov['source_anchor_declaration']['generator_claimed_source_score'] == 1
    assert prov['source_anchor_contract_mismatch'] is True
    assert prov['deterministic_adjustments'] == {}
    assert result['rubric']['criterion_version'] == CRITERION
    schema = json.loads(SCHEMA.read_text())
    jsonschema.validate(obj, schema)
    assert calls[0]['schema'] == schema
    assert calls[0]['settings']['generation_semantic_protocol'] == PROTOCOL
    assert calls[0]['settings']['effective_output_schema_sha256'] == sha256_json(schema)
    assert calls[0]['settings']['response_format'] is None
    prompt = calls[0]['system']
    assert prompt.index('LITERAL SOURCE OBSERVATION') < prompt.index('SELECT ONE BEHAVIOR AND DEFINE PUBLIC M/C')
    # Pin the requests addressing the known defects, not a mocked semantic success.
    assert 'Analysis is not a loophole for importing a correct clinical answer.' in prompt
    assert 'Choosing an exception option does not retrospectively strengthen the explanation.' in prompt
    assert 'Calling that assertion “uncritical” does not make it 0.' in prompt
    assert 'The hypothetical0 is removal of M, not a new preferred answer, new agreement' in prompt
    assert 'Include absent conclusions when C is a conclusion' in prompt
    assert 'supported alternative outcome and a justified information limit' in prompt


def test_v3_correction_keeps_partial_claim_and_does_not_force_failure0(monkeypatch):
    source, good = v3_example()
    bad = copy.deepcopy(good)
    bad['analysis'].pop('limitations')
    calls = []
    def fake(key, model, system, user, config, schema, name, role):
        calls.append(user)
        return json.dumps(bad if len(calls) == 1 else good), {}
    monkeypatch.setattr(B, '_openrouter_json', fake)
    result = B.api_rubric_generator('unused', 'synthetic', v3_settings())(PROMPT.read_text(), source)
    assert result['parse_error'] is None and len(calls) == 2
    assert 'Preserve a supported partial score 1' in calls[1]
    assert 'For failure, source_score must be integer 0' not in calls[1]
    assert result['rubric']['source_provenance']['analysis']['source_score'] == 1
    assert result['rubric']['scores'] == good['candidate_rubric']['scores']


def test_v3_does_not_relabel_v2_criteria_or_change_frozen_v2_files():
    assert hashlib.sha256((BASE / 'trajectory_rubric_generator_v2_5_public_mc_semantic.txt').read_bytes()).hexdigest() == (
        '0a9f9f531d5a4a3014ea8922a2cc348abd136bbe0e15852931b0828ecd62fe2d')
    assert hashlib.sha256((BASE / 'trajectory_rubric_generator_v2_5_public_mc_semantic.schema.json').read_bytes()).hexdigest() == (
        'e59398434a210e44651654135c824b944ffe8a283a0ef9ac87de6b4117cce0ba')
    source, obj = example('failure', 1)
    assert normalize(source, obj)[1] is None
    assert 'criterion_version differs' in normalize(source, obj, generation_semantic_protocol=PROTOCOL)[1]
    obj['candidate_rubric']['criterion_version'] = CRITERION
    assert normalize(source, obj, generation_semantic_protocol=PROTOCOL)[1] is None
    assert 'criterion_version differs' in normalize(source, obj)[1]


def test_v3_cannot_use_an_older_pinned_schema_with_new_protocol():
    old = settings()
    old['generation_semantic_protocol'] = PROTOCOL
    with pytest.raises(ValueError, match='new criterion version schema'):
        B.api_rubric_generator('unused', 'synthetic', old)


def test_gate_dispatch_accepts_v3_without_removing_prior_protocols(monkeypatch):
    from src.online_rl.tree import quality_gate
    seen = []
    def fake(client, cases, **kwargs):
        seen.append(kwargs['protocol'])
        return 'synthetic gate'
    monkeypatch.setattr(quality_gate, 'public_mc_posthoc_gate', fake)
    for protocol in ('public_mc_gate_v1', 'public_mc_gate_v2', 'public_mc_gate_v3'):
        assert B.posthoc_gates(object(), {}, protocol=protocol) == 'synthetic gate'
    assert seen == ['public_mc_gate_v1', 'public_mc_gate_v2', 'public_mc_gate_v3']
