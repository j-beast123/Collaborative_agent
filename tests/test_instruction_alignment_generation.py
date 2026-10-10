"""The joint output must preserve a complete, source-direction-correct guide contract."""
import copy
import json

import pytest

from rubric_test_fixtures import _output, _tin
from src.online_rl.tree import backends as B
from src.online_rl.tree.instruction_audit import ANALYSIS_PROTOCOL


def example(outcome='success'):
    tin = _tin(outcome)
    obj = _output(tin)
    obj['intervention_instruction'] = ('Do not make the requested comparison.' if outcome == 'success'
                                       else 'Compare the two considerations and connect them to the criterion.')
    obj['analysis']['score_construction_check'].update(
        minimal_contribution='The expressed comparison.',
        complete_behavior='The comparison is connected to the decision criterion.',
        instruction_alignment={'target_level': 0 if outcome == 'success' else 2,
                               'instruction_quote': obj['intervention_instruction'],
                               'rationale': 'The guide removes M.' if outcome == 'success' else 'The guide requires M and C.'})
    return tin, obj


def normalize(obj, tin):
    return B.normalize_trajectory_analysis_output(obj, None, tin, require_intervention=True,
        instruction_alignment_protocol=ANALYSIS_PROTOCOL)


@pytest.mark.parametrize('outcome', ['success', 'failure'])
def test_joint_normalizer_preserves_checked_analysis_without_mutating_input(outcome):
    tin, obj = example(outcome)
    before = copy.deepcopy(obj)
    rubric, error = normalize(obj, tin)
    assert error is None
    assert obj == before
    prov = rubric['source_provenance']
    assert prov['instruction_alignment_protocol'] == ANALYSIS_PROTOCOL
    assert prov['analysis']['score_construction_check'] == obj['analysis']['score_construction_check']
    assert prov['model_reported_analysis']['score_construction_check'] == obj['analysis']['score_construction_check']


@pytest.mark.parametrize('field', ['minimal_contribution', 'complete_behavior', 'instruction_alignment'])
def test_missing_analysis_is_error_not_a_synthesized_assertion(field):
    tin, obj = example()
    obj['analysis']['score_construction_check'].pop(field)
    assert normalize(obj, tin)[0] is None


@pytest.mark.parametrize('change', [
    {'target_level': 2}, {'target_level': True}, {'target_level': '0'},
    {'instruction_quote': 'An unobserved clause.'}, {'instruction_quote': '  '},
    {'rationale': ''}, {'rationale': None}, {'extra': 'unvalidated'},
])
def test_malformed_or_wrong_direction_alignment_is_not_normalized(change):
    tin, obj = example()
    obj['analysis']['score_construction_check']['instruction_alignment'].update(change)
    rubric, error = normalize(obj, tin)
    assert rubric is None and error


def test_generator_transports_and_preserves_instruction_with_new_local_schema(monkeypatch):
    from scripts.online_rl.run_tree import ROOT
    from src.online_rl.tree.feedback_grounding import PROTOCOL as FEEDBACK_PROTOCOL, REFLECTION_KIND
    from dataclasses import replace
    from src.online_rl.tree.experiment import load_config
    cfg = load_config('configs/experiments/train120_max2_fixed_p0_v23.json')
    tin, obj = example()
    tin = replace(tin, feedback_kind=REFLECTION_KIND, feedback=json.dumps({'helpful': [], 'unresolved': []}))
    obj['analysis']['feedback_check']['claim_reviews'] = []
    # Empty structured Doctor reflection requires no manufactured claim review.
    calls = []

    def fake(key, model, system, user, settings, schema, name, role):
        calls.append({'schema': schema, 'settings': settings, 'system': system})
        return json.dumps(obj), {}

    monkeypatch.setattr(B, '_openrouter_json', fake)
    settings = B.generation_settings(cfg['rubric'])
    generator = B.api_rubric_generator('unused', 'model', settings)
    result = generator((ROOT / cfg['rubric']['generator_prompt_file']).read_text(), tin)
    assert result.get('parse_error') is None
    assert len(calls) == 1
    assert calls[0]['settings']['instruction_alignment_protocol'] == ANALYSIS_PROTOCOL
    assert result['intervention_instruction'] == obj['intervention_instruction']
    prov = result['rubric']['source_provenance']
    assert prov['instruction_alignment_protocol'] == ANALYSIS_PROTOCOL
    assert prov['feedback_check_protocol'] == FEEDBACK_PROTOCOL
    required = calls[0]['schema']['properties']['analysis']['properties']['score_construction_check']['required']
    assert {'minimal_contribution', 'complete_behavior', 'instruction_alignment'} <= set(required)


def test_new_analysis_requires_joint_contract_and_versioned_local_schema():
    with pytest.raises(ValueError, match='joint output contract'):
        B.api_rubric_generator('unused', 'model', {'instruction_alignment_protocol': ANALYSIS_PROTOCOL})
    with pytest.raises(ValueError, match='Unknown instruction alignment'):
        B.generation_settings({'generation': {'instruction_alignment_protocol': 'unknown'}})


def test_structured_applicability_rejects_string_booleans_before_cache():
    from types import SimpleNamespace
    from src.online_rl.tree.agent_state import visible_rubric
    tin, obj = example()
    rubric, _ = normalize(obj, tin)
    rubric['rubric_id'] = 'r'
    client = SimpleNamespace(model='judge', protocol_sha256='p', log=[],
        judge=lambda *args: ({'r': {'applicable': 'false', 'evidence_basis': '', 'missing_requirement': 'absent'}}, {}))
    scorer = B.two_stage_structured_scorer(client, observed_context=True)
    with pytest.raises(ValueError, match='JSON boolean'):
        scorer.review_state({'question': 'Q', 'options': {}}, [], [visible_rubric(rubric)])
