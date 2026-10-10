"""Offline V7 provenance correction without relaxing quotes or rewriting criteria."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rubric_test_fixtures import _tin, _output
from src.online_rl.tree import backends as B
from src.online_rl.tree.b200_runner import load_config
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.feedback_grounding import REFLECTION_KIND


def example():
    tin = replace(_tin('failure'), feedback_kind=REFLECTION_KIND,
                  feedback=json.dumps({'helpful': [], 'unresolved': []}))
    obj = _output(tin)
    c, a = obj['candidate_rubric'], obj['analysis']
    obj['intervention_instruction'] = 'Compare both considerations and connect them to the decision criterion.'
    c.update(required_behavior=c['scores']['2']['behavior'], minimum_performance='Make the requested comparison.',
             completion_condition='Connect the comparison to the stated decision criterion.',
             criterion_version='public_behavior_app_alignment_v5')
    a['source_score'] = 1
    a['feedback_check']['claim_reviews'] = []
    a['score_construction_check'].update(minimum_ref='candidate_rubric.minimum_performance',
        completion_ref='candidate_rubric.completion_condition',
        instruction_alignment={'target_level': 2, 'instruction_ref': 'intervention_instruction',
                               'rationale': 'Requires both M and C.'})
    return tin, obj


def setup(monkeypatch, replies):
    cfg = load_config('configs/experiments/train120_max4_v7_rubric_only_b200.json')['rubric']
    requests = []
    def send(key, model, system, user, settings, schema, name, role):
        requests.append(dict(model=model, system=system, user=user, settings=settings, role=role))
        return json.dumps(replies[len(requests) - 1]), {'usage': {'total_tokens': 7}}
    monkeypatch.setattr(B, '_openrouter_json', send)
    gen = B.api_rubric_generator('secret-never-log', cfg['generator_model'], B.generation_settings(cfg))
    prompt = Path(cfg['generator_prompt_file']).read_text()
    return gen, prompt, requests


@pytest.mark.parametrize('fault', ['paraphrase', 'wrong_turn', 'wrong_speaker'])
def test_exact_quote_correction_preserves_criterion_and_success_cache(monkeypatch, tmp_path, fault):
    tin, good = example()
    bad = copy.deepcopy(good)
    span = bad['analysis']['evidence_spans'][0]
    if fault == 'paraphrase': span['quote'] = 'Please compare the two considerations.'
    if fault == 'wrong_turn': span['turn_id'] = 2
    if fault == 'wrong_speaker': span['speaker'] = 'agent'
    gen, prompt, requests = setup(monkeypatch, [bad, bad, good])
    cache = CallCache(tmp_path / 'calls.jsonl')
    ident = {'generator': gen.settings, 'leaf': tin.leaf_id}
    result = cache.call('rubric_generation', ident, lambda: gen(prompt, tin))
    assert result['parse_error'] is None and len(requests) == 3
    assert result['rubric']['source_provenance']['analysis']['source_score'] == 1
    assert len(result['usage_attempts']) == 3
    assert len(result['validation_errors']) == 2
    assert 'Exact source text:' in requests[1]['user']
    assert 'Correct ONLY analysis.evidence_spans' in requests[2]['user']
    assert 'Source catalogue:' in requests[2]['user']
    assert 'Please compare these two considerations.' in requests[2]['user']
    for req in requests:
        assert req['model'] == 'gpt-6-luna' and req['system'] == prompt
        assert req['settings']['api_provider'] == 'openai'
        assert req['settings']['reasoning_effort'] == 'none'
        assert req['settings'] == requests[0]['settings']
    assert CallCache(cache.path).call('rubric_generation', ident,
                                    lambda: pytest.fail('valid cache repeated')) == result


@pytest.mark.parametrize('rewrite', [False, True])
def test_invalid_quote_or_unrelated_rewrite_fails_with_audit(monkeypatch, tmp_path, rewrite):
    tin, good = example()
    bad = copy.deepcopy(good)
    bad['analysis']['evidence_spans'][0]['quote'] = 'Invented support.'
    third = copy.deepcopy(good if rewrite else bad)
    if rewrite: third['candidate_rubric']['name'] = 'Different criterion'
    gen, prompt, requests = setup(monkeypatch, [bad, bad, third])
    cache = CallCache(tmp_path / 'calls.jsonl')
    with pytest.raises(B.RubricOutputIncomplete) as error:
        cache.call('rubric_generation', {}, lambda: gen(prompt, tin))
    assert len(requests) == 3 and not cache.store
    audit = error.value.audit
    assert audit['attempts'] == 3 and len(audit['usage_attempts']) == 3
    assert audit['case_id'] == tin.case_id and audit['leaf_id'] == tin.leaf_id
    if rewrite: assert 'outside analysis.evidence_spans' in audit['validation_errors'][-1]
    else: assert 'exactly match' in audit['validation_errors'][-1]
    log = cache.path.with_suffix('.errors.jsonl').read_text()
    assert json.loads(log)['audit'] == audit
    assert 'secret-never-log' not in log


@pytest.mark.parametrize('bad_first', [False, True])
def test_valid_first_or_first_repair_needs_no_third_request(monkeypatch, bad_first):
    tin, good = example()
    bad = copy.deepcopy(good)
    bad['analysis']['evidence_spans'][0]['quote'] = 'Paraphrase'
    gen, prompt, requests = setup(monkeypatch, [bad, good] if bad_first else [good])
    assert gen(prompt, tin)['parse_error'] is None
    assert len(requests) == (2 if bad_first else 1)


def test_non_quote_error_remains_two_requests(monkeypatch):
    tin, bad = example()
    bad['source_case_id'] = 'another-case'
    gen, prompt, requests = setup(monkeypatch, [bad, bad])
    with pytest.raises(B.RubricOutputIncomplete): gen(prompt, tin)
    assert len(requests) == 2
