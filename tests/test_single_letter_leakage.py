import copy
import json
from types import SimpleNamespace

import pytest

from src.online_rl.sherpa.rubrics import leakage_matches
from src.online_rl.sherpa.pipeline import assess


CASE = {'case_id': 'image', 'options': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'},
        'ground_truth': {'correct_option': 'D', 'correct_text': 'D'}}


def rubric(text):
    return {'name': 'Check evidence', 'situation': 'When a label is unverified', 'behavior': text}


def test_article_is_not_an_image_label():
    assert leakage_matches(CASE, rubric('Does the assistant request a description of the image?')) == []


@pytest.mark.parametrize('text', ['Select option a.', 'Identify label b.', 'The choice is A.', 'Answer D.'])
def test_actual_image_label_references_still_rejected(text):
    assert leakage_matches(CASE, rubric(text))


def test_medical_answer_text_still_rejected():
    case = {**CASE, 'options': {'A': 'Lichen planus', 'D': 'Other diagnosis'}}
    assert leakage_matches(case, rubric('Suggest LICHEN PLANUS to the physician.'))


def test_revalidate_cached_raw_without_overwriting_audit_or_calling_generator():
    rubrics = [{**rubric('Does the assistant request a description?'), 'name': f'Check {i}'} for i in range(5)]
    old = {'valid': False, 'rubrics': [], 'removed_count': 5, 'removed_fraction': 1,
           'attempts': [{'raw': json.dumps({'rubrics': rubrics}), 'valid': False, 'error': 'Answer leakage'}]}
    original = copy.deepcopy(old)
    cache = SimpleNamespace(call=lambda *args: old)
    backend = SimpleNamespace(cfg={'method': 'sherpa'}, identity=lambda: {},
                              score_all=lambda *args: [{'reward': 1}, {'reward': 0}])
    result = assess(CASE, [{'outcome': 1}, {'outcome': 0}], [], 'prompt', backend, cache, {})
    assert result['valid'] and result['auc'] == 1
    assert result['generation']['validation_recovery']['source'] == 'stored_raw_generation'
    assert old == original
    for row in rubrics: row['behavior'] = 'Select option A.'
    old['attempts'][0]['raw'] = json.dumps({'rubrics': rubrics})
    backend.score_all = lambda *args: pytest.fail('Must not score leaking rubrics')
    assert not assess(CASE, [], [], 'prompt', backend, cache, {})['valid']
