"""CPU tests for relaxed initial rubric size and explicit old-cache reuse."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from baselines.common.rubrics import criteria, RubricEngine, state_input, PROTOCOL, INITIAL
from baselines.common.test_methods import FakeBackend, FakeReference, TinyModel, fixture
from baselines.repairs.remove_minimum import alias, convert, OLD_INITIAL, OLD_SOURCE, OLD_PROMPTS, NEW_SOURCE, NEW_PROMPTS
from src.online_rl.integrated.records import sha256_json


class SmallGenerator(FakeBackend):
    def __init__(self, count): super().__init__('generator'); self.count = count
    def complete(self, request):
        result = super().complete(request)
        if request.prompt_version.endswith('initial_rubric'):
            value = json.loads(result.text); value['criteria'] = value['criteria'][:self.count]
            return replace(result, text=json.dumps(value))
        return result


class RubricCountTests(unittest.TestCase):
    def test_one_through_six_initial_criteria_work_for_both_methods(self):
        trees, _, cases = fixture(TinyModel())
        for method in ('rar', 'online_rubrics'):
            for count in range(1, 7):
                with self.subTest(method=method, count=count), tempfile.TemporaryDirectory() as temp:
                    engine = RubricEngine(method, SmallGenerator(count), FakeBackend('grader'), temp,
                        weights={'w_outcome': 0., 'w_rubric': 1.},
                        reference=FakeReference() if method == 'online_rubrics' else None)
                    targets, report = engine.score(trees[0], cases[trees[0].case_id])
                    self.assertEqual(len(report['groups']['root']['initial_rubric']), count)
                    self.assertEqual(len(targets['nodes']), 4)
                    self.assertTrue(all(n['R_A'] is not None for n in targets['nodes'].values()))

    def test_empty_initial_still_rejects_but_empty_online_additions_work(self):
        with self.assertRaisesRegex(ValueError, 'empty'): criteria({'criteria': []})
        self.assertEqual(criteria({'criteria': []}, signed=True), [])
        item = {'text': 'criterion', 'category': 'Essential'}
        self.assertEqual(len(criteria({'criteria': [item]*20})), 20)
        with self.assertRaises(ValueError): criteria({'criteria': [item]*21})
        with self.assertRaises(ValueError): criteria({'criteria': [{'text': 'bad', 'category': 'invalid'}]})

    def test_old_success_is_explicitly_reused_without_generator_call(self):
        trees, _, cases = fixture(TinyModel()); tree = trees[0]; case = cases[tree.case_id]
        with tempfile.TemporaryDirectory() as temp:
            gen = FakeBackend('generator'); grader = FakeBackend('grader')
            engine = RubricEngine('rar', gen, grader, temp, weights={'w_outcome': 0., 'w_rubric': 1.})
            identity = {'backend': gen.identity(), 'system': OLD_INITIAL,
                'payload': {**state_input(case, tree.nodes[0]), 'train_reference_answer': case['ground_truth']},
                'params': engine.gen_params, 'protocol': PROTOCOL}
            value = {'criteria': [{'text': f'criterion{i}', 'category': 'Essential'} for i in range(7)]}
            old = {'key': sha256_json(['initial_rubric', identity]), 'kind': 'initial_rubric', 'identity': identity, 'value': value}
            updated = alias(old)
            self.assertEqual(updated['value'], old['value'])
            self.assertNotEqual(updated['key'], old['key'])
            self.assertEqual(updated['reuse_provenance']['original_key'], old['key'])
            self.assertEqual(updated['identity']['system'], INITIAL)
            path = Path(temp)/'cache/initial_rubrics.jsonl'; path.parent.mkdir()
            path.write_text(json.dumps(old)+'\n'+json.dumps(updated)+'\n')
            engine = RubricEngine('rar', gen, grader, temp, weights={'w_outcome': 0., 'w_rubric': 1.})
            self.assertEqual(len(engine.initial(case, tree.nodes[0])), 7)
            self.assertEqual(gen.calls, [])

    def test_transition_changes_only_source_and_prompt_identity(self):
        cfg = {'optimizer': {'lr': 1e-5}, 'comparison_method': {'source_sha256': OLD_SOURCE,
            'engine': {'prompts_sha256': OLD_PROMPTS, 'reward_weights': {'w_rubric': 1., 'w_outcome': 0.}}}}
        new = convert(cfg)
        self.assertEqual(cfg['comparison_method']['source_sha256'], OLD_SOURCE)
        self.assertEqual(new['comparison_method']['source_sha256'], NEW_SOURCE)
        self.assertEqual(new['comparison_method']['engine']['prompts_sha256'], NEW_PROMPTS)
        self.assertEqual(new['optimizer'], cfg['optimizer'])
        self.assertEqual(new['comparison_method']['engine']['reward_weights'], cfg['comparison_method']['engine']['reward_weights'])


if __name__ == '__main__': unittest.main()
