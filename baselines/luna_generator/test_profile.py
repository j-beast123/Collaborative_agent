"""CPU-only fake transport and real reward-engine/profile wiring regressions."""
import copy
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baselines.luna_generator.backend import LunaGeneratorBackend
from baselines.luna_generator.generator_profile import MODEL, ROOT, install_profile, validate_spec
from baselines.common.test_methods import FakeBackend, FakeReference, TinyModel, fixture
from baselines.parallel_api.engine import ParallelRubricEngine
from baselines.parallel_api.openai_profile import GRADE_PARAMS
from src.judges.llm import ChatRequest


def spec_for(method):
    return json.loads((ROOT / 'baselines/luna_generator' / (method + '.json')).read_text())


class ProfileTests(unittest.TestCase):
    def test_configs_preserve_science_and_isolate_old_runs(self):
        allowed = {'output_dir', 'initial_rubric_source', 'initial_rubric_registry',
                   'doctor_port_start', 'cache_namespace', 'generator_profile', 'previous_run'}
        ports = []
        for method in ('rar', 'online_rubrics'):
            old = json.loads((ROOT / 'baselines/parallel_api' / (method + '_openai.json')).read_text())
            new = spec_for(method)
            validate_spec(method, new)
            self.assertEqual({k: v for k, v in old.items() if k not in allowed},
                             {k: v for k, v in new.items() if k not in allowed})
            self.assertEqual(new['previous_run'], old['output_dir'])
            self.assertNotEqual(new['output_dir'], old['output_dir'])
            self.assertNotEqual(new['initial_rubric_registry'], old['initial_rubric_registry'])
            ports.extend([new['doctor_port_start'], new['doctor_port_start'] + 1])
            for field in ('output_dir', 'initial_rubric_registry', 'doctor_port_start', 'initial_rubric_source'):
                bad = {**new, field: old[field] if field != 'initial_rubric_source' else 'old-cache.jsonl'}
                with self.assertRaises(ValueError):
                    validate_spec(method, bad)
        self.assertEqual(len(set(ports)), 4)

    def test_new_generator_reaches_all_roles_and_reuses_successful_cache(self):
        trees, _, cases = fixture(TinyModel())
        for method in ('rar', 'online_rubrics'):
            with tempfile.TemporaryDirectory() as temp:
                grader = FakeBackend('grader')
                reference = FakeReference() if method == 'online_rubrics' else None
                engine = ParallelRubricEngine(method, FakeBackend('generator'), grader, temp,
                    weights={'w_outcome': 0., 'w_rubric': 1.}, reference=reference,
                    api_workers=2, grader_workers=2)
                engine.grade_params = copy.deepcopy(GRADE_PARAMS)
                before = copy.deepcopy(engine.gen_params)
                cfg = {'comparison_method': {'engine': engine.identity}, 'rubric': {'generation': {}},
                       'grpo': {'lr': 1e-5}, 'output_dir': temp}
                original = copy.deepcopy(cfg)
                launcher = SimpleNamespace(setup=lambda *args: (cfg, {}, engine, reference, []))
                install_profile(launcher)
                cfg, _, engine, ref, errors = launcher.setup(method, spec_for(method), {})
                self.assertFalse(errors)
                self.assertIs(ref, reference)
                self.assertIs(engine.grader, grader)
                self.assertEqual(engine.grade_params, GRADE_PARAMS)
                self.assertEqual(cfg['grpo'], original['grpo'])
                before.pop('temperature', None)
                self.assertEqual(engine.gen_params, {**before, 'reasoning': {'effort': 'none'}})
                self.assertNotIn('temperature', cfg['rubric']['generation'])
                self.assertEqual(cfg['comparison_method']['engine']['generator']['model'], MODEL)
                self.assertIn('backend.py', cfg['comparison_method']['generator_profile']['execution_source_sha256'])
                fake = FakeBackend('generator')
                requests = []
                def complete(req):
                    requests.append(req)
                    self.assertEqual(req.params['reasoning'], {'effort': 'none'})
                    self.assertNotIn('temperature', req.params)
                    return fake.complete(req)
                tree = trees[0]
                with patch.object(engine.generator, 'complete', side_effect=complete):
                    result = engine.score(tree, cases[tree.case_id])
                    count = len(requests)
                    self.assertEqual(result, engine.score(tree, cases[tree.case_id]))
                    self.assertEqual(len(requests), count)
                expected = {'initial_rubric', 'pairwise_extract', 'deduplicate'} if reference else {'initial_rubric'}
                self.assertEqual({r.prompt_version.split('.')[-1] for r in requests}, expected)
                cache = '\n'.join(p.read_text() for p in Path(temp).rglob('*.jsonl'))
                self.assertIn(MODEL, cache)
                self.assertEqual(engine.weights, {'w_outcome': 0., 'w_rubric': 1.})

    def test_generator_identity_and_reasoning_change_resume_binding(self):
        from baselines.common.rubric_launch import bind
        with tempfile.TemporaryDirectory() as temp:
            cfg = {'output_dir': temp, 'comparison_method': {
                'generator_profile': {'model': MODEL, 'reasoning': 'none'}}}
            bind(cfg)
            bind(copy.deepcopy(cfg))
            for field, value in [('model', 'previous-generator'), ('reasoning', 'low')]:
                changed = copy.deepcopy(cfg)
                changed['comparison_method']['generator_profile'][field] = value
                with self.assertRaisesRegex(RuntimeError, 'Resume method settings changed'):
                    bind(changed)


class TransportTests(unittest.TestCase):
    def backend(self, temp):
        return LunaGeneratorBackend(audit_path=Path(temp)/'audit.jsonl',
            base_url='https://openrouter.ai/api/v1', model=MODEL, revision=MODEL,
            api_key_env='OPENROUTER_API_KEY', max_retries=2, sleep=lambda _: None)

    def reply(self, model=MODEL, finish='stop', reasoning_tokens=None):
        stream = io.BytesIO(json.dumps({'id': 'synthetic-request', 'model': model,
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'cost': .001,
                      'completion_tokens_details': {'reasoning_tokens': reasoning_tokens}},
            'choices': [{'finish_reason': finish, 'message': {'content': '{"criteria":[]}'}}]}).encode())
        stream.headers = {}
        return stream

    def test_request_routes_none_and_audits_usage_without_key(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = self.backend(temp)
            def respond(req, timeout):
                payload = json.loads(req.data)
                self.assertEqual(payload['model'], MODEL)
                self.assertEqual(payload['reasoning'], {'effort': 'none'})
                self.assertEqual(payload['provider'], {'allow_fallbacks': False, 'require_parameters': True})
                self.assertEqual(req.get_header('Authorization'), 'Bearer fake-luna-key')
                return self.reply()
            with patch('baselines.luna_generator.backend.read_api_key', return_value='fake-luna-key'), \
                 patch('urllib.request.urlopen', side_effect=respond):
                result = backend.complete(ChatRequest('s', 'u', 'rar.initial_rubric', {'reasoning': {'effort': 'none'}}))
            self.assertEqual(result.status, 'ok')
            audit = (Path(temp)/'audit.jsonl').read_text()
            self.assertNotIn('fake-luna-key', audit)
            row = json.loads(audit)
            self.assertEqual(row['response_model'], MODEL)
            self.assertEqual(row['cost_usd'], .001)

    def test_wrong_model_or_reasoning_is_not_silently_accepted(self):
        for kwargs in ({'model': 'different-model'}, {'reasoning_tokens': 10}):
            with tempfile.TemporaryDirectory() as temp, \
                 patch('baselines.luna_generator.backend.read_api_key', return_value='fake'), \
                 patch('urllib.request.urlopen', side_effect=lambda *a, **k: self.reply(**kwargs)) as call:
                result = self.backend(temp).complete(ChatRequest('s', 'u', 'generator', {'reasoning': {'effort': 'none'}}))
                self.assertEqual(result.status, 'failed')
                self.assertEqual(call.call_count, 1)

    def test_incomplete_response_bounded_retry(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch('baselines.luna_generator.backend.read_api_key', return_value='fake'), \
             patch('urllib.request.urlopen', side_effect=lambda *a, **k: self.reply(finish='length')) as call:
            result = self.backend(temp).complete(ChatRequest('s', 'u', 'generator', {}))
            self.assertEqual(result.status, 'failed')
            self.assertEqual(call.call_count, 3)
            self.assertEqual(len((Path(temp)/'audit.jsonl').read_text().splitlines()), 3)

    def test_unsupported_parameters_fail_without_fallback(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch('baselines.luna_generator.backend.read_api_key', return_value='fake-key'):
            def reject(req, timeout):
                raise urllib.error.HTTPError(req.full_url, 400, 'unsupported', {}, io.BytesIO(b'unsupported fake-key'))
            with patch('urllib.request.urlopen', side_effect=reject) as call:
                result = self.backend(temp).complete(ChatRequest('s', 'u', 'generator', {}))
            self.assertEqual(result.status, 'failed')
            self.assertEqual(call.call_count, 1)
            self.assertNotIn('fake-key', (Path(temp)/'audit.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
