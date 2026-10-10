import json
import tempfile
import unittest
from pathlib import Path
from baselines.parallel_api.json_response import decode_json_response
from baselines.parallel_api.engine import ParallelRubricEngine
from baselines.common.rubrics import INITIAL, criteria
from src.judges.llm import ChatResult


GOOD = {'criteria': [{'text': 'One valid criterion', 'category': 'Essential'}]}


class Responses:
    def __init__(self, texts): self.texts = iter(texts); self.requests = []
    def identity(self): return {'model': 'test', 'revision': 'test'}
    def complete(self, request):
        self.requests.append(request)
        return ChatResult('ok', next(self.texts), 'test', 'test', 'test', {'completion_tokens': 10}, 0, 1)


class JSONResponseTests(unittest.TestCase):
    def test_literal_wrappers_preserve_values(self):
        text = json.dumps(GOOD)
        for body in (text, '```json\n'+text+'\n```', "cat << 'EOF'\n"+text+'\nEOF'):
            value, _ = decode_json_response(body)
            self.assertEqual(value, GOOD); criteria(value)

    def test_rejects_prose_extra_commands_multiple_or_truncated_json(self):
        text = json.dumps(GOOD)
        for body in ('', 'Explanation: '+text, text+'\n{}', text[:-1],
                     "cat << 'EOF'\n"+text+'\nEOF\necho done', '```json\n'+text):
            with self.assertRaises(ValueError): decode_json_response(body)

    def run_call(self, backend, out):
        engine = ParallelRubricEngine('rar', backend, backend, out, weights={'w_outcome': 0., 'w_rubric': 1.})
        def call(): return engine.call('initial_rubric', INITIAL, {'case': 'test'}, criteria, generator=True, initial=True)
        return engine, call

    def test_format_retry_audit_same_request_and_success_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); backend = Responses(['', '{broken', "cat << 'EOF'\n"+json.dumps(GOOD)+'\nEOF'])
            engine, call = self.run_call(backend, out)
            self.assertEqual(call(), GOOD); self.assertEqual(call(), GOOD)
            self.assertEqual(len(backend.requests), 3)
            self.assertTrue(all(r == backend.requests[0] for r in backend.requests))
            rows = [json.loads(x) for x in (out/'judge_results.jsonl').read_text().splitlines()]
            self.assertEqual([r['validation_status'] for r in rows], ['invalid_json', 'invalid_json', 'valid'])
            self.assertEqual(rows[-1]['json_normalization'], 'outer_literal_heredoc_removed')
            self.assertEqual(len(engine.initial_cache.store), 1)

    def test_exhaustion_never_caches_invalid_response(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = Responses(['not JSON']*3); engine, call = self.run_call(backend, Path(temp))
            with self.assertRaisesRegex(RuntimeError, 'invalid JSON after 3'): call()
            self.assertEqual(len(backend.requests), 3)
            self.assertFalse(engine.initial_cache.store)

    def test_wrapped_invalid_schema_still_fails_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = Responses(['```json\n{"criteria": []}\n```'])
            engine, call = self.run_call(backend, Path(temp))
            with self.assertRaisesRegex(ValueError, 'empty'): call()
            self.assertEqual(len(backend.requests), 1)
            self.assertFalse(engine.initial_cache.store)


if __name__ == '__main__': unittest.main()
