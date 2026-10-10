"""CPU-only request identity and exact pre-update migration regression tests."""
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.common.backend import AuditedBackend
from baselines.common.local_settings import recipe
from baselines.common.runner import source_hash
from baselines.repairs import generator_id as repair
from src.judges.llm import ChatRequest
from src.online_rl.integrated.records import atomic_write_json, sha256_json


class GeneratorRepairTests(unittest.TestCase):
    def test_exact_requested_returned_model_and_failed_cost_audit(self):
        self.assertEqual(recipe()['rubric']['generator_model'], repair.NEW_MODEL)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'TEST_REWARD_KEY': 'synthetic'}):
            audit = Path(temp) / 'audit.jsonl'
            backend = AuditedBackend(audit_path=audit, base_url='https://openrouter.ai/api/v1',
                model=repair.NEW_MODEL, revision=repair.NEW_MODEL, request_model=repair.NEW_MODEL,
                accepted_response_models=[repair.NEW_MODEL], api_key_env='TEST_REWARD_KEY', max_retries=2)
            for returned in [repair.NEW_MODEL, repair.OLD_MODEL, 'other/gemini-3.8-flash', None]:
                sent = []
                def respond(req, timeout):
                    sent.append(json.loads(req.data))
                    body = {'id': 'synthetic-request', 'model': returned, 'usage': {'cost': .01},
                            'choices': [{'message': {'content': '{}'}}]}
                    stream = io.BytesIO(json.dumps(body).encode()); stream.headers = {}; return stream
                with patch('urllib.request.urlopen', side_effect=respond):
                    result = backend.complete(ChatRequest('system', 'user', 'test', {}))
                self.assertEqual(result.status, 'ok' if returned == repair.NEW_MODEL else 'failed')
                self.assertEqual(len(sent), 1)
                self.assertEqual(sent[0]['model'], repair.NEW_MODEL)
                self.assertFalse(sent[0]['provider']['allow_fallbacks'])
            rows = [json.loads(s) for s in audit.read_text().splitlines()]
            self.assertEqual(sum(r['cost_usd'] for r in rows), .04)
            self.assertTrue(all(r['request_id'] == 'synthetic-request' for r in rows))
            self.assertNotIn('TEST_REWARD_KEY', audit.read_text())

    def fixture(self, root):
        old = {'output_dir': str(root), 'rubric': {'generator_model': repair.OLD_MODEL},
            'five_gpu': {'checkpoint_evaluation_enabled': False},
            'four_gpu': {'doctor_physical_gpus': ['90', '91'], 'assistant_physical_gpus': ['92', '93']},
            'optimizer': {'lr': 1e-5},
            'comparison_method': {'source_sha256': repair.OLD_SOURCE, 'method': 'rar',
                'engine': {'generator': {'model': repair.OLD_MODEL, 'revision': repair.OLD_MODEL,
                    'request_model': repair.OLD_MODEL, 'accepted_response_models': [repair.OLD_MODEL]}}}}
        checkpoint = root / 'checkpoints/checkpoint_000000'
        state = {'case_cursor': 0, 'batch_cursor': 0, 'policy_sha256': 'initial',
            'theta0_policy_sha256': 'initial', 'config_sha256': repair.normalized_hash(old),
            'protocol_sha256': 'dialogue', 'checkpoint': str(checkpoint),
            'experiment_fingerprint': sha256_json(old['comparison_method'])}
        files = {'resolved_launch_config.json': old, 'method_binding.json': old['comparison_method'],
            'run_state.json': state, 'run_manifest.json': {'config': old, 'protocol_sha256': 'dialogue'},
            'initial_policy.json': {'sha': 'initial'},
            'checkpoints/checkpoint_000000/COMPLETE.json': {'completed_cases': 0,
                'experiment_fingerprint': state['experiment_fingerprint'], 'policy_sha256': 'initial',
                'protocol_sha256': 'dialogue'},
            'batches/batch_000001/manifest.json': {'source_policy_sha256': 'initial', 'case_ids': list(range(5))}}
        for index in range(5):
            prefix = f'batches/batch_000001/case_{index:06d}/'
            child = copy.deepcopy(old)
            child['output_dir'] = str(root / prefix)
            files[prefix + 'run_state.json'] = {'case_cursor': 1, 'checkpoint': None,
                'config_sha256': sha256_json(child), 'policy_sha256': 'initial'}
            files[prefix + 'run_manifest.json'] = {'config': child, 'config_sha256': sha256_json(child)}
            for name in ('result.json', 'cases/case_000000.json'):
                files[prefix + name] = {'case_id': index, 'policy_sha256': 'initial', 'update': {'optimizer_step': False}}
        for name, value in files.items(): atomic_write_json(root / name, value)
        (checkpoint / 'adapter.bin').write_bytes(b'immutable initial weights')
        (root / 'reward_api_attempts.jsonl').write_text(json.dumps({'status': 'failed',
            'requested_model': repair.OLD_MODEL, 'response_model': repair.NEW_MODEL,
            'error': 'Returned model does not match the configured accepted model IDs'}) + '\n')
        return repair.convert(old)

    def test_migration_is_idempotent_preserves_trees_and_resumes_interruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = self.fixture(root)
            changes, preserved = repair.plan(cfg)
            self.assertEqual(len(changes), 15)
            native = repair.atomic_write_json
            writes = []
            def interrupt(path, value):
                native(path, value); writes.append(path)
                if len(writes) == 4: raise OSError('synthetic interruption')
            with patch.object(repair, 'source_hash', return_value=repair.NEW_SOURCE):
                with patch.object(repair, 'atomic_write_json', side_effect=interrupt):
                    with self.assertRaises(OSError): repair.apply(cfg)
                result = repair.apply(cfg)
                self.assertEqual(result, repair.apply(cfg))
            self.assertEqual(result['saved_trees'], 5)
            for name, digest in preserved.items(): self.assertEqual(repair.sha(root / name), digest)
            state = repair.read(root / 'run_state.json')
            self.assertEqual(state['config_sha256'], repair.normalized_hash(cfg))
            self.assertEqual(state['experiment_fingerprint'], sha256_json(cfg['comparison_method']))
            for path in root.glob('batches/*/case_*/run_manifest.json'):
                man = repair.read(path); childstate = repair.read(path.with_name('run_state.json'))
                self.assertEqual(childstate['config_sha256'], sha256_json(man['config']))
                self.assertEqual(man['config_sha256'], childstate['config_sha256'])

    def test_unrelated_changes_and_updated_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = self.fixture(root)
            changed = copy.deepcopy(cfg); changed['optimizer']['lr'] = .1
            with self.assertRaisesRegex(RuntimeError, 'beyond'): repair.plan(changed)
            state = repair.read(root / 'run_state.json'); state['case_cursor'] = 5
            atomic_write_json(root / 'run_state.json', state)
            with self.assertRaisesRegex(RuntimeError, 'advanced'): repair.plan(cfg)

    def test_source_lock_rejects_unreviewed_code(self):
        with patch.object(repair, 'source_hash', return_value='unreviewed'):
            with self.assertRaisesRegex(RuntimeError, 'Source differs'):
                repair.apply({})


if __name__ == '__main__': unittest.main()
