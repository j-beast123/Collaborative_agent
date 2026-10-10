"""Fake-checkpoint CPU tests; never touch real run artifacts or providers."""
import copy
import fcntl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.repairs import luna_temperature as repair
from src.online_rl.integrated.records import atomic_write_json as write, sha256_json


def fixture(root):
    out = root/'run'
    order = root/'order.json'
    write(order, [f'c{i}' for i in range(120)])
    params = {'temperature': .7, 'reasoning': {'effort': 'none'}, 'max_tokens': 4096}
    old = {'output_dir': str(out), 'schedule': {'cases': 120},
        'rubric': {'generation': {'temperature': .7, 'reasoning_effort': 'none'}},
        'five_gpu': {'checkpoint_evaluation_enabled': False, 'train_order_file': str(order)},
        'four_gpu': {'doctor_physical_gpus': ['test_luna0', 'test_luna1'],
                     'assistant_physical_gpus': ['test_luna2', 'test_luna3']},
        'comparison_method': {'source_sha256': repair.BASE_SOURCE, 'method': 'rar',
            'engine': {'generator': {'request_model': repair.MODEL},
                       'grader': {'request_model': 'gpt-5.6-luna'}, 'generator_params': copy.deepcopy(params)},
            'generator_profile': {'execution_source_sha256': copy.deepcopy(repair.OLD_SOURCE),
                                  'generator_params': copy.deepcopy(params)}}}
    fp = sha256_json(old['comparison_method'])
    checkpoint = out/'checkpoints/checkpoint_000000'
    write(checkpoint/'COMPLETE.json', {'completed_cases': 0, 'policy_sha256': 'initial',
        'protocol_sha256': 'protocol', 'experiment_fingerprint': fp})
    (checkpoint/'optimizer.pt').write_bytes(b'synthetic initial optimizer')
    state = {'case_cursor': 0, 'batch_cursor': 0, 'checkpoint': str(checkpoint),
        'policy_sha256': 'initial', 'protocol_sha256': 'protocol',
        'experiment_fingerprint': fp, 'config_sha256': repair.normalized_hash(old)}
    write(out/'resolved_launch_config.json', old)
    write(out/'method_binding.json', old['comparison_method'])
    write(out/'run_state.json', state)
    write(out/'run_manifest.json', {'config': old, 'protocol_sha256': 'protocol'})
    pending = out/'batches/batch_000001'
    write(pending/'manifest.json', {'source_policy_sha256': 'initial', 'case_ids': [f'c{i}' for i in range(5)]})
    for i in range(5):
        child = pending/f'case_{i:06d}'
        config = copy.deepcopy(old); config['output_dir'] = str(child); config['schedule']['cases'] = 1
        write(child/'run_manifest.json', {'config': config, 'config_sha256': sha256_json(config)})
        write(child/'run_state.json', {'config_sha256': sha256_json(config), 'case_cursor': 1,
            'checkpoint': None, 'policy_sha256': 'initial'})
        write(child/'result.json', {'case_id': f'c{i}', 'policy_sha256': 'initial', 'update': {}})
        (child/'calls.jsonl').write_text('synthetic original rollout cache\n')
    return old, repair.apply_profile_config(old)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.old, self.cfg = fixture(Path(self.temp.name)); self.out = Path(self.cfg['output_dir'])
        self.protocol = patch.object(repair, 'protocol_digest', return_value='protocol')
        self.protocol.start(); self.addCleanup(self.protocol.stop)

    def test_preserves_all_five_rollouts_and_optimizer_and_is_idempotent(self):
        _, preserved, _ = repair.plan(self.cfg)
        self.assertEqual(repair.apply(self.cfg)['committed_cases'], 0)
        for name, digest in preserved.items():
            self.assertEqual(repair.sha(self.out/name), digest)
        state = repair.read(self.out/'run_state.json')
        self.assertEqual(state['config_sha256'], repair.normalized_hash(self.cfg))
        self.assertEqual(repair.read(self.out/'method_binding.json'), self.cfg['comparison_method'])
        for i in range(5):
            child = self.out/f'batches/batch_000001/case_{i:06d}'
            man = repair.read(child/'run_manifest.json')
            self.assertEqual(man['config']['comparison_method'], self.cfg['comparison_method'])
            self.assertEqual(repair.read(child/'run_state.json')['config_sha256'], sha256_json(man['config']))
        self.assertTrue(repair.apply(self.cfg)['already_applied'])

    def test_partial_repair_can_resume(self):
        original = repair.write_text_atomic
        calls = []
        def interrupt(path, value):
            if calls: raise OSError('synthetic interruption')
            calls.append(path); original(path, value)
        with patch.object(repair, 'write_text_atomic', side_effect=interrupt):
            with self.assertRaises(OSError): repair.apply(self.cfg)
        self.assertEqual(repair.apply(self.cfg)['committed_cases'], 0)

    def test_rejects_already_trained_or_unrelated_config(self):
        changed = copy.deepcopy(self.cfg); changed['schedule']['cases'] = 240
        with self.assertRaisesRegex(RuntimeError, 'Unrelated config'): repair.plan(changed)
        state = repair.read(self.out/'run_state.json'); state.update(case_cursor=5, batch_cursor=1)
        write(self.out/'run_state.json', state)
        with self.assertRaisesRegex(RuntimeError, 'zero committed'): repair.plan(self.cfg)

    def test_rejects_active_runner_and_successful_rewards(self):
        with (self.out/'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'still running'): repair.apply(self.cfg)
        write(self.out/'rewards/cache/initial_rubrics.jsonl', {'unexpected': 'cached reward'})
        with self.assertRaisesRegex(RuntimeError, 'successful reward cache'): repair.plan(self.cfg)


if __name__ == '__main__':
    unittest.main()
