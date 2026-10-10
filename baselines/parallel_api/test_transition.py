"""CPU-only transition tests, using disposable fake checkpoints and no credentials."""
import copy
import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.parallel_api import transition as t
from baselines.parallel_api.openai_profile import apply_profile_config
from src.online_rl.integrated.records import atomic_write_json as write, sha256_json


def fixture(root):
    out = root/'run'; out.mkdir()
    write(root/'order.json', [f'c{i}' for i in range(120)])
    old = {'output_dir': str(out), 'schedule': {'cases': 120}, 'judges': {},
        'five_gpu': {'checkpoint_evaluation_enabled': False, 'train_order_file': str(root/'order.json')},
        'four_gpu': {'doctor_physical_gpus': ['test_openai_0', 'test_openai_1'],
                     'assistant_physical_gpus': ['test_openai_2', 'test_openai_3']},
        'comparison_method': {'source_sha256': t.BASE_SOURCE, 'method': 'rar', 'method_spec': {},
            'engine': {'grader': {'request_model': 'openai/gpt-5.6-luna'}, 'grader_params': {'temperature': 0}},
            'doctor_identity': {'digest': 'fixed'}}}
    fp = sha256_json(old['comparison_method']); ckpt = out/'checkpoints/recovery_000001'
    write(ckpt/'COMPLETE.json', {'completed_cases': 5, 'policy_sha256': 'trained',
        'experiment_fingerprint': fp, 'protocol_sha256': 'protocol', 'update': {'optimizer_steps': 1}})
    (ckpt/'optimizer.pt').write_bytes(b'unchanged optimizer and trained weights')
    state = {'case_cursor': 5, 'batch_cursor': 1, 'checkpoint': str(ckpt), 'policy_sha256': 'trained',
        'theta0_policy_sha256': 'initial', 'protocol_sha256': 'protocol',
        'experiment_fingerprint': fp, 'config_sha256': t.normalized_hash(old)}
    write(out/'resolved_launch_config.json', old); write(out/'method_binding.json', old['comparison_method'])
    write(out/'run_manifest.json', {'config': old, 'protocol_sha256': 'protocol'})
    write(out/'run_state.json', state)
    for i in range(5): write(out/f'cases/case_{i:06d}.json', {'historical': i})
    write(out/'batches/batch_000001/manifest.json', {'old_history': True})
    pending = out/'batches/batch_000002'
    write(pending/'manifest.json', {'source_policy_sha256': 'trained', 'case_ids': [f'c{i}' for i in range(5,10)]})
    # One finished tree and one interrupted tree. Both can be resumed unchanged.
    for i in (5, 6):
        case = pending/f'case_{i:06d}'; cfg = copy.deepcopy(old); cfg['output_dir'] = str(case); cfg['schedule']['cases'] = 1
        write(case/'run_manifest.json', {'config': cfg, 'config_sha256': sha256_json(cfg)})
        write(case/'run_state.json', {'config_sha256': sha256_json(cfg), 'case_cursor': int(i==5),
            'checkpoint': None, 'policy_sha256': 'trained'})
        if i == 5:
            record = {'case_id': 'c5', 'policy_sha256': 'trained', 'update': {'optimizer_step': False}}
            write(case/'result.json', record); write(case/'cases/case_000000.json', record)
        (case/'calls.jsonl').write_text('cached rollout\n')
    (out/'rewards').mkdir(); (out/'rewards/cache.jsonl').write_text('old provider cache remains\n')
    return old, apply_profile_config(old)


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.old, self.cfg = fixture(Path(self.temp.name)); self.out = Path(self.cfg['output_dir'])
        self.protocol = patch.object(t, 'protocol_digest', return_value='protocol'); self.protocol.start(); self.addCleanup(self.protocol.stop)

    def test_checkpoint_progress_cache_and_history_preserved(self):
        edits, preserved, state = t.plan(self.cfg)
        self.assertEqual(state['case_cursor'], 5)
        result = t.apply(self.cfg)
        self.assertEqual(result['committed_cases'], 5)
        for name, digest in preserved.items(): self.assertEqual(t.sha(self.out/name), digest)
        saved = t.read(self.out/'run_state.json')
        self.assertEqual(saved['policy_sha256'], 'trained'); self.assertEqual(saved['batch_cursor'], 1)
        self.assertEqual(saved['config_sha256'], t.normalized_hash(self.cfg))
        self.assertEqual(t.read(self.out/'method_binding.json'), self.cfg['comparison_method'])
        child = self.out/'batches/batch_000002/case_000005'
        man = t.read(child/'run_manifest.json')
        self.assertEqual(t.read(child/'run_state.json')['config_sha256'], sha256_json(man['config']))
        self.assertEqual(man['config']['comparison_method'], self.cfg['comparison_method'])
        self.assertTrue(t.apply(self.cfg)['already_applied'])

    def test_interrupted_transition_resumes_from_journal(self):
        original = t.write_text_atomic; calls = []
        def stop_after_one(path, value):
            if calls: raise OSError('simulated interruption')
            calls.append(path); original(path, value)
        with patch.object(t, 'write_text_atomic', side_effect=stop_after_one):
            with self.assertRaises(OSError): t.apply(self.cfg)
        self.assertFalse((self.out/'repairs'/t.NAME/'COMPLETE.json').exists())
        self.assertEqual(t.apply(self.cfg)['committed_cases'], 5)

    def test_rejects_unrelated_change_and_live_lock(self):
        bad = copy.deepcopy(self.cfg); bad['schedule']['cases'] = 240
        with self.assertRaisesRegex(RuntimeError, 'Unrelated config'): t.plan(bad)
        with (self.out/'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'still running'): t.apply(self.cfg)
        self.assertFalse((self.out/'repairs').exists())

    def test_rejects_uncommitted_and_partial_checkpoints(self):
        orphan = self.out/'checkpoints/recovery_000002'
        write(orphan/'COMPLETE.json', {'completed_cases': 10})
        with self.assertRaisesRegex(RuntimeError, 'Uncommitted checkpoint'): t.plan(self.cfg)
        orphan.rename(orphan.with_name(orphan.name+'.pending'))
        with self.assertRaisesRegex(RuntimeError, 'Incomplete checkpoint'): t.plan(self.cfg)

    def test_changed_artifact_blocks_partial_transition_resume(self):
        with patch.object(t, 'write_text_atomic', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError): t.apply(self.cfg)
        (self.out/'rewards/cache.jsonl').write_text('unexpected change')
        with self.assertRaisesRegex(RuntimeError, 'Preserved artifact changed'): t.apply(self.cfg)


if __name__ == '__main__': unittest.main()
