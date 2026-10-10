"""CPU fixtures for the post-transition response parser repair."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from baselines.parallel_api.test_transition import fixture
from baselines.parallel_api.openai_profile import apply_profile_config
from baselines.repairs import json_response as repair
from src.online_rl.integrated.records import atomic_write_json as write, sha256_json


class ParserRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        old, _ = fixture(Path(self.temp.name)); self.out = Path(old['output_dir'])
        def previous(cfg):
            cfg = apply_profile_config(cfg)
            hashes = cfg['comparison_method']['api_profile']['execution_source_sha256']
            hashes['engine.py'] = repair.OLD_ENGINE; hashes.pop('json_response.py')
            return cfg
        self.old = previous(old); self.cfg = apply_profile_config(self.old)
        fp = sha256_json(self.old['comparison_method'])
        write(self.out/'resolved_launch_config.json', self.old)
        write(self.out/'method_binding.json', self.old['comparison_method'])
        man = repair.read(self.out/'run_manifest.json'); man['config'] = self.old; write(self.out/'run_manifest.json', man)
        state = repair.read(self.out/'run_state.json'); state.update(config_sha256=repair.normalized_hash(self.old), experiment_fingerprint=fp)
        write(self.out/'run_state.json', state)
        proofpath = Path(state['checkpoint'])/'COMPLETE.json'
        proof = repair.read(proofpath); proof['experiment_fingerprint'] = fp; write(proofpath, proof)
        for path in self.out.glob('batches/batch_000002/case_*/run_manifest.json'):
            man = repair.read(path); man['config'] = previous(man['config']); man['config_sha256'] = sha256_json(man['config']); write(path, man)
            sp = path.parent/'run_state.json'; cs = repair.read(sp); cs['config_sha256'] = man['config_sha256']; write(sp, cs)
        pp = patch.object(repair, 'protocol_digest', return_value='protocol'); pp.start(); self.addCleanup(pp.stop)

    def test_exact_source_upgrade_preserves_progress_and_weights(self):
        edits, preserved, state = repair.plan(self.cfg)
        repair.apply(self.cfg)
        for name, digest in preserved.items(): self.assertEqual(repair.sha(self.out/name), digest)
        self.assertEqual(repair.read(self.out/'method_binding.json'), self.cfg['comparison_method'])
        self.assertEqual(repair.read(self.out/'run_state.json')['case_cursor'], 5)

    def test_unrelated_profile_and_source_changes_rejected(self):
        cfg = copy.deepcopy(self.cfg); cfg['comparison_method']['api_profile']['execution_source_sha256']['backend.py'] = 'other'
        with self.assertRaisesRegex(RuntimeError, 'beyond the reviewed'): repair.plan(cfg)
        cfg = copy.deepcopy(self.cfg); cfg['comparison_method']['engine']['grader_params']['reasoning_effort'] = 'none'
        with self.assertRaisesRegex(RuntimeError, 'Unrelated config'): repair.plan(cfg)

    def test_interrupted_repair_replays_journal(self):
        original = repair.write_text_atomic; seen = []
        def stop(path, text):
            if seen: raise OSError('interruption')
            seen.append(path); original(path, text)
        with patch.object(repair, 'write_text_atomic', side_effect=stop):
            with self.assertRaises(OSError): repair.apply(self.cfg)
        self.assertEqual(repair.apply(self.cfg)['committed_cases'], 5)


if __name__ == '__main__': unittest.main()
