"""CPU-only synthetic bytes; no registry requests or model downloads."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from baselines.common.doctor_assets import verify,MANIFEST


class AssetTests(unittest.TestCase):
    def fixture(self,root):
        content=b'synthetic blob';digest=hashlib.sha256(content).hexdigest()
        path=Path(root)/'blobs'/('sha256-'+digest);path.parent.mkdir();path.write_bytes(content)
        entry={'digest':'sha256:'+digest,'size':len(content)}
        manifest=Path(root)/MANIFEST;manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'config':entry,'layers':[entry]}))
        return hashlib.sha256(manifest.read_bytes()).hexdigest(),path
    def test_full_check_and_deduplicated_size(self):
        with tempfile.TemporaryDirectory() as root:
            expected,path=self.fixture(root);r=verify(root,expected=expected)
            self.assertTrue(r['all_blob_hashes_verified']);self.assertEqual(r['blob_count'],1)
            self.assertEqual(r['blob_bytes'],path.stat().st_size)
    def test_manifest_mismatch_rejected_before_blobs(self):
        with tempfile.TemporaryDirectory() as root:
            expected,path=self.fixture(root);path.unlink()
            with self.assertRaisesRegex(ValueError,'manifest digest differs'):verify(root,expected='0'*64)
            self.assertFalse(verify(root,expected=expected,manifest_only=True)['all_blob_hashes_verified'])
    def test_same_size_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            expected,path=self.fixture(root);path.write_bytes(b'X'*path.stat().st_size)
            with self.assertRaisesRegex(ValueError,'Corrupted'):verify(root,expected=expected)


if __name__=='__main__':unittest.main()
