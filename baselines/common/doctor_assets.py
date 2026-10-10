"""Offline Doctor manifest/blob verification. Never loads a model or starts Ollama."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from baselines.common.protocol import settings
EXPECTED=settings()['doctor']['model_digest']
MANIFEST='manifests/registry.ollama.ai/library/gemma3/12b-it-q8_0'


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_manifest(path,expected=EXPECTED):
    raw=Path(path).read_bytes();actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected:raise ValueError(f'Doctor manifest digest differs: expected {expected}, actual {actual}; do not use or download this tag for the experiment')
    value=json.loads(raw);entries={}
    for item in [value['config']]+value['layers']:
        digest=item['digest'];size=item['size']
        if not re.fullmatch(r'sha256:[0-9a-f]{64}',digest) or not isinstance(size,int) or isinstance(size,bool) or size<0:
            raise ValueError('Invalid manifest blob digest or size')
        name='blobs/'+digest.replace(':','-')
        if name in entries and entries[name]!=size:raise ValueError('Conflicting blob sizes')
        entries[name]=size
    return actual,entries


def verify(root,*,manifest=None,manifest_only=False,expected=EXPECTED):
    root=Path(root);actual,entries=read_manifest(manifest or root/MANIFEST,expected)
    if not manifest_only:
        for relative,size in entries.items():
            path=root/relative
            if not path.is_file() or path.stat().st_size!=size:raise ValueError('Missing or incomplete Doctor blob: '+str(path))
            if sha256_file(path)!=Path(relative).name.removeprefix('sha256-'):raise ValueError('Corrupted Doctor blob: '+str(path))
    return {'manifest_sha256':actual,'model':'gemma3:12b-it-q8_0','blob_count':len(entries),
        'blob_bytes':sum(entries.values()),'blob_GiB':sum(entries.values())/1024**3,
        'all_blob_hashes_verified':not manifest_only,'blob_paths':sorted(entries)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',required=True);p.add_argument('--manifest')
    p.add_argument('--manifest-only',action='store_true')
    p.add_argument('--blob-list',help='Write validated relative blob paths for rsync; no models are copied')
    a=p.parse_args()
    result=verify(a.models,manifest=a.manifest,manifest_only=a.manifest_only)
    if a.blob_list:Path(a.blob_list).write_text(''.join(s+'\n' for s in result['blob_paths']))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
