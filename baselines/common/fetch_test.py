"""Restore pinned English Arg + companion data and the exact SHERPA test bytes."""
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from scripts.prepare_casimedicos_arg import ARG_REPO, ARG_REVISION, EXP_REPO, EXP_REVISION, SPLIT_FILES, load_split, sha256
from scripts.online_rl.prepare_casimedicos622 import prepare
from baselines.common.protocol import ROOT, settings
from baselines.collabllm.data import audit_data


def main():
    raw = ROOT / 'data/raw/casimedicos_arg'
    raw.mkdir(parents=True,exist_ok=True)
    url = 'https://datasets-server.huggingface.co/splits?dataset=HiTZ%2Fcasimedicos-arg'
    with urllib.request.urlopen(url,timeout=30) as response:
        metadata=json.load(response)
    english=[r for r in metadata['splits'] if r['config']=='en']
    if {r['split'] for r in english}!={'train','validation','test'}:
        raise ValueError('Unexpected English split metadata')
    (raw/'viewer_splits.json').write_text(json.dumps(metadata,indent=2)+'\n')
    verified={}
    # Existing loader checks exact ID coverage and preserves companion source order.
    companion=ROOT/'data/raw/casimedicos_exp/en'
    companion.mkdir(parents=True,exist_ok=True)
    for split, filenames in SPLIT_FILES.items():
        rows, detail=load_split(split,raw,reuse_downloads=True)
        arg_rows=json.loads((raw/f'casimedicos_arg_en_{split}_ordered.json').read_text())
        arg_ids=[str(r['id']) for r in arg_rows]
        if len(arg_ids)!=len(set(arg_ids)) or len(arg_ids)!=len(rows):
            raise ValueError('Duplicate/missing original Arg IDs')
        source=raw/f'casimedicos_exp_en_{split}.jsonl'
        destination=companion/Path(filenames['companion']).name
        if destination.exists() and destination.read_bytes()!=source.read_bytes():
            raise ValueError('Refusing to overwrite different companion source bytes')
        if not destination.exists():shutil.copyfile(source,destination)
        verified[split]=detail
    output=prepare(companion,ROOT/'data/interim/casimedicos')
    actual=sha256(output); expected=settings()['test_sha256']
    if actual!=expected:
        raise ValueError(f'Restored test hash differs from SHERPA: {actual} != {expected}')
    receipt={'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'requested_dataset':ARG_REPO,'arg_revision':ARG_REVISION,
        'companion_dataset':EXP_REPO,'companion_revision':EXP_REVISION,
        'splits':verified,'rows':622,'output':str(output.relative_to(ROOT)),
        'output_sha256':actual,'expected_sha256':expected,'exact_original_hash_match':True,
        'order':'train then validation then test; unchanged companion rows within each split',
        'annotation_use':'Raw Arg annotations retained separately; not exposed to Agent or Doctor',
        'test_evaluation_performed':False}
    (ROOT/'baselines/common/test_download_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    audit=audit_data()
    (ROOT/'baselines/collabllm/data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if not audit['all_verified']:raise ValueError(audit['errors'])
    print(json.dumps({'rows':622,'sha256':actual,'all_data_verified':True,'output':str(output)},indent=2))


if __name__=='__main__':main()
