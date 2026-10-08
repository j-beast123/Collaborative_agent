"""Normalize all 622 local official English CasiMedicos-Exp cases, without sampling."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def prepare(source,out):
    rows=[];sources={};counts={};seen=set()
    for split,filename,n in (('train','en_train_casimedicos.jsonl',434),('validation','en_dev_casimedicos.jsonl',63),('test','en_test_casimedicos.jsonl',125)):
        path=source/filename
        raw=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(raw)!=n:raise ValueError(f'{split}: expected {n} source rows, found {len(raw)}')
        counts[split]=len(raw);sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for r in raw:
            numbered=sorted((int(k),str(v).strip()) for k,v in r['options'].items() if v is not None and str(v).strip().casefold() not in ('','nan','none','null'))
            if not 2<=len(numbered)<=5:raise ValueError('Invalid answer choices')
            labels={k:'ABCDE'[i] for i,(k,v) in enumerate(numbered)}
            options={labels[k]:v for k,v in numbered}
            gold=labels[int(r['correct_option'])]
            cid=f'casimedicos_en_{split}_{r["id"]}_{r["question_id_specific"]}'
            if cid in seen:raise ValueError('Duplicate CasiMedicos ID')
            seen.add(cid)
            rows.append({'case_id':cid,'source_dataset':'HiTZ/casimedicos-exp','source_split':split,
                         'source_question_id':f'{r["id"]}_{r["question_id_specific"]}',
                         'language':'en','question':r['full_question'],'scenario':'','options':options,
                         'ground_truth':{'correct_option':gold,'correct_text':options[gold]},'subject':r.get('type')})
    if len(rows)!=622:raise ValueError('Expected all 622 English cases')
    out.mkdir(parents=True,exist_ok=True)
    dest=out/'all_en_622.jsonl'
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if dest.exists() and dest.read_text()!=data:raise ValueError('Refusing to replace a different normalized dataset')
    dest.write_text(data)
    manifest={'source_dataset':'HiTZ/casimedicos-exp','language':'en','source_sha256':sources,'source_counts':counts,
              'normalization':'number-sorted nonempty choices mapped to A..E; correct_option mapped by original number',
              'output_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'cases':622,'selection':'all English splits, no sampling'}
    (out/'all_en_622.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    return dest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/home/kai/datasets/casimedicos-exp/source/data/en'))
    p.add_argument('--out',type=Path,default=ROOT/'data/interim/casimedicos')
    a=p.parse_args();prepare(a.source,a.out)
