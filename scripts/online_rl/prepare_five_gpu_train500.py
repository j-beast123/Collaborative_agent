"""Keep the original 487 training cases; replace 13 reserved calibration cases from TRAIN."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.online_rl.cases import load_cases, file_sha256
from src.online_rl.guards import assert_not_protected, protected_ids, protected_question_hashes, question_sha256
from src.online_rl.integrated.episode import schedule
from src.online_rl.integrated.records import atomic_write_json


def prepare(original, source, out, seed=20261008):
    import pyarrow.parquet as pq
    original,source,out=Path(original),Path(source),Path(out)
    base=load_cases(original)
    manifests=[json.loads((ROOT/'data/splits'/name).read_text()) for name in
               ('calibration_only_v1.json','calibration_only_v2_1.json')]
    if any(m['pool_sha256']!=file_sha256(original) for m in manifests):
        raise RuntimeError('Original pool differs from calibration reservation')
    excluded={cid for m in manifests for cid in m['positions_1_indexed'].values()}
    ids=sorted(c['case_id'] for c in base)
    for manifest in manifests:
        for position,cid in manifest['positions_1_indexed'].items():
            if schedule(ids,manifest['schedule_seed'],int(position)-1,1)[0]!=cid:
                raise RuntimeError('Calibration schedule does not match original pool')
    kept=[c for c in base if c['case_id'] not in excluded]
    seen_ids=set(ids)|set().union(*protected_ids().values())
    seen_questions={question_sha256(c['question']) for c in base}|set().union(*protected_question_hashes().values())
    candidates=[];index=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            cid='medmcqa_'+str(row['id']);qh=question_sha256(row['question'])
            if cid not in seen_ids and str(row['id']) not in seen_ids and qh not in seen_questions:
                try: correct='ABCD'[int(row['cop'])]
                except (ValueError,IndexError,TypeError):
                    index+=1;continue
                options={k:row['op'+k.lower()] for k in 'ABCD'}
                if all(isinstance(v,str) and v.strip() for v in options.values()):
                    candidates.append({'case_id':cid,'source_question_id':str(row['id']),
                        'source_dataset':'openlifescienceai/medmcqa','source_split':'train','source_row_index':index,
                        'scenario':'','question':row['question'],'options':options,
                        'ground_truth':{'correct_option':correct,'correct_text':options[correct]},
                        'subject_name':row.get('subject_name'),'topic_name':row.get('topic_name'),
                        'choice_type':row.get('choice_type')})
                    seen_ids.add(cid);seen_questions.add(qh)
            index+=1
    extra=random.Random(seed).sample(candidates,500-len(kept))
    cases=kept+extra
    assert_not_protected(cases,'five-GPU 500-case train pool')
    if len(cases)!=500 or len({c['case_id'] for c in cases})!=500:
        raise RuntimeError('Expected 500 unique cases')
    if {c['case_id'] for c in cases}&excluded:raise RuntimeError('Calibration cases reached train pool')
    order=[schedule(ids,0,i,1)[0] for i in range(len(ids)) if schedule(ids,0,i,1)[0] not in excluded]
    order.extend(c['case_id'] for c in extra)
    out.mkdir(parents=True,exist_ok=True)
    dataset=out/'train500_excluding_calibration_seed20261008.jsonl'
    dataset.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    order_file=out/'train500_order_seed20261008.json'
    atomic_write_json(order_file,order)
    manifest={'original_pool':str(original),'original_sha256':file_sha256(original),
        'source_parquet':str(source),'source_sha256':file_sha256(source),'replacement_seed':seed,
        'kept_original':len(kept),'replacement_cases':len(extra),'total_unique':len(cases),
        'excluded_calibration_case_ids':sorted(excluded),'replacement_case_ids':[c['case_id'] for c in extra],
        'dataset_sha256':file_sha256(dataset),'order_sha256':file_sha256(order_file),
        'order_rule':'Original seeded order excluding calibration; append 13 deterministic TRAIN replacements'}
    atomic_write_json(out/'train500_excluding_calibration.manifest.json',manifest)
    return dataset,order_file,manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original',default='/home/kai/sherpa/data/interim/medmcqa/train_random500_seed20261006.jsonl')
    p.add_argument('--source',default='/home/kai/datasets/medmcqa/source/train.parquet')
    p.add_argument('--out',default=str(ROOT/'data/interim/medmcqa/five_gpu_train500'))
    a=p.parse_args();dataset,order,manifest=prepare(a.original,a.source,a.out)
    print(json.dumps({'dataset':str(dataset),'order':str(order),'total_unique':manifest['total_unique'],
        'kept_original':manifest['kept_original'],'replacement_cases':manifest['replacement_cases']},indent=2))

if __name__=='__main__':main()
