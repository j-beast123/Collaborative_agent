"""Train-only MedMCQA pools and deterministic mixed-case selection."""
import copy
import json
import random
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

from src.online_rl.cases import load_cases, file_sha256
from src.online_rl.guards import protected_ids, protected_question_hashes, question_sha256, assert_not_protected
from src.online_rl.integrated.records import atomic_write_json, read_json, sha256_json


def medmcqa_case(row, index):
    label = 'ABCD'[int(row['cop'])]
    options = {letter: row[key] for letter, key in zip('ABCD', ('opa','opb','opc','opd'))}
    return {'case_id':'medmcqa_'+str(row['id']), 'source_question_id':str(row['id']),
            'source_dataset':'openlifescienceai/medmcqa', 'source_split':'train', 'source_row_index':index,
            'scenario':'', 'question':row['question'], 'options':options,
            'ground_truth':{'correct_option':label,'correct_text':options[label]},
            'subject_name':row.get('subject_name'), 'topic_name':row.get('topic_name'), 'choice_type':row.get('choice_type')}


def excluded(template, root):
    current = load_cases(Path(root)/template['data']['train'])
    ids = set().union(*protected_ids().values()); hashes = set().union(*protected_question_hashes().values())
    ids.update(template['data'].get('excluded_calibration_case_ids', []))
    for case in current:
        ids.update((case['case_id'], str(case.get('source_question_id')))); hashes.add(question_sha256(case['question']))
    return ids, hashes


def prepare_pool(task, template, root, *, source=None):
    home=Path(root)/task['output_dir']; home.mkdir(parents=True, exist_ok=True)
    identity={'pool':{key:task[key] for key in ('dataset','pool_size','seed')},'template_excluded_train_sha256':file_sha256(Path(root)/template['data']['train']),
              'protected_manifest_sha256':file_sha256(Path(root)/'data/splits/protected_eval_v1.json'),
              'excluded_calibration_ids':template['data'].get('excluded_calibration_case_ids',[]),
              'local_source_sha256':file_sha256(source) if source else None}
    path=home/'pool.jsonl'; manifest=home/'pool_manifest.json'
    if manifest.exists():
        previous=read_json(manifest)
        if 'task' in previous['identity']:
            legacy = previous['identity'].pop('task')
            previous['identity']['pool'] = {key:legacy[key] for key in ('dataset','pool_size','seed')}
        if previous['identity']!=identity or file_sha256(path)!=previous['pool_sha256']:
            raise RuntimeError('Pool inputs changed; choose a new Plan B output directory')
        atomic_write_json(manifest,previous)
        return load_cases(path)
    if source:
        rows=load_cases(source)
        if any(row.get('source_split')!='train' for row in rows): raise ValueError('Local source must explicitly contain train rows')
    else:
        url='https://datasets-server.huggingface.co/parquet?'+urllib.parse.urlencode({'dataset':task['dataset']})
        with urllib.request.urlopen(url,timeout=60) as response: metadata=json.load(response)
        shards=[r for r in metadata['parquet_files'] if r['split']=='train' and r['config']=='default']
        if not shards: raise RuntimeError('No MedMCQA train parquet shards found')
        import pyarrow.parquet as pq
        rows=[]; offset=0
        for index,shard in enumerate(shards):
            local=home/'source_cache'/f'train_{index:04d}.parquet'; local.parent.mkdir(exist_ok=True)
            if not local.exists():
                tmp=local.with_suffix('.parquet.tmp')
                with urllib.request.urlopen(shard['url'],timeout=60) as response,tmp.open('wb') as stream: shutil.copyfileobj(response,stream)
                tmp.replace(local)
            fields=['id','question','opa','opb','opc','opd','cop','subject_name','topic_name','choice_type']
            for batch in pq.ParquetFile(local).iter_batches(columns=fields,batch_size=4096):
                for raw in batch.to_pylist(): rows.append(medmcqa_case(raw,offset)); offset+=1
    ids,hashes=excluded(template,root); random.Random(task['seed']).shuffle(rows); selected=[]
    for row in rows:
        key=row['case_id']; source_id=str(row.get('source_question_id')); qhash=question_sha256(row['question'])
        if key in ids or source_id in ids or qhash in hashes: continue
        if row.get('source_split')!='train': raise ValueError('Plan B uses only the train split')
        ids.update((key,source_id)); hashes.add(qhash); selected.append(row)
        if len(selected)==task['pool_size']: break
    if len(selected)<task['pool_size']: raise RuntimeError(f'Only {len(selected)} eligible unique train rows; need {task["pool_size"]}')
    assert_not_protected(selected,'Plan B pool')
    tmp=path.with_suffix('.jsonl.tmp'); tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)); tmp.replace(path)
    atomic_write_json(manifest,{'identity':identity,'pool_sha256':file_sha256(path),'count':len(selected),
                               'train_only':True,'current_training_excluded':True})
    return selected


def screening_identity(task, template, root, doctor):
    home=Path(root)/task['output_dir']
    return {'pool_sha256':file_sha256(home/'pool.jsonl'),'doctor':doctor.protocol,'rollout':template['rollout'],
            'base_model':template['policy']['base_model'],
            'agent_prompt_sha256':file_sha256(Path(root)/template['rollout'].get('agent_system_prompt_file','prompts/agent.txt')),
            'engine':'vllm','base_no_lora':True,'screen_rollouts':8,'seed':task['seed'],
            'min_successes':task['min_successes'],'max_successes':8-task['min_successes']}


def finalize_dataset(task, template, root, identity):
    home=Path(root)/task['output_dir']; pool=load_cases(home/'pool.jsonl'); eligible=[]
    for case in pool:
        path=home/'screen'/case['case_id']/'outcomes.json'
        if not path.exists(): continue
        row=read_json(path)
        if row['identity']!=identity: raise RuntimeError('Cached screening protocol changed')
        if row['accepted']: eligible.append(case)
    if task['selection']=='random120' and any(not (home/'screen'/c['case_id']/'outcomes.json').exists() for c in pool):
        raise RuntimeError('random120 requires screening the entire fixed pool')
    if len(eligible)<120: raise RuntimeError(f'Only {len(eligible)} mixed cases found; need 120')
    selected=eligible[:120] if task['selection']=='first120' else random.Random(task['seed']+1).sample(eligible,120)
    if task['selection']=='first120':
        last=max(pool.index(row) for row in selected)
        if any(not (home/'screen'/c['case_id']/'outcomes.json').exists() for c in pool[:last+1]):
            raise RuntimeError('Complete screening the fixed pool prefix before selecting first120')
    assert_not_protected(selected,'Plan B selected training set'); ids=[c['case_id'] for c in selected]
    manifest=home/'selection.json'
    if manifest.exists() and read_json(manifest)['case_ids']!=ids: raise RuntimeError('Pinned Plan B selection changed')
    dataset=home/'train120.jsonl'; dataset.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    order=ids[:]; random.Random(task['seed']+2).shuffle(order); order_file=home/'train120_order.json'; atomic_write_json(order_file,order)
    prompt_ids=random.Random(task['seed']+3).sample(ids,24); ids_file=home/'prompt24_case_ids.json'; atomic_write_json(ids_file,prompt_ids)
    openings={key:read_json(home/'screen'/key/'outcomes.json')['doctor_opening'] for key in prompt_ids}
    openings_file=home/'prompt24_openings.json'; atomic_write_json(openings_file,openings)
    selection={'identity':identity,'case_ids':ids,'prompt_case_ids':prompt_ids,'eligible_cases':len(eligible),
               'selection':task['selection'],'dataset_sha256':file_sha256(dataset),'screen_rollouts_not_used_for_auc':True}
    atomic_write_json(manifest,selection)
    cfg=copy.deepcopy(template); cfg['output_dir']=str(home/'training')
    cfg['policy']['init_adapter'] = None
    cfg['checkpoint_upload']['path_prefix'] = 'ours/plan_b/' + home.name
    cfg['data'].update(train=str(dataset),train_sha256=file_sha256(dataset),train_manifest=str(manifest))
    cfg['schedule'].update(order_file=str(order_file),order_sha256=file_sha256(order_file),seed=task['seed'])
    cfg['prompt_optimization'].update(output_dir=str(home/'prompt_optimization'),final_artifact=str(home/'prompt_optimization/final_prompt.json'),
        mixed_cases=24,optimization_case_ids_file=str(ids_file),optimization_case_ids_sha256=file_sha256(ids_file),
        fixed_openings_file=str(openings_file),fixed_openings_sha256=file_sha256(openings_file),
        base_collection_identity={'screening':sha256_json(identity),'selected24':file_sha256(ids_file),
                                  'fixed_openings':file_sha256(openings_file),'fresh_rollouts':32})
    for name in ('ours_plan_b.json','prompt_plan_b.json'): atomic_write_json(home/name,cfg)
    return cfg
