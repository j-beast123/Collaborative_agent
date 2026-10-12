import copy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.online_rl.sherpa import plan_b, prompt_search
from src.online_rl.sherpa.rollouts import schedule
from src.online_rl.sherpa.config import config_problems
from src.online_rl.cases import load_cases
from src.online_rl.integrated.records import atomic_write_json
from scripts.online_rl import run_plan_b

ROOT = Path.cwd()


def case(index, question=None):
    return {'case_id': f'medmcqa_plan_b_test_{index}', 'source_question_id': f'plan_b_test_{index}',
            'source_split': 'train', 'source_dataset': 'openlifescienceai/medmcqa',
            'question': question or f'Unique Plan B synthetic question {index}',
            'options': {'A':'one','B':'two'}, 'ground_truth': {'correct_option':'A'}}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_hf_zero_based_labels_and_no_explanation_leak():
    raw = {'id':'uuid','question':'Question','cop':2,'opa':'A text','opb':'B text','opc':'C text','opd':'D text',
           'exp':'Hidden answer explanation'}
    row = plan_b.medmcqa_case(raw,42)
    assert row['ground_truth'] == {'correct_option':'C','correct_text':'C text'}
    assert row['source_split'] == 'train' and row['source_row_index'] == 42
    assert 'exp' not in row


def test_pool_excludes_current_protected_and_duplicate_questions(tmp_path,monkeypatch):
    write_rows(tmp_path/'old.jsonl',[case(0)])
    write_rows(tmp_path/'source.jsonl',[case(i) for i in range(8)]+[case(20,case(0)['question']),case(21,case(1)['question'])])
    atomic_write_json(tmp_path/'data/splits/protected_eval_v1.json',{})
    monkeypatch.setattr(plan_b,'protected_ids',lambda: {'eval':{case(2)['case_id']}})
    monkeypatch.setattr(plan_b,'protected_question_hashes',lambda: {'eval':{plan_b.question_sha256(case(3)['question'])}})
    task={'output_dir':'new','dataset':'openlifescienceai/medmcqa','seed':7,'pool_size':3}
    template={'data':{'train':'old.jsonl'}}
    rows=plan_b.prepare_pool(task,template,tmp_path,source=tmp_path/'source.jsonl')
    assert not {case(i)['case_id'] for i in (0,2,3)} & {row['case_id'] for row in rows}
    assert all(row['question']!=case(0)['question'] for row in rows)
    assert len({row['question'] for row in rows})==3
    assert plan_b.prepare_pool(task,template,tmp_path,source=tmp_path/'source.jsonl')==rows
    with (tmp_path/'new/pool.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(RuntimeError,match='Pool inputs changed'):
        plan_b.prepare_pool(task,template,tmp_path,source=tmp_path/'source.jsonl')


@pytest.mark.parametrize('selection',['first120','random120'])
def test_training120_prompt24_and_only24_new_rollout_cases(tmp_path,monkeypatch,selection):
    template=json.loads((ROOT/'configs/ours.json').read_text()); original=copy.deepcopy(template)
    task={'output_dir':str(tmp_path),'seed':7,'selection':selection}
    pool=[case(i) for i in range(144)];write_rows(tmp_path/'pool.jsonl',pool);identity={'fixed':'screen'}
    for row in pool:
        atomic_write_json(tmp_path/'screen'/row['case_id']/'outcomes.json',
            {'identity':identity,'accepted':True,'doctor_opening':{'visible':'Explain your reasoning.','seed':42}})
    cfg=plan_b.finalize_dataset(task,template,ROOT,identity)
    assert template==original
    selected=load_cases(cfg['data']['train']); assert len(selected)==120
    ids=json.loads(Path(cfg['prompt_optimization']['optimization_case_ids_file']).read_text())
    assert len(set(ids))==24 and set(ids)<={row['case_id'] for row in selected}
    assert len(schedule(cfg,selected,ROOT))==90 and not config_problems(cfg,ROOT)
    policy=SimpleNamespace(original_base_only=True,doctor=SimpleNamespace(protocol='test-doctor'))
    calls=[]
    def collect(row,cfg,policy,cache,step,**kwargs):
        calls.append((row['case_id'],kwargs))
        return None,[{'outcome':i%2,'trajectory_id':str(i)} for i in range(32)]
    monkeypatch.setattr(prompt_search,'collect',collect)
    result=prompt_search.collect_base(cfg,policy,selected,ROOT)
    assert result['complete'] and result['completed_cases']==24 and result['total_cases']==24
    assert {key for key,_ in calls}==set(ids)
    assert all(options['fixed_opening']['seed']==42 and 'seed_override' in options for _,options in calls)
    assert len(list(Path(cfg['prompt_optimization']['output_dir']).glob('base_rollouts/*/rollouts.json')))==24
    calls.clear();prompt_search.collect_base(cfg,policy,selected,ROOT);assert not calls
    assessed_ids=[]
    monkeypatch.setattr(prompt_search,'reflections',lambda *args: [])
    def assess(row,*args):
        assessed_ids.append(row['case_id'])
        return {'valid':True,'auc':0.6,'scores':[],'generation':{'rubrics':[]}}
    monkeypatch.setattr(prompt_search,'assess',assess)
    backend=SimpleNamespace(identity=lambda:{},propose=lambda parent,*args:{'text':parent})
    final=prompt_search.optimize(cfg,policy,backend,selected,ROOT)
    assert len(final['optimization_case_ids'])==24 and set(assessed_ids)==set(ids)
    text,_=prompt_search.frozen_prompt(cfg,ROOT,policy=policy,backend=backend)
    assert text==Path(cfg['prompt_optimization']['initial_prompt_file']).read_text()
    Path(cfg['prompt_optimization']['optimization_case_ids_file']).write_text('[]')
    with pytest.raises(RuntimeError,match='case selection changed'):
        prompt_search.optimization_order(cfg,selected,ROOT)


def test_screen_uses_shared_eight_rollout_function_and_reuses_outcomes(tmp_path,monkeypatch):
    from src.online_rl.sherpa import rollouts
    calls=[]
    def collect(row,cfg,policy,cache,step,**kwargs):
        calls.append(kwargs)
        cache.call('doctor_turn',{'round':1,'path':''},lambda:{'visible':'Opening','seed':12})
        return None,[{'outcome':int(i<3)} for i in range(8)]
    monkeypatch.setattr(rollouts,'collect',collect)
    task={'seed':7,'min_successes':2};result=run_plan_b.screen_case(case(0),task,{},None,tmp_path,{'fixed':1})
    assert result['accepted'] and result['successes']==3
    assert calls[0]['branching']==8 and not calls[0]['behavior']
    assert result['doctor_opening']['seed']==12
    assert 'trajectories' not in result
    run_plan_b.screen_case(case(0),task,{},None,tmp_path,{'fixed':1});assert len(calls)==1


def test_busy_gpu_aborts_before_loading_models(tmp_path,monkeypatch):
    template=json.loads((ROOT/'configs/ours.json').read_text())
    write_rows(tmp_path/'pool.jsonl',[case(0)])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','planb_test_busy0,planb_test_busy1')
    task={'output_dir':str(tmp_path),'seed':7,'min_successes':2,'selection':'first120','case_workers':16}
    lock_path=Path('/tmp/sherpa_tree_physical_gpuplanb_test_busy0.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='current training is untouched'):
            run_plan_b.screen(task,template)


def test_shared_rollout_opening_override_does_not_change_default_protocol(monkeypatch):
    from src.online_rl.sherpa import rollouts
    from src.online_rl.tree.cache import CallCache
    from src.online_rl.integrated.records import sha256_json
    cfg=json.loads((ROOT/'configs/ours.json').read_text())
    policy=SimpleNamespace(original_base_only=True,doctor=SimpleNamespace(protocol='doctor',model='test'))
    captured=[]
    def build(row,**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(leaves=[SimpleNamespace(leaf_id=str(i),dialogue=[],outcome=1,
            final_option='A',stop_reason='done',path=[]) for i in range(kwargs['branching'])])
    monkeypatch.setattr(rollouts,'build_tree',build)
    rollouts.collect(case(0),cfg,policy,CallCache(None),0,behavior=False,branching=8)
    expected=sha256_json({'rollout':cfg['rollout'],'doctor':'doctor','base_model':cfg['policy']['base_model'],'sampling_step':0})
    assert captured[0]['protocol_sha256']==expected and 'opening' not in captured[0]
    opening={'visible':'Keep this opening','seed':42}
    rollouts.collect(case(0),cfg,policy,CallCache(None),0,behavior=False,fixed_opening=opening)
    assert captured[1]['opening'](None,99)==opening
    assert captured[1]['protocol_sha256']!=expected
