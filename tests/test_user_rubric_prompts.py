"""Explicit prompt revision without overwriting optimized evidence or API work."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.online_rl.integrated.records import sha256_text
from src.online_rl.sherpa.prompt_search import frozen_prompt
from src.online_rl.sherpa.backend import ChecklistBackend
from src.online_rl.tree.experiment import load_config

ROOT = Path(__file__).resolve().parents[1]


def source_artifact(tmp_path):
    (tmp_path/'original.txt').write_text('OPTIMIZED ORIGINAL')
    artifact = {'schema':'sherpa_auc_prompt_v1','test_used':False,
                'prompt_file':str(tmp_path/'original.txt'), 'prompt_sha256':sha256_text('OPTIMIZED ORIGINAL'),
                'base_identity':{'base_model':'fake'}, 'backend':{'judge':'original'}}
    (tmp_path/'source.json').write_text(json.dumps(artifact))
    return artifact


def test_explicit_revision_records_truthful_provenance_and_new_backend(tmp_path):
    source = source_artifact(tmp_path)
    raw = (tmp_path/'source.json').read_bytes()
    (tmp_path/'revised.txt').write_text('USER REVISION')
    cfg = {'prompt_optimization':{'final_artifact':'source.json'},
           'rubric_pipeline':{'generator_prompt_file':'revised.txt'}}
    backend = SimpleNamespace(identity=lambda:{'judge':'new'})
    text, artifact = frozen_prompt(cfg, tmp_path, backend=backend)
    assert text == 'USER REVISION' and artifact['prompt_sha256'] == sha256_text(text)
    assert artifact['prompt_source'] == 'explicit_user_revision' and artifact['auc_validated'] is False
    assert artifact['source_optimized_prompt_sha256'] == source['prompt_sha256']
    assert 'mean_auc' not in artifact and artifact['backend'] == {'judge':'new'}
    assert (tmp_path/'source.json').read_bytes() == raw
    del cfg['rubric_pipeline']['generator_prompt_file']
    with pytest.raises(RuntimeError, match='contract changed'):
        frozen_prompt(cfg, tmp_path, backend=backend)


def test_revision_never_falls_back_on_missing_empty_or_corrupt_source(tmp_path):
    source_artifact(tmp_path)
    cfg = {'prompt_optimization':{'final_artifact':'source.json'},
           'rubric_pipeline':{'generator_prompt_file':'revised.txt'}}
    with pytest.raises(FileNotFoundError):
        frozen_prompt(cfg, tmp_path)
    (tmp_path/'revised.txt').write_text(' ')
    with pytest.raises(RuntimeError, match='empty'):
        frozen_prompt(cfg, tmp_path)
    (tmp_path/'revised.txt').write_text('USER REVISION')
    (tmp_path/'original.txt').write_text('TAMPERED')
    with pytest.raises(RuntimeError, match='changed'):
        frozen_prompt(cfg, tmp_path)


def test_new_prompts_reach_generator_and_judge_with_existing_visibility(monkeypatch):
    cfg = load_config('configs/ours_group_mean.json')
    text, artifact = frozen_prompt(cfg, ROOT)
    assert text == (ROOT/'prompts/sherpa/trajectory_checklist_generator_v4.txt').read_text()
    assert artifact['prompt_source'] == 'explicit_user_revision'
    assert load_config('configs/ours.json')['rubric_pipeline'].get('generator_prompt_file') is None
    assert load_config('configs/plan_c.json')['rubric_pipeline']['judge_prompt_file'].startswith('prompts/plan_c/')
    backend = ChecklistBackend.__new__(ChecklistBackend)
    backend.cfg, backend.generator, backend.key, backend.settings = cfg, 'fake', 'fake', {}
    backend.judge_model, backend.judge_key, backend.judge_settings = 'fake','fake',{}
    backend.judge_prompt = (ROOT/cfg['rubric_pipeline']['judge_prompt_file']).read_text()
    calls = []
    rubrics = [{'name':f'check_{i}', 'situation':'always','behavior':'Is an assumption checked?'} for i in range(5)]
    def call(key, model, system, user, settings, schema, name, role):
        calls.append((system,json.loads(user),role))
        return json.dumps({'scores':[1,0,1,0,1]} if role == 'rubric_judge' else {'rubrics':rubrics}), {}
    monkeypatch.setattr('src.online_rl.sherpa.backend._generator_json',call)
    case = {'case_id':'fake','question':'Synthetic question','options':{'A':'alpha','B':'beta'},
            'ground_truth':{'correct_option':'A'}}
    generated = backend.generate(case, [], text)
    assert generated['valid'] and calls[0][0].startswith(text)
    candidate = {'trajectory_id':'t','outcome':1,'dialogue':[
        {'speaker':'agent','stage':'A1','round':1,'text':'Check the premise.'},
        {'speaker':'doctor','stage':'FINAL','round':0,'text':'SECRET FINAL'}]}
    result = backend.score(case, candidate, rubrics)
    assert result['reward'] == .6
    assert calls[1][0] == backend.judge_prompt
    assert 'Evidence rule:' in calls[1][0]
    assert 'SECRET FINAL' not in json.dumps(calls[1][1]) and 'outcome' not in calls[1][1]
